"""Regime-aware entry challenger: global fallback + trend/volatility experts."""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import app_config as settings
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from polybot.models.counterfactual_actions import action_vector
from polybot.models.market_regime import RegimeThresholds, classify, fit_thresholds, one_hot


def load(path: Path) -> list[dict[str, Any]]:
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    rows=db.execute(
        """SELECT c.*,t.features_json AS snapshot_features_json FROM counterfactual_action_examples c
           JOIN training_examples t ON t.snapshot_id=c.snapshot_id
           WHERE c.phase='entry' AND c.action LIKE 'BUY_%'
             AND ABS(c.candidate_notional_usdc-?)<0.001
           ORDER BY c.event_slug,c.observed_at,c.action""",
        (settings.PAPER_ENTRY_NOTIONAL_USDC,),
    ).fetchall();db.close();result=[]
    for row in rows:
        features=json.loads(row["snapshot_features_json"]);outcome=str(row["outcome"])
        result.append({"event":str(row["event_slug"]),"observed":str(row["observed_at"]),
                       "outcome":outcome,"filled":int(row["filled"]),"pnl":float(row["target_net_pnl_usdc"]),
                       "notional":float(row["candidate_notional_usdc"]),"features":features,
                       "base_x":action_vector(str(row["event_slug"]),outcome,str(row["observed_at"]),features,
                                              str(row["action"]),float(row["candidate_price"]),float(row["candidate_notional_usdc"]))})
    return result


def _matrix(rows,thresholds):
    regimes=[classify(r["features"],thresholds) for r in rows]
    x=np.asarray([r["base_x"]+one_hot(regime) for r,regime in zip(rows,regimes,strict=True)])
    return x,regimes


def _fit_models(rows):
    thresholds=fit_thresholds([r["features"] for r in rows]);x,regimes=_matrix(rows,thresholds)
    filled=np.asarray([r["filled"] for r in rows]);pnl=np.asarray([r["pnl"] for r in rows]);mask=filled==1
    def fit(index):
        local_filled=filled[index];local_x=x[index];local_pnl=pnl[index];local_mask=local_filled==1
        if len(local_x)<500 or len(np.unique(local_filled))<2 or local_mask.sum()<200:return None
        fill=HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=12,l2_regularization=5,random_state=42).fit(local_x,local_filled)
        value=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=12,l2_regularization=5,random_state=42).fit(local_x[local_mask],local_pnl[local_mask])
        severe=(local_pnl[local_mask]<=-settings.ACTION_TAIL_LOSS_FRACTION*np.asarray([rows[i]["notional"] for i in np.flatnonzero(index)])[local_mask]).astype(int)
        risk=(HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=12,l2_regularization=5,random_state=42).fit(local_x[local_mask],severe)
              if len(np.unique(severe))==2 else None)
        return {"fill":fill,"value":value,"risk":risk,"rows":int(len(local_x))}
    all_index=np.ones(len(rows),dtype=bool);global_model=fit(all_index);experts={}
    if global_model is None:
        raise ValueError(
            "Insufficient fill/non-fill diversity in the training fold for the global model"
        )
    # Отдельные ансамбли для девяти режимов слишком тяжёлы и нестабильны в редких
    # режимах. Используем общую нелинейную модель с режимными one-hot признаками,
    # а эксперты хранят только shrinkage-поправки по остаткам на train-части.
    base_fill=global_model["fill"].predict_proba(x)[:,1]
    base_value=global_model["value"].predict(x)
    for key in sorted({r["expert"] for r in regimes}):
        index=np.asarray([regime["expert"]==key for regime in regimes])
        filled_index=index & mask
        count=int(index.sum());filled_count=int(filled_index.sum())
        if count < 100 or filled_count < 50:continue
        shrink=count/(count+1000.0)
        experts[key]={
            "rows":count,
            "fill_delta":float(shrink*np.mean(filled[index]-base_fill[index])),
            "value_delta":float(shrink*np.mean(pnl[filled_index]-base_value[filled_index])),
        }
    return {"thresholds":thresholds,"global":global_model,"experts":experts,"feature_count":x.shape[1]}


def _predict(bundle,rows):
    x,regimes=_matrix(rows,bundle["thresholds"]);result=[]
    for index,(row,regime) in enumerate(zip(rows,regimes,strict=True)):
        vector=x[index:index+1];global_model=bundle["global"];expert=bundle["experts"].get(regime["expert"])
        def score(model):
            pfill=float(model["fill"].predict_proba(vector)[0,1]);ev=pfill*float(model["value"].predict(vector)[0]);
            ptail=float(model["risk"].predict_proba(vector)[0,1]) if model.get("risk") else 0.0
            return ev,ptail
        global_ev,global_tail=score(global_model)
        if expert:
            base_pfill=float(global_model["fill"].predict_proba(vector)[0,1])
            pfill=min(1.0,max(0.0,base_pfill+expert["fill_delta"]))
            value=float(global_model["value"].predict(vector)[0])+expert["value_delta"]
            ev=pfill*value;tail=global_tail
        else:ev,tail=global_ev,global_tail
        result.append((ev,tail,regime))
    return result


def _policy(rows,prediction,threshold,max_tail):
    grouped=defaultdict(lambda:defaultdict(list))
    for row,pred in zip(rows,prediction,strict=True):grouped[row["event"]][row["observed"]].append((row,*pred))
    pnls=[];directions=defaultdict(int);regimes=defaultdict(lambda:{"trades":0,"pnl":0.0});nonfills=0
    for event,times in grouped.items():
        selected=None
        for observed,candidates in sorted(times.items()):
            elapsed=datetime.fromisoformat(observed).timestamp()-int(event.rsplit('-',1)[-1])
            if not settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN<=elapsed<=300-settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:continue
            best=max(candidates,key=lambda item:item[1])
            if best[1]>=threshold and best[2]<=max_tail:selected=best;break
        if not selected:continue
        row,_,_,regime=selected
        if not row["filled"]:nonfills+=1;continue
        pnl=float(row["pnl"]);pnls.append(pnl);directions[row["outcome"]]+=1
        key=regime["expert"];regimes[key]["trades"]+=1;regimes[key]["pnl"]+=pnl
    losses=-sum(x for x in pnls if x<0);profit=sum(x for x in pnls if x>0)
    return {"trades":len(pnls),"net_pnl":sum(pnls),"expectancy":mean(pnls) if pnls else 0.0,
            "profit_factor":profit/losses if losses else (999.0 if profit else 0.0),"up":directions["Up"],"down":directions["Down"],
            "nonfills":nonfills,"regimes":dict(regimes)}


def run(path:Path,min_train=300,validation_events=100,fold_events=200):
    rows=load(path);events=list(dict.fromkeys(r["event"] for r in rows));folds=[]
    output=settings.REGIME_ENTRY_V5_DIR;output.mkdir(parents=True,exist_ok=True)
    for start in range(min_train+validation_events,len(events),fold_events):
        train_events=set(events[:start-validation_events]);validation=set(events[start-validation_events:start]);test_events=set(events[start:start+fold_events])
        train=[r for r in rows if r["event"] in train_events];valid=[r for r in rows if r["event"] in validation];test=[r for r in rows if r["event"] in test_events]
        bundle=_fit_models(train);vp=_predict(bundle,valid);tp=_predict(bundle,test);choices=[]
        candidate_thresholds = sorted(
            {float(settings.ACTION_VALUE_MIN_EXPECTED_PNL_USDC)}
            | {
                float(value)
                for value in np.quantile([item[0] for item in vp], np.linspace(0.45, 0.95, 12))
                if float(value) >= settings.ACTION_VALUE_MIN_EXPECTED_PNL_USDC
            }
        )
        for threshold in candidate_thresholds:
            for max_tail in (.35,.50,.75,1.0):
                metric=_policy(valid,vp,float(threshold),max_tail)
                if metric["trades"]>=20 and min(metric["up"],metric["down"])>=5:choices.append((metric["net_pnl"],metric["profit_factor"],threshold,max_tail))
        if choices:_,_,threshold,max_tail=max(choices)
        else:threshold,max_tail=99.0,0.0
        folds.append({"test_events":len(test_events),"threshold":float(threshold),"max_tail_probability":float(max_tail),
                      "experts":sorted(bundle["experts"]),**_policy(test,tp,float(threshold),float(max_tail))})
        (output/'partial_folds.json').write_text(json.dumps(folds,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"fold_done={len(folds)} pnl={folds[-1]['net_pnl']:.4f}", flush=True)
        del train,valid,test,bundle,vp,tp;gc.collect()
    full=_fit_models(rows)
    report={"version":"entry_value_v5_regime_experts","protocol":"expanding temporal walk-forward; train-only regime quantiles; global nonlinear model + shrinkage residual experts by trend/volatility; phase/distance features; GTD bid/mid/ask fill simulation and fees",
            "events":len(events),"folds":folds,"total_test_events":sum(f["test_events"] for f in folds),"total_trades":sum(f["trades"] for f in folds),
            "total_net_pnl":sum(f["net_pnl"] for f in folds),"positive_folds":sum(f["net_pnl"]>0 for f in folds),
            "up":sum(f["up"] for f in folds),"down":sum(f["down"] for f in folds),"experts":sorted(full["experts"]),
            "thresholds":full["thresholds"].to_dict()}
    report["promotion_gate"]={"passed":bool(len(folds)>=5 and report["positive_folds"]/len(folds)>=.8 and report["total_net_pnl"]>0 and min(report["up"],report["down"])>=30),
                              "candidate_only":True,"requirements":">=5 folds; >=80% positive; total PnL>0; >=30 Up and Down"}
    joblib.dump({"bundle":full,"report":report},output/'entry_value_v5_regime_experts.joblib')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--db',type=Path,required=True);args=parser.parse_args();print(json.dumps(run(args.db),ensure_ascii=False,indent=2))


if __name__=='__main__':
    from polybot.runtime import run_sync
    run_sync(__file__,main)
