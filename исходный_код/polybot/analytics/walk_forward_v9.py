"""Walk-forward v9: entry EV/tail risk -> GTD fill -> model CLOSE/HOLD -> net PnL."""

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
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from polybot.models.counterfactual_actions import action_vector
from polybot.models.exit_features import feature_map, vector as exit_vector
from polybot.trading.fees import total_fee_usdc


def _time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _load(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    db = sqlite3.connect(path); db.row_factory = sqlite3.Row
    entries = []
    rows = db.execute(
        """SELECT c.*,t.features_json FROM counterfactual_action_examples c
           JOIN training_examples t ON t.snapshot_id=c.snapshot_id
           WHERE c.phase='entry' AND c.action LIKE 'BUY_%'
             AND ABS(c.candidate_notional_usdc-?)<0.001
           ORDER BY c.event_slug,c.observed_at,c.action,c.candidate_price""",
        (settings.PAPER_ENTRY_NOTIONAL_USDC,),
    ).fetchall()
    for row in rows:
        features = json.loads(row["features_json"])
        outcome = str(row["outcome"])
        entries.append({
            "event": str(row["event_slug"]), "observed": str(row["observed_at"]),
            "outcome": outcome, "price": float(row["candidate_price"]),
            "fill_price": float(row["fill_price"] or row["candidate_price"]),
            "filled": int(row["filled"]), "pnl": float(row["target_net_pnl_usdc"]),
            "notional": float(row["candidate_notional_usdc"]), "features": features,
            "x": action_vector(str(row["event_slug"]), outcome, str(row["observed_at"]), features,
                               str(row["action"]), float(row["candidate_price"]), float(row["candidate_notional_usdc"])),
        })
    exit_pairs = []
    holds = db.execute(
        """SELECT * FROM action_counterfactuals WHERE action='HOLD' AND horizon_seconds>=300
           AND status='resolved' AND net_pnl_usdc IS NOT NULL ORDER BY event_slug,observed_at"""
    ).fetchall()
    for hold in holds:
        close = db.execute(
            """SELECT net_pnl_usdc FROM action_counterfactuals WHERE decision_id=? AND action='CLOSE'
               AND status='evaluated' AND net_pnl_usdc IS NOT NULL ORDER BY id LIMIT 1""",
            (hold["decision_id"],),
        ).fetchone()
        if close is None: continue
        state = json.loads(hold["features_json"])
        values = feature_map(state, str(hold["outcome"]), hold["current_bid"], hold["shares"], hold["cost_usdc"])
        exit_pairs.append({
            "event": str(hold["event_slug"]), "outcome": str(hold["outcome"]),
            "x": exit_vector(values), "advantage": float(close[0])-float(hold["net_pnl_usdc"]),
        })
    db.close(); return entries, exit_pairs


def _load_future(path: Path, events: set[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not events: return {}
    db=sqlite3.connect(path); future=defaultdict(list)
    placeholders=','.join('?' for _ in events)
    for row in db.execute(
        f"SELECT event_slug,outcome,observed_at,features_json FROM training_examples WHERE event_slug IN ({placeholders}) ORDER BY observed_at",
        tuple(events),
    ):
        features=json.loads(row[3])
        if features.get('best_bid') is not None:
            future[(str(row[0]),str(row[1]))].append({'observed':str(row[2]),'features':features})
    db.close();return future


def _fit_entry(rows: list[dict[str, Any]]):
    x=np.asarray([r["x"] for r in rows]); filled=np.asarray([r["filled"] for r in rows]); pnl=np.asarray([r["pnl"] for r in rows])
    fill=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=4,random_state=42).fit(x,filled)
    mask=filled==1
    value=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,l2_regularization=4,random_state=42).fit(x[mask],pnl[mask])
    tail=(pnl[mask] <= -settings.ACTION_TAIL_LOSS_FRACTION*np.asarray([r["notional"] for r in rows])[mask]).astype(int)
    risk=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=5,random_state=42).fit(x[mask],tail)
    return fill,value,risk


def _fit_exit(rows: list[dict[str, Any]]):
    if len({r["event"] for r in rows}) < 60: return None
    x=np.asarray([r["x"] for r in rows]); y=np.asarray([r["advantage"] for r in rows]); label=(y>=settings.EXIT_VALUE_MARGIN_USDC).astype(int)
    if len(np.unique(label))<2:return None
    classifier=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=5,random_state=42).fit(x,label)
    value=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,l2_regularization=5,random_state=42).fit(x,y)
    return classifier,value


def _scores(models, rows):
    fill,value,risk=models; x=np.asarray([r["x"] for r in rows])
    pfill=fill.predict_proba(x)[:,1]; ev=pfill*value.predict(x); ptail=risk.predict_proba(x)[:,1]
    tail_exposure=pfill*ptail*np.asarray([r["notional"] for r in rows])
    return ev,ptail,tail_exposure


def _ranked(rows, scores, tails):
    grouped=defaultdict(list)
    for row,score,tail in zip(rows,scores,tails,strict=True): grouped[row["event"]].append((row,float(score),float(tail)))
    ranked={}
    for event,candidates in grouped.items():
        by_time=defaultdict(list)
        for item in candidates:by_time[item[0]["observed"]].append(item)
        ranked[event]=[max(items,key=lambda z:z[1]) for _,items in sorted(by_time.items())]
    return ranked


def _policy(ranked, threshold, max_tail_probability, exit_models, future):
    pnls=[]; hold_pnls=[]; directions=defaultdict(int); closes=0; nonfills=0
    for event, candidates in ranked.items():
        chosen=None
        for row,score,tail in candidates:
            elapsed=_time(row["observed"])-int(event.rsplit('-',1)[-1])
            if not settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN<=elapsed<=300-settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:continue
            if score>=threshold and tail<=max_tail_probability: chosen=(row,score,tail);break
        if chosen is None:continue
        row=chosen[0]
        if not row["filled"]: nonfills+=1;continue
        hold=float(row["pnl"]); realized=hold
        if exit_models:
            classifier,value=exit_models; shares=row["notional"]/row["fill_price"]
            for tick in future.get((event,row["outcome"]),[]):
                if _time(tick["observed"])<=_time(row["observed"]):continue
                bid=tick["features"].get("best_bid")
                if bid is None:continue
                values=feature_map(tick["features"],row["outcome"],bid,shares,row["notional"])
                x=np.asarray([exit_vector(values)])
                if classifier.predict_proba(x)[0,1]>=0.65 and value.predict(x)[0]>=settings.EXIT_VALUE_MARGIN_USDC:
                    realized=shares*float(bid)-row["notional"]-total_fee_usdc(shares,float(bid));closes+=1;break
        pnls.append(realized);hold_pnls.append(hold);directions[row["outcome"]]+=1
    return {"trades":len(pnls),"net_pnl":sum(pnls),"expectancy":mean(pnls) if pnls else 0.0,
            "hold_net_pnl":sum(hold_pnls),"exit_vs_hold":sum(pnls)-sum(hold_pnls),"early_closes":closes,
            "up":directions["Up"],"down":directions["Down"],"nonfills":nonfills}


def run(path:Path,min_train:int=300,validation_events:int=100,fold_events:int=100,resume:bool=True):
    entries,exit_rows=_load(path); events=list(dict.fromkeys(r["event"] for r in entries)); folds=[]
    if resume and settings.WALK_FORWARD_V9_REPORT_PATH.exists():
        try:
            previous=json.loads(settings.WALK_FORWARD_V9_REPORT_PATH.read_text(encoding='utf-8'))
            folds=list(previous.get('folds') or [])
        except (OSError,json.JSONDecodeError):pass
    starts=list(range(min_train+validation_events,len(events),fold_events))
    for start in starts[len(folds):]:
        train_events=set(events[:start-validation_events]); valid_events=set(events[start-validation_events:start]); test_events=set(events[start:start+fold_events])
        train=[r for r in entries if r["event"] in train_events]; valid=[r for r in entries if r["event"] in valid_events]; test=[r for r in entries if r["event"] in test_events]
        models=_fit_entry(train); vbase,vt,vexposure=_scores(models,valid); tbase,tt,texposure=_scores(models,test)
        choices=[]
        for penalty in (0.0,0.10,0.25,0.50,0.75,1.0):
            valid_ranked=_ranked(valid,vbase-penalty*vexposure,vt)
            for max_tail in (0.35,0.50,0.75,1.0):
                for threshold in np.arange(-0.5,1.51,0.1):
                    m=_policy(valid_ranked,float(threshold),max_tail,None,{})
                    if m["trades"]>=20 and min(m["up"],m["down"])>=5:
                        choices.append((m["net_pnl"],m["expectancy"],penalty,max_tail,threshold))
        if choices:
            _,_,penalty,max_tail,threshold=max(choices)
        else:
            penalty,max_tail,threshold=0.0,1.0,2.0
        test_ranked=_ranked(test,tbase-penalty*texposure,tt)
        exit_models=_fit_exit([r for r in exit_rows if r["event"] in train_events])
        future=_load_future(path,test_events) if exit_models else {}
        folds.append({"test_events":len(test_events),"threshold":float(threshold),
                      "tail_penalty":float(penalty),"max_tail_probability":float(max_tail),
                      **_policy(test_ranked,float(threshold),float(max_tail),exit_models,future)})
        settings.WALK_FORWARD_V9_REPORT_PATH.write_text(json.dumps({"status":"running","folds":folds},ensure_ascii=False,indent=2),encoding="utf-8")
        del train,valid,test,models,valid_ranked,test_ranked,exit_models,future
        gc.collect()
    report={"version":"entry_value_v4_tail_risk_full_chain","protocol":"expanding temporal walk-forward; GTD labels; fees; tail gate; future-book CLOSE/HOLD; untouched test folds",
            "events":len(events),"folds":folds,"total_test_events":sum(f["test_events"] for f in folds),
            "total_trades":sum(f["trades"] for f in folds),"total_net_pnl":sum(f["net_pnl"] for f in folds),
            "total_hold_pnl":sum(f["hold_net_pnl"] for f in folds),"exit_vs_hold":sum(f["exit_vs_hold"] for f in folds),
            "positive_folds":sum(f["net_pnl"]>0 for f in folds)}
    report["promotion_gate"]={"passed":bool(len(folds)>=5 and report["positive_folds"]/len(folds)>=.8 and report["total_net_pnl"]>0 and report["exit_vs_hold"]>=0 and min(sum(f["up"] for f in folds),sum(f["down"] for f in folds))>=30),
                              "candidate_only":True,"requirements":">=5 folds; >=80% positive; total PnL>0; exit>=HOLD; >=30 Up and Down"}
    settings.WALK_FORWARD_V9_REPORT_PATH.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--db',type=Path,default=settings.DATABASE_PATH);args=parser.parse_args();print(json.dumps(run(args.db),ensure_ascii=False,indent=2))


if __name__=='__main__':
    from polybot.runtime import run_sync
    run_sync(__file__,main)
