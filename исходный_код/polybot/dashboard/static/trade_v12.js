/* V12: отложенный подробный replay. Ничего не вычисляет в торговом процессе. */
function tradePoints(rows, timeKey, valueKey) {
  return (rows || []).map(row => ({t: +new Date(row[timeKey]), v: Number(row[valueKey])}))
    .filter(point => Number.isFinite(point.t) && Number.isFinite(point.v));
}

function drawReplayChart(canvasId, series, options = {}, markers = []) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(560, rect.width || 560), height = options.height || 250;
  const pad = {left: 66, right: 22, top: 25, bottom: 34};
  canvas.width = width * dpr; canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  ctx.font = '10px Segoe UI';
  const points = series.flatMap(item => item.points || []);
  const configuredStart = options.start ? +new Date(options.start) : NaN;
  const configuredEnd = options.end ? +new Date(options.end) : NaN;
  const minT = Number.isFinite(configuredStart) ? configuredStart : Math.min(...points.map(x => x.t));
  const maxT = Number.isFinite(configuredEnd) ? configuredEnd : Math.max(...points.map(x => x.t));
  if (!points.length || !Number.isFinite(minT) || !Number.isFinite(maxT)) {
    ctx.fillStyle = '#8996a8'; ctx.fillText('Для этого графика данных пока нет', 20, 42); return;
  }
  let values = points.map(x => x.v);
  if (Number.isFinite(options.reference)) values.push(Number(options.reference));
  if (Number.isFinite(options.zero)) values.push(Number(options.zero));
  let minV = options.min ?? Math.min(...values), maxV = options.max ?? Math.max(...values);
  if (minV === maxV) { minV -= .5; maxV += .5; }
  const margin = options.fixedRange ? 0 : (maxV - minV) * .08;
  minV -= margin; maxV += margin;
  const x = t => pad.left + (width - pad.left - pad.right) * (t - minT) / Math.max(1, maxT - minT);
  const y = v => pad.top + (height - pad.top - pad.bottom) * (maxV - v) / Math.max(1e-9, maxV - minV);
  ctx.fillStyle = '#0a0f16'; ctx.fillRect(pad.left, pad.top, width - pad.left - pad.right, height - pad.top - pad.bottom);
  for (let i = 0; i <= 4; i++) {
    const py = pad.top + (height - pad.top - pad.bottom) * i / 4;
    ctx.strokeStyle = '#222c39'; ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke();
    const value = maxV - (maxV - minV) * i / 4;
    ctx.fillStyle = '#8996a8'; ctx.fillText(options.format ? options.format(value) : value.toFixed(2), 4, py + 3);
  }
  for (let i = 0; i <= 5; i++) {
    const t = minT + (maxT - minT) * i / 5, px = x(t);
    ctx.fillStyle = '#8996a8'; ctx.fillText(new Date(t).toLocaleTimeString('ru-RU', {hour:'2-digit', minute:'2-digit', second:'2-digit'}), Math.max(pad.left, Math.min(width - 70, px - 27)), height - 10);
  }
  if (Number.isFinite(options.reference)) {
    ctx.setLineDash([6, 4]); ctx.strokeStyle = '#70a7ff'; ctx.beginPath();
    ctx.moveTo(pad.left, y(options.reference)); ctx.lineTo(width - pad.right, y(options.reference)); ctx.stroke(); ctx.setLineDash([]);
  }
  series.forEach(item => {
    if (!item.points?.length) return;
    ctx.setLineDash(item.dash || []); ctx.strokeStyle = item.color; ctx.lineWidth = item.width || 1.8; ctx.beginPath();
    item.points.forEach((point, index) => index ? ctx.lineTo(x(point.t), y(point.v)) : ctx.moveTo(x(point.t), y(point.v)));
    ctx.stroke(); ctx.setLineDash([]);
    if (item.scatter) item.points.forEach(point => {ctx.fillStyle=item.color;ctx.beginPath();ctx.arc(x(point.t),y(point.v),4,0,Math.PI*2);ctx.fill();});
  });
  markers.forEach(marker => {
    const t = +new Date(marker.t); if (!Number.isFinite(t) || t < minT || t > maxT) return;
    const px = x(t); ctx.strokeStyle = marker.color || '#bd8cff'; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, height-pad.bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = marker.color || '#bd8cff'; ctx.fillText(marker.label || '', Math.min(width-120, px+3), pad.top+11+(marker.row||0)*11);
  });
}

async function loadTrade() {
  const key = $('tradeSelect').value;
  if (!key) return;
  const d = await json(`/api/trade/${key}`); currentTradeKey = key;
  const p = d.position || {}, target = d.target || {}, r = d.resolution || {}, fees = d.fees || {};
  const decision = d.decision || {}, orders = d.orders || [];
  const delta = r.early_exit_vs_hold_usdc;
  $('tradeMeta').innerHTML =
    `<div class="detail"><span>Событие</span><strong>${target.event_url ? `<a class="event-link" href="${esc(target.event_url)}" target="_blank" rel="noopener">${esc(target.event_slug)}</a>` : esc(p.event_slug || '—')}</strong></div>` +
    detail('Режим', String(d.source || '').toUpperCase()) + detail('Сторона', p.outcome || '—') +
    detail('Вход', `${money(r.original_cost_usdc ?? p.cost_usdc)} @ ${Number(p.average_price || 0).toFixed(3)}`) +
    detail('Выход', p.close_price == null ? 'HOLD/не закрыта' : `${Number(p.close_price).toFixed(3)} · ${p.exit_timing || p.close_reason || ''}`) +
    detail('Комиссия входа', money(fees.entry_usdc || 0)) + detail('Комиссия выхода', money(fees.exit_usdc || 0)) +
    detail('Комиссии всего', `${money(fees.total_usdc || 0)} · ${fees.source || 'ledger'}`) +
    detail('Финальный исход', r.winning_outcome ? `${r.winning_outcome} · ${r.selected_outcome_won ? 'WIN' : 'LOSS'}` : 'ещё не рассчитан') +
    detail('Фактический net PnL', r.actual_pnl_usdc == null ? '—' : money(r.actual_pnl_usdc)) +
    detail('PnL при HOLD', r.hold_to_resolution_pnl_usdc == null ? '—' : money(r.hold_to_resolution_pnl_usdc)) +
    detail('Выход vs HOLD', !r.had_early_exit ? 'раннего выхода не было' : delta == null ? 'ждём исход' : `${delta >= 0 ? '+' : ''}${money(delta)}`) +
    detail('Entry-модель', decision.model_name || '—') + detail('Причина входа', decision.reason || '—');

  const start = target.start_time, end = target.end_time;
  const markers = [
    {t:p.opened_at,label:`ВХОД ${p.outcome || ''} @${Number(p.average_price||0).toFixed(3)}`,color:'#70a7ff'},
    {t:r.had_early_exit ? p.closed_at : null,label:'РАННИЙ ВЫХОД',color:'#ff6f75',row:1},
    {t:r.resolved_at,label:r.winning_outcome ? `ИСХОД ${r.winning_outcome}` : 'КОНЕЦ',color:'#bd8cff',row:2},
  ].filter(x => x.t);
  const btc = tradePoints(d.reference, 'collected_at', 'reference_price');
  drawReplayChart('tradeBtcChart', [{name:'BTC',points:btc,color:'#ffbe55'}], {start,end,reference:Number(target.target_price),format:v=>`$${v.toFixed(0)}`}, markers);

  const market = d.market || [];
  const marketSeries = [
    ['Up bid','Up','best_bid','#36d6c9',[]],['Up ask','Up','best_ask','#36d6c9',[4,3]],
    ['Down bid','Down','best_bid','#ffbe55',[]],['Down ask','Down','best_ask','#ffbe55',[4,3]],
  ].map(([name,outcome,key,color,dash]) => ({name,color,dash,points:tradePoints(market.filter(x=>x.outcome===outcome),'collected_at',key)}));
  const orderSeries = orders.filter(x=>Number(x.requested_price)>0).map((order,index)=>({
    name:order.action, color:String(order.action).toUpperCase().includes('BUY')?'#70a7ff':'#ff6f75', scatter:true,
    points:[{t:+new Date(order.fill_observed_at || order.created_at),v:Number(order.filled_price ?? order.requested_price)}]
  }));
  drawReplayChart('tradeMarketChart',[...marketSeries,...orderSeries],{start,end,min:0,max:1,fixedRange:true,format:v=>v.toFixed(2)},markers);

  const decisions = d.decisions || [], entry = decisions.filter(x=>x.role==='entry'), exit = decisions.filter(x=>x.role==='exit');
  const decisionSeries = rows => [
    {name:'P Up',color:'#36d6c9',points:tradePoints(rows,'observed_at','predicted_up_probability')},
    {name:'P Down',color:'#ffbe55',points:tradePoints(rows,'observed_at','predicted_down_probability')},
    {name:'Confidence',color:'#bd8cff',dash:[5,3],points:tradePoints(rows,'observed_at','confidence')},
  ];
  const actionMarkers = rows => rows.filter(x=>x.executed || !['WAIT','HOLD'].includes(String(x.action).toUpperCase())).map((x,i)=>({t:x.observed_at,label:`${x.action}${x.executed?' · EXEC':''}`,color:x.executed?'#a8ee64':'#70a7ff',row:i%3}));
  drawReplayChart('tradeEntryConfidenceChart',decisionSeries(entry),{start,end,min:0,max:1,fixedRange:true,format:v=>`${(v*100).toFixed(0)}%`},actionMarkers(entry));
  drawReplayChart('tradeExitConfidenceChart',decisionSeries(exit),{start,end,min:0,max:1,fixedRange:true,format:v=>`${(v*100).toFixed(0)}%`},actionMarkers(exit));
  drawReplayChart('tradePnlChart',[
    {name:'Факт',color:'#a8ee64',points:tradePoints(d.pnl,'t','v')},
    {name:'HOLD',color:'#ffbe55',dash:[7,5],points:tradePoints(d.hold_pnl,'t','v')},
  ],{start,end,zero:0,format:v=>money(v)},markers);
  $('tradeLegend').innerHTML = '<span><i style="background:#36d6c9"></i>Up</span><span><i style="background:#ffbe55"></i>Down / HOLD</span><span><i style="background:#70a7ff"></i>Вход</span><span><i style="background:#ff6f75"></i>Выход</span><span><i style="background:#bd8cff"></i>Confidence / исход</span>';
}
