/* V11: понятный разбор сделки с фактическим исходом и контрфактом HOLD. */
async function loadTrade() {
  const key = $('tradeSelect').value;
  if (!key) return;
  const d = await json(`/api/trade/${key}`);
  currentTradeKey = key;
  const p = d.position || {};
  const decision = d.decision || {};
  const target = d.target || {};
  const resolution = d.resolution || {};
  const orders = d.orders || [];
  const exitPlan = d.exit_plan || {};
  const delta = resolution.early_exit_vs_hold_usdc;
  const executed = orders.filter(order => Number(order.shares) > 0).length;
  const latency = orders.length
    ? Math.round(orders.reduce((sum, order) => sum + Number(order.latency_ms || 0), 0) / orders.length)
    : 0;

  $('tradeMeta').innerHTML =
    detail('Режим', String(d.source || '').toUpperCase()) +
    detail('Сторона', p.outcome || '—') +
    detail('Вход', `${money(resolution.original_cost_usdc ?? p.cost_usdc)} @ ${Number(p.average_price || 0).toFixed(3)}`) +
    detail('Финальный исход', resolution.winning_outcome
      ? `${resolution.winning_outcome} · ${resolution.selected_outcome_won ? 'позиция выиграла' : 'позиция проиграла'}`
      : 'ещё не рассчитан') +
    detail('Фактический net PnL', resolution.actual_pnl_usdc == null ? '—' : money(resolution.actual_pnl_usdc)) +
    detail('PnL при HOLD до конца', resolution.hold_to_resolution_pnl_usdc == null ? '—' : money(resolution.hold_to_resolution_pnl_usdc)) +
    detail('Ранний выход vs HOLD', !resolution.had_early_exit ? 'не применимо · позиция удержана до расчёта' : (delta == null ? 'ожидается расчёт события' : `${delta >= 0 ? '+' : ''}${money(delta)} · ${delta >= 0 ? 'сохранено' : 'недополучено'}`)) +
    detail('Исполнение', orders.length ? `${executed}/${orders.length} исполнено · avg ${latency} ms` : 'нет записей') +
    detail('Модель', decision.model_name || '—') +
    detail('Price to Beat', target.target_price ? `$${Number(target.target_price).toFixed(2)}` : '—') +
    detail('Уверенность выбранного направления', decision.confidence == null ? '—' : `${(Number(decision.confidence) * 100).toFixed(1)}% · ${Number(decision.confidence) < 0.20 ? 'слабая' : Number(decision.confidence) < 0.50 ? 'умеренная' : Number(decision.confidence) < 0.80 ? 'сильная' : 'очень сильная'}`) +
    detail('Как считается confidence', decision.predicted_up_probability == null ? 'модельная вероятность выбранного исхода относительно Price to Beat' : `P(Up) ${(Number(decision.predicted_up_probability) * 100).toFixed(1)}% · P(Down) ${(Number(decision.predicted_down_probability) * 100).toFixed(1)}%; для входа берётся вероятность выбранной стороны после калибровки и смешивания с рынком`) +
    detail('Пятиступенчатый выход', exitPlan.enabled ? `пройдено ${Number(exitPlan.completed_stage || 0)}/5 · выход только при сигнале риска и преимуществе CLOSE над HOLD` : 'выключен') +
    detail('Пороги риска по ступеням', (exitPlan.risk_probability_thresholds || []).map((v, i) => `${i + 1}: P(позиции)≤${(Number(v) * 100).toFixed(0)}%`).join(' · ') || '—') +
    detail('Открыта', p.opened_at ? new Date(p.opened_at).toLocaleString('ru-RU') : '—') +
    detail('Закрыта', p.closed_at ? new Date(p.closed_at).toLocaleString('ru-RU') : 'позиция открыта') +
    detail('Обоснование', decision.reason || '—');
  drawTrade(d);
}

function drawTrade(d) {
  const canvas = $('tradeChart');
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const dpr = devicePixelRatio || 1;
  const width = Math.max(620, rect.width);
  const height = 420;
  const left = 76;
  const right = 22;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '10px Segoe UI';

  const position = d.position || {};
  const resolution = d.resolution || {};
  const eventStart = d.target?.start_time ? +new Date(d.target.start_time) : null;
  const eventEnd = d.target?.end_time ? +new Date(d.target.end_time) : null;
  const target = Number(d.target?.target_price);
  const btc = (d.reference || []).filter(x => x.reference_price != null)
    .map(x => ({t: +new Date(x.collected_at), v: +x.reference_price}));
  const contract = (d.contract || []).filter(x => x.best_bid != null || x.midpoint != null)
    .map(x => ({t: +new Date(x.collected_at), v: +(x.best_bid ?? x.midpoint)}));
  const pnl = (d.pnl || []).map(x => ({t: +new Date(x.t), v: +x.v}));
  const holdPnl = (d.hold_pnl || []).map(x => ({t: +new Date(x.t), v: +x.v}));
  const times = [...btc, ...contract, ...pnl, ...holdPnl].map(x => x.t).filter(Number.isFinite);
  if (position.opened_at) times.push(+new Date(position.opened_at));
  if (resolution.had_early_exit && position.closed_at) times.push(+new Date(position.closed_at));
  if (Number.isFinite(eventStart)) times.push(eventStart);
  if (Number.isFinite(eventEnd)) times.push(eventEnd);
  if (!times.length) {
    ctx.fillStyle = '#8996a8';
    ctx.fillText('Для этой сделки временной ряд ещё не сохранён', 20, 40);
    return;
  }
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const x = time => left + (width - left - right) * (time - minTime) / Math.max(1, maxTime - minTime);
  const panels = [
    {points: btc, y: 28, h: 104, color: '#ffbe55', label: 'BTC / Price to Beat', target: Number.isFinite(target) ? target : null, format: v => `$${v.toFixed(0)}`},
    {points: contract, y: 158, h: 82, color: '#36d6c9', label: `Контракт ${position.outcome || ''}`, min: 0, max: 1, format: v => v.toFixed(2)},
    {points: pnl, secondary: holdPnl, y: 286, h: 92, color: '#a8ee64', secondaryColor: '#ffbe55', label: 'Фактический PnL / контрфакт HOLD, $', zero: 0, format: v => `$${v.toFixed(2)}`},
  ];

  for (const panel of panels) {
    const values = panel.points.map(point => point.v);
    if (panel.secondary) values.push(...panel.secondary.map(point => point.v));
    if (panel.target != null) values.push(panel.target);
    if (panel.zero != null) values.push(panel.zero);
    const low = panel.min ?? Math.min(...values, 0);
    const high = panel.max ?? Math.max(...values, 1);
    const range = Math.max(1e-9, high - low);
    const y = value => panel.y + panel.h - (value - low) / range * panel.h;
    ctx.fillStyle = '#101721';
    ctx.fillRect(left, panel.y, width - left - right, panel.h);
    ctx.strokeStyle = '#273443';
    ctx.strokeRect(left, panel.y, width - left - right, panel.h);
    ctx.fillStyle = '#a5afbd';
    ctx.fillText(panel.label, 6, panel.y + 12);
    ctx.fillText(panel.format(high), 6, panel.y + 28);
    ctx.fillText(panel.format(low), 6, panel.y + panel.h - 3);
    if (panel.zero != null && low <= 0 && high >= 0) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#596675';
      ctx.beginPath(); ctx.moveTo(left, y(0)); ctx.lineTo(width - right, y(0)); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (panel.target != null) {
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = '#70a7ff';
      ctx.beginPath(); ctx.moveTo(left, y(panel.target)); ctx.lineTo(width - right, y(panel.target)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#70a7ff';
      ctx.fillText(`TARGET ${panel.format(panel.target)}`, width - 150, y(panel.target) - 4);
    }
    if (panel.points.length) {
      ctx.strokeStyle = panel.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      panel.points.forEach((point, index) => index ? ctx.lineTo(x(point.t), y(point.v)) : ctx.moveTo(x(point.t), y(point.v)));
      ctx.stroke();
    }
    if (panel.secondary?.length) {
      ctx.setLineDash([7, 5]);
      ctx.strokeStyle = panel.secondaryColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      panel.secondary.forEach((point, index) => index ? ctx.lineTo(x(point.t), y(point.v)) : ctx.moveTo(x(point.t), y(point.v)));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  const markers = [
    [position.opened_at, '#70a7ff', 'ВХОД'],
    [resolution.had_early_exit ? position.closed_at : null, '#ff6f75', 'ВЫХОД'],
    [resolution.resolved_at, '#bd8cff', resolution.winning_outcome ? `ИСХОД ${resolution.winning_outcome}` : 'РАСЧЁТ'],
  ];
  markers.forEach(([time, color, label], index) => {
    if (!time) return;
    const px = x(+new Date(time));
    ctx.strokeStyle = color;
    ctx.beginPath(); ctx.moveTo(px, 18); ctx.lineTo(px, height - 28); ctx.stroke();
    ctx.fillStyle = color;
    const elapsed = Number.isFinite(eventStart) ? Math.max(0, (+new Date(time) - eventStart) / 1000) : null;
    ctx.fillText(`${label}${elapsed == null ? '' : ` · ${Math.floor(elapsed / 60)}:${String(Math.round(elapsed % 60)).padStart(2, '0')}`}`, Math.min(width - 150, Math.max(left + 2, px + 4)), 12 + index * 11);
  });
  (d.orders || []).filter(order => order.exit_stage && Number(order.shares) > 0).forEach(order => {
    const px = x(+new Date(order.created_at));
    ctx.strokeStyle = '#f08cff';
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, 145); ctx.lineTo(px, height - 28); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f08cff';
    ctx.fillText(`EXIT ${order.exit_stage}/5`, Math.min(width - 72, px + 3), 151);
  });
  ctx.fillStyle = '#8996a8';
  const tickStart = Number.isFinite(eventStart) ? eventStart : minTime;
  const tickEnd = Number.isFinite(eventEnd) ? eventEnd : maxTime;
  for (let i = 0; i <= 5; i++) {
    const tickTime = tickStart + (tickEnd - tickStart) * i / 5;
    const px = x(tickTime);
    ctx.fillText(`${i}:00`, Math.max(left, Math.min(width - right - 28, px - 12)), height - 8);
  }
  const delta = resolution.early_exit_vs_hold_usdc;
  $('tradeLegend').innerHTML =
    '<span><i style="background:#ffbe55"></i>BTC/reference</span>' +
    '<span><i style="background:#70a7ff"></i>Price to Beat и вход</span>' +
    '<span><i style="background:#36d6c9"></i>Цена контракта</span>' +
    '<span><i style="background:#a8ee64"></i>Mark-to-market PnL</span>' +
    '<span><i style="background:#ffbe55"></i>HOLD до финального исхода · пунктир</span>' +
    '<span><i style="background:#bd8cff"></i>Финальный исход</span>' +
    (!resolution.had_early_exit ? '<strong>Раннего выхода не было · сравнение с HOLD не применяется</strong>' : (delta == null ? '<strong>Результат vs HOLD появится после расчёта</strong>' : `<strong class="${delta >= 0 ? 'positive' : 'negative'}">Ранний выход vs HOLD: ${delta >= 0 ? '+' : ''}${money(delta)}</strong>`));
}
