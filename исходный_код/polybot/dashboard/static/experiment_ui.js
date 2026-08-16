/* PAPER-only value experiment UI and request de-duplication. */
(() => {
  const originalRefresh = window.refresh;
  let refreshInFlight = null;
  if (typeof originalRefresh === 'function') {
    window.refresh = function coordinatedRefresh() {
      if (refreshInFlight) return refreshInFlight;
      refreshInFlight = Promise.resolve(originalRefresh()).finally(() => { refreshInFlight = null; });
      return refreshInFlight;
    };
  }

  const originalRenderOverview = window.renderOverview;
  if (typeof originalRenderOverview !== 'function') return;
  window.renderOverview = function renderOverviewWithValuePipeline(data) {
    originalRenderOverview(data);
    const value = data?.trading?.action_value || {};
    const panel = document.getElementById('actionValueArchitecture');
    if (!panel) return;
    const number = (x, digits = 3) => x == null ? '—' : Number(x).toFixed(digits);
    panel.innerHTML = `
      <strong>2 · ${value.name || 'Action-value gate'}</strong>
      <span>${value.formula || 'P(fill) × E(net PnL | fill)'}</span>
      <div class="metric-line"><small>Fill ROC-AUC <b>${number(value.fill_roc_auc)}</b></small><small>Test PnL <b>${value.test_pnl_usdc == null ? '—' : '$' + Number(value.test_pnl_usdc).toFixed(2)}</b></small><small>CI lower <b>${value.test_ci_lower_usdc == null ? '—' : '$' + Number(value.test_ci_lower_usdc).toFixed(2)}</b></small></div>
      <small class="${value.promotion_passed ? '' : 'experiment-warning'}">${value.promotion_passed ? 'Promotion gate пройден' : 'PAPER-эксперимент · promotion gate не пройден · LIVE заблокирован'}</small>`;
    const policy = data?.ml_policy || {};
    const policyMetrics = policy.metrics || {};
    if (policy.enabled) {
      panel.innerHTML += `<hr><strong>ML-policy · автономный выбор действий</strong>
        <span>WAIT / BUY_UP / BUY_DOWN × bid/mid/ask × $1/$3/$5/$10; exit-модель выбирает HOLD/CLOSE</span>
        <div class="metric-line"><small>Решений <b>${policyMetrics.decisions || 0}</b></small><small>Исполнено <b>${policyMetrics.executed || 0}</b></small><small>Avg utility gap <b>${number(policyMetrics.average_utility_gap, 4)}</b></small><small>Avg confidence <b>${(100 * Number(policyMetrics.average_confidence || 0)).toFixed(1)}%</b></small></div>
        <div class="metric-line"><small>Оценённых сделок <b>${policyMetrics.quality?.evaluated_trades || 0}</b></small><small>Utility MAE <b>${number(policyMetrics.quality?.utility_mae_usdc, 3)}</b></small><small>Utility bias <b>${number(policyMetrics.quality?.utility_bias_usdc, 3)}</b></small><small>Знак PnL угадан <b>${policyMetrics.quality?.utility_sign_accuracy == null ? '—' : (100 * policyMetrics.quality.utility_sign_accuracy).toFixed(1) + '%'}</b></small></div>
        <small>Ручные confidence/edge/time/spread/tail/ступенчатые торговые gates отключены. Ограничения капитала и валидности данных сохранены.</small>`;
    }
    const v4 = value.candidate_v4 || {};
    if (v4.version) {
      panel.innerHTML += `<hr><strong>Challenger · ${v4.version}</strong>
        <span>EV − tail-risk penalty · Up/Down ${v4.test_up ?? '—'}/${v4.test_down ?? '—'}</span>
        <div class="metric-line"><small>Test PnL <b>${v4.test_pnl_usdc == null ? '—' : '$' + Number(v4.test_pnl_usdc).toFixed(2)}</b></small><small>Tail P <b>${v4.tail_probability_mean == null ? '—' : (100 * Number(v4.tail_probability_mean)).toFixed(1) + '%'}</b></small><small>Tail loss <b>${v4.expected_tail_loss_mean_usdc == null ? '—' : '$' + Number(v4.expected_tail_loss_mean_usdc).toFixed(2)}</b></small></div>
        <small class="${v4.promotion_passed ? '' : 'experiment-warning'}">${v4.promotion_passed ? 'promotion gate пройден' : 'candidate only · gate не пройден'}</small>`;
    }
    const walk = data?.models?.walk_forward_v9 || {};
    const regime = data?.models?.regime_entry_v5 || {};
    if (walk.folds?.length) {
      const grid = document.getElementById('modelHealthGrid');
      if (grid) {
        const card = document.createElement('div'); card.className = 'health-model-card';
        card.innerHTML = `<strong>Walk-forward v9 · full chain</strong><span>${walk.total_test_events || 0} OOS событий · ${walk.total_trades || 0} сделок · PnL $${Number(walk.total_net_pnl || 0).toFixed(2)}</span><span>Early exit vs HOLD $${Number(walk.exit_vs_hold || 0).toFixed(2)} · положительных фолдов ${walk.positive_folds || 0}/${walk.folds.length}</span><span class="${walk.promotion_gate?.passed ? '' : 'experiment-warning'}">${walk.promotion_gate?.passed ? 'gate passed' : 'candidate only · full-chain gate не пройден'}</span>`;
        grid.prepend(card);
      }
    }
    if (regime.folds?.length) {
      const grid = document.getElementById('modelHealthGrid');
      if (grid) {
        const card = document.createElement('div');
        card.className = 'health-model-card';
        const passed = Boolean(regime.promotion_gate?.passed);
        card.innerHTML = `<strong>Entry v5 · рыночные режимы · PAPER challenger</strong><span>${regime.total_test_events || 0} OOS событий · ${regime.total_trades || 0} сделок · PnL $${Number(regime.total_net_pnl || 0).toFixed(2)}</span><span>Up/Down ${regime.up || 0}/${regime.down || 0} · положительных фолдов ${regime.positive_folds || 0}/${regime.folds.length}</span><span>Режимы: тренд/флэт × волатильность; фаза окна и расстояние до Price to Beat входят в признаки</span><span class="${passed ? '' : 'experiment-warning'}">${passed ? 'promotion gate пройден' : 'candidate only · gate не пройден, активная PAPER-модель не изменена'}</span>`;
        grid.prepend(card);
      }
    }
    const liveButton = document.getElementById('modeToggle');
    if (liveButton && value.paper_experiment) {
      liveButton.disabled = false;
      liveButton.title = 'Ручное переключение доступно после технического LIVE-preflight; shadow-эксперимент продолжится';
    }
  };
})();
