# custom_entry_v25 live-down champion v2

Дата фиксации: 2026-09-07. Это снимок работающей LIVE-конфигурации, а не новое обучение.

## Что зафиксировано

- Entry: `custom`, веса `custom_entry_v25`.
- Exit selector: `catboost`; фактическое early-exit execution в этой конфигурации выключено.
- Strategy: `custom_entry_v25_exit_shadow_only_counterfactual_v21`.
- Режим на момент снимка: LIVE, entry-сторона Down.
- Веса и сырые PAPER/LIVE events хранятся в `модели/архив_версий/experiments/custom_entry_v25_live_down_champion_v2_20260907T173544Z`.

## Контрольные SHA-256

- custom entry: `98b8afa4124a79d0a10614c23ff8ab44b74e1f9a811d4f59941a636d0728e119`
- catboost model: `efdaf8dec09525fb8f1bcc0cec8652eea6e6f3b5d931eb5e43996ae0624188ca`
- catboost bundle: `615dd0e2c298d8deda2b2873b62f7f733ad08c1bf1d515bb475d73f1837b3f65`

## Как восстановить

1. Клонировать ветку `best-models`.
2. Скопировать артефакты из папки `artifacts` в пути, указанные `app_config.py`, или использовать уже версионированные копии.
3. Создать `api_config.py` из `api_config.example.py`; секреты намеренно не хранятся в Git.
4. Запустить `запустить_проект.py`.
5. Сверить SHA-256 весов с `snapshot.json` до включения LIVE.

## Важное ограничение снимка

На момент снимка CLOB V2 возвращал `fee_rate_bps=0` в trade history, хотя конкретный BTC 5m market имел `feeSchedule={rate: 0.07, exponent: 1, takerOnly: true}`. Поэтому PnL в сырых JSONL до reconciliation завышен на taker fee. В код добавлена сверка по feeSchedule; перед любым promotion-решением результаты нужно пересчитать net after fees.

Контрольный пример: fill `5 @ 0.500 = $2.50`, taker fee `$0.0875`, полная стоимость `$2.5875`, эффективная цена `0.5175`.

## План улучшения Up без риска для champion

Текущий Down-champion не меняется. Up готовится как отдельный challenger:

- признаки канонизируются относительно кандида: signed distance до Price to Beat, signed returns, momentum, spread/depth его контракта;
- зеркальная аугментация допустима только с реальной ценой противоположного контракта и пересчитанным net PnL; label нельзя просто механически перевернуть;
- train/calibration/test разбиваются по времени и event slug, чтобы Up/Down одного события не разошлись по фолдам;
- калибровка, value/PnL, PR-AUC, Brier, max drawdown и Up-only метрики считаются отдельно;
- challenger сначала работает в shadow на тех же событиях. Подключение Up не меняет Down-решения.

## План exit-challenger

Для каждого тика после фактического entry нужно сравнивать три контрфакта по исполнимому bid и после комисии: `EXIT_NOW`, `EXIT_LATER`, `HOLD_TO_RESOLUTION`. В признаки входят remaining time, signed distance до target, скорость/ускорение, volatility, bid spread/depth/imbalance, entry price, MFE/MAE и drawdown from peak.

Первая версия разрешает только полный exit и только в shadow. Promotion возможен, если на walk-forward он улучшает net PnL и max drawdown против HOLD, а precision редких exit-сигналов проходит заданный gate.
