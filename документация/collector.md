# Collect-only pipeline

The pipeline uses only public `GET` APIs and public WebSockets. It does not sign, create, cancel, or modify orders.

```powershell
.\.venv\Scripts\python.exe .\запуск\collect_only_pipeline.py --event "https://polymarket.com/event/btc-updown-5m-1785756900" --duration 60
.\.venv\Scripts\python.exe .\запуск\validate_collector.py
```

SQLite data is saved in `данные/market_data.sqlite3` with WAL enabled.

Run continuously and let the collector find the next BTC 5-minute market:

```powershell
.\запуск\start_continuous_collector.ps1
```

Or stop automatically after one hour:

```powershell
.\.venv\Scripts\python.exe .\запуск\continuous_btc_collector.py --max-seconds 3600
```

Resolved outcomes are converted to local supervised examples by the continuous collector after a market closes. You can also run this step manually:

```powershell
.\.venv\Scripts\python.exe .\запуск\label_training_data.py
```

- `events`, `markets`: Gamma event metadata, outcomes, and CLOB token IDs.
- `market_snapshots`: Polymarket best bid/ask, midpoint, spread, and raw order book.
- `external_prices`: Bybit, OKX, Pyth, and Chainlink RTDS updates when available.
- `raw_messages`: replayable raw public WebSocket messages.
- `collector_runs`: run state for operational checks.
- `training_examples`: features from an observed snapshot and the later binary resolved outcome.
