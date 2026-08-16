# Аудит API-подключений

Проверено: `2026-08-09 12:35:42 +0300`

> Секреты в отчёт не записываются.

| Подключение | Статус | Результат | Что исправить |
|---|---|---|---|
| Polymarket Gamma | **OK** | HTTP 200, 578 ms | — |
| Polymarket Data | **OK** | HTTP 200, 512 ms | — |
| Polymarket CLOB | **OK** | HTTP 200, 488 ms | — |
| Polymarket geoblock | **OK** | HTTP 200, 596 ms | — |
| Bybit public market data | **OK** | HTTP 200, 390 ms | — |
| OKX public market data | **OK** | HTTP 200, 703 ms | — |
| Pyth BTC/USD oracle | **OK** | HTTP 200, 529 ms | — |
| Telegram Bot API | **OK** | HTTP 200, 522 ms | — |
| OpenAI API | **OPTIONAL** | OPENAI_API_KEY пуст. | Создать API key на platform.openai.com. |
| Polymarket Market WebSocket | **OK** | WebSocket handshake successful | — |
| Polymarket RTDS WebSocket | **OK** | WebSocket handshake successful | — |
| Bybit WebSocket | **OK** | WebSocket handshake successful | — |
| OKX WebSocket | **OK** | WebSocket handshake successful | — |
| Polymarket trading auth | **OK** | Все обязательные поля заполнены. | Запустить generate_polymarket_credentials.py и указать корректный POLYMARKET_SIGNATURE_TYPE. |
| Direct Chainlink Data Streams | **INCOMPLETE** | CHAINLINK_STREAM_URL пуст; CHAINLINK_FEED_ID имеет неожиданный формат | Уточнить endpoint и схему auth в Chainlink Self Service. До этого использовать Chainlink через Polymarket RTDS. |
| Telegram MTProto | **INCOMPLETE** | API_ID/API_HASH/PHONE заполнены не полностью. | Получить api_id/api_hash на my.telegram.org/apps и указать отдельный номер collector-аккаунта. |
| Local Qwen | **CONFIGURED** | Model configured: Qwen/Qwen2.5-1.5B-Instruct. Model files download on first use. | При нехватке RAM уменьшить модель или использовать quantized runtime. |

## Интерпретация

- `OK` — сетевое подключение и базовый формат ответа проверены.
- `CONFIGURED` — конфигурация заполнена, но полноценный authenticated flow ещё не выполнялся.
- `INCOMPLETE` — обязательные поля отсутствуют или неоднозначны.
- `OPTIONAL` — источник не блокирует основной Polymarket pipeline.
- `ERROR` — сетевой запрос или проверка API завершились ошибкой.

## Важные замечания

- Binance полностью исключён из проекта.
- CryptoRank исключён как платный необязательный источник.
- Canonical resolution source для BTC Up/Down остаётся Chainlink через Polymarket RTDS.
- Pyth BTC/USD добавлен как независимый oracle для сверки цены и confidence interval.
- LLM формирует структурированный сигнал, но не имеет прямого доступа к размещению ордеров.
