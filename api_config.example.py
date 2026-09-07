"""Безопасный шаблон API-конфигурации.

Скопируйте этот файл в ``api_config.py`` и заполните только локально.
Файл ``api_config.py`` исключён из Git, потому что содержит торговые ключи.
"""

# Polymarket: публичные адреса API
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
POLYMARKET_TARGET_EVENT_URL = ""
POLYMARKET_DATA_URL = "https://data-api.polymarket.com"
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
POLYMARKET_CRYPTO_PRICE_URL = "https://polymarket.com/api/crypto/crypto-price"
POLYMARKET_MARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_USER_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
POLYMARKET_RTDS_WS = "wss://ws-live-data.polymarket.com"
POLYMARKET_GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
POLYGON_CHAIN_ID = 137

# Polymarket: секреты и параметры кошелька
POLYMARKET_PRIVATE_KEY = ""
POLYMARKET_FUNDER_ADDRESS = ""
POLYMARKET_API_KEY = ""
POLYMARKET_API_KEY_ADDRESS = ""
POLYMARKET_API_SECRET = ""
POLYMARKET_API_PASSPHRASE = ""
POLYMARKET_SIGNATURE_TYPE = 1
LIVE_KEYS_ROTATED_AFTER_AUDIT = False
LIVE_EXECUTOR_IMPLEMENTED = True
LIVE_TRADING_ENABLED = False
KILL_SWITCH = True

# Chainlink (источник отключён в текущей конфигурации)
CHAINLINK_SOURCE = ""
CHAINLINK_SYMBOL = "btc/usd"
CHAINLINK_CLIENT_ID = ""
CHAINLINK_CLIENT_SECRET = ""
CHAINLINK_FEED_ID = ""
CHAINLINK_STREAM_URL = ""

# Bybit
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"
BYBIT_REST_URL = "https://api.bybit.com"
BYBIT_SYMBOL = "BTCUSDT"
BYBIT_API_KEY = ""
BYBIT_API_SECRET = ""

# OKX
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_REST_URL = "https://www.okx.com"
OKX_SYMBOL = "BTC-USDT"
OKX_API_KEY = ""
OKX_API_SECRET = ""
OKX_API_PASSPHRASE = ""

# Telegram
TELEGRAM_API_ID = ""
TELEGRAM_API_HASH = ""
TELEGRAM_PHONE = ""
TELEGRAM_SESSION_NAME = "polybot"
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_ALERT_CHAT_ID = ""

# Pyth/oracle
# Актуальный Hermes endpoint; запросы требуют Bearer API key.
PYTH_HERMES_URL = "https://pyth.dourolabs.app/hermes"
PYTH_API_KEY = ""
PYTH_BTC_USD_FEED_ID = ""

# Опциональные удалённые LLM
OPENAI_API_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = ""
OPENAI_MODEL = ""
HUGGINGFACE_API_TOKEN = ""
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = ""
GROQ_API_KEY = ""
GROQ_MODEL = ""

# Polymarket Builder / Relayer
POLYMARKET_BUILDER_ADDRESS = ""
POLYMARKET_BUILDER_CODE = ""
POLYMARKET_RELAYER_API_KEY = ""
POLYMARKET_RELAYER_API_KEY_ADDRESS = ""
