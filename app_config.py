"""Параметры и гиперпараметры локального проекта Polymarket.

Этот файл можно редактировать вручную: здесь находятся режимы сбора, модели,
торговые ограничения и настройки дашборда. Секреты хранятся в api_config.py.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Storage and retention
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "данные"
LOG_DIR = PROJECT_ROOT / "журналы"
EXPORT_DIR = DATA_DIR / "экспорт_для_просмотра"
MODEL_DIR = PROJECT_ROOT / "модели"
DATABASE_PATH = DATA_DIR / "market_data.sqlite3"
PARQUET_ARCHIVE_DIR = DATA_DIR / "parquet_архив"
PARQUET_ARCHIVE_ENABLED = True
PARQUET_ROLLOVER_SECONDS = 3600
# Live-записи накапливаются только в пределах одного poll-цикла и фиксируются
# одной транзакцией. Часовой RAM-cache запрещён: он теряет час данных при сбое.
COLLECTOR_WRITE_CACHE_MAX_SECONDS = 5.0
RAW_DATA_RETENTION_DAYS = 7
NORMALIZED_DATA_RETENTION_DAYS = 90
STORE_RAW_MESSAGES = False
DATABASE_MAINTENANCE_SECONDS = 3600
SQLITE_BUSY_TIMEOUT_MS = 10_000
SQLITE_WRITE_RETRY_ATTEMPTS = 8
SQLITE_WRITE_RETRY_DELAY_SECONDS = 0.25

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
COLLECTOR_ENABLED = True
COLLECTOR_BTC_5M_SLUG_PREFIX = "btc-updown-5m"
COLLECTOR_DEFAULT_EVENT_URL = "https://polymarket.com/event/btc-updown-5m-1785756900"
COLLECTOR_POLL_SECONDS = 2.0
COLLECTOR_DISCOVERY_SECONDS = 15
COLLECTOR_LABEL_INTERVAL_SECONDS = 60
COLLECTOR_MAX_BOOK_LEVELS = 20
COLLECTOR_USE_WEBSOCKETS = True
ENABLE_POLYMARKET_MARKET_WS = False  # REST books are enough at 5s cadence; full WS produced hundreds of messages/sec
STORE_POLYMARKET_RAW_WS = False  # price_change messages caused multi-GB growth; normalized books remain enabled
RAW_MESSAGE_SAMPLE_SECONDS = 5.0

ENABLE_POLYMARKET = True
ENABLE_BYBIT = True
ENABLE_OKX = True
ENABLE_PYTH = False  # Hermes требует API key; включить после заполнения PYTH_API_KEY.
ENABLE_CHAINLINK_RTDS = False  # disabled: latency is unsuitable for the BTC 5m strategy
ENABLE_TELEGRAM_NEWS = False

# ---------------------------------------------------------------------------
# Freshness, validation, and latency controls
# ---------------------------------------------------------------------------
MAX_POLYMARKET_AGE_SECONDS = 15
MAX_EXCHANGE_AGE_SECONDS = 15
MAX_PYTH_AGE_SECONDS = 90
MAX_CHAINLINK_AGE_SECONDS = 90
MAX_TARGET_REFERENCE_AGE_SECONDS = 20
REQUIRE_OFFICIAL_EVENT_TARGET = True
MAX_SOURCE_PRICE_DEVIATION_PCT = 0.35
MAX_ALLOWED_SPREAD = 0.08
MIN_REQUIRED_PRICE_SOURCES = 2
BLOCK_ON_ORACLE_CONFLICT = True

# ---------------------------------------------------------------------------
# LLM and ML hyperparameters
# ---------------------------------------------------------------------------
LLM_ENABLED = False
LLM_PRIMARY_PROVIDER = "qwen_local"
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GEMMA_MODEL_ID = "google/gemma-3-1b-it"
QWEN_DEVICE = "cpu"
QWEN_LOCAL_FILES_ONLY = False
QWEN_MAX_NEW_TOKENS = 72
QWEN_TEMPERATURE = 0.0
QWEN_TOP_P = 0.8
QWEN_MIN_CONFIDENCE = 0.70
LLM_TORCH_DTYPE = "float16"
LLM_DECISION_CACHE_SECONDS = 15
# Локальная LLM медленная: офлайн-аудит явно ограничен малой выборкой и не считается статистически значимым.
LLM_OFFLINE_MAX_EVENTS = 10
GEMMA_OFFLINE_MAX_EVENTS = 3
QWEN_OFFLINE_MAX_EVENTS = 10

# Модели загружаются лениво: в памяти одновременно находится не более одной LLM.
DEFAULT_TRADING_MODEL = "custom"
DEFAULT_ENTRY_MODEL = "custom"
# Рабочий тандем: custom выбирает вход, CatBoost независимо переоценивает
# вероятность удерживаемого исхода для решения HOLD/CLOSE.
DEFAULT_EXIT_MODEL = "catboost"
MODEL_SWITCH_MAX_DELAY_SECONDS = 300
MODEL_VERSION_ARCHIVE_DIR = MODEL_DIR / "архив_версий"
MODEL_VERSION_REGISTRY_PATH = MODEL_DIR / "реестр_версий.json"
CATBOOST_ARTIFACT_PATH = MODEL_DIR / "catboost" / "catboost_btc_5m.cbm"
CATBOOST_METADATA_PATH = MODEL_DIR / "catboost" / "catboost_bundle.joblib"
QWEN_LOCAL_PATH = MODEL_DIR / "qwen2.5-1.5b-instruct" / "weights"
QWEN_GGUF_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
QWEN_GGUF_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
QWEN_GGUF_PATH = MODEL_DIR / "qwen2.5-1.5b-instruct" / "gguf" / QWEN_GGUF_FILENAME
GEMMA_LOCAL_PATH = MODEL_DIR / "gemma-3-1b" / "weights"
MODEL_COMPARISON_REPORT_PATH = MODEL_DIR / "offline_comparison.json"
OFFLINE_TOURNAMENT_DATASET_PATH = EXPORT_DIR / "offline_tournament_labeled.jsonl"
OFFLINE_TOURNAMENT_PARQUET_PATH = EXPORT_DIR / "offline_tournament_labeled.parquet"
WALK_FORWARD_REPORT_PATH = MODEL_DIR / "walk_forward_report.json"
ACTION_VALUE_DATASET_PATH = EXPORT_DIR / "action_value_pnl.jsonl"
COUNTERFACTUAL_ACTION_DATASET_PATH = EXPORT_DIR / "counterfactual_entry_actions.jsonl"
COUNTERFACTUAL_ACTION_PARQUET_PATH = EXPORT_DIR / "counterfactual_entry_actions.parquet"
PNL_MODEL_ARTIFACT_PATH = MODEL_DIR / "своя_дообученная" / "btc_5m_net_pnl.joblib"
PNL_MODEL_REPORT_PATH = MODEL_DIR / "pnl_model_report.json"
# Экспериментальная двухступенчатая value-модель разрешена только в PAPER.
# При включённом флаге сервер и trading-engine блокируют переход в LIVE.
PAPER_ACTION_VALUE_EXPERIMENT_ENABLED = False
PAPER_ACTION_VALUE_EXPERIMENT_NAME = "two_stage_value_v3 · P(fill) × E(PnL|fill)"
EXIT_MODEL_ARTIFACT_PATH = MODEL_DIR / "exit_model" / "btc_5m_exit_value.joblib"
EXIT_MODEL_REPORT_PATH = MODEL_DIR / "exit_model_report.json"
QWEN_LORA_ADAPTER_PATH = MODEL_DIR / "qwen_qlora_experiment_v1" / "adapter"
WALK_FORWARD_V8_REPORT_PATH = MODEL_DIR / "walk_forward_v8_report.json"
PNL_DATASET_MIN_ENTRY_PRICE = 0.05
PNL_DATASET_MAX_ENTRY_PRICE = 0.95
COUNTERFACTUAL_ENTRY_NOTIONALS_USDC = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
COUNTERFACTUAL_LIMIT_LEVELS = ("ask", "midpoint", "bid")
COUNTERFACTUAL_WAIT_LOOKAHEAD_SECONDS = 30
CONSENSUS_NUMERIC_MODEL = "custom"
CONSENSUS_LLM_MODEL = "qwen"
CONSENSUS_ENTRY_CONFIDENCE = 0.72
# Выход легче входа: обе модели должны оценить шанс выигрыша удерживаемой стороны не выше этого порога.
CONSENSUS_MAX_HELD_WIN_PROBABILITY_FOR_EXIT = 0.45

TRAINING_ENABLED = True
TRAINING_TEST_SIZE = 0.20
TRAINING_CALIBRATION_SIZE = 0.20
TRAINING_RANDOM_STATE = 42
TRAINING_MIN_EXAMPLES = 1_000
TRAINING_MIN_INDEPENDENT_EVENTS = 100
# Обучение запускается отдельно: оно не должно конкурировать со сбором данных
# и торговым контуром за оперативную память и SQLite.
AUTO_RETRAIN_ENABLED = False
RETRAIN_EVERY_NEW_EVENTS = 100
MODEL_CANDIDATE_DIR = MODEL_DIR / "candidates"
TRAINING_ARTIFACT_PATH = MODEL_DIR / "своя_дообученная" / "btc_5m_direction.joblib"
QWEN_TRAINING_DATASET_PATH = EXPORT_DIR / "qwen_trade_instructions.jsonl"
PREPARED_TRAINING_DIR = DATA_DIR / "подготовленные_обучения"

# GPU-ready pipeline. Эти пути и параметры работают и на CPU: CUDA-зависимости
# устанавливаются позднее отдельной командой уже на новом компьютере.
SEQUENCE_DATASET_DIR = DATA_DIR / "sequence_dataset"
SEQUENCE_DATASET_PATH = SEQUENCE_DATASET_DIR / "btc_5m_sequences.parquet"
SEQUENCE_DATASET_MANIFEST_PATH = SEQUENCE_DATASET_DIR / "manifest.json"
EXIT_SEQUENCE_DATASET_PATH = SEQUENCE_DATASET_DIR / "btc_5m_exit_sequences_v14.parquet"
EXIT_SEQUENCE_MANIFEST_PATH = SEQUENCE_DATASET_DIR / "exit_manifest_v14.json"
EXIT_SHADOW_REPORT_PATH = MODEL_DIR / "exit_shadow_tournament_v14.json"
EXIT_SEQUENCE_SHADOW_ARTIFACT_PATH = MODEL_DIR / "shadow" / "exit_sequence_v14.joblib"
GPU_MODEL_CATALOG_PATH = PROJECT_ROOT / "настройка_проекта" / "gpu_model_catalog.json"
GPU_TRAINING_PLAN_PATH = PROJECT_ROOT / "настройка_проекта" / "gpu_training_plan.json"
GPU_REQUIREMENTS_PATH = PROJECT_ROOT / "настройка_проекта" / "requirements-gpu.txt"
SEQUENCE_SAMPLE_SECONDS = 5
SEQUENCE_LENGTHS_SECONDS = (30, 60, 120)
SEQUENCE_PRIMARY_LENGTH_SECONDS = 120
GPU_TRAINING_RANDOM_SEED = 42
GPU_TRAINING_BATCH_SIZE = 256
GPU_TRAINING_MAX_EPOCHS = 80
GPU_TRAINING_EARLY_STOPPING_PATIENCE = 10
GPU_MIXED_PRECISION = True
GPU_NUM_WORKERS = 4
# Контекст только из полностью завершённых предыдущих пятиминуток. Нулевое окно —
# честный baseline без истории; 3 и 12 сравниваются на одинаковых walk-forward фолдах.
EVENT_HISTORY_WINDOWS = (0, 3, 12)
EVENT_HISTORY_LIVE_WINDOWS = (3, 12)
HISTORY_CONTEXT_REPORT_PATH = MODEL_DIR / "history_context_walk_forward.json"
HISTORY_MODEL_CANDIDATE_DIR = MODEL_CANDIDATE_DIR / "history_context_v1"
LOSS_REVERSAL_REPORT_PATH = MODEL_DIR / "loss_reversal_hypothesis.json"
MIGRATION_MANIFEST_PATH = PROJECT_ROOT / "настройка_проекта" / "migration_manifest.json"

# ---------------------------------------------------------------------------
# Trading controls. Live trading remains disabled by default.
# ---------------------------------------------------------------------------
TRADING_MODE = "paper"  # collect_only, paper, shadow, live
STRATEGY_VERSION = "custom_entry_v25_catboost_exit_v22_full_exit_grid_v20"
STRATEGY_RUN_LABEL = "v20_full_exit_adaptive_limit_entry_grid_paper"
LIVE_TRADING_ENABLED = True
KILL_SWITCH = False
LIVE_EXECUTOR_IMPLEMENTED = True
LIVE_KEYS_ROTATED_AFTER_AUDIT = False
LIVE_SCALE_DEFAULT = 0.10
LIVE_SCALE_MIN = 0.10
LIVE_SCALE_MAX = 3.0
TRADE_SIZE_MULTIPLIER_DEFAULT = 1.0
TRADE_SIZE_MULTIPLIER_MIN = 0.10
TRADE_SIZE_MULTIPLIER_MAX = 3.0
LIVE_CANARY_MAX_LOSS_USDC = 5.0
LIVE_MIN_POSITION_USDC = 1.0
MAX_POSITION_USDC = 10.0
MAX_DAILY_LOSS_USDC = 20.0  # secondary guard; direction/validation stops have priority
MAX_CONSECUTIVE_LOSSES = 3
MAX_CONSECUTIVE_WRONG_DIRECTIONS = 3
PAPER_LOSS_STREAK_COOLDOWN_SECONDS = 30 * 60
# Сильная задержка входных данных останавливает торговлю, а не только блокирует
# отдельное решение. Сбор данных при этом продолжает работать для диагностики.
VALIDATION_HARD_STOP_ENABLED = True
VALIDATION_HARD_STOP_AGE_SECONDS = 45.0
VALIDATION_HARD_STOP_CONSECUTIVE_CYCLES = 3
# После завершения 5m-события старая позиция продолжает ждать resolution в фоне,
# но не должна блокировать новую версионированную PAPER-сессию на целый час.
STALE_POSITION_NEW_SESSION_GRACE_SECONDS = 60
# PAPER-only предварительный расчёт после окончания пятиминутки. Он освобождает
# торговый цикл, но никогда не подменяет официальный label в обучающих данных.
PAPER_POST_EVENT_SETTLEMENT_ENABLED = True
PAPER_POST_EVENT_SETTLEMENT_DELAY_SECONDS = 10
PAPER_POST_EVENT_WIN_BID_THRESHOLD = 0.99
PAPER_POST_EVENT_LOSS_ASK_THRESHOLD = 0.01
PAPER_POST_EVENT_MAX_QUOTE_DISTANCE_SECONDS = 20
MAX_OPEN_POSITIONS = 1
MAX_FRESH_ENTRIES_PER_EVENT = 1
MIN_ENTRY_CONFIDENCE = 0.55
EARLY_ENTRY_ENABLED = True
EARLY_EXIT_ENABLED = True
# Вероятность относится к исходу относительно Price to Beat, а не к цене контракта.
MAX_HELD_WIN_PROBABILITY_FOR_EXIT = 0.30
PARTIAL_EXIT_CONFIDENCE = 0.62
PARTIAL_EXIT_FRACTION = 0.50
PARTIAL_EXIT_ENABLED = False
FULL_EXIT_ONLY_ENABLED = True  # Exit-модель выбирает только HOLD либо полное CLOSE.
ALLOW_POSITION_ADD = False
TAKE_PROFIT_PCT = 0.12
STOP_LOSS_PCT = 0.08
ESTIMATED_SLIPPAGE_BPS = 25
POLYMARKET_CRYPTO_TAKER_FEE_RATE = 0.07
POLYMARKET_BUILDER_FEE_BPS = 0
ENTRY_ORDER_TYPE = "GTD"
TAKE_PROFIT_ORDER_TYPE = "GTD"
EXIT_ORDER_TYPE = "GTD"
GTD_EFFECTIVE_LIFETIME_SECONDS = 20
FAK_PRICE_CAP_SLIPPAGE_BPS = 35

# Лесенка лимитных входов. Модель по-прежнему определяет общий размер позиции;
# исполнитель лишь распределяет его по более выгодным ценовым уровням. Число
# уровней автоматически уменьшается, если бюджет не позволяет соблюсти минимум
# CLOB (обычно 5 shares) и минимум $1 на каждую заявку.
ENTRY_GRID_ENABLED = True
ENTRY_GRID_LIVE_ENABLED = False  # Сначала валидируем fills в PAPER, затем отдельный canary.
ENTRY_GRID_MIN_ORDERS = 3
ENTRY_GRID_MAX_ORDERS = 5
ENTRY_GRID_PRICE_STEP = 0.03
ENTRY_GRID_MIN_ORDER_USDC = 1.0

# Paper trading: complete simulation with no signed or submitted orders.
PAPER_INITIAL_BALANCE_USDC = 300.0
PAPER_ENTRY_NOTIONAL_USDC = 3.0
PAPER_ADD_NOTIONAL_USDC = 1.5
PAPER_MAX_EVENT_EXPOSURE_USDC = 10.0
PAPER_DECISION_PROVIDER = "model_registry"  # активная модель выбирается через runtime_controls
PAPER_POLL_SECONDS = 2.0
# Время внутри пятиминутки является признаком модели, а не жёстким правилом.
# При False вход возможен в любой момент, пока рынок технически не завершён.
MODEL_TIME_GATES_ENABLED = False
PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN = 15
PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE = 15
PAPER_FORCE_EXIT_SECONDS_BEFORE_CLOSE = 0  # without reversal signal, hold until resolution
PAPER_MIN_BTC_MOVE_PCT = 0.08
PAPER_MAX_SOURCE_DISAGREEMENT_PCT = 0.20
PAPER_SHARP_MOVE_LOOKBACK_SECONDS = 10
PAPER_SHARP_MOVE_BLOCK_PCT = 0.22
PAPER_CONSENSUS_MISPRICING_PROB = 0.08
PAPER_MIN_SECONDS_BETWEEN_ACTIONS = 10
PAPER_LIMIT_FILL_TOLERANCE = 0.01
PAPER_MIN_ENTRY_PRICE = 0.10
PAPER_MAX_ENTRY_PRICE = 0.80
DEFAULT_CLOB_MIN_ORDER_SIZE_SHARES = 5.0
PAPER_DECISION_LOG_SECONDS = 2.0
PAPER_MIN_ENTRY_NET_EDGE = 0.04
# Дополнительный запас сверх комиссии и рыночной цены: вход разрешён только
# когда калиброванная вероятность покрывает fee/slippage и этот safety margin.
ENTRY_VALUE_SAFETY_MARGIN = 0.025
# Ни один денежный entry-value артефакт пока не прошёл честный temporal promotion gate.
# Поэтому v15 использует его только как shadow-кандидата, а в PAPER принимает решение
# по калиброванной вероятности направления и net EV после комиссии.
ACTION_VALUE_ENABLED = False
# Полная value-модель пока не прошла gate, но её отдельный fill-классификатор
# честно валидирован и используется только для P(fill) лимитной заявки.
FILL_PROBABILITY_MODEL_ENABLED = True
FILL_PROBABILITY_MIN_ROC_AUC = 0.75
FILL_PROBABILITY_MAX_BRIER = 0.20
# Даже небольшой положительный net-PnL допустим, если value-gate уже покрыл
# комиссию, spread и запас ошибки. Главная целевая метрика остаётся суммой PnL сессии.
ACTION_VALUE_MIN_EXPECTED_PNL_USDC = 0.02
MIN_ACCEPTABLE_NET_PNL_USDC = 0.10
ACTION_VALUE_MODEL_WEIGHT = 1.0
ACTION_VALUE_MIN_R2 = 0.0
ACTION_PROBABILITY_MODEL_WEIGHT = 0.75
# Entry-value v4: денежный EV уменьшается на ожидаемый хвостовой убыток.
# Порог крупного убытка задаётся долей капитала заявки, а не направлением Up/Down.
ACTION_TAIL_RISK_ENABLED = True
ACTION_TAIL_LOSS_FRACTION = 0.50
ACTION_TAIL_RISK_PENALTY = 1.00
ACTION_TAIL_MAX_PROBABILITY = 0.35
ACTION_TAIL_MAX_EXPECTED_LOSS_USDC = 1.50
ACTION_VALUE_V4_REPORT_PATH = MODEL_DIR / "entry_value_v4_tail_risk_report.json"
WALK_FORWARD_V9_REPORT_PATH = MODEL_DIR / "walk_forward_v9_full_chain_report.json"
REGIME_ENTRY_V5_DIR = MODEL_CANDIDATE_DIR / "entry_value_v5_regime_experts"
REGIME_ENTRY_V5_ARTIFACT_PATH = REGIME_ENTRY_V5_DIR / "entry_value_v5_regime_experts.joblib"
REGIME_ENTRY_V5_REPORT_PATH = REGIME_ENTRY_V5_DIR / "report.json"
LOW_PROBABILITY_TRADING_ENABLED = False  # Только после отдельной проверки стратегии buy-low/sell-higher.
# Размер позиции определяется непрерывной risk-adjusted Kelly-функцией. Ступени
# сохранены только как совместимый fallback для старых артефактов.
ADAPTIVE_POSITION_SIZING_ENABLED = True
POSITION_SIZING_MODE = "calibrated_fractional_kelly_v1"
POSITION_SIZE_EDGE_TIERS = ((0.04, 2.0), (0.08, 3.0), (0.18, 5.0), (0.35, 10.0))
POSITION_SIZE_MIN_USDC = 1.0
POSITION_SIZE_KELLY_FRACTION = 1.0
POSITION_SIZE_KELLY_POWER = 1.5
POSITION_SIZE_FILL_EXPONENT = 1.0
POSITION_SIZE_MAX_EXPECTED_LOSS_USDC = 1.25
ENTRY_SIGNAL_CONFIRMATIONS = 3
REVERSAL_SIGNAL_CONFIRMATIONS = 3
SIGNAL_CONFIRMATION_MIN_SECONDS = 6

# Пятиступенчатый выход. Каждая ступень фиксирует 20% исходного количества:
# доли ниже относятся к оставшейся позиции (20%, 25%, 33.3%, 50%, 100%).
FIVE_STAGE_EXIT_ENABLED = True  # Ступени остаются признаками риска, но исполнение всегда полное.
# Прибыль сама по себе больше не является сигналом выхода. Закрываемся ступенями
# только при ухудшении вероятности удерживаемого исхода и переходе BTC за Price to Beat.
EXIT_ON_PROFIT_ALONE = False
EXIT_STAGE_REMAINING_FRACTIONS = (0.20, 0.25, 1 / 3, 0.50, 1.00)
EXIT_STAGE_PROFIT_RETURN_PCT = (0.25, 0.50, 0.75, 1.00, 1.50)
# Выход должен быть легче входа: первая защитная ступень доступна уже при
# потере моделью преимущества, но фактическая продажа всё равно требует
# подтверждения adverse-side и преимущества CLOSE над HOLD после комиссии.
EXIT_STAGE_MAX_HELD_PROBABILITY = (0.58, 0.48, 0.38, 0.28, 0.18)
EXIT_RISK_SIGNAL_CONFIRMATIONS = 5
EXIT_RISK_CONFIRMATION_MIN_SECONDS = 10
EXIT_PROFIT_SIGNAL_CONFIRMATIONS = 2
EXIT_PROFIT_CONFIRMATION_MIN_SECONDS = 2
EXIT_RISK_REQUIRES_ADVERSE_TARGET_SIDE = True
EXIT_VALUE_ENABLED = True
EXIT_VALUE_MARGIN_USDC = 0.05
# Аналитический fallback не должен закрывать позицию из-за преимущества на уровне шума.
# На завершённой текущей сессии порог $0.50 дал лучший результат среди проверенных порогов.
EXIT_FALLBACK_MIN_ADVANTAGE_USDC = 0.15
EXIT_FALLBACK_MIN_ADVANTAGE_FRACTION = 0.05
EXIT_MODEL_MIN_ROWS = 300
EXIT_WAIT_HORIZONS_SECONDS = (15, 30, 60)
EXIT_TIMING_VALUE_MARGIN_USDC = 0.05
# Реалистичный paper execution: очередь, глубина, задержка и partial/non-fill.
EXECUTION_SIMULATION_ENABLED = True
EXECUTION_QUEUE_AHEAD_FRACTION = 0.50
EXECUTION_MAX_BOOK_PARTICIPATION = 0.25
EXECUTION_BASE_LATENCY_MS = 350
EXECUTION_LATENCY_JITTER_MS = 450
EXECUTION_MIN_FILL_PROBABILITY = 0.15
EXECUTION_RANDOM_SEED = 42
MAX_EXECUTION_BOOK_AGE_SECONDS = 8.0
# P(fill) в PAPER является модельной оценкой по стакану/latency, а не фактом.
# Реальный fill-rate и фактическое движение стакана выводятся отдельно.
EXECUTION_FILL_PROBABILITY_KIND = "book_depth_latency_heuristic_v2"

# Shadow-эксперимент следующей пятиминутки. До promotion gate заявки на ещё не
# начавшийся рынок не отправляются в LIVE.
NEXT_EVENT_CONTEXT_ENABLED = True
NEXT_EVENT_FORECAST_SAMPLE_SECONDS = 15.0
NEXT_EVENT_PREOPEN_SHADOW_ENABLED = True
NEXT_EVENT_PREOPEN_LIVE_ENABLED = False
NEXT_EVENT_LIMIT_PRICE_CAP = 0.80
NEXT_EVENT_LIMIT_PRICE_FLOOR = 0.10
NEXT_EVENT_ORDER_REPLACE_TICKS = 2
COUNTERFACTUAL_SAMPLE_SECONDS = 15
COUNTERFACTUAL_ACTIONS = ("BUY_UP", "BUY_DOWN", "WAIT", "HOLD", "CLOSE")
# Все три модели получают одинаковые независимые события; shadow не управляет капиталом.
SHADOW_TOURNAMENT_ENABLED = True
# Qwen/Gemma запускаются вручную и не задерживают локальный collector.
SHADOW_MODELS = ("catboost", "custom")
SHADOW_SAMPLE_SECONDS = 15
MODEL_HEALTH_WINDOWS = (30, 100, 300)
DRIFT_ALERT_ROC_AUC = 0.52
DRIFT_ALERT_BRIER = 0.25
DRIFT_ALERT_DIRECTION_SHARE = 0.80
DIRECTION_COLLAPSE_BLOCK_ENABLED = False
ML_AUTONOMOUS_POLICY_ENABLED = True
ML_AUTONOMOUS_POLICY_PAPER_ONLY = True
ML_POLICY_DIRECTION_PRESERVING = True
# Выбран на validation-периоде нового custom+history; test остаётся отрицательным,
# поэтому версия разрешена только для PAPER-проверки стабильности.
ML_POLICY_MIN_DIRECTION_CONFIDENCE = 0.55
ML_POLICY_LIMIT_LEVELS = ("bid", "midpoint", "ask")
# Модель сравнивает весь допустимый диапазон размеров. $10 — бюджет/жёсткий
# потолок на одно событие, а не обязательный размер каждой ставки.
ML_POLICY_NOTIONALS_USDC = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
ML_POLICY_LARGE_SIZE_MANUAL_GATE_ENABLED = False
ML_POLICY_FIVE_DOLLAR_MIN_OUTCOME_PROBABILITY = 0.82
ML_POLICY_FIVE_DOLLAR_MIN_UTILITY_USDC = 0.18
# WAIT конкурирует с BUY как полноценное действие и включает запас на ошибку оценки value.
ML_POLICY_WAIT_UTILITY_USDC = 0.05
ML_POLICY_SKIP_SIGNAL_CONFIRMATION = True
ML_POLICY_REQUIRE_FRESH_DATA = True
# Активный PAPER-сбор: сначала модель свободно ждёт сильный сигнал, но в конце
# разрешённого окна должна выбрать лучший исполнимый вход. Почти случайный
# прогноз, плохие данные и технически невалидный стакан по-прежнему разрешают WAIT.
PAPER_ACTIVE_COLLECTION_ENABLED = False
PAPER_ACTIVE_COLLECTION_FORCE_ENTRY_REMAINING_SECONDS = 90
PAPER_ACTIVE_COLLECTION_MIN_DIRECTION_CONFIDENCE = 0.54
DIRECTION_COLLAPSE_WINDOW_EVENTS = 30
DIRECTION_COLLAPSE_MIN_EVENTS = 15
DIRECTION_COLLAPSE_MAX_SHARE = 0.90
COUNTERFACTUAL_HORIZONS_SECONDS = (15, 30, 45, 60, 90, 120, 180, 240)
DECISION_RETENTION_DAYS = 30
EQUITY_SNAPSHOT_RETENTION_DAYS = 30
WAIT_DECISION_SAMPLE_SECONDS = 15

# Live launcher remains locked until paper results are statistically meaningful.
LIVE_MIN_RESOLVED_EVENTS = 300
LIVE_MIN_WILSON_WIN_RATE = 0.52
LIVE_MIN_PROFIT_FACTOR = 1.20
LIVE_MAX_DRAWDOWN_PCT = 0.10
LIVE_MIN_NET_PNL_USDC = 30.0
LIVE_MIN_EXPECTED_VALUE_PER_TRADE = 0.03
LIVE_MAX_BRIER_SCORE = 0.18
LIVE_MIN_CALIBRATION_EVENTS = 300
LIVE_MIN_DIRECTION_ENTRIES = 30
LIVE_MAX_SINGLE_DIRECTION_SHARE = 0.80
LIVE_REQUIRE_POSITIVE_PNL_CI95 = True
LIVE_REQUIRED_CONFIRMATION = "REAL_TRADING_UNLOCKED"

# ---------------------------------------------------------------------------
# Dashboard and monitoring
# ---------------------------------------------------------------------------
# 0.0.0.0 даёт доступ с телефона в той же Wi-Fi сети. Удалённые клиенты
# обязательно проходят Basic Auth в dashboard/app.py.
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8765
DASHBOARD_USERNAME = "polybot"
DASHBOARD_PASSWORD_FILE = DATA_DIR / ".dashboard_password"
DASHBOARD_REFRESH_SECONDS = 5
DASHBOARD_DATABASE_ROWS = 100
HEALTH_STALE_AFTER_SECONDS = 30
SYSTEM_METRICS_ENABLED = True
REPORTING_TARGET_DAILY_RETURN_PCT = 0.05  # reporting benchmark, never a reason to force trades
