-- One-time reset: rebuild empty pa_analysis with semantic column names.
--
-- Use only when pa_analysis is empty or its analysis history can be discarded.
-- SQLAlchemy create_all() does not alter existing columns, so an existing
-- empty table with legacy non-semantic diagnosis/decision columns must be dropped once.
--
-- Run:
--   psql -h <host> -p <port> -U postgres -d freqtrade_monitor -f reset_pa_analysis_semantic_columns.sql

DROP TABLE IF EXISTS pa_analysis;

CREATE TABLE pa_analysis (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    market VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    candle_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'pending',
    kline_table TEXT,
    feature_table TEXT,
    l1_features JSON,
    market_diagnosis JSON,
    selected_strategies JSON,
    experience_cases JSON,
    trade_decision JSON,
    validation_status VARCHAR,
    validation_errors JSON,
    prompt_metadata JSON,
    raw_responses JSON,
    CONSTRAINT ux_pa_analysis_identity UNIQUE (symbol, timeframe, candle_time)
);
