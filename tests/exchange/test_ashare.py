import json
import logging
import sys
from datetime import time, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from freqtrade.constants import CandleType
from freqtrade.exchange.ashare import (
    SUPPORTED_TIMEFRAMES,
    Ashare,
    _fetch_sina_1h,
    _fetch_tencent_daily,
    _is_market_open,
    _parse_pair,
    _to_sina_symbol,
    _to_tencent_symbol,
)
from tests.conftest import EXMS, get_patched_exchange


# ======================================================================
# _parse_pair
# ======================================================================


class TestParsePair:
    def test_valid_sz(self):
        code, exchange = _parse_pair("000001/SZ")
        assert code == "000001"
        assert exchange == "SZ"

    def test_valid_sh(self):
        code, exchange = _parse_pair("600519/SH")
        assert code == "600519"
        assert exchange == "SH"

    def test_invalid_no_slash(self):
        with pytest.raises(ValueError, match="Invalid A-share pair format"):
            _parse_pair("000001SZ")

    def test_invalid_multiple_slash(self):
        with pytest.raises(ValueError, match="Invalid A-share pair format"):
            _parse_pair("000001/SZ/extra")


# ======================================================================
# _to_tencent_symbol / _to_sina_symbol
# ======================================================================


class TestSymbolConversion:
    def test_tencent_sz(self):
        assert _to_tencent_symbol("000001/SZ") == "sz000001"

    def test_tencent_sh(self):
        assert _to_tencent_symbol("600519/SH") == "sh600519"

    def test_sina_sz(self):
        assert _to_sina_symbol("000858/SZ") == "sz000858"

    def test_sina_sh(self):
        assert _to_sina_symbol("600519/SH") == "sh600519"


# ======================================================================
# _is_market_open
# ======================================================================


class TestIsMarketOpen:
    @patch("freqtrade.exchange.ashare.datetime")
    def test_weekday_morning_trading(self, mock_datetime):
        from datetime import datetime as real_dt
        mock_now = real_dt(2026, 6, 8, 10, 0, tzinfo=timezone(timedelta(hours=8)))
        mock_datetime.now.return_value = mock_now
        assert _is_market_open() is True

    @patch("freqtrade.exchange.ashare.datetime")
    def test_weekday_afternoon_trading(self, mock_datetime):
        from datetime import datetime as real_dt
        mock_now = real_dt(2026, 6, 9, 14, 0, tzinfo=timezone(timedelta(hours=8)))
        mock_datetime.now.return_value = mock_now
        assert _is_market_open() is True

    @patch("freqtrade.exchange.ashare.datetime")
    def test_weekday_lunch_closed(self, mock_datetime):
        from datetime import datetime as real_dt
        mock_now = real_dt(2026, 6, 10, 12, 0, tzinfo=timezone(timedelta(hours=8)))
        mock_datetime.now.return_value = mock_now
        assert _is_market_open() is False

    @patch("freqtrade.exchange.ashare.datetime")
    def test_weekday_early_morning_closed(self, mock_datetime):
        from datetime import datetime as real_dt
        mock_now = real_dt(2026, 6, 11, 8, 0, tzinfo=timezone(timedelta(hours=8)))
        mock_datetime.now.return_value = mock_now
        assert _is_market_open() is False

    @patch("freqtrade.exchange.ashare.datetime")
    def test_saturday_closed(self, mock_datetime):
        from datetime import datetime as real_dt
        mock_now = real_dt(2026, 6, 13, 10, 0, tzinfo=timezone(timedelta(hours=8)))
        mock_datetime.now.return_value = mock_now
        assert _is_market_open() is False

    @patch("freqtrade.exchange.ashare.datetime")
    def test_sunday_closed(self, mock_datetime):
        from datetime import datetime as real_dt
        mock_now = real_dt(2026, 6, 14, 14, 0, tzinfo=timezone(timedelta(hours=8)))
        mock_datetime.now.return_value = mock_now
        assert _is_market_open() is False

    @patch("freqtrade.exchange.ashare.datetime")
    def test_weekday_after_close(self, mock_datetime):
        from datetime import datetime as real_dt
        mock_now = real_dt(2026, 6, 12, 16, 0, tzinfo=timezone(timedelta(hours=8)))
        mock_datetime.now.return_value = mock_now
        assert _is_market_open() is False


# ======================================================================
# _fetch_tencent_daily
# ======================================================================


def _make_tencent_response(n=5):
    """模拟腾讯 K 线 API 返回的 JSON"""
    bars = []
    for i in range(n):
        bars.append([
            f"2026-06-{10 - i:02d}",
            f"{10.0 + i * 0.1:.3f}",
            f"{10.5 + i * 0.1:.3f}",
            f"{11.0 + i * 0.1:.3f}",
            f"{9.5 + i * 0.1:.3f}",
            f"{1000000 + i * 100000:.3f}",
        ])
    return {"code": 0, "msg": "", "data": {"sz000001": {"qfqday": bars}}}


class TestFetchTencentDaily:
    @patch("freqtrade.exchange.ashare.httpx.Client")
    def test_parses_json_to_dataframe(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_tencent_response(5)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        df = _fetch_tencent_daily("000001/SZ")
        assert df is not None
        assert len(df) == 5
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    @patch("freqtrade.exchange.ashare.httpx.Client")
    def test_empty_data_returns_none(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "data": {"sz000001": {"qfqday": []}}}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        df = _fetch_tencent_daily("000001/SZ")
        assert df is None

    @patch("freqtrade.exchange.ashare.httpx.Client")
    def test_dtype_conversion(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_tencent_response(3)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        df = _fetch_tencent_daily("000001/SZ")
        assert df is not None
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            assert pd.api.types.is_numeric_dtype(df[col])

    @patch("freqtrade.exchange.ashare.httpx.Client")
    def test_uses_correct_url_and_params(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_tencent_response(3)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        _fetch_tencent_daily("600519/SH", count=100)
        call_args = mock_client.get.call_args
        assert "fqkline" in call_args[1]["params"]["param"] or call_args[0][0] is not None
        param_str = call_args[1]["params"]["param"]
        assert "sh600519" in param_str
        assert "day" in param_str
        assert "100" in param_str


# ======================================================================
# _fetch_sina_1h
# ======================================================================


def _make_sina_jsonp(n=5):
    """模拟新浪 JSONP 响应"""
    bars = []
    for i in range(n):
        bars.append({
            "day": f"2026-06-{10 - i:02d} 14:00:00",
            "open": f"{10.0 + i * 0.1:.2f}",
            "high": f"{11.0 + i * 0.1:.2f}",
            "low": f"{9.5 + i * 0.1:.2f}",
            "close": f"{10.5 + i * 0.1:.2f}",
            "volume": f"{1000000 + i * 100000}",
        })
    json_str = json.dumps(bars)
    return f"/*<script>location.href='//sina.com';</script>*/\ncallback({json_str})"


class TestFetchSina1h:
    @patch("freqtrade.exchange.ashare.httpx.Client")
    def test_parses_jsonp_to_dataframe(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = _make_sina_jsonp(5)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        df = _fetch_sina_1h("000001/SZ")
        assert df is not None
        assert len(df) == 5
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    @patch("freqtrade.exchange.ashare.httpx.Client")
    def test_empty_bars_returns_none(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = "callback([])"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        df = _fetch_sina_1h("000001/SZ")
        assert df is None

    @patch("freqtrade.exchange.ashare.httpx.Client")
    def test_dtype_conversion(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = _make_sina_jsonp(3)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        df = _fetch_sina_1h("000001/SZ")
        assert df is not None
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            assert pd.api.types.is_numeric_dtype(df[col])


# ======================================================================
# Ashare Exchange class
# ======================================================================


class TestAshareExchange:
    def test_init_ccxt_returns_namespace(self, default_conf, mocker):
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        assert exchange._api is not None
        assert exchange._api.name == "ashare"

    def test_name_property(self, default_conf, mocker):
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        assert exchange.name == "ashare"

    def test_id_property(self, default_conf, mocker):
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        assert exchange.id == "ashare"

    def test_timeframes_property(self, default_conf, mocker):
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        tfs = exchange.timeframes
        assert "1h" in tfs
        assert "1d" in tfs

    def test_exchange_has(self, default_conf, mocker):
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        assert exchange.exchange_has("fetchOHLCV") is True
        assert exchange.exchange_has("createOrder") is False

    def test_reload_markets(self, default_conf, mocker):
        default_conf["exchange"]["pair_whitelist"] = ["000001/SZ", "600519/SH"]
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        exchange._markets = {}
        exchange.reload_markets()
        assert "000001/SZ" in exchange._markets
        assert exchange._markets["000001/SZ"]["base"] == "000001"
        assert exchange._markets["000001/SZ"]["active"] is True

    @patch("freqtrade.exchange.ashare._time.sleep")
    @patch("freqtrade.exchange.ashare._fetch_tencent_daily")
    def test_refresh_ohlcv_daily(self, mock_fetch, mock_sleep, default_conf, mocker):
        mock_df = pd.DataFrame({
            "date": pd.date_range("2026-06-08", periods=5, freq="1D", tz="UTC"),
            "open": [10.0] * 5, "high": [11.0] * 5, "low": [9.0] * 5,
            "close": [10.5] * 5, "volume": [1e6] * 5,
        })
        mock_fetch.return_value = mock_df
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        exchange.refresh_latest_ohlcv([("000001/SZ", "1d", CandleType.SPOT)])
        key = ("000001/SZ", "1d", CandleType.SPOT)
        assert key in exchange._klines
        assert len(exchange._klines[key]) == 5

    @patch("freqtrade.exchange.ashare._time.sleep")
    @patch("freqtrade.exchange.ashare._fetch_sina_1h")
    def test_refresh_ohlcv_1h(self, mock_fetch, mock_sleep, default_conf, mocker):
        mock_df = pd.DataFrame({
            "date": pd.date_range("2026-06-08", periods=5, freq="1h", tz="UTC"),
            "open": [10.0] * 5, "high": [11.0] * 5, "low": [9.0] * 5,
            "close": [10.5] * 5, "volume": [1e6] * 5,
        })
        mock_fetch.return_value = mock_df
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        exchange.refresh_latest_ohlcv([("000001/SZ", "1h", CandleType.SPOT)])
        key = ("000001/SZ", "1h", CandleType.SPOT)
        assert key in exchange._klines

    def test_refresh_ohlcv_unsupported_timeframe(self, default_conf, mocker):
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        exchange.refresh_latest_ohlcv([("000001/SZ", "5m", CandleType.SPOT)])
        assert len(exchange._klines) == 0

    @patch("freqtrade.exchange.ashare._time.sleep")
    @patch("freqtrade.exchange.ashare._fetch_tencent_daily", side_effect=Exception("network"))
    def test_all_retries_exhausted(self, mock_fetch, mock_sleep, default_conf, mocker, caplog):
        caplog.set_level(logging.WARNING)
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        exchange.refresh_latest_ohlcv([("000001/SZ", "1d", CandleType.SPOT)])
        assert mock_fetch.call_count == 3
        assert any("Failed to fetch A-share OHLCV" in r.message for r in caplog.records)
        assert len(exchange._klines) == 0

    @patch("freqtrade.exchange.ashare._time.sleep")
    @patch("freqtrade.exchange.ashare._fetch_tencent_daily")
    def test_retry_then_succeeds(self, mock_fetch, mock_sleep, default_conf, mocker):
        mock_df = pd.DataFrame({
            "date": pd.date_range("2026-06-08", periods=3, freq="1D", tz="UTC"),
            "open": [10.0] * 3, "high": [11.0] * 3, "low": [9.0] * 3,
            "close": [10.5] * 3, "volume": [1e6] * 3,
        })
        mock_fetch.side_effect = [Exception("timeout"), mock_df]
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        exchange.refresh_latest_ohlcv([("000001/SZ", "1d", CandleType.SPOT)])
        assert mock_fetch.call_count == 2
        assert ("000001/SZ", "1d", CandleType.SPOT) in exchange._klines

    @patch("freqtrade.exchange.ashare._time.sleep")
    @patch("freqtrade.exchange.ashare._fetch_tencent_daily", return_value=None)
    def test_none_result_preserves_cache(self, mock_fetch, mock_sleep, default_conf, mocker):
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        exchange.refresh_latest_ohlcv([("000001/SZ", "1d", CandleType.SPOT)])
        assert len(exchange._klines) == 0

    @patch("freqtrade.exchange.ashare._time.sleep")
    @patch("freqtrade.exchange.ashare._fetch_tencent_daily")
    def test_inter_pair_delay(self, mock_fetch, mock_sleep, default_conf, mocker):
        mock_df = pd.DataFrame({
            "date": pd.date_range("2026-06-08", periods=3, freq="1D", tz="UTC"),
            "open": [10.0] * 3, "high": [11.0] * 3, "low": [9.0] * 3,
            "close": [10.5] * 3, "volume": [1e6] * 3,
        })
        mock_fetch.return_value = mock_df
        exchange = get_patched_exchange(mocker, default_conf, exchange="ashare")
        pair_list = [("000001/SZ", "1d", CandleType.SPOT), ("600519/SH", "1d", CandleType.SPOT)]
        exchange.refresh_latest_ohlcv(pair_list)
        mock_sleep.assert_any_call(1)


# ======================================================================
# check_exchange custom exchange whitelist
# ======================================================================


class TestCheckExchangeCustomExchange:
    def test_ashare_passes_check(self, default_conf, mocker):
        from freqtrade.exchange.check_exchange import check_exchange

        default_conf["runmode"] = "dry_run"
        default_conf["exchange"]["name"] = "ashare"
        result = check_exchange(default_conf, check_for_bad=False)
        assert result is True
