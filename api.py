import requests
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class INDstocksAPI:
    def __init__(self):
        self.base_url       = "https://api.indstocks.com"
        self.token          = os.getenv("INDSTOCKS_TOKEN")
        self.headers        = {
            "Authorization": self.token,
            "Content-Type":  "application/json"
        }
        self.max_retries    = 3
        self._token_invalid = False

    def refresh_token(self):
        load_dotenv(override=True)
        new_token = os.environ.get("INDSTOCKS_TOKEN")
        if new_token and new_token != self.token:
            self.token                    = new_token
            self.headers["Authorization"] = new_token
            self._token_invalid           = False
            logger.info("✅ API token refreshed successfully")
            return True
        elif new_token == self.token:
            logger.warning(
                "Token unchanged — update INDSTOCKS_TOKEN "
                "in environment and restart"
            )
        return False

    def is_token_valid(self) -> bool:
        return not self._token_invalid

    def _request(self, method: str, endpoint: str, **kwargs):
        """
        Core request handler with retry + rate limit logic.
        Supports params= for query string and json= or json_body= for body.
        json_body= is an alias that maps to json= (for clarity at call sites).
        """
        if self._token_invalid:
            logger.error("Token invalid — skipping request.")
            return None

        # Allow callers to use json_body= as a named alias for json=
        if "json_body" in kwargs:
            kwargs["json"] = kwargs.pop("json_body")

        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method, url,
                    headers=self.headers,
                    timeout=10,
                    **kwargs
                )
                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)

                elif response.status_code == 403:
                    self._token_invalid = True
                    logger.error(
                        "❌ Token expired (403). "
                        "Generate a new token on INDstocks dashboard "
                        "and update INDSTOCKS_TOKEN in your environment."
                    )
                    raise Exception("TokenExpired")

                else:
                    logger.error(
                        f"API Error {response.status_code}: {response.text}"
                    )
                    return None

            except Exception as e:
                if "TokenExpired" in str(e):
                    raise
                logger.warning(f"Request error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)

        return None

    # ── Market Data ───────────────────────────────────────────

    def get_ltp(self, scrip_codes: str) -> dict:
        """
        Get last traded price.
        scrip_codes: comma-separated NSE_/NFO_ format e.g. 'NSE_2885,NFO_51011'
        Returns dict keyed by scrip_code → live_price float.
        """
        data = self._request(
            "GET", "/market/quotes/ltp",
            params={"scrip-codes": scrip_codes}
        )
        if data and "data" in data:
            return {
                k: v.get("live_price")
                for k, v in data["data"].items()
            }
        return {}

    def get_full_quote(self, scrip_codes: str) -> dict:
        """Get full OHLCV + market depth."""
        data = self._request(
            "GET", "/market/quotes/full",
            params={"scrip-codes": scrip_codes}
        )
        return data["data"] if data else {}

    def get_historical(self, scrip_code: str, interval: str,
                       start_ms: int, end_ms: int):
        """
        Get OHLCV candles.

        scrip_code must be in SEGMENT_TOKEN format:
          NSE equity  → NSE_<security_id>  e.g. NSE_3045
          NFO futures → NFO_<security_id>  e.g. NFO_51011

        interval: '1minute','5minute','15minute','60minute','1day' etc.
        start_ms / end_ms: Unix epoch milliseconds (IST).

        Returns a pandas DataFrame with columns:
          [timestamp, open, high, low, close, volume]
        or None if no data.
        """
        import pandas as pd
        data = self._request(
            "GET", f"/market/historical/{interval}",
            params={
                "scrip-codes": scrip_code,
                "start_time":  start_ms,
                "end_time":    end_ms,
            }
        )
        if data and data.get("data") and "candles" in data["data"]:
            candles = data["data"]["candles"]
            if not candles:
                return None
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high",
                         "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms"
            )
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        return None

    # ── Orders ────────────────────────────────────────────────

    def place_order(self, txn_type: str, security_id: str,
                    qty: int, order_type: str = "MARKET",
                    limit_price: float = None,
                    exchange: str = "NSE",
                    segment: str = "EQUITY",
                    product: str = "CNC",
                    is_amo: bool = False,
                    algo_id: str = "99999") -> dict:
        """
        Place a standard order.

        Confirmed required fields from API docs:
          txn_type, exchange, segment, security_id, qty,
          order_type, validity, product, is_amo, algo_id

        product enum: "MARGIN", "INTRADAY", "CNC"
          — "MIS" is NOT a valid API value, use "INTRADAY"
        """
        payload = {
            "txn_type":    txn_type,
            "exchange":    exchange,
            "segment":     segment,
            "security_id": str(security_id),
            "qty":         int(qty),
            "order_type":  order_type,
            "validity":    "DAY",
            "product":     product,
            "is_amo":      is_amo,
            "algo_id":     algo_id,
        }
        if order_type == "LIMIT" and limit_price is not None:
            payload["limit_price"] = round(float(limit_price), 2)

        data = self._request("POST", "/order", json=payload)
        if data and data.get("status") == "success":
            logger.info(
                f"✅ {txn_type} order placed: {data['data']}"
            )
            return data["data"]
        logger.error(
            f"❌ place_order failed: {txn_type} {security_id} "
            f"qty={qty} | response={data}"
        )
        return None

    def place_smart_order(self, txn_type: str, security_id: str,
                          qty: int, limit_price: float,
                          sl_trigger: float, sl_limit: float,
                          tgt_trigger: float, tgt_limit: float,
                          exchange: str = "NSE") -> dict:
        """
        Place a GTT (Good Till Triggered) smart order.
        DERIVATIVE segment only — confirmed from API docs.
        """
        payload = {
            "txn_type":          txn_type,
            "exchange":          exchange,
            "segment":           "DERIVATIVE",
            "product":           "MARGIN",
            "order_type":        "LIMIT",
            "validity":          "DAY",
            "security_id":       str(security_id),
            "qty":               int(qty),
            "limit_price":       round(float(limit_price), 2),
            "sl_trigger_price":  round(float(sl_trigger), 2),
            "sl_limit_price":    round(float(sl_limit), 2),
            "tgt_trigger_price": round(float(tgt_trigger), 2),
            "tgt_limit_price":   round(float(tgt_limit), 2),
            "algo_id":           "99999",
        }
        data = self._request("POST", "/smart/order", json=payload)
        if data and data.get("status") == "success":
            return data["data"]
        return None

    def cancel_order(self, order_id: str,
                     segment: str = "EQUITY") -> dict:
        data = self._request(
            "POST", "/order/cancel",
            json={"order_id": order_id, "segment": segment}
        )
        return data["data"] if data else {}

    def get_order_book(self) -> list:
        data = self._request("GET", "/order-book")
        return data["data"] if data else []

    # ── Portfolio ─────────────────────────────────────────────

    def get_holdings(self) -> list:
        data = self._request("GET", "/portfolio/holdings")
        return data["data"] if data else []

    def get_positions(self, segment: str = "equity",
                      product: str = "cnc") -> dict:
        """
        segment: 'equity' or 'derivative' (lowercase per API docs)
        product: 'cnc', 'intraday', 'margin' (lowercase per API docs)
        """
        data = self._request(
            "GET", "/portfolio/positions",
            params={"segment": segment.lower(),
                    "product": product.lower()}
        )
        return data["data"] if data else {}

    def get_funds(self) -> dict:
        data = self._request("GET", "/funds")
        return data["data"] if data else {}

    def get_true_balance(self, trade_mode: str = "MIS") -> float:
        """
        Returns the single real balance for the given trade mode.

        The broker returns the same underlying cash split across
        multiple fields — never add them together.

        MIS    → eq_mis   (intraday equity)
        CNC    → eq_cnc   (delivery equity)
        MTF    → eq_mtf   (margin trading facility)
        MARGIN → future   (F&O margin)
        """
        funds = self.get_funds()
        if not funds:
            return 0.0
        avl = funds.get("detailed_avl_balance", {})
        mapping = {
            "MIS":    float(avl.get("eq_mis",  0.0)),
            "CNC":    float(avl.get("eq_cnc",  0.0)),
            "MTF":    float(avl.get("eq_mtf",  0.0)),
            "MARGIN": float(avl.get("future",  0.0)),
        }
        return mapping.get(
            trade_mode.upper(),
            float(avl.get("eq_mis", 0.0))
        )

    def check_margin(self, security_id: str, qty: int,
                     price: float, segment: str = "EQUITY",
                     txn_type: str = "BUY",
                     product: str = "CNC",
                     exchange: str = "NSE") -> dict:
        """
        Calculate margin required before placing an order.

        Confirmed API field names from docs:
          securityID  (camelCase)
          txnType     (camelCase)
          quantity    (string)
          price       (string)
          product     (MARGIN / INTRADAY / CNC)
          segment     (DERIVATIVE / EQUITY)
          exchange    (NSE / BSE)

        Returns data dict with total_margin and charges breakdown,
        or None on failure.
        """
        data = self._request(
            "GET", "/margin",
            json={
                "segment":    segment,
                "txnType":    txn_type,
                "quantity":   str(int(qty)),
                "price":      str(round(float(price), 2)),
                "product":    product,
                "securityID": str(security_id),
                "exchange":   exchange,
            }
        )
        if data and data.get("status") == "success":
            return data["data"]
        return None

    def get_margin_per_unit(self, security_id: str,
                            price: float,
                            segment: str = "EQUITY",
                            txn_type: str = "BUY",
                            product: str = "INTRADAY",
                            exchange: str = "NSE") -> float:
        """
        Returns margin required for a single unit (qty=1).

        product must already be a valid API product string:
          INTRADAY, CNC, or MARGIN
        Use _api_product() to convert TRADE_MODE before calling.

        Returns 0.0 on failure (caller should fallback to raw price).
        """
        margin = self.check_margin(
            security_id = security_id,
            qty         = 1,
            price       = price,
            segment     = segment,
            txn_type    = txn_type,
            product     = product,
            exchange    = exchange,
        )
        if margin:
            total = float(margin.get("total_margin", 0.0))
            logger.debug(
                f"Margin/unit [{security_id}] @ ₹{price} "
                f"({product}): ₹{total:.2f}"
            )
            return total
        return 0.0

    @staticmethod
    def _api_product(trade_mode: str,
                     segment: str = "EQUITY") -> str:
        """
        Converts internal TRADE_MODE → API product string.

        The API does NOT accept "MIS" — it requires "INTRADAY".
        Confirmed product enum from docs: MARGIN, INTRADAY, CNC

        TRADE_MODE   segment      → API product
        ─────────────────────────────────────────
        MIS          EQUITY       → INTRADAY
        CNC          EQUITY       → CNC
        MTF          EQUITY       → CNC
        MARGIN       DERIVATIVE   → MARGIN
        anything     DERIVATIVE   → MARGIN
        """
        seg = (segment or "EQUITY").upper()
        if seg == "DERIVATIVE":
            return "MARGIN"
        mode = (trade_mode or "MIS").upper()
        return {
            "MIS":    "INTRADAY",
            "CNC":    "CNC",
            "MTF":    "CNC",
            "MARGIN": "MARGIN",
        }.get(mode, "INTRADAY")
