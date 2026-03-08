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
        """
        Reload token from environment.
        On Render, updating the env var and redeploying
        restarts the process with the new token at boot.
        This method handles the case where token is updated
        without a full redeploy (local dev or runtime update).
        """
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
                "in Render Environment and redeploy"
            )
        return False

    def is_token_valid(self) -> bool:
        return not self._token_invalid

    def _request(self, method, endpoint, **kwargs):
        """Core request handler with retry + rate limit logic."""
        if self._token_invalid:
            logger.error(
                "Token is invalid — skipping request. "
                "Update INDSTOCKS_TOKEN in Render Environment."
            )
            return None

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
                        "Generate a new token on INDstocks dashboard, "
                        "update INDSTOCKS_TOKEN in Render Environment, "
                        "then redeploy."
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

    def get_ltp(self, scrip_codes: str):
        """Get last traded price. scrip_codes = 'NSE_2885,NSE_11536'"""
        data = self._request(
            "GET", "/market/quotes/ltp",
            params={"scrip-codes": scrip_codes}
        )
        if data:
            return {k: v["live_price"] for k, v in data["data"].items()}
        return {}

    def get_full_quote(self, scrip_codes: str):
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
        interval: '1minute','5minute','15minute','60minute','1day'
        timestamps in Unix milliseconds (IST)
        """
        import pandas as pd
        data = self._request(
            "GET", f"/market/historical/{interval}",
            params={
                "scrip-codes": scrip_code,
                "start_time":  start_ms,
                "end_time":    end_ms
            }
        )
        if data and "candles" in data["data"]:
            df = pd.DataFrame(
                data["data"]["candles"],
                columns=["timestamp", "open", "high",
                         "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms"
            )
            return df
        return None

    # ── Orders ────────────────────────────────────────────────

    def place_order(self, txn_type, security_id, qty,
                    order_type="MARKET", limit_price=None,
                    exchange="NSE", segment="EQUITY",
                    product="CNC", is_amo=False):
        """Place a standard order."""
        payload = {
            "txn_type":    txn_type,
            "exchange":    exchange,
            "segment":     segment,
            "security_id": security_id,
            "qty":         qty,
            "order_type":  order_type,
            "validity":    "DAY",
            "product":     product,
            "is_amo":      is_amo,
            "algo_id":     "99999"
        }
        if order_type == "LIMIT" and limit_price:
            payload["limit_price"] = limit_price

        data = self._request("POST", "/order", json=payload)
        if data:
            logger.info(f"{txn_type} order placed: {data['data']}")
            return data["data"]
        return None

    def place_smart_order(self, txn_type, security_id, qty,
                          limit_price, sl_trigger, sl_limit,
                          tgt_trigger, tgt_limit,
                          exchange="NSE"):
        """Place GTT order with auto stop-loss + target."""
        payload = {
            "txn_type":          txn_type,
            "exchange":          exchange,
            "segment":           "DERIVATIVE",
            "product":           "MARGIN",
            "order_type":        "LIMIT",
            "validity":          "DAY",
            "security_id":       security_id,
            "qty":               qty,
            "limit_price":       limit_price,
            "sl_trigger_price":  sl_trigger,
            "sl_limit_price":    sl_limit,
            "tgt_trigger_price": tgt_trigger,
            "tgt_limit_price":   tgt_limit,
            "algo_id":           "99999"
        }
        return self._request("POST", "/smart/order", json=payload)

    def cancel_order(self, order_id, segment="EQUITY"):
        return self._request(
            "POST", "/order/cancel",
            json={"order_id": order_id, "segment": segment}
        )

    def get_order_book(self):
        data = self._request("GET", "/order-book")
        return data["data"] if data else []

    # ── Portfolio ─────────────────────────────────────────────

    def get_holdings(self):
        data = self._request("GET", "/portfolio/holdings")
        return data["data"] if data else []

    def get_positions(self, segment="equity", product="cnc"):
        data = self._request(
            "GET", "/portfolio/positions",
            params={"segment": segment, "product": product}
        )
        return data["data"] if data else {}

    def get_funds(self):
        data = self._request("GET", "/funds")
        return data["data"] if data else {}

    def get_true_balance(self, trade_mode: str = "MIS") -> float:
        """
        Returns the single real balance for the given trade mode.
        This is the ACTUAL cash available — never add segments together,
        they all represent the same underlying money split differently.

        MIS    → eq_mis   (intraday equity, includes broker leverage)
        CNC    → eq_cnc   (delivery equity, no leverage)
        MTF    → eq_mtf   (margin trading facility)
        MARGIN → future   (F&O margin)
        """
        funds = self.get_funds()
        if not funds:
            return 0.0
        avl = funds.get("detailed_avl_balance", {})
        mapping = {
            "MIS":    avl.get("eq_mis",     0.0),
            "CNC":    avl.get("eq_cnc",     0.0),
            "MTF":    avl.get("eq_mtf",     0.0),
            "MARGIN": avl.get("future",     0.0),
        }
        return float(mapping.get(trade_mode.upper(), avl.get("eq_mis", 0.0)))

    def check_margin(self, security_id, qty, price,
                     segment="EQUITY", txn_type="BUY",
                     product="CNC", exchange="NSE"):
        """
        Check margin required before placing order.
        Returns full margin dict from broker or None on failure.

        Key field: data["total_margin"] — actual funds needed for this order.
        This accounts for broker leverage automatically, so:
          total_margin < cash_balance → order will go through
          total_margin > cash_balance → insufficient margin
        """
        data = self._request(
            "GET", "/margin",
            json={
                "segment":    segment,
                "txnType":    txn_type,
                "quantity":   str(qty),
                "price":      str(price),
                "product":    product,
                "securityID": str(security_id),
                "exchange":   exchange
            }
        )
        if data and data.get("status") == "success":
            return data["data"]
        return None

    @staticmethod
    def _api_product(trade_mode: str, segment: str = "EQUITY") -> str:
        """
        Converts internal TRADE_MODE → API product string.

        The INDstocks margin and order API does NOT accept "MIS" —
        it requires "INTRADAY". This mapping handles all modes:

        TRADE_MODE   segment      → API product
        ──────────────────────────────────────────
        MIS          EQUITY       → INTRADAY
        CNC          EQUITY       → CNC
        MTF          EQUITY       → CNC
        MARGIN       DERIVATIVE   → MARGIN
        anything     DERIVATIVE   → MARGIN
        """
        mode = (trade_mode or "MIS").upper()
        seg  = (segment or "EQUITY").upper()

        if seg == "DERIVATIVE":
            return "MARGIN"

        return {
            "MIS":    "INTRADAY",
            "CNC":    "CNC",
            "MTF":    "CNC",
            "MARGIN": "MARGIN",
        }.get(mode, "INTRADAY")

    def get_margin_per_unit(self, security_id, price,
                             segment="EQUITY", txn_type="BUY",
                             product="MIS", exchange="NSE") -> float:
        """
        Returns margin required for a SINGLE unit of this instrument.
        Use this to calculate max affordable quantity:
            max_qty = available_balance / margin_per_unit

        For MIS equity: margin_per_unit = price / leverage_multiplier
        For MARGIN (F&O): margin_per_unit = SPAN + exposure margin per unit
        Returns 0.0 on failure (caller should fallback to price).
        """
        api_product = self._api_product(product, segment)  # MIS → INTRADAY

        margin = self.check_margin(
            security_id = security_id,
            qty         = 1,
            price       = price,
            segment     = segment,
            txn_type    = txn_type,
            product     = api_product,
            exchange    = exchange
        )
        if margin:
            total = float(margin.get("total_margin", 0.0))
            logger.debug(
                f"Margin/unit [{security_id}] @ ₹{price} "
                f"({product}→{api_product}): ₹{total:.2f}"
            )
            return total
        return 0.0
