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

    def check_margin(self, security_id, qty, price,
                     segment="EQUITY", txn_type="BUY",
                     product="CNC"):
        """Check margin before placing order."""
        data = self._request(
            "GET", "/margin",
            json={
                "segment":    segment,
                "txnType":    txn_type,
                "quantity":   str(qty),
                "price":      str(price),
                "product":    product,
                "securityID": security_id,
                "exchange":   "NSE"
            }
        )
        return data["data"] if data else None
