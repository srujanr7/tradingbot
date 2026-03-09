import websocket
import json
import threading
import logging
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PriceFeed:
    """
    Streams live LTP or full quotes via WebSocket.
    Calls on_tick(instrument, ltp) on every price update.
    Auto-reconnects on disconnect.
    """
    WS_URL = "wss://ws-prices.indstocks.com/api/v1/ws/prices"

    def __init__(self, instruments: list, mode: str = "ltp", on_tick=None):
        self.instruments   = instruments
        self.mode          = mode
        self.on_tick       = on_tick
        self.token         = os.getenv("INDSTOCKS_TOKEN")
        self.ws            = None
        self._running      = False
        self.latest_prices = {}
        self._logged_count = 0   # for raw message diagnostics

    def _on_open(self, ws):
        logger.info("✅ Price feed connected")
        ws.send(json.dumps({
            "action":      "subscribe",
            "mode":        self.mode,
            "instruments": self.instruments
        }))

    def _on_message(self, ws, message):
        try:
            # Log first 3 raw messages to help diagnose broker format
            if self._logged_count < 3:
                logger.info(f"RAW WS MESSAGE #{self._logged_count + 1}: {message[:500]}")
                self._logged_count += 1

            data = json.loads(message)

            # Handle double-encoded JSON (broker sends a JSON string inside a string)
            if isinstance(data, str):
                data = json.loads(data)

            # Handle list payloads (some brokers batch multiple ticks in one message)
            if isinstance(data, list):
                for item in data:
                    self._process_tick(item)
                return

            self._process_tick(data)

        except json.JSONDecodeError as e:
            logger.error(f"Price feed JSON decode error: {e} | raw={message[:200]}")
        except Exception as e:
            logger.error(f"Price feed message error: {e} | raw={message[:200]}")

    def _process_tick(self, data):
        """
        Process a single normalised tick dict.
        Handles both 'ltp' and 'quote' modes.
        Silently ignores non-dict payloads (e.g. heartbeats).
        """
        try:
            if not isinstance(data, dict):
                return

            mode = data.get("mode")
            if not mode:
                return

            if mode == "ltp":
                token = data.get("instrument")
                ltp   = data.get("data", {}).get("ltp")
                if token is None or ltp is None:
                    return
                self.latest_prices[token] = float(ltp)
                if self.on_tick:
                    self.on_tick(token, float(ltp))

            elif mode == "quote":
                token = data.get("instrument")
                price = data.get("data", {}).get("live_price")
                if token is None:
                    return
                if price is not None:
                    self.latest_prices[token] = float(price)
                if self.on_tick:
                    self.on_tick(token, data.get("data", {}))

        except Exception as e:
            logger.error(f"Tick processing error: {e} | data={str(data)[:200]}")

    def _on_error(self, ws, error):
        logger.error(f"Price feed error: {error}")

    def _on_close(self, ws, code, msg):
        logger.warning(f"Price feed closed: {code} {msg}")
        if self._running:
            logger.info("Price feed reconnecting in 5s...")
            time.sleep(5)
            self._connect()

    def _connect(self):
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            header=[f"Authorization: {self.token}"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def start(self):
        self._running = True
        thread = threading.Thread(target=self._connect, daemon=True)
        thread.start()
        logger.info("Price feed thread started")
        return thread

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()

    def get_ltp(self, token: str) -> float:
        return self.latest_prices.get(token)


class OrderFeed:
    """
    REST-only order status tracker.
    Polls GET /order until terminal state is reached.
    No WebSocket dependency — works reliably at 15min candle timeframe.
    """

    def __init__(self, on_update=None):
        self.on_update    = on_update
        self.token        = os.getenv("INDSTOCKS_TOKEN")
        self.order_states = {}

    def start(self):
        logger.info("✅ Order feed started (REST polling mode)")

    def stop(self):
        pass

    def _fetch_order(self, order_id: str, segment: str) -> dict:
        """
        Calls GET /order and normalises the response into
        the same shape the rest of the bot expects.
        """
        try:
            resp = requests.get(
                "https://api.indstocks.com/order",
                headers={
                    "Authorization": self.token,
                    "Content-Type":  "application/json"
                },
                json={
                    "order_id": order_id,
                    "segment":  segment
                },
                timeout=10
            )
            if resp.status_code == 200:
                raw    = resp.json().get("data", {})
                status = raw.get("status", "")

                status_map = {
                    "PARTIALLY FILLED":             "PARTIALLY_EXECUTED",
                    "PARTIALLY FILLED - CANCELLED": "PARTIALLY_EXECUTED",
                    "PARTIALLY FILLED - EXPIRED":   "PARTIALLY_EXECUTED",
                }
                status = status_map.get(status, status)

                traded_qty    = int(raw.get("traded_qty", 0) or 0)
                requested_qty = int(raw.get("requested_qty", 0) or 0)
                traded_price  = raw.get("traded_price", 0)
                try:
                    avg_price = float(traded_price) if traded_price else 0.0
                except (ValueError, TypeError):
                    avg_price = 0.0

                result = {
                    "order_id":           order_id,
                    "order_status":       status,
                    "filled_quantity":    traded_qty,
                    "remaining_quantity": max(requested_qty - traded_qty, 0),
                    "average_price":      avg_price,
                }

                prev = self.order_states.get(order_id, {})
                if prev.get("order_status") != status:
                    self.order_states[order_id] = result
                    logger.info(f"Order {order_id} → {status}")
                    if self.on_update:
                        self.on_update(result)

                return result

            else:
                logger.warning(
                    f"GET /order returned {resp.status_code} "
                    f"for {order_id}"
                )
        except Exception as e:
            logger.error(f"Order fetch error ({order_id}): {e}")

        return {}

    def wait_for_fill(self, order_id: str, timeout: int = 30,
                      segment: str = "EQUITY") -> dict:
        """
        Polls GET /order every second until a terminal state
        is reached or timeout expires.
        """
        terminal = {
            "SUCCESS", "FAILED", "CANCELLED",
            "PARTIALLY_EXECUTED", "EXPIRED", "ABORTED"
        }
        start = time.time()

        while time.time() - start < timeout:
            state = self._fetch_order(order_id, segment)
            if state.get("order_status") in terminal:
                return state
            time.sleep(1.0)

        logger.warning(f"wait_for_fill timeout for order {order_id}")
        return self.order_states.get(order_id, {})
