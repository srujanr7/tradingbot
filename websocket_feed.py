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

    def _on_open(self, ws):
        logger.info("✅ Price feed connected")
        ws.send(json.dumps({
            "action":      "subscribe",
            "mode":        self.mode,
            "instruments": self.instruments
        }))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "mode" not in data:
                return
            if data["mode"] == "ltp":
                token = data["instrument"]
                ltp   = data["data"]["ltp"]
                self.latest_prices[token] = ltp
                if self.on_tick:
                    self.on_tick(token, ltp)
            elif data["mode"] == "quote":
                token = data["instrument"]
                self.latest_prices[token] = data["data"].get("live_price")
                if self.on_tick:
                    self.on_tick(token, data["data"])
        except Exception as e:
            logger.error(f"Price feed parse error: {e}")

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
    Streams real-time order status updates.
    Falls back to REST polling if WebSocket is unavailable.
    """
    WS_URL = "wss://ws-order-updates.indstocks.com"

    def __init__(self, on_update=None):
        self.on_update    = on_update
        self.token        = os.getenv("INDSTOCKS_TOKEN")
        self.ws           = None
        self._running     = False
        self._ws_alive    = False
        self.order_states = {}

    def _on_open(self, ws):
        logger.info("✅ Order feed connected")
        self._ws_alive = True
        ws.send(json.dumps({
            "action": "subscribe",
            "mode":   "order_updates"
        }))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # Ignore heartbeats — they won't have "type": "order"
            if data.get("type") == "order":
                order_id = data["order_id"]
                status   = data["order_status"]
                self.order_states[order_id] = data
                logger.info(f"Order update → {order_id}: {status}")
                if self.on_update:
                    self.on_update(data)
        except Exception as e:
            logger.error(f"Order feed parse error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"Order feed error: {error}")
        self._ws_alive = False

    def _on_close(self, ws, code, msg):
        self._ws_alive = False
        logger.warning(f"Order feed closed: {code} {msg}")
        if self._running:
            logger.info("Order feed reconnecting in 5s...")
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
        logger.info("Order feed thread started")
        return thread

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()

    def _poll_order_rest(self, order_id: str, segment: str = "EQUITY") -> dict:
        """
        REST fallback when WebSocket is unavailable.
        Matches GET /order which takes order_id + segment as JSON body.
        Normalises response to same shape as WebSocket order update.
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
                # Normalise REST status names to match WebSocket names
                status_map = {
                    "PARTIALLY FILLED":            "PARTIALLY_EXECUTED",
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
                return {
                    "order_id":           order_id,
                    "order_status":       status,
                    "filled_quantity":    traded_qty,
                    "remaining_quantity": max(requested_qty - traded_qty, 0),
                    "average_price":      avg_price,
                }
            else:
                logger.warning(
                    f"Order REST poll returned {resp.status_code} "
                    f"for {order_id}"
                )
        except Exception as e:
            logger.error(f"Order REST poll error: {e}")
        return {}

    def wait_for_fill(self, order_id: str, timeout: int = 30,
                      segment: str = "EQUITY") -> dict:
        """
        Waits for order to reach a terminal state.
        Uses WebSocket cache if connection is alive,
        otherwise polls REST every 1s as fallback.
        """
        terminal = {
            "SUCCESS", "FAILED", "CANCELLED",
            "PARTIALLY_EXECUTED", "EXPIRED", "ABORTED"
        }
        start = time.time()

        while time.time() - start < timeout:
            # Check WebSocket cache first
            state = self.order_states.get(order_id, {})
            if state.get("order_status") in terminal:
                return state

            # WS is down — poll REST directly
            if not self._ws_alive:
                rest_state = self._poll_order_rest(order_id, segment)
                if rest_state.get("order_status") in terminal:
                    self.order_states[order_id] = rest_state
                    logger.info(
                        f"Order {order_id} confirmed via REST: "
                        f"{rest_state['order_status']}"
                    )
                    return rest_state

            time.sleep(1.0)

        logger.warning(f"wait_for_fill timeout for order {order_id}")
        return self.order_states.get(order_id, {})
