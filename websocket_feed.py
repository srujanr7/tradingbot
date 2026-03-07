import websocket
import json
import threading
import logging
import time
import os
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
        self.instruments = instruments
        self.mode = mode
        self.on_tick = on_tick
        self.token = os.getenv("INDSTOCKS_TOKEN")
        self.ws = None
        self._running = False
        self.latest_prices = {}

    def _on_open(self, ws):
        logger.info("✅ Price feed connected")
        ws.send(json.dumps({
            "action": "subscribe",
            "mode": self.mode,
            "instruments": self.instruments
        }))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "mode" not in data:
                return
            if data["mode"] == "ltp":
                token = data["instrument"]
                ltp = data["data"]["ltp"]
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
            logger.info("Reconnecting in 5s...")
            time.sleep(5)
            self._connect()

    def _connect(self):
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            header=[f"Authorization: {self.token}"],  # FIX 1: list of strings
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
    WS_URL = "wss://ws-order-updates.indstocks.com"

    def __init__(self, on_update=None):
        self.on_update    = on_update
        self.token        = os.getenv("INDSTOCKS_TOKEN")
        self.ws           = None
        self._running     = False
        self.order_states = {}
        self._ws_alive    = False  # track if WS actually connected

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

    def _poll_order_rest(self, order_id: str) -> dict:
        """
        REST fallback when WebSocket is unavailable.
        Calls GET /orders/{order_id} directly.
        """
        import requests
        try:
            resp = requests.get(
                f"https://api.indstocks.com/orders/{order_id}",
                headers={"Authorization": self.token},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                # Normalise to same shape as WS order update
                return {
                    "order_id":           order_id,
                    "order_status":       data.get("order_status", ""),
                    "filled_quantity":    data.get("filled_qty", 0),
                    "remaining_quantity": data.get("remaining_qty", 0),
                    "average_price":      data.get("average_price", 0),
                }
        except Exception as e:
            logger.error(f"Order REST poll error: {e}")
        return {}

    def wait_for_fill(self, order_id: str, timeout: int = 30) -> dict:
        """
        Wait for order to reach terminal state.
        Uses WebSocket cache if WS is alive, otherwise falls back to REST polling.
        """
        terminal = {"SUCCESS", "FAILED", "CANCELLED", "PARTIALLY_EXECUTED"}
        start    = time.time()

        while time.time() - start < timeout:
            # Try WS cache first
            state = self.order_states.get(order_id, {})
            if state.get("order_status") in terminal:
                return state

            # If WS is down, poll REST directly
            if not self._ws_alive:
                rest_state = self._poll_order_rest(order_id)
                if rest_state.get("order_status") in terminal:
                    self.order_states[order_id] = rest_state
                    return rest_state

            time.sleep(0.5)

        # Final check — return whatever we have
        return self.order_states.get(order_id, {})
