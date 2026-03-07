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
        """
        instruments: ["NSE:2885", "NFO:51011"]
        mode: "ltp" or "quote"
        """
        self.instruments = instruments
        self.mode = mode
        self.on_tick = on_tick
        self.token = os.getenv("INDSTOCKS_TOKEN")
        self.ws = None
        self._running = False
        self.latest_prices = {}   # {instrument_token: ltp}

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

            # Ignore heartbeats
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
            header={"Authorization": self.token},
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def start(self):
        """Start feed in background thread"""
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
        """Non-blocking price lookup from latest cache"""
        return self.latest_prices.get(token)


class OrderFeed:
    """
    Streams real-time order status updates.
    Calls on_update(order_data) on every order state change.
    """
    WS_URL = "wss://ws-order-updates.indstocks.com"

    def __init__(self, on_update=None):
        self.on_update = on_update
        self.token = os.getenv("INDSTOCKS_TOKEN")
        self.ws = None
        self._running = False
        self.order_states = {}   # {order_id: latest_status}

    def _on_open(self, ws):
        logger.info("✅ Order feed connected")
        ws.send(json.dumps({
            "action": "subscribe",
            "mode": "order_updates"
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

    def _on_close(self, ws, code, msg):
        if self._running:
            time.sleep(5)
            self._connect()

    def _connect(self):
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            header={"Authorization": self.token},
            on_open=self._on_open,
            on_message=self._on_message,
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

    def wait_for_fill(self, order_id: str, timeout: int = 30) -> dict:
        """Block until order fills or timeout. Returns final order state."""
        start = time.time()
        while time.time() - start < timeout:
            state = self.order_states.get(order_id, {})
            if state.get("order_status") in ("SUCCESS", "FAILED",
                                              "CANCELLED", "PARTIALLY_EXECUTED"):
                return state
            time.sleep(0.5)
        return self.order_states.get(order_id, {})
