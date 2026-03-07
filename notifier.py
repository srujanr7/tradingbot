import requests
import logging
import os
import threading
import time
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends alerts to Telegram and listens for commands.
    All sends are non-blocking (background thread).
    """

    BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    def __init__(self):
        self.chat_id       = TELEGRAM_CHAT_ID
        self.bot_paused    = False   # Controlled via /stop and /resume
        self._last_update  = 0       # For command polling
        self._command_callbacks = {} # Registered command handlers
        self._running      = False

    # ─────────────────────────────────────────────────────────
    # CORE SEND
    # ─────────────────────────────────────────────────────────

    def _send(self, message: str):
        """Send message in background thread so bot never blocks."""
        def _do_send():
            try:
                response = requests.post(
                    f"{self.BASE_URL}/sendMessage",
                    json={
                        "chat_id":    self.chat_id,
                        "text":       message,
                        "parse_mode": "HTML"   # Supports <b>, <i>, <code>
                    },
                    timeout=10
                )
                if response.status_code != 200:
                    logger.error(f"Telegram send failed: {response.text}")
            except Exception as e:
                logger.error(f"Telegram error: {e}")

        threading.Thread(target=_do_send, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # ALERT TYPES
    # ─────────────────────────────────────────────────────────

    def bot_started(self):
        self._send(
            "🚀 <b>BOT STARTED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Time    : {self._now()}\n"
            "Status  : All systems running\n"
            "Mode    : Live trading\n\n"
            "Commands:\n"
            "/status   → current positions\n"
            "/stop     → pause trading\n"
            "/resume   → resume trading\n"
            "/pnl      → today's PnL\n"
            "/positions → open positions"
        )

    def bot_stopped(self, reason: str = "Manual shutdown"):
        self._send(
            "🔴 <b>BOT STOPPED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Reason  : {reason}\n"
            f"Time    : {self._now()}"
        )

    def trade_executed(self, side: str, name: str, segment: str,
                       qty: int, price: float,
                       confidence: float, xgb: str, rl: str):
        emoji = "🟢" if side == "BUY" else "🔴"
        self._send(
            f"{emoji} <b>{side} EXECUTED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Segment    : {segment}\n"
            f"Qty        : {qty}\n"
            f"Price      : ₹{price:,.2f}\n"
            f"XGB Signal : {xgb}\n"
            f"RL Signal  : {rl}\n"
            f"Confidence : {confidence*100:.1f}%\n"
            f"Time       : {self._now()}"
        )

    def trade_closed(self, name: str, qty: int,
                     entry: float, exit_price: float,
                     pnl: float, daily_pnl: float):
        emoji  = "✅" if pnl >= 0 else "❌"
        pct    = ((exit_price - entry) / entry) * 100
        self._send(
            f"{emoji} <b>POSITION CLOSED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Qty        : {qty}\n"
            f"Entry      : ₹{entry:,.2f}\n"
            f"Exit       : ₹{exit_price:,.2f}\n"
            f"PnL        : ₹{pnl:+,.2f} ({pct:+.2f}%)\n"
            f"Daily PnL  : ₹{daily_pnl:+,.2f}\n"
            f"Time       : {self._now()}"
        )

    def order_failed(self, side: str, name: str,
                     qty: int, reason: str):
        self._send(
            "🚨 <b>ORDER FAILED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Side    : {side}\n"
            f"Name    : {name}\n"
            f"Qty     : {qty}\n"
            f"Reason  : {reason}\n"
            f"Time    : {self._now()}"
        )

    def insufficient_margin(self, name: str, required: float,
                             available: float):
        self._send(
            "⚠️ <b>INSUFFICIENT MARGIN</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Required   : ₹{required:,.2f}\n"
            f"Available  : ₹{available:,.2f}\n"
            f"Action     : Order skipped\n"
            f"Time       : {self._now()}"
        )

    def risk_warning(self, daily_pnl: float,
                     limit: float, balance: float):
        pct = (abs(daily_pnl) / balance) * 100
        self._send(
            "⚠️ <b>RISK WARNING</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Daily Loss   : ₹{daily_pnl:,.2f} ({pct:.1f}%)\n"
            f"Limit        : ₹{limit:,.2f}\n"
            f"Status       : Approaching limit\n"
            f"Time         : {self._now()}"
        )

    def kill_switch(self, daily_pnl: float, limit: float):
        self._send(
            "🔴 <b>KILL SWITCH ACTIVATED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Daily Loss : ₹{daily_pnl:,.2f}\n"
            f"Limit Hit  : ₹{limit:,.2f}\n"
            f"Action     : All trading PAUSED\n"
            f"Time       : {self._now()}\n\n"
            "Send /resume tomorrow after reset."
        )

    def squareoff_alert(self, positions: dict):
        lines = ""
        for key, pos in positions.items():
            if pos:
                lines += f"  {key.upper()} — {pos['qty']} units @ ₹{pos['entry']:,.2f}\n"
        if not lines:
            lines = "  No open positions\n"
        self._send(
            "⏰ <b>AUTO SQUARE-OFF</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Time    : {self._now()}\n"
            f"Closed  :\n{lines}"
        )

    def model_retrained(self, instrument: str, samples: int,
                        duration_seconds: float, next_retrain: str):
        self._send(
            "🧠 <b>MODEL RETRAINED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument   : {instrument}\n"
            f"Samples      : {samples:,}\n"
            f"Duration     : {duration_seconds:.0f}s\n"
            f"Next Retrain : {next_retrain}\n"
            f"Time         : {self._now()}"
        )

    def token_expired(self):
        self._send(
            "❌ <b>INDSTOCKS TOKEN EXPIRED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Action  : Trading paused\n"
            f"Time    : {self._now()}\n\n"
            "Go to indstocks.com → API section\n"
            "→ Regenerate token\n"
            "→ Update INDSTOCKS_TOKEN in Railway variables"
        )

    def daily_summary(self, total_pnl: float, trades: int,
                      wins: int, balance: float):
        losses   = trades - wins
        win_rate = (wins / trades * 100) if trades > 0 else 0
        emoji    = "✅" if total_pnl >= 0 else "❌"
        self._send(
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Total PnL  : ₹{total_pnl:+,.2f}\n"
            f"Trades     : {trades}\n"
            f"Wins       : {wins}  |  Losses: {losses}\n"
            f"Win Rate   : {win_rate:.1f}%\n"
            f"Balance    : ₹{balance:,.2f}\n"
            f"Date       : {datetime.now().strftime('%d %b %Y')}"
        )

    def error(self, error_type: str, message: str):
        self._send(
            "🚨 <b>ERROR</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Type    : <code>{error_type}</code>\n"
            f"Message : {message}\n"
            f"Time    : {self._now()}"
        )

    # ─────────────────────────────────────────────────────────
    # TWO-WAY COMMAND CONTROL
    # ─────────────────────────────────────────────────────────

    def start_command_listener(self, bot_ref):
        """
        Polls Telegram every 3 seconds for incoming commands.
        bot_ref = the running bot object so commands can control it.
        Runs in background thread.
        """
        self._running = True

        def _poll():
            while self._running:
                try:
                    response = requests.get(
                        f"{self.BASE_URL}/getUpdates",
                        params={
                            "offset":  self._last_update + 1,
                            "timeout": 2
                        },
                        timeout=10
                    )
                    if response.status_code == 200:
                        updates = response.json().get("result", [])
                        for update in updates:
                            self._last_update = update["update_id"]
                            msg = update.get("message", {})
                            text = msg.get("text", "").strip().lower()
                            self._handle_command(text, bot_ref)
                except Exception as e:
                    logger.error(f"Command poll error: {e}")
                time.sleep(3)

        thread = threading.Thread(target=_poll, daemon=True)
        thread.start()
        logger.info("✅ Telegram command listener started")
        return thread

    def _handle_command(self, command: str, bot_ref):
        """Route incoming Telegram commands to bot actions."""

        if command == "/stop":
            self.bot_paused = True
            self._send(
                "⛔ <b>TRADING PAUSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "No new orders will be placed.\n"
                f"Time : {self._now()}\n\n"
                "Send /resume to restart."
            )

        elif command == "/resume":
            self.bot_paused = False
            self._send(
                "✅ <b>TRADING RESUMED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Bot is now active again.\n"
                f"Time : {self._now()}"
            )

        elif command == "/status":
            positions = bot_ref.positions if hasattr(bot_ref, "positions") else {}
            risk      = bot_ref.risk      if hasattr(bot_ref, "risk")      else None
            lines = ""
            for key, pos in positions.items():
                if pos:
                    lines += (f"  {key.upper()}: {pos['qty']} units "
                              f"@ ₹{pos['entry']:,.2f}\n")
            if not lines:
                lines = "  No open positions\n"
            pnl_line = f"₹{risk.daily_pnl:+,.2f}" if risk else "N/A"
            self._send(
                "📊 <b>BOT STATUS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Running    : {'⛔ PAUSED' if self.bot_paused else '✅ ACTIVE'}\n"
                f"Daily PnL  : {pnl_line}\n"
                f"Positions  :\n{lines}"
                f"Time       : {self._now()}"
            )

        elif command == "/positions":
            positions = bot_ref.positions if hasattr(bot_ref, "positions") else {}
            lines = ""
            for key, pos in positions.items():
                if pos:
                    lines += (f"  {key.upper()}: {pos['qty']} units "
                              f"@ ₹{pos['entry']:,.2f}\n")
            if not lines:
                lines = "  No open positions\n"
            self._send(
                "📋 <b>OPEN POSITIONS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"{lines}"
                f"Time : {self._now()}"
            )

        elif command == "/pnl":
            risk = bot_ref.risk if hasattr(bot_ref, "risk") else None
            if risk:
                emoji = "✅" if risk.daily_pnl >= 0 else "❌"
                self._send(
                    f"{emoji} <b>TODAY'S PnL</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"Daily PnL  : ₹{risk.daily_pnl:+,.2f}\n"
                    f"Trades     : {risk.trades_today}\n"
                    f"Time       : {self._now()}"
                )

        elif command == "/help":
            self._send(
                "ℹ️ <b>AVAILABLE COMMANDS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "/status    → bot status + positions\n"
                "/positions → open positions\n"
                "/pnl       → today's PnL\n"
                "/stop      → pause all trading\n"
                "/resume    → resume trading\n"
                "/help      → show this message"
            )

    def stop_listener(self):
        self._running = False

    # ─────────────────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now().strftime("%d %b %Y %H:%M:%S")