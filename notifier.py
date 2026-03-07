import requests
import logging
import threading
import time
from datetime import datetime, timedelta
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends alerts to Telegram and listens for commands.
    All sends are non-blocking (background thread).
    """

    BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    def __init__(self):
        self.chat_id      = TELEGRAM_CHAT_ID
        self.bot_paused   = False
        self._last_update = 0
        self._running     = False
        self._bot_ref     = None

    # ─────────────────────────────────────────────────────────
    # CORE SEND
    # ─────────────────────────────────────────────────────────

    def send(self, message: str):
        self._send(message)

    def _send(self, message: str):
        def _do_send():
            try:
                response = requests.post(
                    f"{self.BASE_URL}/sendMessage",
                    json={
                        "chat_id":    self.chat_id,
                        "text":       message,
                        "parse_mode": "HTML"
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
            "<b>📊 Info</b>\n"
            "/status    → system health\n"
            "/risk      → risk dashboard\n"
            "/funds     → full balance breakdown\n"
            "/pnl       → today's PnL\n"
            "/positions → open positions\n\n"
            "<b>⚙️ Controls</b>\n"
            "/stop      → pause trading\n"
            "/resume    → resume trading\n"
            "/pause 30  → pause for N mins\n\n"
            "<b>💰 Risk Controls</b>\n"
            "/setmode MIS     → intraday equity\n"
            "/setmode CNC     → delivery equity\n"
            "/setmode MTF     → leveraged delivery\n"
            "/setmode MARGIN  → futures &amp; options\n"
            "/setcapital 2000 → limit capital ₹\n"
            "/setlimit 5000   → max ₹ per trade\n"
            "/setdaily 3000   → max daily loss ₹\n"
            "/setslots 3 2    → equity/FNO slots\n"
            "/settrades 10    → max trades/day\n\n"
            "Send /help anytime for full list"
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
        emoji = "✅" if pnl >= 0 else "❌"
        pct   = ((exit_price - entry) / entry) * 100
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
        pct = (abs(daily_pnl) / balance) * 100 if balance else 0
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
                lines += (
                    f"  {key.upper()} — "
                    f"{pos['qty']} units @ ₹{pos['entry']:,.2f}\n"
                )
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
            "1️⃣ INDstocks → Algo Access → New Token\n"
            "2️⃣ SSH into VM → run update_token.sh\n"
            "3️⃣ Bot restarts automatically"
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
    # COMMAND LISTENER
    # ─────────────────────────────────────────────────────────

    def start_command_listener(self, bot_ref=None):
        self._bot_ref = bot_ref
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
                            msg  = update.get("message", {})
                            text = msg.get("text", "").strip()
                            if text:
                                self._handle_command(text)
                except Exception as e:
                    logger.error(f"Command poll error: {e}")
                time.sleep(3)

        threading.Thread(target=_poll, daemon=True).start()
        logger.info("✅ Telegram command listener started")

    def _handle_command(self, text: str):
        parts = text.strip().split()
        cmd   = parts[0].lower() if parts else ""
        args  = parts[1:]
        ref   = self._bot_ref

        # ── /help ─────────────────────────────────────────
        if cmd == "/help":
            self._send(
                "ℹ️ <b>AVAILABLE COMMANDS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "<b>📊 Info</b>\n"
                "/status      → system health\n"
                "/risk        → risk dashboard\n"
                "/funds       → full balance breakdown\n"
                "/pnl         → today's PnL\n"
                "/positions   → open positions\n\n"
                "<b>⚙️ Controls</b>\n"
                "/stop            → pause all trading\n"
                "/resume          → resume trading\n"
                "/pause 30        → pause for N minutes\n\n"
                "<b>💰 Risk Controls</b>\n"
                "/setmode MIS     → intraday equity\n"
                "/setmode CNC     → delivery equity\n"
                "/setmode MTF     → leveraged delivery\n"
                "/setmode MARGIN  → futures &amp; options\n"
                "/setcapital 2000 → limit capital to ₹2,000\n"
                "/setcapital 0    → remove capital limit\n"
                "/setlimit 5000   → max ₹ per trade\n"
                "/setlimit 0      → remove trade limit\n"
                "/setdaily 3000   → max daily loss ₹\n"
                "/setdaily 0      → revert to 3% auto\n"
                "/setslots 3 2    → equity / FNO slots\n"
                "/settrades 10    → max trades today"
            )

        # ── /funds ────────────────────────────────────────
        elif cmd == "/funds":
            if not ref:
                return
            f = ref.get_all_balances()
            if not f:
                self._send("❌ Could not fetch funds. Check token.")
                return
            avl = f.get("detailed_avl_balance", {})
            self._send(
                "💵 <b>FUNDS BREAKDOWN</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "<b>📈 Equity</b>\n"
                f"Intraday (MIS)   : ₹{avl.get('eq_mis', 0):,.2f}\n"
                f"Delivery (CNC)   : ₹{avl.get('eq_cnc', 0):,.2f}\n"
                f"Leveraged (MTF)  : ₹{avl.get('eq_mtf', 0):,.2f}\n\n"
                "<b>📊 Derivatives</b>\n"
                f"Futures (MARGIN) : ₹{avl.get('future', 0):,.2f}\n"
                f"Options Buy      : ₹{avl.get('option_buy', 0):,.2f}\n"
                f"Options Sell     : ₹{avl.get('option_sell', 0):,.2f}\n\n"
                "<b>💰 Account</b>\n"
                f"Opening Balance  : ₹{f.get('sod_balance', 0):,.2f}\n"
                f"Funds Added      : ₹{f.get('funds_added', 0):,.2f}\n"
                f"Funds Withdrawn  : ₹{f.get('funds_withdrawn', 0):,.2f}\n"
                f"Withdrawal Avail : ₹{f.get('withdrawal_balance', 0):,.2f}\n\n"
                "<b>📉 Today's Activity</b>\n"
                f"Realized PnL     : ₹{f.get('realized_pnl', 0):+,.2f}\n"
                f"Unrealized PnL   : ₹{f.get('unrealized_pnl', 0):+,.2f}\n"
                f"Brokerage        : ₹{f.get('brokerage', 0):,.2f}\n"
                f"Equity Charges   : ₹{f.get('eq_charges', 0):,.2f}\n"
                f"FNO Charges      : ₹{f.get('fno_charges', 0):,.2f}\n\n"
                "<b>🔒 Pledge</b>\n"
                f"Pledge Received  : ₹{f.get('pledge_received', 0):,.2f}\n"
                f"Pledge Remaining : ₹{f.get('pledge_remained', 0):,.2f}\n\n"
                f"Time : {self._now()}"
            )

        # ── /status ───────────────────────────────────────
        elif cmd == "/status":
            if not ref:
                return
            st = ref.posmgr.status()
            self._send(
                "📊 <b>BOT STATUS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode       : {'⏸ Paused' if self.bot_paused else '✅ Live'}\n"
                f"Trade Mode : {ref.TRADE_MODE} "
                f"{'(intraday)' if ref.TRADE_MODE == 'MIS' else '(delivery)' if ref.TRADE_MODE == 'CNC' else '(leveraged)' if ref.TRADE_MODE == 'MTF' else '(derivatives)'}\n"
                f"Positions  : {st['open_positions']} open\n"
                f"Equity     : {st['equity_open']}/{ref.MAX_EQUITY_POS} slots\n"
                f"FNO        : {st['fno_open']}/{ref.MAX_FNO_POS} slots\n"
                f"Win Rate   : {st['win_rate']:.1f}%\n"
                f"Daily PnL  : ₹{ref.risk.daily_pnl:+,.0f}\n"
                f"Trades     : {ref.risk.trades_today}/{ref.risk.max_trades_per_day}\n"
                f"Time       : {self._now()}"
            )

        # ── /risk ─────────────────────────────────────────
        elif cmd == "/risk":
            if not ref:
                return
            seg   = "DERIVATIVE" if ref.TRADE_MODE == "MARGIN" else "EQUITY"
            bal   = ref.get_balance(seg)
            limit = ref.risk.get_daily_limit(bal)
            used  = abs(min(ref.risk.daily_pnl, 0))
            pct   = (used / limit * 100) if limit else 0
            st    = ref.posmgr.status()
            pt    = ref.risk.per_trade_limit
            dc    = ref.risk.daily_loss_cap
            tc    = ref.TRADE_CAPITAL
            self._send(
                "💰 <b>RISK DASHBOARD</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Trade Mode       : {ref.TRADE_MODE} "
                f"{'(intraday)' if ref.TRADE_MODE == 'MIS' else '(delivery)' if ref.TRADE_MODE == 'CNC' else '(leveraged)' if ref.TRADE_MODE == 'MTF' else '(derivatives)'}\n"
                f"Active Balance   : ₹{bal:,.0f}\n"
                f"Capital Limit    : {'₹' + f'{tc:,.0f}' if tc else 'No limit (use all)'}\n"
                f"Daily Loss Limit : ₹{limit:,.0f} {'(custom)' if dc else '(3% auto)'}\n"
                f"Daily Loss Used  : ₹{used:,.0f} ({pct:.0f}%)\n"
                f"Per Trade Limit  : {'₹' + f'{pt:,.0f}' if pt else 'No limit set'}\n"
                f"Max Trades/Day   : {ref.risk.max_trades_per_day}\n"
                f"Trades Today     : {ref.risk.trades_today}\n\n"
                f"<b>Positions</b>\n"
                f"Equity : {st['equity_open']}/{ref.MAX_EQUITY_POS} slots used\n"
                f"FNO    : {st['fno_open']}/{ref.MAX_FNO_POS} slots used\n\n"
                f"Status : {'⏸ Paused' if self.bot_paused else '✅ Active'}\n"
                f"Time   : {self._now()}"
            )

        # ── /pnl ──────────────────────────────────────────
        elif cmd == "/pnl":
            if not ref:
                return
            emoji = "✅" if ref.risk.daily_pnl >= 0 else "❌"
            f     = ref.get_all_balances()
            self._send(
                f"{emoji} <b>TODAY'S PnL</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Daily PnL      : ₹{ref.risk.daily_pnl:+,.2f}\n"
                f"Realized PnL   : ₹{f.get('realized_pnl', 0):+,.2f}\n"
                f"Unrealized PnL : ₹{f.get('unrealized_pnl', 0):+,.2f}\n"
                f"Brokerage      : ₹{f.get('brokerage', 0):,.2f}\n"
                f"Trades         : {ref.risk.trades_today}\n"
                f"Time           : {self._now()}"
            )

        # ── /positions ────────────────────────────────────
        elif cmd == "/positions":
            if not ref:
                return
            positions = ref.posmgr.positions
            if not positions:
                self._send("📭 No open positions")
                return
            lines = ["📋 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━"]
            for sc, pos in positions.items():
                lines.append(
                    f"<b>{pos['name']}</b>\n"
                    f"  Qty     : {pos['qty']}\n"
                    f"  Entry   : ₹{pos['entry']:,.2f}\n"
                    f"  Segment : {pos['segment']}"
                )
            lines.append(f"Time : {self._now()}")
            self._send("\n\n".join(lines))

        # ── /stop ─────────────────────────────────────────
        elif cmd == "/stop":
            self.bot_paused = True
            self._send(
                "⛔ <b>TRADING PAUSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "No new orders will be placed.\n"
                f"Time : {self._now()}\n\n"
                "Send /resume to restart."
            )

        # ── /resume ───────────────────────────────────────
        elif cmd == "/resume":
            self.bot_paused = False
            self._send(
                "✅ <b>TRADING RESUMED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Bot is now active again.\n"
                f"Time : {self._now()}"
            )

        # ── /pause <minutes> ──────────────────────────────
        elif cmd == "/pause":
            minutes = int(args[0]) if args and args[0].isdigit() else 30
            self.bot_paused = True
            resume_at = datetime.now() + timedelta(minutes=minutes)
            self._send(
                "⏸ <b>BOT PAUSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Duration  : {minutes} minutes\n"
                f"Resumes   : {resume_at.strftime('%I:%M %p')}\n"
                f"Time      : {self._now()}"
            )
            def _auto_resume():
                time.sleep(minutes * 60)
                self.bot_paused = False
                self._send(
                    "▶️ <b>BOT AUTO-RESUMED</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "Pause period ended.\n"
                    f"Time : {self._now()}"
                )
            threading.Thread(target=_auto_resume, daemon=True).start()

        # ── /setmode <MIS|CNC|MTF|MARGIN> ────────────────
        elif cmd == "/setmode":
            if not args:
                self._send(
                    "Usage: <code>/setmode MIS</code>\n\n"
                    "<b>Available modes:</b>\n"
                    "MIS    → intraday equity (eq_mis)\n"
                    "CNC    → delivery equity (eq_cnc)\n"
                    "MTF    → leveraged delivery (eq_mtf)\n"
                    "MARGIN → futures &amp; options (future)"
                )
                return
            mode = args[0].upper()
            if mode not in ("MIS", "CNC", "MTF", "MARGIN"):
                self._send(
                    "❌ Invalid mode.\n"
                    "Valid: <code>MIS</code>, <code>CNC</code>, "
                    "<code>MTF</code>, <code>MARGIN</code>"
                )
                return
            ref.TRADE_MODE = mode
            seg = "DERIVATIVE" if mode == "MARGIN" else "EQUITY"
            bal = ref.get_balance(seg)
            mode_desc = {
                "MIS":    "Intraday equity — auto-closes 3:10PM",
                "CNC":    "Delivery equity — hold overnight",
                "MTF":    "Leveraged delivery — hold with margin",
                "MARGIN": "Futures &amp; options — derivatives"
            }
            self._send(
                f"✅ <b>Trade mode set to {mode}</b>\n"
                f"{mode_desc[mode]}\n"
                f"Available balance : ₹{bal:,.0f}"
            )

        # ── /setcapital <amount> ──────────────────────────
        elif cmd == "/setcapital":
            if not args:
                self._send(
                    "Usage: <code>/setcapital 2000</code>\n"
                    "Limits total capital the bot can use.\n"
                    "Use <code>/setcapital 0</code> to use all funds."
                )
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.TRADE_CAPITAL = None
                    self._send(
                        "✅ Capital limit <b>removed</b>\n"
                        "Bot will use all available funds."
                    )
                else:
                    ref.TRADE_CAPITAL = amount
                    self._send(
                        f"✅ <b>Capital limit set to ₹{amount:,.0f}</b>\n"
                        f"Bot will only deploy up to ₹{amount:,.0f}."
                    )
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setcapital 2000</code>")

        # ── /setlimit <amount> ────────────────────────────
        elif cmd == "/setlimit":
            if not args:
                self._send(
                    "Usage: <code>/setlimit 5000</code>\n"
                    "Sets max ₹ value for any single trade.\n"
                    "Use <code>/setlimit 0</code> to remove."
                )
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.risk.per_trade_limit = None
                    self._send("✅ Per-trade limit <b>removed</b>")
                else:
                    ref.risk.per_trade_limit = amount
                    self._send(
                        f"✅ <b>Per-trade limit: ₹{amount:,.0f}</b>\n"
                        "Takes effect on next cycle."
                    )
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setlimit 5000</code>")

        # ── /setdaily <amount> ────────────────────────────
        elif cmd == "/setdaily":
            if not args:
                self._send(
                    "Usage: <code>/setdaily 3000</code>\n"
                    "Sets max ₹ daily loss before bot stops.\n"
                    "Use <code>/setdaily 0</code> to revert to 3%."
                )
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.risk.daily_loss_cap = None
                    self._send("✅ Daily loss cap removed — reverting to 3%")
                else:
                    ref.risk.daily_loss_cap = amount
                    self._send(
                        f"✅ <b>Daily loss cap: ₹{amount:,.0f}</b>\n"
                        "Takes effect immediately."
                    )
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setdaily 3000</code>")

        # ── /setslots <equity> <fno> ──────────────────────
        elif cmd == "/setslots":
            if len(args) < 2:
                self._send("Usage: <code>/setslots 3 2</code>")
                return
            try:
                eq  = int(args[0])
                fno = int(args[1])
                ref.posmgr.max_equity = eq
                ref.posmgr.max_fno    = fno
                ref.MAX_EQUITY_POS    = eq
                ref.MAX_FNO_POS       = fno
                self._send(
                    f"✅ <b>Slots updated</b>\n"
                    f"Equity : max {eq}\n"
                    f"FNO    : max {fno}"
                )
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setslots 3 2</code>")

        # ── /settrades <n> ────────────────────────────────
        elif cmd == "/settrades":
            if not args:
                self._send("Usage: <code>/settrades 10</code>")
                return
            try:
                n = int(args[0])
                ref.risk.max_trades_per_day = n
                self._send(
                    f"✅ <b>Max trades/day: {n}</b>\n"
                    f"Used today: {ref.risk.trades_today}"
                )
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/settrades 10</code>")

        elif cmd.startswith("/"):
            self._send(
                f"❓ Unknown command: <code>{cmd}</code>\n"
                "Send /help to see all commands."
            )

    def stop_listener(self):
        self._running = False

    def _now(self) -> str:
        return datetime.now().strftime("%d %b %Y %H:%M:%S")
