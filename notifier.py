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
    All sends are fire-and-forget background threads.
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
        def _do():
            try:
                r = requests.post(
                    f"{self.BASE_URL}/sendMessage",
                    json={
                        "chat_id":    self.chat_id,
                        "text":       message,
                        "parse_mode": "HTML"
                    },
                    timeout=10
                )
                if r.status_code != 200:
                    logger.error(f"Telegram send failed: {r.text}")
            except Exception as e:
                logger.error(f"Telegram error: {e}")
        threading.Thread(target=_do, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # ALERT TYPES
    # ─────────────────────────────────────────────────────────

    def bot_started(self):
        self._send(
            "🚀 <b>BOT STARTED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Time   : {self._now()}\n"
            "Status : All systems running\n\n"
            "<b>📊 Info</b>\n"
            "/status    → system health\n"
            "/risk      → risk dashboard\n"
            "/funds     → balance breakdown\n"
            "/pnl       → today's PnL\n"
            "/positions → open positions\n"
            "/models    → model status\n"
            "/performance → all-time stats\n\n"
            "<b>⚙️ Controls</b>\n"
            "/stop      → pause trading\n"
            "/resume    → resume trading\n"
            "/pause 30  → pause N minutes\n\n"
            "<b>💰 Risk Controls</b>\n"
            "/setmode MIS|CNC|MTF|MARGIN\n"
            "/setcapital 2000\n"
            "/setlimit 5000\n"
            "/setdaily 3000\n"
            "/setslots 3 2\n"
            "/settrades 10\n\n"
            "Send /help for full list"
        )

    def bot_stopped(self, reason: str = "Manual shutdown"):
        self._send(
            "🔴 <b>BOT STOPPED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Reason : {reason}\n"
            f"Time   : {self._now()}"
        )

    def trade_executed(self, side: str, name: str,
                       segment: str, qty: int, price: float,
                       confidence: float, xgb: str, rl: str,
                       lgbm: str = "", lstm: str = "",
                       pattern: str = "", regime: str = ""):
        emoji = "🟢" if side == "BUY" else "🔴"
        extra = ""
        if lgbm:
            extra += f"LGBM       : {lgbm}\n"
        if lstm:
            extra += f"LSTM       : {lstm}\n"
        if pattern and pattern != "NONE":
            extra += f"Pattern    : {pattern}\n"
        if regime:
            extra += f"Regime     : {regime}\n"
        self._send(
            f"{emoji} <b>{side} EXECUTED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Segment    : {segment}\n"
            f"Qty        : {qty}\n"
            f"Price      : ₹{price:,.2f}\n"
            f"XGB        : {xgb}\n"
            f"RL         : {rl}\n"
            f"{extra}"
            f"Confidence : {confidence*100:.1f}%\n"
            f"Time       : {self._now()}"
        )

    def trade_closed(self, name: str, qty: int,
                     entry: float, exit_price: float,
                     pnl: float, daily_pnl: float):
        emoji = "✅" if pnl >= 0 else "❌"
        pct   = (exit_price - entry) / entry * 100
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
            f"Side   : {side}\n"
            f"Name   : {name}\n"
            f"Qty    : {qty}\n"
            f"Reason : {reason}\n"
            f"Time   : {self._now()}"
        )

    def insufficient_margin(self, name: str,
                             required: float, available: float):
        self._send(
            "⚠️ <b>INSUFFICIENT MARGIN</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument : {name}\n"
            f"Required   : ₹{required:,.2f}\n"
            f"Available  : ₹{available:,.2f}\n"
            f"Action     : Skipped\n"
            f"Time       : {self._now()}"
        )

    def risk_warning(self, daily_pnl: float,
                     limit: float, balance: float):
        pct = abs(daily_pnl) / balance * 100 if balance else 0
        self._send(
            "⚠️ <b>RISK WARNING</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Daily Loss : ₹{daily_pnl:,.2f} ({pct:.1f}%)\n"
            f"Limit      : ₹{limit:,.2f}\n"
            f"Time       : {self._now()}"
        )

    def kill_switch(self, daily_pnl: float, limit: float):
        self._send(
            "🔴 <b>KILL SWITCH ACTIVATED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Daily Loss : ₹{daily_pnl:,.2f}\n"
            f"Limit Hit  : ₹{limit:,.2f}\n"
            f"Action     : Trading PAUSED\n"
            f"Time       : {self._now()}\n\n"
            "Send /resume tomorrow after reset."
        )

    def squareoff_alert(self, positions: dict):
        lines = ""
        for key, pos in positions.items():
            if pos:
                lines += (
                    f"  {key.upper()} — "
                    f"{pos['qty']} @ ₹{pos['entry']:,.2f}\n"
                )
        if not lines:
            lines = "  No open positions\n"
        self._send(
            "⏰ <b>AUTO SQUARE-OFF</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Time   : {self._now()}\n"
            f"Closed :\n{lines}"
        )

    def model_retrained(self, instrument: str, samples: int,
                        duration_seconds: float,
                        next_retrain: str):
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
            "❌ <b>TOKEN EXPIRED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Action : Trading paused\n"
            f"Time   : {self._now()}\n\n"
            "1️⃣ INDstocks → Algo Access → New Token\n"
            "2️⃣ SSH → run update_token.sh\n"
            "3️⃣ Bot restarts automatically"
        )

    def daily_summary(self, total_pnl: float, trades: int,
                      wins: int, balance: float):
        losses   = trades - wins
        win_rate = wins / trades * 100 if trades > 0 else 0
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
                    r = requests.get(
                        f"{self.BASE_URL}/getUpdates",
                        params={
                            "offset":  self._last_update + 1,
                            "timeout": 2
                        },
                        timeout=10
                    )
                    if r.status_code == 200:
                        for update in r.json().get("result", []):
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

        if cmd == "/help":
            self._send(
                "ℹ️ <b>AVAILABLE COMMANDS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                "<b>📊 Info</b>\n"
                "/status       → system health\n"
                "/risk         → risk dashboard\n"
                "/funds        → balance breakdown\n"
                "/pnl          → today's PnL\n"
                "/positions    → open positions\n"
                "/models       → model status\n"
                "/performance  → all-time stats\n"
                "/intervals    → active intervals\n\n"
                "<b>⚙️ Controls</b>\n"
                "/stop             → pause trading\n"
                "/resume           → resume trading\n"
                "/pause 30         → pause N minutes\n\n"
                "<b>💰 Risk Controls</b>\n"
                "/setmode MIS      → intraday equity\n"
                "/setmode CNC      → delivery equity\n"
                "/setmode MTF      → leveraged delivery\n"
                "/setmode MARGIN   → futures &amp; options\n"
                "/setcapital 2000  → limit capital ₹\n"
                "/setcapital 0     → remove limit\n"
                "/setlimit 5000    → max ₹ per trade\n"
                "/setlimit 0       → remove limit\n"
                "/setdaily 3000    → max daily loss ₹\n"
                "/setdaily 0       → revert to 3% auto\n"
                "/setslots 3 2     → equity/FNO slots\n"
                "/settrades 10     → max trades today"
            )

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
                f"Withdrawal Avail : ₹{f.get('withdrawal_balance', 0):,.2f}\n\n"
                "<b>📉 Today</b>\n"
                f"Realized PnL     : ₹{f.get('realized_pnl', 0):+,.2f}\n"
                f"Unrealized PnL   : ₹{f.get('unrealized_pnl', 0):+,.2f}\n"
                f"Brokerage        : ₹{f.get('brokerage', 0):,.2f}\n\n"
                f"Time : {self._now()}"
            )

        elif cmd == "/status":
            if not ref:
                return
            st = ref.posmgr.status()
            self._send(
                "📊 <b>BOT STATUS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode       : {'⏸ Paused' if self.bot_paused else '✅ Live'}\n"
                f"Trade Mode : {ref.TRADE_MODE}\n"
                f"Positions  : {st['open_positions']} open\n"
                f"Equity     : {st['equity_open']}/{ref.MAX_EQUITY_POS}\n"
                f"FNO        : {st['fno_open']}/{ref.MAX_FNO_POS}\n"
                f"Win Rate   : {st['win_rate']:.1f}%\n"
                f"Daily PnL  : ₹{ref.risk.daily_pnl:+,.0f}\n"
                f"Trades     : {ref.risk.trades_today}/{ref.risk.max_trades_per_day}\n"
                f"Time       : {self._now()}"
            )

        elif cmd == "/risk":
            if not ref:
                return
            seg   = "DERIVATIVE" if ref.TRADE_MODE == "MARGIN" else "EQUITY"
            bal   = ref.get_balance(seg)
            limit = ref.risk.get_daily_limit(bal)
            used  = abs(min(ref.risk.daily_pnl, 0))
            pct   = used / limit * 100 if limit else 0
            st    = ref.posmgr.status()
            self._send(
                "💰 <b>RISK DASHBOARD</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Trade Mode       : {ref.TRADE_MODE}\n"
                f"Active Balance   : ₹{bal:,.0f}\n"
                f"Capital Limit    : {'₹' + f'{ref.TRADE_CAPITAL:,.0f}' if ref.TRADE_CAPITAL else 'No limit'}\n"
                f"Daily Loss Limit : ₹{limit:,.0f} {'(custom)' if ref.risk.daily_loss_cap else '(3% auto)'}\n"
                f"Daily Loss Used  : ₹{used:,.0f} ({pct:.0f}%)\n"
                f"Per Trade Limit  : {'₹' + f'{ref.risk.per_trade_limit:,.0f}' if ref.risk.per_trade_limit else 'No limit'}\n"
                f"Max Trades/Day   : {ref.risk.max_trades_per_day}\n"
                f"Trades Today     : {ref.risk.trades_today}\n\n"
                f"Equity : {st['equity_open']}/{ref.MAX_EQUITY_POS} slots\n"
                f"FNO    : {st['fno_open']}/{ref.MAX_FNO_POS} slots\n\n"
                f"Status : {'⏸ Paused' if self.bot_paused else '✅ Active'}\n"
                f"Time   : {self._now()}"
            )

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

        elif cmd == "/positions":
            if not ref:
                return
            positions = ref.posmgr.positions
            if not positions:
                self._send("📭 No open positions")
                return
            lines = ["📋 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━"]
            for sc, pos in positions.items():
                meta    = pos.get("signal_meta", {})
                pattern = meta.get("pattern", "")
                regime  = meta.get("regime", "")
                lines.append(
                    f"<b>{pos['name']}</b>\n"
                    f"  Qty     : {pos['qty']}\n"
                    f"  Entry   : ₹{pos['entry']:,.2f}\n"
                    f"  Segment : {pos['segment']}\n"
                    f"  Conf    : {meta.get('confidence', 0):.1%}\n"
                    + (f"  Pattern : {pattern}\n" if pattern and pattern != "NONE" else "")
                    + (f"  Regime  : {regime}\n" if regime else "")
                )
            lines.append(f"Time : {self._now()}")
            self._send("\n\n".join(lines))

        elif cmd == "/models":
            if not ref:
                return
            import os
            model_dir = "ml/models"
            if not os.path.exists(model_dir):
                self._send("No models trained yet.")
                return
            files = os.listdir(model_dir)
            xgb   = len([f for f in files if f.startswith("xgb")])
            lgbm  = len([f for f in files if f.startswith("lgbm")])
            lstm  = len([f for f in files if f.startswith("lstm")])
            ppo   = len([f for f in files if f.startswith("ppo")])
            # MetaModel weights
            from ml.meta_model import MetaModel
            weights = MetaModel().weights
            self._send(
                "🤖 <b>MODEL STATUS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"XGBoost  : {xgb} instruments\n"
                f"LightGBM : {lgbm} instruments\n"
                f"LSTM     : {lstm} instruments\n"
                f"PPO      : {ppo} instruments\n\n"
                "<b>Meta Weights (auto-updating)</b>\n"
                f"XGB  : {weights.get('xgb', 0):.3f}\n"
                f"PPO  : {weights.get('ppo', 0):.3f}\n"
                f"A2C  : {weights.get('a2c', 0):.3f}\n"
                f"Sent : {weights.get('sentiment', 0):.3f}\n\n"
                f"Retrain  : Every Sunday 18:30\n"
                f"Interval : Every Sunday 17:00\n"
                f"Time     : {self._now()}"
            )

        elif cmd == "/intervals":
            if not ref:
                return
            lines = ["⏱ <b>ACTIVE INTERVALS</b>\n━━━━━━━━━━━━━━━━━━━━"]
            for sc, trainer in ref.trainers.items():
                lines.append(f"  {sc} → {trainer.interval}")
            lines.append(f"\nTime : {self._now()}")
            self._send("\n".join(lines))

        elif cmd == "/performance":
            if not ref:
                return
            from ml.trade_memory import TradeMemory
            stats = TradeMemory().get_stats()
            pf    = stats.get("profit_factor", 0)
            self._send(
                "📈 <b>ALL-TIME PERFORMANCE</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Total Trades   : {stats['total']}\n"
                f"Wins           : {stats['wins']}\n"
                f"Losses         : {stats['losses']}\n"
                f"Win Rate       : {stats['win_rate']}%\n"
                f"Avg Win        : ₹{stats['avg_win']:+,.2f}\n"
                f"Avg Loss       : ₹{stats['avg_loss']:+,.2f}\n"
                f"Profit Factor  : {pf:.2f}x\n"
                f"Best Trade     : ₹{stats['best_trade']:+,.2f}\n"
                f"Worst Trade    : ₹{stats['worst_trade']:+,.2f}\n"
                f"Total PnL      : ₹{stats['total_pnl']:+,.2f}\n"
                f"Time           : {self._now()}"
            )

        elif cmd == "/stop":
            self.bot_paused = True
            self._send(
                "⛔ <b>TRADING PAUSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "No new orders will be placed.\n"
                f"Time : {self._now()}\n\n"
                "Send /resume to restart."
            )

        elif cmd == "/resume":
            self.bot_paused = False
            self._send(
                "✅ <b>TRADING RESUMED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Time : {self._now()}"
            )

        elif cmd == "/pause":
            minutes   = int(args[0]) if args and args[0].isdigit() else 30
            self.bot_paused = True
            resume_at = datetime.now() + timedelta(minutes=minutes)
            self._send(
                "⏸ <b>BOT PAUSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Duration : {minutes} minutes\n"
                f"Resumes  : {resume_at.strftime('%I:%M %p')}\n"
                f"Time     : {self._now()}"
            )
            def _auto_resume():
                time.sleep(minutes * 60)
                self.bot_paused = False
                self._send(
                    "▶️ <b>AUTO-RESUMED</b>\n"
                    f"Time : {self._now()}"
                )
            threading.Thread(target=_auto_resume, daemon=True).start()

        elif cmd == "/setmode":
            if not args:
                self._send(
                    "Usage: <code>/setmode MIS</code>\n\n"
                    "MIS    → intraday equity\n"
                    "CNC    → delivery equity\n"
                    "MTF    → leveraged delivery\n"
                    "MARGIN → futures &amp; options"
                )
                return
            mode = args[0].upper()
            if mode not in ("MIS", "CNC", "MTF", "MARGIN"):
                self._send("❌ Valid: MIS, CNC, MTF, MARGIN")
                return
            ref.TRADE_MODE = mode
            seg = "DERIVATIVE" if mode == "MARGIN" else "EQUITY"
            bal = ref.get_balance(seg)
            self._send(
                f"✅ <b>Mode → {mode}</b>\n"
                f"Available : ₹{bal:,.0f}"
            )

        elif cmd == "/setcapital":
            if not args:
                self._send("Usage: <code>/setcapital 2000</code> or 0 to remove")
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.TRADE_CAPITAL = None
                    self._send("✅ Capital limit removed — using all funds.")
                else:
                    ref.TRADE_CAPITAL = amount
                    self._send(f"✅ Capital limit: ₹{amount:,.0f}")
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setcapital 2000</code>")

        elif cmd == "/setlimit":
            if not args:
                self._send("Usage: <code>/setlimit 5000</code> or 0 to remove")
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.risk.per_trade_limit = None
                    self._send("✅ Per-trade limit removed.")
                else:
                    ref.risk.per_trade_limit = amount
                    self._send(f"✅ Per-trade limit: ₹{amount:,.0f}")
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setlimit 5000</code>")

        elif cmd == "/setdaily":
            if not args:
                self._send("Usage: <code>/setdaily 3000</code> or 0 for 3% auto")
                return
            try:
                amount = float(args[0])
                if amount <= 0:
                    ref.risk.daily_loss_cap = None
                    self._send("✅ Daily cap removed — reverting to 3%.")
                else:
                    ref.risk.daily_loss_cap = amount
                    self._send(f"✅ Daily loss cap: ₹{amount:,.0f}")
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setdaily 3000</code>")

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
                    f"✅ Slots updated\n"
                    f"Equity : {eq} | FNO : {fno}"
                )
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/setslots 3 2</code>")

        elif cmd == "/settrades":
            if not args:
                self._send("Usage: <code>/settrades 10</code>")
                return
            try:
                n = int(args[0])
                ref.risk.max_trades_per_day = n
                self._send(
                    f"✅ Max trades/day: {n}\n"
                    f"Used today: {ref.risk.trades_today}"
                )
            except ValueError:
                self._send("❌ Invalid. Usage: <code>/settrades 10</code>")

        elif cmd.startswith("/"):
            self._send(
                f"❓ Unknown: <code>{cmd}</code>\n"
                "Send /help to see all commands."
            )

    def stop_listener(self):
        self._running = False

    def _now(self) -> str:
        return datetime.now().strftime("%d %b %Y %H:%M:%S")
