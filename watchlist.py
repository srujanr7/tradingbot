import logging
import time
import io
import threading
import requests
import os
import pandas as pd
from api import INDstocksAPI

logger = logging.getLogger(__name__)


class FullMarketScanner:
    """
    Two-tier scanner:

    Tier 1 (every 30 min):
      - Downloads full instruments CSV from API
      - Filters by liquidity, segment, series
      - Quick-scores all stocks by volume + momentum
      - Shortlists top 50 candidates

    Tier 2 (every 60 sec, called by bot.py):
      - Runs ML signals only on shortlisted candidates
      - Returns ranked actionable signals
    """

    TIER1_SHORTLIST  = 50
    MIN_VOLUME       = 100_000
    MIN_PRICE_EQUITY = 50.0
    MIN_PRICE_FNO    = 1.0
    TIER1_INTERVAL   = 30 * 60
    BATCH_SIZE       = 100   # safe limit for GET query string length

    def __init__(self, api: INDstocksAPI,
                 top_n_equity: int = 10,
                 top_n_fno: int    = 5,
                 top_n_index: int  = 2):
        self.api           = api
        self.top_n_equity  = top_n_equity
        self.top_n_fno     = top_n_fno
        self.top_n_index   = top_n_index

        self.universe_equity = []
        self.universe_fno    = []
        self.universe_index  = []

        self.shortlist_equity = []
        self.shortlist_fno    = []
        self.shortlist_index  = []

        self._last_tier1 = 0
        self._lock       = threading.Lock()

    # ── Tier 0: Load full instrument universe ─────────────────

    def load_universe(self):
        logger.info("📥 Loading full instrument universe from API...")

        try:
            equity_df = self._fetch_instruments("equity")
            if equity_df is not None:
                self.universe_equity = self._parse_equity(equity_df)
                logger.info(
                    f"✅ Equity universe: "
                    f"{len(self.universe_equity)} instruments"
                )
        except Exception as e:
            logger.error(f"Equity universe load error: {e}")

        try:
            fno_df = self._fetch_instruments("fno")
            if fno_df is not None:
                self.universe_fno = self._parse_fno(fno_df)
                logger.info(
                    f"✅ FNO universe: "
                    f"{len(self.universe_fno)} instruments"
                )
        except Exception as e:
            logger.error(f"FNO universe load error: {e}")

        try:
            index_df = self._fetch_instruments("index")
            if index_df is not None:
                self.universe_index = self._parse_index(index_df)
                logger.info(
                    f"✅ Index universe: "
                    f"{len(self.universe_index)} instruments"
                )
        except Exception as e:
            logger.error(f"Index universe load error: {e}")

        logger.info(
            f"📦 Universe loaded — "
            f"Equity: {len(self.universe_equity)} | "
            f"FNO: {len(self.universe_fno)} | "
            f"Index: {len(self.universe_index)}"
        )

    def _fetch_instruments(self, source: str) -> pd.DataFrame:
        token = os.getenv("INDSTOCKS_TOKEN")
        try:
            response = requests.get(
                "https://api.indstocks.com/market/instruments",
                headers={"Authorization": token},
                params={"source": source},
                timeout=30
            )
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                df.columns = df.columns.str.strip()
                logger.info(f"  {source}: {len(df)} rows downloaded")
                return df
            else:
                logger.error(
                    f"Instruments fetch failed ({source}): "
                    f"{response.status_code}"
                )
                return None
        except Exception as e:
            logger.error(f"Instruments fetch error ({source}): {e}")
            return None

    def _parse_equity(self, df: pd.DataFrame) -> list:
        df.columns = df.columns.str.strip()
        try:
            filtered = df[
                (df["EXCH"]            == "NSE") &
                (df["SERIES"]          == "EQ") &
                (df["INSTRUMENT_NAME"] == "EQUITY")
            ].copy()
        except KeyError as e:
            logger.error(
                f"Equity filter error — missing column: {e}. "
                f"Available: {df.columns.tolist()}"
            )
            return []

        instruments = []
        for _, row in filtered.iterrows():
            try:
                sid  = str(row["SECURITY_ID"])
                name = str(
                    row.get("SYMBOL_NAME") or
                    row.get("TRADING_SYMBOL") or
                    sid
                )
                instruments.append({
                    "name":            name,
                    "scrip_code":      f"NSE_{sid}",
                    "security_id":     sid,
                    "ws_token":        f"NSE:{sid}",
                    "segment":         "EQUITY",
                    "product":         "CNC",
                    "exchange":        "NSE",
                    "instrument_type": "EQUITY",
                    "tick_size":       float(row.get("TICK_SIZE", 0.05)),
                    "lot_units":       int(row.get("LOT_UNITS", 1)),
                })
            except Exception as e:
                logger.warning(f"Skipping equity row: {e}")
                continue

        return instruments

    def _parse_fno(self, df: pd.DataFrame) -> list:
        """
        Filter FNO CSV to current-month futures only.
        scrip_code uses NFO_ prefix to match /market/quotes API format.
        ws_token uses NFO: prefix for WebSocket subscriptions.
        Both confirmed from API docs: SEGMENT_TOKEN and SEGMENT:TOKEN formats.
        """
        df.columns = df.columns.str.strip()
        try:
            filtered = df[
                (df["INSTRUMENT_NAME"].isin(["FUTSTK", "FUTIDX"])) &
                (df["EXPIRY_FLAG"] == "M")
            ].copy()
        except KeyError as e:
            logger.error(
                f"FNO filter error — missing column: {e}. "
                f"Available: {df.columns.tolist()}"
            )
            return []

        instruments = []
        for _, row in filtered.iterrows():
            try:
                sid  = str(row["SECURITY_ID"])
                name = str(
                    row.get("SYMBOL_NAME") or
                    row.get("TRADING_SYMBOL") or
                    sid
                )
                inst_name = str(row.get("INSTRUMENT_NAME", "FUTSTK"))
                instruments.append({
                    "name":            f"{name} Fut",
                    "scrip_code":      f"NFO_{sid}",
                    "security_id":     sid,
                    "ws_token":        f"NFO:{sid}",
                    "segment":         "DERIVATIVE",
                    "product":         "MARGIN",
                    "exchange":        "NSE",
                    "instrument_type": "FUTURES",
                    "lot_size":        int(row.get("LOT_UNITS", 1)),
                    "expiry":          str(row.get("EXPIRY_DATE", "")),
                    "tick_size":       float(row.get("TICK_SIZE", 0.05)),
                })
            except Exception as e:
                logger.warning(f"Skipping FNO row: {e}")
                continue

        return instruments

    def _parse_index(self, df: pd.DataFrame) -> list:
        """
        Parse index instruments.
        Falls back to SECURITY_ID as name if no name column exists
        (index CSV only has EXCH, SEGMENT, SECURITY_ID).
        """
        df.columns = df.columns.str.strip()
        logger.info(f"Index CSV columns: {df.columns.tolist()}")

        name_col = None
        for col in ["SYMBOL_NAME", "TRADING_SYMBOL",
                    "CUSTOM_SYMBOL", "INSTRUMENT_NAME"]:
            if col in df.columns:
                name_col = col
                logger.info(f"Using index name column: {name_col}")
                break

        if name_col is None:
            logger.warning(
                f"No name column found in index CSV. "
                f"Available: {df.columns.tolist()}. "
                f"Falling back to SECURITY_ID."
            )
            name_col = "SECURITY_ID"

        instruments = []
        for _, row in df.iterrows():
            try:
                sid = str(row["SECURITY_ID"])
                instruments.append({
                    "name":            str(row[name_col]),
                    "scrip_code":      f"NIDX_{sid}",
                    "security_id":     sid,
                    "ws_token":        f"NIDX:{sid}",
                    "segment":         "INDEX",
                    "product":         None,
                    "exchange":        "NSE",
                    "instrument_type": "INDEX",
                })
            except Exception as e:
                logger.warning(f"Skipping index row: {e}")
                continue

        logger.info(f"Index universe: {len(instruments)} instruments")
        return instruments

    # ── Tier 1: Score and shortlist ───────────────────────────

    def tier1_scan(self):
        logger.info("🔍 Tier 1: Scoring full market universe...")
        start = time.time()

        equity_shortlist = self._score_batch(
            self.universe_equity,
            min_price=self.MIN_PRICE_EQUITY,
            top_n=self.TIER1_SHORTLIST
        )
        fno_shortlist = self._score_batch(
            self.universe_fno,
            min_price=self.MIN_PRICE_FNO,
            top_n=self.TIER1_SHORTLIST // 5
        )
        index_shortlist = self.universe_index[:10]

        with self._lock:
            self.shortlist_equity = equity_shortlist
            self.shortlist_fno    = fno_shortlist
            self.shortlist_index  = index_shortlist
            self._last_tier1      = time.time()

        elapsed   = time.time() - start
        top_names = [s["name"] for s in equity_shortlist[:5]]
        logger.info(
            f"✅ Tier 1 complete in {elapsed:.1f}s | "
            f"Equity: {len(equity_shortlist)} | "
            f"FNO: {len(fno_shortlist)} | "
            f"Top 5: {top_names}"
        )

    def _score_batch(self, instruments: list,
                     min_price: float,
                     top_n: int) -> list:
        """
        Batch-fetch quotes for up to BATCH_SIZE instruments at a time.
        Score by volume + volatility + momentum.
        Return top N scored instruments.
        scrip_code format NSE_TOKEN / NFO_TOKEN matches API docs exactly.
        """
        if not instruments:
            return []

        scored     = []
        batch_size = self.BATCH_SIZE
        total      = len(instruments)

        for i in range(0, total, batch_size):
            batch       = instruments[i:i + batch_size]
            scrip_codes = ",".join([inst["scrip_code"] for inst in batch])
            batch_num   = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            try:
                quotes = self.api._request(
                    "GET", "/market/quotes/full",
                    params={"scrip-codes": scrip_codes}
                )

                if not quotes or "data" not in quotes:
                    logger.warning(
                        f"Batch {batch_num}/{total_batches}: "
                        f"empty/bad response "
                        f"(sample: {scrip_codes[:60]}...)"
                    )
                    time.sleep(0.5)
                    continue

                hits = 0
                for inst in batch:
                    code = inst["scrip_code"]
                    data = quotes["data"].get(code, {})
                    if not data:
                        continue

                    price  = data.get("live_price", 0)
                    volume = data.get("volume", 0)
                    change = abs(data.get("day_change_percentage", 0))
                    high   = data.get("day_high", price)
                    low    = data.get("day_low",  price)

                    if price < min_price:
                        continue
                    if volume < self.MIN_VOLUME:
                        continue

                    vol_score = min(volume / 1_000_000, 40)
                    atr_pct   = (
                        (high - low) / price * 100
                        if price > 0 else 0
                    )
                    atr_score = min(atr_pct * 10, 40)
                    mom_score = min(change * 4, 20)
                    total_score = vol_score + atr_score + mom_score

                    scored.append({
                        **inst,
                        "score":  round(total_score, 2),
                        "ltp":    price,
                        "volume": volume,
                        "change": change
                    })
                    hits += 1

                logger.debug(
                    f"Batch {batch_num}/{total_batches}: "
                    f"{hits}/{len(batch)} instruments scored"
                )

            except Exception as e:
                logger.error(
                    f"Batch {batch_num}/{total_batches} score error "
                    f"(sample: {scrip_codes[:60]}...): {e}"
                )

            time.sleep(0.5)

        scored.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            f"Scoring complete: {len(scored)} passed filters, "
            f"returning top {min(top_n, len(scored))}"
        )
        return scored[:top_n]

    # ── Public API ────────────────────────────────────────────

    def get_active(self) -> list:
        if time.time() - self._last_tier1 > self.TIER1_INTERVAL:
            threading.Thread(
                target=self.tier1_scan,
                daemon=True
            ).start()

        with self._lock:
            return (
                self.shortlist_equity[:self.top_n_equity] +
                self.shortlist_fno[:self.top_n_fno]
            )

    def get_ws_tokens(self) -> list:
        with self._lock:
            all_insts = (
                self.shortlist_equity +
                self.shortlist_fno +
                self.shortlist_index
            )
            return [inst["ws_token"] for inst in all_insts]

    def start_background_refresh(self):
        def _loop():
            while True:
                try:
                    self.tier1_scan()
                except Exception as e:
                    logger.error(f"Tier 1 background error: {e}")
                time.sleep(self.TIER1_INTERVAL)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        logger.info("✅ Background market scanner started")
        return thread
