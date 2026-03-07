import logging
import time
import io
import threading
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

    # How many candidates pass Tier 1 into deep scan
    TIER1_SHORTLIST   = 50

    # Minimum volume to even consider a stock (avoid illiquid traps)
    MIN_VOLUME        = 100_000

    # Minimum LTP — avoid penny stocks
    MIN_PRICE_EQUITY  = 50.0
    MIN_PRICE_FNO     = 1.0

    # Refresh Tier 1 every N seconds
    TIER1_INTERVAL    = 30 * 60   # 30 minutes

    def __init__(self, api: INDstocksAPI,
                 top_n_equity: int = 10,
                 top_n_fno: int    = 5,
                 top_n_index: int  = 2):
        self.api           = api
        self.top_n_equity  = top_n_equity
        self.top_n_fno     = top_n_fno
        self.top_n_index   = top_n_index

        # Full instrument universe (loaded from CSV)
        self.universe_equity = []
        self.universe_fno    = []
        self.universe_index  = []

        # Tier 1 shortlist (scored + filtered)
        self.shortlist_equity = []
        self.shortlist_fno    = []
        self.shortlist_index  = []

        self._last_tier1     = 0
        self._lock           = threading.Lock()

    # ── Tier 0: Load full instrument universe from API ────────

    def load_universe(self):
        """
        Downloads all instruments from INDstocks API.
        Called once on startup and then every 24 hours
        (instruments don't change intraday).
        """
        logger.info("📥 Loading full instrument universe from API...")

        equity_df = self._fetch_instruments("equity")
        fno_df    = self._fetch_instruments("fno")
        index_df  = self._fetch_instruments("index")

        if equity_df is not None:
            self.universe_equity = self._parse_equity(equity_df)
            logger.info(f"✅ Equity universe: {len(self.universe_equity)} instruments")

        if fno_df is not None:
            self.universe_fno = self._parse_fno(fno_df)
            logger.info(f"✅ FNO universe: {len(self.universe_fno)} instruments")

        if index_df is not None:
            self.universe_index = self._parse_index(index_df)
            logger.info(f"✅ Index universe: {len(self.universe_index)} instruments")

    def _fetch_instruments(self, source: str) -> pd.DataFrame:
        """Download instruments CSV from API and parse into DataFrame."""
        import requests
        import os
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
                logger.info(f"  {source}: {len(df)} rows downloaded")
                return df
            else:
                logger.error(f"Instruments fetch failed ({source}): {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Instruments fetch error ({source}): {e}")
            return None

    def _parse_equity(self, df: pd.DataFrame) -> list:
        """
        Filter equity CSV to only tradeable EQ series stocks on NSE.
        Returns list of instrument dicts.
        """
        # Only NSE, EQ series, actual equity instruments
        filtered = df[
            (df["EXCH"]            == "NSE") &
            (df["SERIES"]          == "EQ") &
            (df["INSTRUMENT_NAME"] == "EQUITY")
        ].copy()

        instruments = []
        for _, row in filtered.iterrows():
            sid = str(row["SECURITY_ID"])
            instruments.append({
                "name":        row["SYMBOL_NAME"],
                "scrip_code":  f"NSE_{sid}",
                "security_id": sid,
                "ws_token":    f"NSE:{sid}",
                "segment":     "EQUITY",
                "product":     "CNC",
                "exchange":    "NSE",
                "tick_size":   float(row.get("TICK_SIZE", 0.05)),
                "lot_units":   int(row.get("LOT_UNITS", 1)),
            })
        return instruments

    def _parse_fno(self, df: pd.DataFrame) -> list:
        """
        Filter FNO CSV to only current-month futures (FUTSTK, FUTIDX).
        Excludes options to keep it manageable.
        """
        filtered = df[
            (df["EXCH"].isin(["NSE", "NFO"])) &
            (df["INSTRUMENT_NAME"].isin(["FUTSTK", "FUTIDX"])) &
            (df["EXPIRY_FLAG"] == "M")   # Monthly expiry only
        ].copy()

        instruments = []
        for _, row in filtered.iterrows():
            sid = str(row["SECURITY_ID"])
            instruments.append({
                "name":        f"{row['SYMBOL_NAME']} Fut",
                "scrip_code":  f"NFO_{sid}",
                "security_id": sid,
                "ws_token":    f"NFO:{sid}",
                "segment":     "DERIVATIVE",
                "product":     "MARGIN",
                "exchange":    "NSE",
                "lot_size":    int(row.get("LOT_UNITS", 1)),
                "expiry":      row.get("EXPIRY_DATE", ""),
                "tick_size":   float(row.get("TICK_SIZE", 0.05)),
            })
        return instruments

    def _parse_index(self, df: pd.DataFrame) -> list:
        """Parse index instruments for market regime tracking."""
        instruments = []
        for _, row in df.iterrows():
            sid = str(row["SECURITY_ID"])
            instruments.append({
                "name":        row["SYMBOL_NAME"],
                "scrip_code":  f"NIDX_{sid}",
                "security_id": sid,
                "ws_token":    f"NIDX:{sid}",
                "segment":     "INDEX",
                "product":     None,
                "exchange":    "NSE",
            })
        return instruments

    # ── Tier 1: Score and shortlist ───────────────────────────

    def tier1_scan(self):
        """
        Quick scan of full universe using only LTP + volume data.
        Much faster than full ML scan — uses batch quote API.
        Shortlists top N candidates for deep ML scan.
        """
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
            top_n=self.TIER1_SHORTLIST // 5   # Fewer FNO candidates
        )
        index_shortlist = self.universe_index[:10]  # All indices, small list

        with self._lock:
            self.shortlist_equity = equity_shortlist
            self.shortlist_fno    = fno_shortlist
            self.shortlist_index  = index_shortlist
            self._last_tier1      = time.time()

        elapsed = time.time() - start
        logger.info(
            f"✅ Tier 1 complete in {elapsed:.1f}s | "
            f"Equity: {len(equity_shortlist)} | "
            f"FNO: {len(fno_shortlist)} | "
            f"Top equity: {[s['name'] for s in equity_shortlist[:5]]}"
        )

    def _score_batch(self, instruments: list,
                     min_price: float, top_n: int) -> list:
        """
        Batch-fetch LTP for up to 1000 instruments at a time,
        score by volume + momentum, return top N.
        """
        if not instruments:
            return []

        scored = []
        batch_size = 500   # API supports up to 1000, use 500 to be safe

        for i in range(0, len(instruments), batch_size):
            batch = instruments[i:i + batch_size]
            scrip_codes = ",".join([inst["scrip_code"] for inst in batch])

            try:
                quotes = self.api._request(
                    "GET", "/market/quotes/full",
                    params={"scrip-codes": scrip_codes}
                )
                if not quotes or "data" not in quotes:
                    continue

                for inst in batch:
                    code  = inst["scrip_code"]
                    data  = quotes["data"].get(code, {})
                    if not data:
                        continue

                    price  = data.get("live_price", 0)
                    volume = data.get("volume", 0)
                    change = abs(data.get("day_change_percentage", 0))
                    high   = data.get("day_high", price)
                    low    = data.get("day_low",  price)

                    # Filter out illiquid and penny stocks
                    if price < min_price:
                        continue
                    if volume < self.MIN_VOLUME:
                        continue

                    # Score: volume (40pts) + volatility (40pts) + momentum (20pts)
                    vol_score  = min(volume / 1_000_000, 40)
                    atr_pct    = ((high - low) / price * 100) if price > 0 else 0
                    atr_score  = min(atr_pct * 10, 40)
                    mom_score  = min(change * 4, 20)
                    total      = vol_score + atr_score + mom_score

                    scored.append({
                        **inst,
                        "score":   round(total, 2),
                        "ltp":     price,
                        "volume":  volume,
                        "change":  change
                    })

            except Exception as e:
                logger.error(f"Batch score error: {e}")

            time.sleep(0.5)   # Rate limit between batches

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    # ── Public API used by bot.py ─────────────────────────────

    def get_active(self) -> list:
        """
        Returns current shortlist for ML scanning.
        Auto-refreshes Tier 1 if stale.
        """
        if time.time() - self._last_tier1 > self.TIER1_INTERVAL:
            threading.Thread(
                target=self.tier1_scan, daemon=True
            ).start()

        with self._lock:
            return (
                self.shortlist_equity[:self.top_n_equity] +
                self.shortlist_fno[:self.top_n_fno]
            )

    def get_ws_tokens(self) -> list:
        """
        Returns all WebSocket tokens from shortlist.
        Called once on startup to subscribe price feed.
        Updated when shortlist refreshes.
        """
        with self._lock:
            all_insts = (
                self.shortlist_equity +
                self.shortlist_fno +
                self.shortlist_index
            )
            return [inst["ws_token"] for inst in all_insts]

    def start_background_refresh(self):
        """Run Tier 1 scan on a background thread every 30 min."""
        def _loop():
            while True:
                try:
                    self.tier1_scan()
                except Exception as e:
                    logger.error(f"Tier 1 error: {e}")
                time.sleep(self.TIER1_INTERVAL)

        threading.Thread(target=_loop, daemon=True).start()
        logger.info("✅ Background market scanner started")