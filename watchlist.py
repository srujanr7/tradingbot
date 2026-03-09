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
      - Equity scored via quotes API (NSE_ prefix confirmed working)
      - FNO shortlisted from CSV only (quotes API rejects NFO_ prefix)
      - Shortlists top 50 equity + top 10 FNO

    Tier 2 (every 60 sec, called by bot.py):
      - Runs ML signals only on shortlisted candidates
      - Returns ranked actionable signals

    CSV columns (confirmed from API docs):
      EXCH, SEGMENT, SECURITY_ID, INSTRUMENT_NAME, EXPIRY_CODE,
      TRADING_SYMBOL, LOT_UNITS, CUSTOM_SYMBOL, EXPIRY_DATE,
      STRIKE_PRICE, OPTION_TYPE, TICK_SIZE, EXPIRY_FLAG,
      SEM_EXCH_INSTRUMENT_TYPE, SERIES, SYMBOL_NAME
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
                logger.info(
                    f"Equity CSV columns: {equity_df.columns.tolist()}"
                )
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
                logger.info(
                    f"FNO CSV columns: {fno_df.columns.tolist()}"
                )
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
                logger.info(
                    f"Index CSV columns: {index_df.columns.tolist()}"
                )
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

    def _best_name(self, row, fallback: str) -> str:
        """
        Pick the most human-readable name from a CSV row.
        Priority: SYMBOL_NAME → CUSTOM_SYMBOL → TRADING_SYMBOL → fallback
        """
        for col in ("SYMBOL_NAME", "CUSTOM_SYMBOL", "TRADING_SYMBOL"):
            val = row.get(col)
            if val and str(val).strip() and str(val).strip() != "nan":
                return str(val).strip()
        return fallback

    def _parse_equity(self, df: pd.DataFrame) -> list:
        """
        Filter: EXCH=NSE, SERIES=EQ, INSTRUMENT_NAME=EQUITY
        scrip_code → NSE_<SECURITY_ID>  (confirmed working with quotes API)
        ws_token   → NSE:<SECURITY_ID>  (confirmed from WS docs)
        """
        df.columns = df.columns.str.strip()

        required = {"EXCH", "SERIES", "INSTRUMENT_NAME", "SECURITY_ID"}
        missing  = required - set(df.columns)
        if missing:
            logger.error(
                f"Equity CSV missing required columns: {missing}. "
                f"Available: {df.columns.tolist()}"
            )
            return []

        try:
            filtered = df[
                (df["EXCH"]            == "NSE") &
                (df["SERIES"]          == "EQ") &
                (df["INSTRUMENT_NAME"] == "EQUITY")
            ].copy()
        except Exception as e:
            logger.error(f"Equity filter error: {e}")
            return []

        logger.info(f"Equity after filter: {len(filtered)} rows")

        instruments = []
        for _, row in filtered.iterrows():
            try:
                sid  = str(row["SECURITY_ID"]).strip()
                name = self._best_name(row, fallback=sid)
                instruments.append({
                    "name":            name,
                    "scrip_code":      f"NSE_{sid}",
                    "security_id":     sid,
                    "ws_token":        f"NSE:{sid}",
                    "segment":         "EQUITY",
                    "product":         "CNC",
                    "exchange":        "NSE",
                    "instrument_type": "EQUITY",
                    "tick_size":       float(row.get("TICK_SIZE") or 0.05),
                    "lot_units":       int(row.get("LOT_UNITS") or 1),
                })
            except Exception as e:
                logger.warning(f"Skipping equity row: {e}")
                continue

        return instruments

    def _parse_fno(self, df: pd.DataFrame) -> list:
        """
        Filter FNO CSV to current-month futures only.

        NOTE: The quotes API (/market/quotes/full) returns 400 for
        NFO_ scrip codes. FNO instruments are therefore shortlisted
        from CSV data alone via _shortlist_fno_from_csv() — no
        quotes API call is made for FNO instruments.

        scrip_code → NFO_<SECURITY_ID>  (stored for order placement)
        ws_token   → NFO:<SECURITY_ID>  (WebSocket subscription)
        """
        df.columns = df.columns.str.strip()

        required = {"INSTRUMENT_NAME", "EXPIRY_FLAG", "SECURITY_ID"}
        missing  = required - set(df.columns)
        if missing:
            logger.error(
                f"FNO CSV missing required columns: {missing}. "
                f"Available: {df.columns.tolist()}"
            )
            return []

        try:
            filtered = df[
                (df["INSTRUMENT_NAME"].isin(["FUTSTK", "FUTIDX"])) &
                (df["EXPIRY_FLAG"] == "M") &
                (pd.to_datetime(df["EXPIRY_DATE"]) >= pd.Timestamp.today())
            ].copy()
        except Exception as e:
            logger.error(f"FNO filter error: {e}")
            return []

        logger.info(f"FNO after filter: {len(filtered)} rows")

        instruments = []
        for _, row in filtered.iterrows():
            try:
                sid  = str(row["SECURITY_ID"]).strip()
                base = self._best_name(row, fallback=sid)
                instruments.append({
                    "name":            f"{base} Future",
                    "scrip_code":      f"NFO_{sid}",
                    "security_id":     sid,
                    "ws_token":        f"NFO:{sid}",
                    "segment":         "DERIVATIVE",
                    "product":         "MARGIN",
                    "exchange":        "NSE",
                    "instrument_type": "FUTURES",
                    "lot_size":        int(row.get("LOT_UNITS") or 1),
                    "expiry":          str(row.get("EXPIRY_DATE") or ""),
                    "tick_size":       float(row.get("TICK_SIZE") or 0.05),
                })
            except Exception as e:
                logger.warning(f"Skipping FNO row: {e}")
                continue

        return instruments

    def _parse_index(self, df: pd.DataFrame) -> list:
        """
        Index CSV only has EXCH, SEGMENT, SECURITY_ID guaranteed.
        _best_name() falls back to SECURITY_ID when no name column exists.
        """
        df.columns = df.columns.str.strip()

        if "SECURITY_ID" not in df.columns:
            logger.error(
                f"Index CSV missing SECURITY_ID. "
                f"Available: {df.columns.tolist()}"
            )
            return []

        instruments = []
        for _, row in df.iterrows():
            try:
                sid  = str(row["SECURITY_ID"]).strip()
                name = self._best_name(row, fallback=sid)
                instruments.append({
                    "name":            name,
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
        logger.info(
            "🔍 Tier 1: equity via quotes API | "
            "FNO via CSV-only shortlist (no quotes API call)"
        )
        start = time.time()

        # Equity — scored via /market/quotes/full (NSE_ prefix works)
        equity_shortlist = self._score_batch_equity(
            self.universe_equity,
            min_price = self.MIN_PRICE_EQUITY,
            top_n     = self.TIER1_SHORTLIST
        )

        # FNO — CSV-only, no quotes API call (NFO_ prefix causes 400)
        fno_shortlist = self._shortlist_fno_from_csv(
            self.universe_fno,
            top_n = self.TIER1_SHORTLIST // 5
        )

        index_shortlist = self.universe_index[:10]

        with self._lock:
            self.shortlist_equity = equity_shortlist
            self.shortlist_fno    = fno_shortlist
            self.shortlist_index  = index_shortlist
            self._last_tier1      = time.time()

        elapsed  = time.time() - start
        top_eq   = [s["name"] for s in equity_shortlist[:5]]
        top_fno  = [s["name"] for s in fno_shortlist[:3]]
        logger.info(
            f"✅ Tier 1 complete in {elapsed:.1f}s | "
            f"Equity: {len(equity_shortlist)} | "
            f"FNO: {len(fno_shortlist)} | "
            f"Top equity: {top_eq} | "
            f"Top FNO: {top_fno}"
        )

    def _score_batch_equity(self, instruments: list,
                            min_price: float,
                            top_n: int) -> list:
        """
        Batch-fetch quotes for equity instruments only.
        NSE_<id> scrip code format confirmed working for equity.
        Score by volume + volatility + momentum.
        Returns top N scored instruments.
        """
        if not instruments:
            return []

        scored        = []
        total         = len(instruments)
        total_batches = (total + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for i in range(0, total, self.BATCH_SIZE):
            batch       = instruments[i:i + self.BATCH_SIZE]
            scrip_codes = ",".join(inst["scrip_code"] for inst in batch)
            batch_num   = i // self.BATCH_SIZE + 1

            try:
                quotes = self.api._request(
                    "GET", "/market/quotes/full",
                    params={"scrip-codes": scrip_codes}
                )

                if not quotes or "data" not in quotes:
                    logger.warning(
                        f"Equity batch {batch_num}/{total_batches}: "
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

                    price  = float(data.get("live_price") or 0)
                    volume = float(data.get("volume") or 0)
                    change = abs(float(
                        data.get("day_change_percentage") or 0
                    ))
                    high   = float(data.get("day_high") or price)
                    low    = float(data.get("day_low")  or price)

                    if price < min_price:
                        continue
                    if volume < self.MIN_VOLUME:
                        continue

                    vol_score   = min(volume / 1_000_000, 40)
                    atr_pct     = (
                        (high - low) / price * 100
                        if price > 0 else 0
                    )
                    atr_score   = min(atr_pct * 10, 40)
                    mom_score   = min(change * 4, 20)
                    total_score = vol_score + atr_score + mom_score

                    scored.append({
                        **inst,
                        "score":  round(total_score, 2),
                        "ltp":    price,
                        "volume": volume,
                        "change": change,
                    })
                    hits += 1

                logger.debug(
                    f"Equity batch {batch_num}/{total_batches}: "
                    f"{hits}/{len(batch)} scored"
                )

            except Exception as e:
                logger.error(
                    f"Equity batch {batch_num}/{total_batches} error "
                    f"(sample: {scrip_codes[:60]}...): {e}"
                )

            time.sleep(0.5)

        scored.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            f"Equity scoring complete: {len(scored)} passed filters, "
            f"returning top {min(top_n, len(scored))}"
        )
        return scored[:top_n]

    def _shortlist_fno_from_csv(self, instruments: list,
                                top_n: int) -> list:
        """
        Selects top FNO futures from the CSV universe without
        calling the quotes API (which returns 400 for NFO_ prefix).

        Priority:
          1. Index futures (NIFTY, BANKNIFTY, FINNIFTY, etc.)
             — always most liquid, always included first
          2. Stock futures sorted by lot_size descending
             — higher lot_size = more institutionally traded

        Each instrument gets score=0 and ltp=0 as defaults.
        Real price is fetched via WS or REST LTP when run_cycle
        scans the instrument.
        """
        if not instruments:
            return []

        index_names = {
            "NIFTY", "BANKNIFTY", "FINNIFTY",
            "MIDCPNIFTY", "SENSEX", "BANKEX"
        }

        index_futs = []
        stock_futs = []

        for inst in instruments:
            name_upper = inst["name"].upper()
            is_index   = any(n in name_upper for n in index_names)
            if is_index:
                index_futs.append(inst)
            else:
                stock_futs.append(inst)

        stock_futs.sort(
            key=lambda x: x.get("lot_size", 0),
            reverse=True
        )

        combined = index_futs + stock_futs

        result = []
        for inst in combined[:top_n]:
            result.append({
                **inst,
                "score":  0.0,
                "ltp":    0.0,
                "volume": 0,
                "change": 0.0,
            })

        logger.info(
            f"FNO shortlist (CSV-only): "
            f"{len(index_futs)} index futures + "
            f"{len(stock_futs)} stock futures available | "
            f"returning top {len(result)}"
        )
        return result

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


