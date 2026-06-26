from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "DATA" / "aluminum_AL0_daily_2005_20260622.json"
OUT_DIR = ROOT / "research" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
MIN_DAYS = 252
COST = 0.0003


@dataclass
class Candidate:
    name: str
    position: np.ndarray
    family: str
    params: dict


def load_data() -> pd.DataFrame:
    payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["candles"])
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "openInterest", "settlement"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def safe_sign(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.sign(arr)
    out[~np.isfinite(out)] = 0
    return out


def annualized_vol(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window).std() * math.sqrt(TRADING_DAYS)


def apply_vol_target(signal: np.ndarray, vol: pd.Series, target: float, max_exposure: float) -> np.ndarray:
    scale = target / vol.to_numpy(dtype=float)
    scale = np.where(np.isfinite(scale), scale, 0)
    scale = np.clip(scale, 0, max_exposure)
    pos = signal * scale
    pos[~np.isfinite(pos)] = 0
    return np.clip(pos, -max_exposure, max_exposure)


def returns_from_position(close: pd.Series, raw_position: np.ndarray, cost: float = COST) -> pd.DataFrame:
    price_ret = close.pct_change().fillna(0).to_numpy(dtype=float)
    position = np.roll(raw_position, 1)
    position[0] = 0
    position = np.where(np.isfinite(position), position, 0)
    turnover = np.abs(position - np.roll(position, 1))
    turnover[0] = abs(position[0])
    strategy_ret = position * price_ret - turnover * cost
    return pd.DataFrame(
        {
            "position": position,
            "turnover": turnover,
            "ret": strategy_ret,
        }
    )


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    return float(np.nanmin(dd))


def min_rolling_ann(ret: np.ndarray, window: int = MIN_DAYS) -> float:
    log_ret = np.log1p(np.clip(ret, -0.999999, None))
    cum = np.r_[0, np.cumsum(log_ret)]
    if len(ret) < window:
        return np.nan
    window_log = cum[window:] - cum[:-window]
    return float(np.exp(window_log * TRADING_DAYS / window).min() - 1)


def min_any_subperiod_ann(ret: np.ndarray, min_days: int = MIN_DAYS) -> tuple[float, int, int]:
    log_ret = np.log1p(np.clip(ret, -0.999999, None))
    cum = np.r_[0, np.cumsum(log_ret)]
    n = len(ret)
    best = math.inf
    best_start = 0
    best_end = min_days

    for start in range(0, n - min_days + 1):
        ends = np.arange(start + min_days, n + 1)
        logs = cum[ends] - cum[start]
        lengths = ends - start
        ann = np.exp(logs * TRADING_DAYS / lengths) - 1
        idx = int(np.argmin(ann))
        value = float(ann[idx])
        if value < best:
            best = value
            best_start = start
            best_end = int(ends[idx])

    return best, best_start, best_end


def metrics(df: pd.DataFrame, candidate: Candidate, include_any: bool = False) -> dict:
    bt = returns_from_position(df["close"], candidate.position)
    ret = bt["ret"].to_numpy(dtype=float)
    equity = np.cumprod(1 + ret)
    total_log = np.log(equity[-1]) if equity[-1] > 0 else -np.inf
    ann_ret = math.exp(total_log * TRADING_DAYS / max(len(ret) - 1, 1)) - 1 if np.isfinite(total_log) else -1
    ann_vol = float(np.nanstd(ret) * math.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    out = {
        "name": candidate.name,
        "family": candidate.family,
        "params": json.dumps(candidate.params, ensure_ascii=False),
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "min_rolling_1y_ann": min_rolling_ann(ret),
        "total_return": float(equity[-1] - 1),
        "annual_turnover": float(np.nanmean(bt["turnover"]) * TRADING_DAYS),
        "avg_abs_position": float(np.nanmean(np.abs(bt["position"]))),
    }
    if include_any:
        worst, start, end = min_any_subperiod_ann(ret)
        out.update(
            {
                "min_any_ge_1y_ann": worst,
                "worst_start": str(df["date"].iloc[start].date()),
                "worst_end": str(df["date"].iloc[end - 1].date()),
                "worst_days": int(end - start),
            }
        )
    return out


def build_candidates(df: pd.DataFrame) -> list[Candidate]:
    close = df["close"]
    ret = close.pct_change()
    vol_windows = [10, 20, 40, 60]
    targets = [0.10, 0.15, 0.20, 0.25]
    max_exposures = [1.0, 1.5, 2.0, 3.0]
    candidates: list[Candidate] = []

    def add(name: str, family: str, base_signal: np.ndarray, params: dict) -> None:
        for vw in vol_windows:
            vol = annualized_vol(ret, vw).shift(0)
            for target in targets:
                for mx in max_exposures:
                    pos = apply_vol_target(base_signal, vol, target, mx)
                    p = {**params, "vol_window": vw, "target_vol": target, "max_exposure": mx}
                    candidates.append(Candidate(f"{name}|vw={vw}|tv={target}|mx={mx}", pos, family, p))

    # Price / moving average trend.
    for period in [5, 10, 20, 40, 60, 80, 120, 160, 200, 250]:
        ma = close.rolling(period).mean()
        sig = safe_sign(close / ma - 1)
        add(f"price_ma_{period}", "price_ma_trend", sig, {"period": period})

    # Moving average cross.
    for fast in [5, 10, 20, 40, 60]:
        for slow in [40, 60, 80, 120, 160, 200, 250]:
            if fast >= slow:
                continue
            ma_fast = close.rolling(fast).mean()
            ma_slow = close.rolling(slow).mean()
            sig = safe_sign(ma_fast / ma_slow - 1)
            add(f"ma_cross_{fast}_{slow}", "ma_cross", sig, {"fast": fast, "slow": slow})

    # Time-series momentum.
    for period in [5, 10, 20, 40, 60, 80, 120, 160, 200, 250]:
        sig = safe_sign(close / close.shift(period) - 1)
        add(f"momentum_{period}", "momentum", sig, {"period": period})

    # Donchian breakout with state holding.
    for period in [20, 40, 60, 80, 120, 160, 200, 250]:
        prev_high = close.shift(1).rolling(period).max()
        prev_low = close.shift(1).rolling(period).min()
        state = np.zeros(len(df))
        current = 0
        for i in range(len(df)):
            if np.isfinite(prev_high.iloc[i]) and close.iloc[i] > prev_high.iloc[i]:
                current = 1
            elif np.isfinite(prev_low.iloc[i]) and close.iloc[i] < prev_low.iloc[i]:
                current = -1
            state[i] = current
        add(f"donchian_{period}", "donchian", state, {"period": period})

    # Bollinger trend and mean reversion.
    for period in [20, 40, 60, 80, 120]:
        ma = close.rolling(period).mean()
        sd = close.rolling(period).std()
        z = (close - ma) / sd
        for threshold in [0.3, 0.5, 0.8, 1.0]:
            sig_trend = np.where(z > threshold, 1, np.where(z < -threshold, -1, 0))
            sig_mr = -sig_trend
            add(
                f"boll_trend_{period}_{threshold}",
                "bollinger_trend",
                sig_trend,
                {"period": period, "threshold": threshold},
            )
            add(
                f"boll_mr_{period}_{threshold}",
                "bollinger_mean_reversion",
                sig_mr,
                {"period": period, "threshold": threshold},
            )

    # Trend with volume/open-interest confirmation. Flat if confirmation disagrees.
    for fast, slow in [(10, 60), (20, 60), (20, 120), (40, 160), (60, 200)]:
        trend = safe_sign(close.rolling(fast).mean() / close.rolling(slow).mean() - 1)
        for confirm_window in [5, 20, 60]:
            oi_confirm = safe_sign(df["openInterest"].pct_change(confirm_window))
            vol_confirm = safe_sign(df["volume"].pct_change(confirm_window))
            sig_oi = np.where(trend == oi_confirm, trend, 0)
            sig_vol = np.where(trend == vol_confirm, trend, 0)
            add(
                f"trend_oi_confirm_{fast}_{slow}_{confirm_window}",
                "trend_oi_confirm",
                sig_oi,
                {"fast": fast, "slow": slow, "confirm_window": confirm_window},
            )
            add(
                f"trend_volume_confirm_{fast}_{slow}_{confirm_window}",
                "trend_volume_confirm",
                sig_vol,
                {"fast": fast, "slow": slow, "confirm_window": confirm_window},
            )

    return candidates


def main() -> None:
    df = load_data()
    candidates = build_candidates(df)
    print(f"Loaded {len(df)} rows: {df.date.iloc[0].date()} -> {df.date.iloc[-1].date()}")
    print(f"Candidates: {len(candidates)}")

    rows = [metrics(df, c, include_any=False) for c in candidates]
    summary = pd.DataFrame(rows).sort_values(
        ["min_rolling_1y_ann", "annual_return"], ascending=[False, False]
    )
    summary.to_csv(OUT_DIR / "strategy_search_summary.csv", index=False, encoding="utf-8-sig")

    top_names = set(summary.head(40)["name"])
    top_by_return = set(summary.sort_values("annual_return", ascending=False).head(20)["name"])
    selected = [c for c in candidates if c.name in top_names or c.name in top_by_return]
    detailed_rows = [metrics(df, c, include_any=True) for c in selected]
    detailed = pd.DataFrame(detailed_rows).sort_values(
        ["min_any_ge_1y_ann", "min_rolling_1y_ann", "annual_return"],
        ascending=[False, False, False],
    )
    detailed.to_csv(OUT_DIR / "strategy_search_detailed.csv", index=False, encoding="utf-8-sig")

    best = detailed.iloc[0].to_dict()
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
