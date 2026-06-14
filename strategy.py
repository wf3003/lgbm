#!/usr/bin/env python3
"""
LGBM 多币种择时 — 40维特征 + 自适应参数 + 在线学习
"""

import os, time, json, traceback, threading, pickle
from datetime import datetime
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ══════ 配置 ══════
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT",
    "TRX/USDT", "DOGE/USDT", "XLM/USDT", "ADA/USDT", "LINK/USDT",
    "TON/USDT", "BCH/USDT", "HBAR/USDT", "LTC/USDT", "SUI/USDT",
    "SHIB/USDT", "AVAX/USDT", "NEAR/USDT", "FIL/USDT", "ATOM/USDT",
]
TIMEFRAME = "15m"
LABEL_OFFSET = 24         # 6h
BASE_POSITION_PCT = 0.05  # 基础仓位 5%
BASE_CONFIDENCE = 0.02    # 基础开仓阈值
BASE_MAX_POSITIONS = 20   # 最大持仓
TAKE_PROFIT = 12
STOP_LOSS = -5
LEVERAGE = 4
FORCE_CLOSE_PCT = -15
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
TRADE_HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
OPEN_FEATURES_FILE = os.path.join(DATA_DIR, "open_features.pkl")

# ══════ 交易所 ══════
PROXY = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
_kw = {
    "options": {"defaultType": "swap", "fetchMarketsByDefault": False},
    "apiKey": os.getenv("OKX_API_KEY"), "secret": os.getenv("OKX_SECRET"),
    "password": os.getenv("OKX_PASSWORD"), "enableRateLimit": True, "timeout": 15000,
}
if PROXY:
    _kw["proxies"] = {"http": PROXY, "https": PROXY}
EXCHANGE = ccxt.okx(_kw)
EXCHANGE.set_sandbox_mode(os.getenv("OKX_SANDBOX", "true").lower() == "true")
MARKETS_LOADED = False

GLOBAL_LATEST_SL = {}
GLOBAL_LATEST_TP = {}
PEAK_PNL = {}  # symbol → 历史峰值 PnL%（用于移动止盈）
OPEN_FEATURES = {}  # symbol → 开仓时的特征向量，平仓时喂给模型学习
OPEN_TIME = {}  # symbol → 开仓时间
TRADE_HISTORY = []        # [{symbol, side, entry, exit, pnl_pct, features, label, ts}]
ONLINE_SAMPLES_X = []     # 在线学习累积样本
ONLINE_SAMPLES_Y = []

# ══════ 模型持久化 ══════

def save_model(model):
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"model": model, "online_x": ONLINE_SAMPLES_X, "online_y": ONLINE_SAMPLES_Y}, f)
        print("  💾 模型已保存")
    except Exception as e:
        print(f"  ⚠️ 模型保存失败: {e}")

def load_model():
    global ONLINE_SAMPLES_X, ONLINE_SAMPLES_Y, OPEN_FEATURES
    try:
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
        # 加载在线样本
        try:
            with open(os.path.join(DATA_DIR, "online_samples.pkl"), "rb") as f:
                samples = pickle.load(f)
            ONLINE_SAMPLES_X = samples.get("online_x", [])
            ONLINE_SAMPLES_Y = samples.get("online_y", [])
        except:
            pass
        print(f"  📦 模型已加载 | 在线样本: {len(ONLINE_SAMPLES_X)} 条")
        # 加载开仓特征
        try:
            with open(OPEN_FEATURES_FILE, "rb") as f2:
                data = pickle.load(f2)
            OPEN_FEATURES = data.get("features", data) if isinstance(data, dict) else {}
            OPEN_TIME = data.get("times", {}) if isinstance(data, dict) else {}
            print(f"  📋 开仓特征: {len(OPEN_FEATURES)} 个币种")
        except:
            pass
        # 加载峰值
        try:
            with open(os.path.join(DATA_DIR, "peak_pnl.pkl"), "rb") as f:
                PEAK_PNL.update(pickle.load(f))
            print(f"  📈 峰值恢复: {len(PEAK_PNL)} 个币种")
        except:
            pass
        return data.get("model")
    except:
        return None

# ══════ 自适应参数 ══════

def load_trade_history():
    global TRADE_HISTORY
    try:
        with open(TRADE_HISTORY_FILE) as f:
            TRADE_HISTORY = json.load(f)
    except:
        TRADE_HISTORY = []

def save_trade_history():
    try:
        with open(TRADE_HISTORY_FILE, "w") as f:
            json.dump(TRADE_HISTORY[-50:], f, ensure_ascii=False, default=str)
    except:
        pass

def record_trade(sym, side, entry, exit_price, qty, pnl_pct, leverage=None, features=None, label=None):
    """记录已平仓交易"""
    global ONLINE_SAMPLES_X, ONLINE_SAMPLES_Y
    pnl_usd = 0
    if entry > 0 and exit_price > 0 and qty > 0:
        ct_val = 1.0
        try:
            mkt = EXCHANGE.market(sym.replace("/","/")+":USDT")
            ct_val = float(mkt.get("contractSize") or mkt.get("info",{}).get("ctVal",1))
        except: pass
        if side == "long":
            pnl_usd = (exit_price - entry) * qty * ct_val
        else:
            pnl_usd = (entry - exit_price) * qty * ct_val
    TRADE_HISTORY.append({
        "symbol": sym, "side": side, "entry": entry, "exit": exit_price,
        "qty": qty, "leverage": leverage or LEVERAGE,
        "pnl_pct": pnl_pct, "pnl_usd": round(pnl_usd, 2),
        "ts": datetime.now().isoformat(),
    })
    # 在线学习：保存特征和标签（0=亏,1=赚）
    if features is not None and label is not None:
        # 按盈亏幅度加权：大赚大亏的样本权重更高
        weight = min(abs(pnl_pct) / TAKE_PROFIT, 2.0)  # 赚满12%=权重1.0，24%=2.0封顶
        feat_vec = np.nan_to_num(features, nan=0, posinf=0, neginf=0).flatten()
        # 加权样本：权重>1的样本重复加入
        repeat = max(1, int(weight))
        for _ in range(repeat):
            ONLINE_SAMPLES_X.append(feat_vec.copy())
            ONLINE_SAMPLES_Y.append(1 if pnl_pct > 0 else 0)
        # 限制在线样本数量，保留最近 500 条
        if len(ONLINE_SAMPLES_X) > 500:
            ONLINE_SAMPLES_X = ONLINE_SAMPLES_X[-500:]
            ONLINE_SAMPLES_Y = ONLINE_SAMPLES_Y[-500:]
    save_trade_history()
    # 在线样本单独持久化（不覆盖模型）
    try:
        with open(os.path.join(DATA_DIR, "online_samples.pkl"), "wb") as f:
            pickle.dump({"online_x": ONLINE_SAMPLES_X, "online_y": ONLINE_SAMPLES_Y}, f)
        print(f"  🧬 在线样本已写盘: {len(ONLINE_SAMPLES_X)} 条")
    except Exception as e:
        print(f"  ⚠️ 样本写盘失败: {e}")

def adapt_params():
    """根据近期胜率自适应调整"""
    if len(TRADE_HISTORY) < 3:
        return BASE_POSITION_PCT, BASE_CONFIDENCE, BASE_MAX_POSITIONS

    recent = TRADE_HISTORY[-10:]
    wins = sum(1 for t in recent if t["pnl_pct"] > 0)
    losses = len(recent) - wins
    wr = wins / len(recent) if recent else 0.5

    # 连亏3笔 → 降仓位、提阈值
    last3 = TRADE_HISTORY[-3:]
    consecutive_losses = sum(1 for t in last3 if t["pnl_pct"] <= 0)

    if consecutive_losses >= 3:
        pos_pct = 0.03
        conf = 0.05
        max_pos = BASE_MAX_POSITIONS
        print(f"  ⚠️ 连亏{consecutive_losses}笔 → 仓位3% 阈值0.05")
    elif wr >= 0.7 and len(recent) >= 5:
        pos_pct = 0.05
        conf = 0.02
        max_pos = 20
        print(f"  🔥 近期胜率{wr*100:.0f}% → 仓位5% 最大{max_pos}仓")
    else:
        pos_pct = BASE_POSITION_PCT
        conf = BASE_CONFIDENCE
        max_pos = BASE_MAX_POSITIONS
        print(f"  📊 近期胜率{wr*100:.0f}% ({wins}胜{losses}负) → 默认参数")

    return pos_pct, conf, max_pos


# ══════ 数据获取 ══════

def to_swap(sym: str) -> str:
    return sym.replace("/", "/") + ":USDT"

def fetch_ohlcv(sym: str) -> pd.DataFrame | None:
    global MARKETS_LOADED
    if not MARKETS_LOADED:
        try:
            EXCHANGE.load_markets(reload=True, params={"instType": "SWAP"})
        except:
            EXCHANGE.load_markets()
        MARKETS_LOADED = True
    try:
        raw = EXCHANGE.fetch_ohlcv(to_swap(sym), TIMEFRAME, limit=300)
        if not raw or len(raw) < 100:
            return None
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        print(f"  [{sym}] 数据获取失败: {e}")
        return None

def fetch_fundamentals(sym: str) -> dict:
    """从 OKX 获取基本面数据"""
    try:
        swap = to_swap(sym)
        fr_data = EXCHANGE.fetch_funding_rate(swap)
        fr = fr_data.get("fundingRate", 0) or 0
        # OI 变更率 (通过 ticker 的百分比变化估算)
        tk = EXCHANGE.fetch_ticker(swap)
        oi_change = float((tk.get("info", {}) or {}).get("openInterestUsd", 0) or 0)
        hi24 = float(tk.get("high", 0) or 0)
        lo24 = float(tk.get("low", 0) or 0)
        vol24 = float(tk.get("quoteVolume", 0) or 0)
        return {"funding_rate": fr, "oi": oi_change, "hi24": hi24, "lo24": lo24, "vol24": vol24}
    except:
        return {}


# ══════ 特征工程 (33技术 + 7基本面 = 40维) ══════

def compute_features(df: pd.DataFrame, fundamentals: dict = None) -> pd.DataFrame | None:
    try:
        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
        close_s1 = c.shift(1)
        n = len(c)

        ret = np.log(c / close_s1.clip(1e-10))
        hl_log_sq = (np.log(h / l.clip(1e-10))) ** 2
        abs_ret = np.abs(ret)
        price_diff_20 = np.abs(c - c.shift(20))

        def r(x, fn, w): return x.rolling(w).apply(fn, raw=True) if hasattr(x.rolling(w), "apply") else x.rolling(w).agg(fn)

        # ── 技术因子 (33维) ──
        parkinson_vol = np.sqrt(r(hl_log_sq, lambda x: np.mean(x), 14) / (4 * np.log(2)))
        gk_vol = np.sqrt(r(0.5 * hl_log_sq - (2 * np.log(2) - 1) * (np.log(c / o.clip(1e-10))) ** 2, lambda x: np.mean(x), 14))
        vol_regime = r(ret, lambda x: np.std(x), 7) / r(ret, lambda x: np.std(x), 60).clip(1e-10)
        vol_term = r(ret, lambda x: np.std(x), 5) / r(ret, lambda x: np.std(x), 60).clip(1e-10)

        hl_range = h - l
        close_loc = (c - l) / hl_range.clip(1e-10)
        oc_range = (c - o) / hl_range.clip(1e-10)
        abs_chg_sum20 = r(abs_ret, lambda x: np.nansum(np.abs(x)), 20)
        price_eff = price_diff_20 / abs_chg_sum20.clip(1e-10)

        amihud = r(abs_ret / v.clip(1), lambda x: np.mean(x), 20) * 1e6
        cv20 = r(c * v, lambda x: np.sum(x), 20)
        sv20 = r(v, lambda x: np.sum(x), 20)
        vwap_bias = (c - cv20 / sv20.clip(1e-10)) / c.clip(1e-10)
        up_vol = r(v * (ret > 0).astype(float), lambda x: np.sum(x), 20)
        dn_vol = r(v * (ret < 0).astype(float), lambda x: np.sum(x), 20)
        ud_vol_ratio = up_vol / dn_vol.clip(1e-10)
        vol_breakout = v / r(v, lambda x: np.max(x), 20).clip(1e-10)

        ret7 = np.log(c / c.shift(7).clip(1e-10))
        ret14 = np.log(c / c.shift(14).clip(1e-10))
        ret30 = np.log(c / c.shift(30).clip(1e-10))
        mom_quality = r(ret, lambda x: np.mean(x), 20) / r(ret, lambda x: np.std(x), 20).clip(1e-10)

        def ta_adx(hh, ll, cc, w):
            up = hh.diff(); dn = -ll.diff()
            p_dm = np.where((up > dn) & (up > 0), up, 0)
            n_dm = np.where((dn > up) & (dn > 0), dn, 0)
            tr = np.maximum(hh - ll, np.abs(hh - cc.shift(1)))
            tr = np.maximum(tr, np.abs(ll - cc.shift(1)))
            atr1 = pd.Series(tr).ewm(span=w, adjust=False).mean()
            pdi = pd.Series(p_dm).ewm(span=w, adjust=False).mean() / atr1.clip(1e-10) * 100
            ndi = pd.Series(n_dm).ewm(span=w, adjust=False).mean() / atr1.clip(1e-10) * 100
            dx = np.abs(pdi - ndi) / (pdi + ndi).clip(1e-10) * 100
            return pd.Series(dx).ewm(span=w, adjust=False).mean()

        adx = ta_adx(h, l, c, 14)
        ma_cross = r(c, lambda x: np.mean(x), 5) / r(c, lambda x: np.mean(x), 20).clip(1e-10) - 1
        trend_str = (c - c.shift(20)) / abs_chg_sum20.clip(1e-10)
        donchian_5 = (2 * c - r(h, lambda x: np.max(x), 5) - r(l, lambda x: np.min(x), 5)) / o.shift(4).clip(1e-10)
        donchian_20 = (2 * c - r(h, lambda x: np.max(x), 20) - r(l, lambda x: np.min(x), 20)) / o.shift(19).clip(1e-10)
        donchian_60 = (2 * c - r(h, lambda x: np.max(x), 60) - r(l, lambda x: np.min(x), 60)) / o.shift(59).clip(1e-10)

        def ta_rsi(cc, w):
            d = cc.diff(); g = d.clip(lower=0).ewm(span=w, adjust=False).mean()
            lr = (-d.clip(upper=0)).ewm(span=w, adjust=False).mean()
            return 100 - 100 / (1 + g / lr.clip(1e-10))
        rsi_val = ta_rsi(c, 14)
        bb_std = r(c, lambda x: np.std(x), 20)
        bb_mid = r(c, lambda x: np.mean(x), 20)
        bb_pct = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)
        ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
        macd_norm = ((ema12 - ema26).ewm(span=9, adjust=False).mean()) / c.clip(1e-10) * 100

        skew = r(ret, lambda x: x.skew() if hasattr(x, 'skew') else 0, 20)
        kurt = r(ret, lambda x: x.kurt() if hasattr(x, 'kurt') else 0, 20)
        max_dd = (c - r(c, lambda x: np.max(x), 20)) / r(c, lambda x: np.max(x), 20).clip(1e-10)
        range_exp = hl_range / r(hl_range, lambda x: np.mean(x), 10).clip(1e-10)
        bar_bias = np.where(hl_range == 0, 0, ((c - l) - (h - c)) / hl_range * v)
        intrabar_vol_bias = r(pd.Series(bar_bias), lambda x: np.sum(x), 5)
        rsi_mid = ta_rsi(c, 24)
        rsi_long = ta_rsi(c, 120)
        ma_cross_mid = r(c, lambda x: np.mean(x), 10) / r(c, lambda x: np.mean(x), 60).clip(1e-10) - 1
        def _tl(): return np.where(l < close_s1, l, close_s1)
        def _th(): return np.where(h > close_s1, h, close_s1)
        _tr = _th() - _tl()
        rsi_multi_6 = (c - r(pd.Series(_tl()), lambda x: np.sum(x), 6)) / r(pd.Series(_tr), lambda x: np.sum(x), 6).clip(1e-10)
        rsi_multi_12 = (c - r(pd.Series(_tl()), lambda x: np.sum(x), 12)) / r(pd.Series(_tr), lambda x: np.sum(x), 12).clip(1e-10)
        rsi_multi_24 = (c - r(pd.Series(_tl()), lambda x: np.sum(x), 24)) / r(pd.Series(_tr), lambda x: np.sum(x), 24).clip(1e-10)
        multi_tf_rsu = (rsi_multi_6 * 288 + rsi_multi_12 * 144 + rsi_multi_24 * 72) * 100 / 504

        feats = pd.DataFrame(index=df.index)
        for name, val in [
            ("parkinson_vol", parkinson_vol), ("gk_vol", gk_vol),
            ("vol_regime", vol_regime), ("vol_term", vol_term),
            ("close_loc", close_loc), ("oc_range", oc_range), ("price_eff", price_eff),
            ("amihud", amihud), ("vwap_bias", vwap_bias),
            ("ud_vol_ratio", ud_vol_ratio), ("vol_breakout", vol_breakout),
            ("ret7", ret7), ("ret14", ret14), ("ret30", ret30), ("mom_quality", mom_quality),
            ("adx", adx), ("ma_cross", ma_cross), ("trend_str", trend_str),
            ("donchian_5", donchian_5), ("donchian_20", donchian_20), ("donchian_60", donchian_60),
            ("rsi", rsi_val), ("bb_pct", bb_pct), ("macd_norm", macd_norm),
            ("skew", skew), ("kurt", kurt), ("max_dd", max_dd),
            ("range_exp", range_exp), ("intrabar_vol_bias", intrabar_vol_bias),
            ("rsi_mid", rsi_mid), ("rsi_long", rsi_long),
            ("ma_cross_mid", ma_cross_mid), ("multi_tf_rsu", multi_tf_rsu),
        ]:
            feats[name] = val

        # ── 基本面因子 (7维) ──
        f = fundamentals or {}
        fr_val = f.get("funding_rate", 0) or 0
        feats["funding_rate"] = pd.Series([fr_val] * n, index=df.index)
        feats["funding_zscore"] = pd.Series([0.0] * n, index=df.index)
        oi_val = f.get("oi", 0) or 0
        feats["oi_raw"] = pd.Series([float(oi_val)] * n, index=df.index)
        feats["vol_trend"] = r(v, lambda x: np.mean(x), 10) / r(v, lambda x: np.mean(x), 30).clip(1e-10)
        hi24, lo24 = f.get("hi24", 0) or 0, f.get("lo24", 0) or 0
        mid24 = (hi24 + lo24) / 2 if hi24 > 0 and lo24 > 0 else float(c.iloc[-1])
        liq = abs(float(c.iloc[-1]) - mid24) / max(hi24, 1e-10) if hi24 > 0 else 0.0
        feats["liq_pressure"] = pd.Series([liq] * n, index=df.index)
        feats["price_oi_div"] = ret7 / (v / r(v, lambda x: np.mean(x), 7).clip(1e-10) + 1e-10)
        feats["oi_change_pct"] = pd.Series([0.0] * n, index=df.index)

        # ── QQE MOD (2维) ──
        qqe = compute_qqe_mod(c, 6, 5, 3.0, 1.61)
        if qqe is not None:
            feats["qqe_primary"] = qqe["primary"]   # 主 QQE 趋势线(去中心化)
            feats["qqe_signal"] = qqe["signal"]      # 综合信号: 1=多, -1=空, 0=无

        # ── SuperTrend (2维) ──
        st = compute_supertrend(h, l, c, 10, 3.0)
        if st is not None:
            feats["st_trend"] = st["trend"]
            feats["st_signal"] = st["signal"]

        # ── A-V2 Heikin Ashi 趋势 (1维) ──
        feats["av2_trend"] = compute_av2(o, h, l, c, 9)

        # ── SMC 结构位置 (1维) ──
        smc = compute_smc_position(h, l, c, 50)
        if smc is not None:
            feats["smc_position"] = smc

        # ── Squeeze Momentum (2维) ──
        sqz = compute_squeeze(c, h, l, v, 20, 2.0, 1.5)
        if sqz is not None:
            feats["sqz_momentum"] = sqz["momentum"]
            feats["sqz_state"] = sqz["state"]

        # ── MACD Histogram 4色状态 (1维, -2~+2) ──
        feats["macd_hist_state"] = compute_macd_hist_state(c, 12, 26, 9)

        # ── KDJ (1维, J值) ──
        feats["kdj_j"] = compute_kdj(h, l, c, 9, 3, 3)

        # ── TRIX (1维) ──
        feats["trix"] = compute_trix(c, 14)

        # ── OBV 量价背离 (1维) ──
        feats["obv_divergence"] = compute_obv_divergence(c, v, 20)

        # ── CPR 中枢 (2维) ──
        cpr = compute_cpr(h, l, c)
        if cpr is not None:
            feats["cpr_width"] = cpr["width"]       # CPR宽度/价格(缩窄=突破前)
            feats["cpr_position"] = cpr["position"]  # 价格在CPR的相对位置

        return feats
    except Exception as e:
        print(f"  [feat] 特征计算失败: {e}")
        return None


# ══════ 标签 + 模型 ══════

def ta_rsi(cc, w):
    d = cc.diff(); g = d.clip(lower=0).ewm(span=w, adjust=False).mean()
    lr = (-d.clip(upper=0)).ewm(span=w, adjust=False).mean()
    return 100 - 100 / (1 + g / lr.clip(1e-10))

def compute_squeeze(close, high, low, volume, length=20, mult=2.0, mult_kc=1.5):
    """Squeeze Momentum: BB + Keltner 挤压检测"""
    try:
        basis = close.rolling(length).mean()
        dev = mult_kc * close.rolling(length).std()
        upper_bb = basis + dev; lower_bb = basis - dev
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        ma_kc = close.rolling(length).mean()
        range_ma = tr.rolling(length).mean()
        upper_kc = ma_kc + range_ma * mult_kc
        lower_kc = ma_kc - range_ma * mult_kc
        # 挤压状态: -1=挤压中, 0=无, 1=挤压释放
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        state = pd.Series(0, index=close.index)
        state[sqz_on] = -1; state[sqz_off] = 1
        # 动量值：线性回归
        hh = high.rolling(length).max(); ll = low.rolling(length).min()
        avg_hl = (hh + ll) / 2
        src = close - ((avg_hl + close.rolling(length).mean()) / 2)
        momentum = src.rolling(length).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0, raw=False)
        return {"momentum": momentum, "state": state}
    except:
        return None

def compute_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ: J 值（短线拐点信号）"""
    try:
        ll = low.rolling(n).min(); hh = high.rolling(n).max()
        rsv = (close - ll) / (hh - ll + 1e-10) * 100
        k = rsv.ewm(span=m1, adjust=False).mean()
        d = k.ewm(span=m2, adjust=False).mean()
        return 3 * k - 2 * d
    except:
        return pd.Series(50.0, index=close.index)

def compute_trix(close, period=14):
    """TRIX: 三重指数平滑去噪趋势"""
    try:
        ema1 = close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = (ema3 - ema3.shift(1)) / ema3.shift(1).clip(1e-10) * 100
        return trix
    except:
        return pd.Series(0.0, index=close.index)

def compute_cpr(high, low, close):
    """CPR: Central Pivot Range"""
    try:
        p = (high + low + close) / 3
        bc = (high + low) / 2
        tc = 2 * p - bc
        width = (tc - bc) / close.clip(1e-10)  # 缩窄=突破前
        tc_bc_range = tc - bc + 1e-10
        position = (close - bc) / tc_bc_range    # 0~1, >1=突破上轨, <0=跌破下轨
        return {"width": width, "position": position}
    except:
        return None

def compute_obv_divergence(close, volume, period=20):
    """OBV 量价背离：价格涨但OBV跌=负背离(假涨)"""
    try:
        obv = pd.Series(0.0, index=close.index)
        direction = (close.diff() >= 0).astype(int)
        direction.iloc[0] = 1
        obv.iloc[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if direction.iloc[i]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        price_pct = close / close.shift(period) - 1
        obv_pct = obv / obv.shift(period) - 1
        divergence = (price_pct.sign() != obv_pct.sign()).astype(float) * (price_pct - obv_pct).abs()
        return divergence
    except:
        return pd.Series(0.0, index=close.index)

def compute_macd_hist_state(close, fast=12, slow=26, sig=9):
    """MACD 柱状图四色状态: +2多加速 +1多减速 -1空减速 -2空加速"""
    try:
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=sig, adjust=False).mean()
        hist = macd - signal
        state = pd.Series(0, index=close.index)
        state[(hist > 0) & (hist > hist.shift(1))] = 2
        state[(hist > 0) & (hist <= hist.shift(1))] = 1
        state[(hist <= 0) & (hist > hist.shift(1))] = -1
        state[(hist <= 0) & (hist <= hist.shift(1))] = -2
        return state
    except:
        return pd.Series(0, index=close.index)

def compute_qqe_mod(close, rsi_len=6, smooth=5, factor=3.0, factor2=1.61):
    """QQE MOD — 双 QQE + BB 综合信号"""
    try:
        n = len(close)
        def ema(x, s): return pd.Series(x).ewm(span=s, adjust=False).mean()
        def calc_qqe(src, rl, sf, qf):
            rsi_v = ta_rsi(src, rl)
            srsi = ema(rsi_v, sf)
            atr_rsi = (srsi.shift(1) - srsi).abs()
            satr = ema(atr_rsi, rl * 2 - 1)
            datr = satr * qf
            # 自适应带
            lb = pd.Series(0.0, index=close.index)
            sb = pd.Series(0.0, index=close.index)
            for i in range(1, n):
                new_lb = srsi.iloc[i] - datr.iloc[i]
                new_sb = srsi.iloc[i] + datr.iloc[i]
                if srsi.iloc[i-1] > lb.iloc[i-1] and srsi.iloc[i] > lb.iloc[i-1]:
                    lb.iloc[i] = max(lb.iloc[i-1], new_lb)
                else:
                    lb.iloc[i] = new_lb
                if srsi.iloc[i-1] < sb.iloc[i-1] and srsi.iloc[i] < sb.iloc[i-1]:
                    sb.iloc[i] = min(sb.iloc[i-1], new_sb)
                else:
                    sb.iloc[i] = new_sb
            # 方向
            trend = pd.Series(0, index=close.index)
            for i in range(1, n):
                if srsi.iloc[i] > sb.iloc[i-1]: trend.iloc[i] = 1
                elif srsi.iloc[i] < lb.iloc[i-1]: trend.iloc[i] = -1
                else: trend.iloc[i] = trend.iloc[i-1]
            trend_line = pd.Series([lb.iloc[i] if trend.iloc[i] == 1 else sb.iloc[i] for i in range(n)], index=close.index)
            return trend_line - 50, srsi - 50, trend_line

        # 主 QQE + BB
        ptl, prsi, _ = calc_qqe(close, rsi_len, smooth, factor)
        # 副 QQE
        stl, srsi2, _ = calc_qqe(close, rsi_len, smooth, factor2)
        # BB on primary
        bb_basis = ptl.rolling(50).mean()
        bb_std = ptl.rolling(50).std()
        bb_upper = bb_basis + 0.35 * bb_std
        bb_lower = bb_basis - 0.35 * bb_std
        # 综合信号
        sig = pd.Series(0, index=close.index)
        sig[(srsi2 > 3) & (prsi > bb_upper)] = 1
        sig[(srsi2 < -3) & (prsi < bb_lower)] = -1
        return {"primary": ptl, "signal": sig}
    except:
        return None

def compute_supertrend(high, low, close, period=10, multiplier=3.0):
    """经典 Supertrend 指标"""
    try:
        n = len(close)
        src = (high + low) / 2
        tr = pd.Series(np.maximum(high - low, np.abs(high - close.shift(1))), index=close.index)
        atr = tr.rolling(period).mean().bfill()
        up = src - multiplier * atr
        dn = src + multiplier * atr
        trend = pd.Series(1, index=close.index)
        up_arr = up.values.copy(); dn_arr = dn.values.copy()
        trend_arr = trend.values.copy(); close_arr = close.values.copy()
        for i in range(1, n):
            up_arr[i] = max(up_arr[i], up_arr[i-1]) if close_arr[i-1] > up_arr[i-1] else up_arr[i]
            dn_arr[i] = min(dn_arr[i], dn_arr[i-1]) if close_arr[i-1] < dn_arr[i-1] else dn_arr[i]
            if trend_arr[i-1] == -1 and close_arr[i] > dn_arr[i-1]:
                trend_arr[i] = 1
            elif trend_arr[i-1] == 1 and close_arr[i] < up_arr[i-1]:
                trend_arr[i] = -1
            else:
                trend_arr[i] = trend_arr[i-1]
        trend_s = pd.Series(trend_arr, index=close.index)
        signal = pd.Series(0, index=close.index)
        changes = trend_s.diff()
        signal[changes == 2] = 1       # -1→1 = 买入信号
        signal[changes == -2] = -1     # 1→-1 = 卖出信号
        return {"trend": trend_s, "signal": signal}
    except:
        return None

def compute_av2(open_, high, low, close, period=9):
    """A-V2: Heikin Ashi MA 趋势强度 (-100 ~ +100)"""
    try:
        ha_close = (open_ + high + low + close) / 4
        ha_open = (ha_close.shift(1) + ha_close.shift(1)) / 2
        ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
        ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
        ema_ha_open = ha_open.ewm(span=period, adjust=False).mean()
        ema_ha_close = ha_close.ewm(span=period, adjust=False).mean()
        ema_ha_high = ha_high.ewm(span=period, adjust=False).mean()
        ema_ha_low = ha_low.ewm(span=period, adjust=False).mean()
        hl_range = ema_ha_high - ema_ha_low
        trend = 100 * (ema_ha_close - ema_ha_open) / hl_range.clip(1e-10)
        return trend
    except:
        return pd.Series(0.0, index=close.index)

def compute_smc_position(high, low, close, size=50):
    """SMC: 价格在 swing 结构中的相对位置 (0~1)"""
    try:
        n = len(close)
        top = pd.Series(0.0, index=close.index)
        bottom = pd.Series(0.0, index=close.index)
        h_arr, l_arr, c_arr = high.values, low.values, close.values
        cur_top, cur_bottom = c_arr[0], c_arr[0]
        for i in range(size, n):
            # 检测新 swing high (突破最近50根最高)
            if h_arr[i] > h_arr[i-size:i].max():
                cur_top = h_arr[i]
            # 检测新 swing low (突破最近50根最低)
            if l_arr[i] < l_arr[i-size:i].min():
                cur_bottom = l_arr[i]
            top.iloc[i] = cur_top
            bottom.iloc[i] = cur_bottom
        # 填充前 size 根
        top.iloc[:size] = top.iloc[size]
        bottom.iloc[:size] = bottom.iloc[size]
        # 归一化位置
        rng = top - bottom
        pos = (close - bottom) / rng.clip(1e-10)
        return pos.clip(0, 1)
    except:
        return None

def compute_label(df: pd.DataFrame) -> np.ndarray:
    future_ret = df["close"].shift(-LABEL_OFFSET) / df["open"].shift(-1) - 1
    return (future_ret > 0).astype(int).values

def train_lgbm(X, y, feature_names=None):
    from lightgbm import LGBMClassifier
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    if feature_names:
        X = pd.DataFrame(X, columns=feature_names)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    model = LGBMClassifier(
        n_estimators=200, max_depth=6, num_leaves=31,
        learning_rate=0.05, class_weight="balanced",
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return model

def update_model_online(model, online_X, online_y, feature_names=None):
    """用历史交易结果增量更新模型"""
    if len(online_X) < 3:
        return model
    X_new = np.nan_to_num(np.array(online_X), nan=0, posinf=0, neginf=0)
    y_new = np.array(online_y)
    if feature_names and len(feature_names) == X_new.shape[1]:
        X_new = pd.DataFrame(X_new, columns=feature_names)
    # 近期样本权重 ×1.5
    n = len(X_new)
    sw = np.ones(n)
    if n >= 5:
        sw[-min(48, n):] = 1.5
    try:
        model.fit(X_new, y_new, sample_weight=sw, init_model=model, eval_set=[(X_new[-5:], y_new[-5:])])
    except:
        pass  # partial_fit 在某些情况下可能失败，静默跳过
    return model


# ══════ 交易执行 ══════

def get_positions():
    try:
        pos_list = EXCHANGE.fetch_positions()
        active = []
        for p in pos_list:
            qty = float(p.get("contracts", 0) or 0)
            if qty <= 0: continue
            sym = (p.get("symbol", "")).replace(":USDT", "")
            entry = float(p.get("entryPrice", 0))
            mark = float(p.get("markPrice", 0) or 0)
            side = p.get("side", "long")
            lev = float(p.get("leverage", LEVERAGE))
            # 获取合约面值
            ct_val = 1.0
            try:
                mkt = EXCHANGE.market(sym.replace("/","/")+":USDT")
                ct_val = float(mkt.get("contractSize") or mkt.get("info",{}).get("ctVal",1))
            except: pass
            # 模拟盘自己算 PnL
            if entry > 0 and mark > 0:
                if side == "long":
                    pnl_pct = (mark / entry - 1) * lev * 100
                    pnl = (mark - entry) * qty * ct_val
                else:
                    pnl_pct = (entry / mark - 1) * lev * 100
                    pnl = (entry - mark) * qty * ct_val
            else:
                pnl_pct = 0; pnl = 0
            active.append({
                "symbol": sym, "side": side, "qty": qty,
                "entry": entry, "mark": mark,
                "unrealized_pnl": pnl,
                "percentage": pnl_pct,
                "margin": float(p.get("initialMargin", 0)),
                "leverage": lev,
            })
        return active
    except:
        return []

def open_position(sym: str, side: str, price: float, pos_pct: float = BASE_POSITION_PCT):
    try:
        swap = to_swap(sym)
        mkt = EXCHANGE.market(swap)
        ct_val = float(mkt.get("contractSize") or mkt.get("info", {}).get("ctVal", 0.01))
        bal = EXCHANGE.fetch_balance()
        free = bal["USDT"]["free"]
        margin_target = free * pos_pct
        contracts = max(int(margin_target * LEVERAGE / (price * ct_val)), 1)
        try:
            EXCHANGE.set_leverage(LEVERAGE, swap, {"mgnMode": "isolated"})
        except:
            pass
        order_side = "buy" if side == "long" else "sell"
        order = EXCHANGE.create_market_order(
            swap, order_side, contracts,
            params={"tdMode": "isolated", "posSide": side}
        )
        avg = order.get("average") or order.get("price") or price
        print(f"  ✅ {sym} {side} {contracts}张 @${avg:.4f}")
        return avg, contracts
    except Exception as e:
        print(f"  ❌ {sym} 开仓失败: {e}")
        return None, 0

def close_position(sym: str, side: str, qty: float, entry_price: float = 0, leverage: float = None, latest_features: np.ndarray = None, label: int = None):
    if leverage is None:
        leverage = LEVERAGE
    try:
        swap = to_swap(sym)
        order_side = "sell" if side == "long" else "buy"
        params = {"reduceOnly": True, "tdMode": "isolated"}
        # 平仓前先取标记价作为成交价兜底
        exit_price = 0
        try:
            positions = EXCHANGE.fetch_positions()
            for p in positions:
                if (p.get("symbol","").replace(":USDT","") == sym and p.get("contracts",0) > 0):
                    exit_price = float(p.get("markPrice", 0) or 0)
                    break
        except:
            pass

        try:
            order = EXCHANGE.create_market_order(swap, order_side, qty, params={"posSide": side, **params})
            exit_price = order.get("average") or order.get("price") or exit_price
        except Exception as e1:
            if "51169" in str(e1) or "51000" in str(e1):
                try:
                    order = EXCHANGE.create_market_order(swap, order_side, qty, params=params)
                    exit_price = order.get("average") or order.get("price") or exit_price
                except Exception as e2:
                    if "51169" in str(e2):
                        print(f"  🔒 {sym} 仓位已不存在")
                        # (position managed by exchange)
                        return
                    raise e2
            else:
                raise e1

        print(f"  🔒 平仓: {sym} {side} {qty}张")
        # (position managed by exchange)

        # 用 exit_price 算 PnL（平仓后交易所已无此仓位，不能再用 get_positions 查）
        pnl_pct = 0
        lev = leverage if leverage else LEVERAGE
        print(f"  🔍 平仓详情: entry={entry_price:.6f} exit={exit_price:.6f} lev={lev}x")
        if exit_price > 0 and entry_price > 0:
            if side == "long":
                pnl_pct = (exit_price / entry_price - 1) * lev * 100
            else:
                pnl_pct = (entry_price / exit_price - 1) * lev * 100

        record_trade(sym, side, entry_price, exit_price, qty, pnl_pct, lev, latest_features, label)
        print(f"  📝 交易记录: {sym} PnL={pnl_pct:.1f}% (${(exit_price - entry_price) * qty if side == 'long' else (entry_price - exit_price) * qty:.2f})")
    except Exception as e:
        print(f"  ❌ {sym} 平仓失败: {str(e)[:100]}")

def monitor_sl_tp():
    while True:
        try:
            positions = EXCHANGE.fetch_positions()
            for p in positions:
                qty = float(p.get("contracts", 0) or 0)
                if qty <= 0: continue
                sym = (p.get("symbol", "")).replace(":USDT", "")

                entry = float(p.get("entryPrice", 0))
                mark = float(p.get("markPrice", 0) or 0)
                side = p.get("side", "long")
                lev = float(p.get("leverage", LEVERAGE))

                # 自己算 PnL%（模拟盘 exchange 返回的 percentage 是 0）
                if entry > 0 and mark > 0:
                    if side == "long":
                        pnl_pct = (mark / entry - 1) * lev * 100
                    else:
                        pnl_pct = (entry / mark - 1) * lev * 100
                else:
                    pnl_pct = 0

                # 确保每个仓位都有止损止盈
                if sym not in GLOBAL_LATEST_SL:
                    GLOBAL_LATEST_SL[sym] = STOP_LOSS
                    GLOBAL_LATEST_TP[sym] = TAKE_PROFIT
                sl = GLOBAL_LATEST_SL.get(sym, 0)

                # 更新峰值
                peak = PEAK_PNL.get(sym, 0)
                if pnl_pct > peak:
                    PEAK_PNL[sym] = pnl_pct
                    peak = pnl_pct
                    # 峰值更新时持久化
                    try:
                        with open(os.path.join(DATA_DIR, "peak_pnl.pkl"), "wb") as f:
                            pickle.dump(PEAK_PNL, f)
                    except: pass

                if pnl_pct <= FORCE_CLOSE_PCT:
                    open_feat = OPEN_FEATURES.pop(sym, None)
                    print(f"🛑 强平: {sym} {pnl_pct:.1f}% ≤ {FORCE_CLOSE_PCT}%")
                    close_position(sym, side, qty, entry, lev, open_feat, 1 if pnl_pct > 0 else 0)
                    PEAK_PNL.pop(sym, None)
                    OPEN_TIME.pop(sym, None)
                    GLOBAL_LATEST_SL.pop(sym, None); GLOBAL_LATEST_TP.pop(sym, None)
                elif sl != 0 and pnl_pct <= sl:
                    open_feat = OPEN_FEATURES.pop(sym, None)
                    print(f"📉 止损: {sym} {pnl_pct:.1f}% ≤ {sl}%")
                    close_position(sym, side, qty, entry, lev, open_feat, 1 if pnl_pct > 0 else 0)
                    PEAK_PNL.pop(sym, None)
                    OPEN_TIME.pop(sym, None)
                    GLOBAL_LATEST_SL.pop(sym, None); GLOBAL_LATEST_TP.pop(sym, None)
                elif peak >= 17 and pnl_pct <= peak - 5:
                    open_feat = OPEN_FEATURES.pop(sym, None)
                    print(f"📈 移动止盈: {sym} 峰值{peak:.1f}%→{pnl_pct:.1f}% (回撤>{5}%)")
                    close_position(sym, side, qty, entry, lev, open_feat, 1 if pnl_pct > 0 else 0)
                    PEAK_PNL.pop(sym, None)
                    OPEN_TIME.pop(sym, None)
                    GLOBAL_LATEST_SL.pop(sym, None); GLOBAL_LATEST_TP.pop(sym, None)
                elif peak >= 12 and pnl_pct <= 12:
                    open_feat = OPEN_FEATURES.pop(sym, None)
                    print(f"📈 保底止盈: {sym} 峰值{peak:.1f}%→{pnl_pct:.1f}% (锁利12%)")
                    close_position(sym, side, qty, entry, lev, open_feat, 1 if pnl_pct > 0 else 0)
                    PEAK_PNL.pop(sym, None)
                    OPEN_TIME.pop(sym, None)
                    GLOBAL_LATEST_SL.pop(sym, None); GLOBAL_LATEST_TP.pop(sym, None)
        except Exception as e:
            print(f"⚠️ 监控异常: {str(e)[:100]}")
        time.sleep(2)


# ══════ 主循环 ══════

def main():
    load_trade_history()
    # 尝试加载已有模型
    model = load_model()
    feat_names = []

    print(f"🚀 LGBM 进化版 | {len(SYMBOLS)} 币种 | 46维特征 | 自适应+在线学习")
    print(f"   历史交易: {len(TRADE_HISTORY)} 笔 | 在线样本: {len(ONLINE_SAMPLES_X)} 条")
    print(f"   模型: {'已加载' if model else '需首次训练'} | 止盈 +{TAKE_PROFIT}% | 止损 {STOP_LOSS}%")

    existing = get_positions()
    if existing:
        print(f"  📋 管理 {len(existing)} 个已有仓位")
        for p in existing:
            GLOBAL_LATEST_SL[p["symbol"]] = STOP_LOSS
            GLOBAL_LATEST_TP[p["symbol"]] = TAKE_PROFIT
            if p["symbol"] not in OPEN_TIME:
                OPEN_TIME[p["symbol"]] = datetime.now().isoformat()

    thread = threading.Thread(target=monitor_sl_tp, daemon=True)
    thread.start()

    while True:
        try:
            # ── 自适应参数 ──
            pos_pct, confidence, max_positions = adapt_params()

            print(f"\n{'='*60}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} 执行 | 仓位{pos_pct*100:.0f}% 阈值{confidence} 最大{max_positions}仓")

            # 先给所有持仓补特征（避免监控触发时还没缓存）
            for p in get_positions():
                if p["symbol"] not in OPEN_FEATURES:
                    df2 = fetch_ohlcv(p["symbol"])
                    fund2 = fetch_fundamentals(p["symbol"])
                    feats2 = compute_features(df2, fund2) if df2 is not None else None
                    if feats2 is not None and len(feats2) > 0:
                        OPEN_FEATURES[p["symbol"]] = feats2.iloc[-1:].values.astype(np.float64)
            if OPEN_FEATURES:
                try:
                    with open(OPEN_FEATURES_FILE, "wb") as f:
                        pickle.dump(OPEN_FEATURES, f)
                except: pass

            all_features = {}; all_labels = {}; latest_features = {}; prices = {}; syms_ok = []

            for sym in SYMBOLS:
                df = fetch_ohlcv(sym)
                if df is None: continue
                fund = fetch_fundamentals(sym)
                feats = compute_features(df, fund)
                if feats is None or len(feats) < 100: continue
                lab = compute_label(df)
                valid = ~np.isnan(lab) & ~np.isnan(feats.values).any(axis=1)
                X_valid = feats[valid].values.astype(np.float64)
                y_valid = lab[valid].astype(int)
                if len(X_valid) < 200: continue
                all_features[sym] = X_valid; all_labels[sym] = y_valid
                latest_features[sym] = X_valid[-1:]
                prices[sym] = float(df["close"].iloc[-1])
                syms_ok.append(sym)

            if len(syms_ok) < 5:
                print(f"  ⚠️ 有效币种不足: {len(syms_ok)}")
                time.sleep(60); continue

            first_sym = syms_ok[0]
            sample_feats = compute_features(fetch_ohlcv(first_sym), fetch_fundamentals(first_sym))
            feat_names = list(sample_feats.columns) if sample_feats is not None else []

            # 首次运行才全量训练
            if model is None:
                X_all = np.vstack([all_features[s] for s in syms_ok])
                y_all = np.hstack([all_labels[s] for s in syms_ok])
                print(f"  首次训练: {len(X_all)} 样本 ({len(syms_ok)} 币种) | 特征: {len(feat_names)}维")
                model = train_lgbm(X_all, y_all, feat_names if feat_names else None)
            else:
                print(f"  📊 使用已有模型 | 特征: {len(feat_names)}维")

            # ── 在线学习：用历史交易增量更新（每轮都做） ──
            if model is not None and ONLINE_SAMPLES_X:
                model = update_model_online(model, ONLINE_SAMPLES_X, ONLINE_SAMPLES_Y, feat_names if feat_names else None)
                print(f"  🧠 在线学习: {len(ONLINE_SAMPLES_X)} 条交易经验")

            if model is not None:
                imp = model.feature_importances_
                fn = model.feature_name_ if hasattr(model, "feature_name_") else range(len(imp))
                top5 = sorted(zip(fn, imp), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top5: {', '.join(f'{n}({v:.0f})' for n,v in top5)}")
                # 保存模型
                save_model(model)

            # ── 信号 ──
            signals = []
            fn_list = list(model.feature_name_) if hasattr(model, "feature_name_") else None
            for sym in syms_ok:
                X_sym = np.nan_to_num(latest_features[sym], nan=0, posinf=0, neginf=0)
                if fn_list:
                    X_sym = pd.DataFrame(X_sym, columns=fn_list)
                pred = model.predict_proba(X_sym)[0, 1]

                rsi_val = 50
                if fn_list and "rsi" in fn_list:
                    rsi_idx = fn_list.index("rsi")
                    rsi_val = float(X_sym.iloc[0, rsi_idx] if isinstance(X_sym, pd.DataFrame) else X_sym[0, rsi_idx])

                # 多指标共振：≥3个同向才开仓
                if fn_list:
                    bullish = 0; bearish = 0
                    row = X_sym.iloc[0] if isinstance(X_sym, pd.DataFrame) else pd.Series(X_sym[0], index=fn_list)
                    # RSI: >60多, <40空 (避开中性区50±10)
                    if rsi_val > 60: bullish += 1
                    elif rsi_val < 40: bearish += 1
                    # MACD 状态: +2多加速, -2空加速 (避开±1弱信号)
                    ms = row.get("macd_hist_state", 0)
                    if ms >= 2: bullish += 1
                    elif ms <= -2: bearish += 1
                    # SuperTrend: 1多, -1空 (二值,无需改)
                    st = row.get("st_trend", 0)
                    if st > 0: bullish += 1
                    else: bearish += 1
                    # QQE: >0多, <0空
                    qs = row.get("qqe_signal", 0)
                    if qs > 0: bullish += 1
                    elif qs < 0: bearish += 1
                    # A-V2: >10多, <-10空 (避开中性区0±10)
                    av2 = row.get("av2_trend", 0)
                    if av2 > 10: bullish += 1
                    elif av2 < -10: bearish += 1

                if pred > 0.5 + confidence and rsi_val > 50 and bullish >= 3:
                    signals.append({"symbol": sym, "action": "long", "score": float(pred),
                        "price": prices[sym], "rsi": rsi_val, "bullish": bullish, "bearish": bearish,
                        "resonance": f"多{bullish}空{bearish}"})
                elif pred < 0.5 - confidence and rsi_val < 50 and bearish >= 3:
                    signals.append({"symbol": sym, "action": "short", "score": float(pred),
                        "price": prices[sym], "rsi": rsi_val, "bullish": bullish, "bearish": bearish,
                        "resonance": f"多{bullish}空{bearish}"})

            signals.sort(key=lambda x: abs(x["score"] - 0.5), reverse=True)

            # 始终打印信号列表
            if signals:
                print(f"  🎯 本轮信号 ({len(signals)}个):")
                for s in signals[:10]:
                    print(f"     {s['action']:5s} {s['symbol']:12s} pred={s['score']:.3f} RSI={s.get('rsi','?'):.0f} 共振={s.get('resonance','?')}")
            elif syms_ok:
                print(f"  🔍 信号详情 (多:pred>{0.5+confidence:.2f} RSI>50 | 空:pred<{0.5-confidence:.2f} RSI<50 | ≥3共振):")
                for sym in syms_ok[:5]:
                    X_sym2 = np.nan_to_num(latest_features[sym], nan=0, posinf=0, neginf=0)
                    if fn_list: X_sym2 = pd.DataFrame(X_sym2, columns=fn_list)
                    pred2 = model.predict_proba(X_sym2)[0, 1]
                    rsi2 = float(X_sym2.iloc[0, fn_list.index("rsi")]) if fn_list and "rsi" in fn_list else 50
                    row = X_sym2.iloc[0] if fn_list else pd.Series()
                    ms = row.get("macd_hist_state", 0)
                    st = row.get("st_trend", 0)
                    qs = row.get("qqe_signal", 0)
                    av2 = row.get("av2_trend", 0)
                    bull = (1 if rsi2>55 else 0) + (1 if ms>0 else 0) + (1 if st>0 else 0) + (1 if qs>0 else 0) + (1 if av2>0 else 0)
                    bear = (1 if rsi2<45 else 0) + (1 if ms<0 else 0) + (1 if st<0 else 0) + (1 if qs<0 else 0) + (1 if av2<0 else 0)
                    score_info = []
                    for name, val, bullish in [("RSI",rsi2,rsi2>55),("MACD",ms,ms>0),("ST",st,st>0),("QQE",qs,qs>0),("AV2",av2,av2>0)]:
                        score_info.append(f"{name}:{'多'if bullish else'空'}")
                    print(f"     {sym}: pred={pred2:.3f} RSI={rsi2:.0f} 共振=多{bull}空{bear} | {' '.join(score_info)}")

            try:
                with open(os.path.join(DATA_DIR, "signals.json"), "w") as f:
                    json.dump({"signals": signals, "updated": datetime.now().isoformat()}, f, ensure_ascii=False, default=str)
            except: pass

            # ── 交易：管理所有交易所仓位 ──
            all_positions = get_positions()
            active_syms = {p["symbol"] for p in all_positions}
            available = max(0, max_positions - len(active_syms))
            print(f"  持仓: {len(active_syms)}/{max_positions} | 信号: {len(signals)} | 在线样本: {len(ONLINE_SAMPLES_X)}")

            for sig in signals[:available]:
                if sig["symbol"] in active_syms:
                    print(f"  ⏭️ {sig['symbol']} 已有仓位,跳过")
                    continue
                # 余额检查
                try:
                    bal = EXCHANGE.fetch_balance()
                    free = bal["USDT"]["free"]
                    if free < 50:
                        print(f"  ⚠️ 余额不足(${free:.0f})，跳过 {sig['symbol']}")
                        continue
                except:
                    pass
                avg, qty = open_position(sig["symbol"], sig["action"], sig["price"], pos_pct)
                if avg is not None:
                    GLOBAL_LATEST_SL[sig["symbol"]] = STOP_LOSS
                    GLOBAL_LATEST_TP[sig["symbol"]] = TAKE_PROFIT
                    # 保存开仓时的特征，等平仓时喂给模型
                    if sig["symbol"] in latest_features:
                        OPEN_FEATURES[sig["symbol"]] = latest_features[sig["symbol"]]
                    OPEN_TIME[sig["symbol"]] = datetime.now().isoformat()
                    try:
                        with open(OPEN_FEATURES_FILE, "wb") as f:
                            pickle.dump({"features": OPEN_FEATURES, "times": OPEN_TIME}, f)
                    except:
                        pass

            now = datetime.now()
            next_15 = ((now.minute // 15) + 1) * 15
            wait = (next_15 - now.minute) * 60 - now.second + 10
            if wait > 0:
                print(f"  🕒 等待 {wait}s...")
                time.sleep(wait)

        except Exception as e:
            print(f"  ❌ 主循环异常: {e}")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
