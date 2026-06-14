#!/usr/bin/env python3
"""LGBM Dashboard — 账户+持仓+交易记录+信号"""
import json, os, time, threading, http.server, urllib.parse, ccxt, pickle
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EXCHANGE = None
STATUS = {}

def get_exchange():
    global EXCHANGE
    if EXCHANGE:
        try: EXCHANGE.fetch_time(); return EXCHANGE
        except: EXCHANGE = None
    try:
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
        kw = {"options": {"defaultType": "swap", "fetchMarketsByDefault": False},
              "apiKey": os.getenv("OKX_API_KEY"), "secret": os.getenv("OKX_SECRET"),
              "password": os.getenv("OKX_PASSWORD"), "enableRateLimit": True, "timeout": 15000}
        if proxy: kw["proxies"] = {"http": proxy, "https": proxy}
        ex = ccxt.okx(kw)
        ex.set_sandbox_mode(os.getenv("OKX_SANDBOX", "true").lower() == "true")
        try: ex.load_markets(reload=True, params={'instType': 'SWAP'})
        except: ex.load_markets()
        EXCHANGE = ex; return ex
    except Exception as e:
        print(f"[dash] ex: {e}"); return None

def read_json(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

def refresh():
    global STATUS
    while True:
        ex = get_exchange()
        result = {"ok": True, "account": {}, "positions": [], "trades": [], "signals": [],
                  "online_samples": 0, "model_loaded": False}
        if ex:
            try:
                bal = ex.fetch_balance()
                usdt = bal.get("USDT", {})
                eq = float(usdt.get("total", 0))
                free = float(usdt.get("free", 0))
                used = eq - free if eq > free else 0
                pos_list = ex.fetch_positions()
                positions = []
                upl_total = 0
                for p in pos_list:
                    qty = float(p.get("contracts", 0) or 0)
                    if qty <= 0: continue
                    sym = (p.get("symbol","")).replace(":USDT","")
                    entry = float(p.get("entryPrice", 0))
                    mark = float(p.get("markPrice", 0) or 0)
                    side = p.get("side", "long")
                    lev = float(p.get("leverage", 10))
                    # 获取合约面值
                    ct_val = 1.0
                    try:
                        swap = sym.replace("/","/")+":USDT"
                        mkt = ex.market(swap)
                        ct_val = float(mkt.get("contractSize") or mkt.get("info",{}).get("ctVal",1))
                    except: pass
                    if entry > 0 and mark > 0:
                        if side == "long":
                            pnl_pct = (mark / entry - 1) * lev * 100
                            pnl = (mark - entry) * qty * ct_val
                        else:
                            pnl_pct = (entry / mark - 1) * lev * 100
                            pnl = (entry - mark) * qty * ct_val
                    else:
                        pnl_pct = 0; pnl = 0
                    upl_total += pnl
                    positions.append({
                        "symbol": (p.get("symbol","")).replace(":USDT",""),
                        "side": side, "qty": qty, "entry": round(entry, 6),
                        "mark": round(mark, 6), "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "leverage": int(lev), "margin": round(float(p.get("initialMargin", 0)), 2),
                    })
                result["account"] = {"totalEquity": round(eq, 2), "availableBalance": round(free, 2),
                                     "unrealizedPnl": round(upl_total, 2), "marginUsed": round(used, 2)}
                result["positions"] = positions
                # 开仓时间
                open_times = {}
                try:
                    with open(os.path.join(DATA_DIR, "open_features.pkl"), "rb") as f:
                        d = pickle.load(f)
                        open_times = d.get("times", {}) if isinstance(d, dict) else {}
                except: pass
                for p in positions:
                    p["open_time"] = open_times.get(p["symbol"], "")
            except Exception as e:
                result["ok"] = False; result["error"] = str(e)[:200]

        # LGBM data
        result["trades"] = read_json(os.path.join(DATA_DIR, "trade_history.json")) or []
        sigs = read_json(os.path.join(DATA_DIR, "signals.json")) or {}
        result["signals"] = sigs.get("signals", []) if isinstance(sigs, dict) else []
        result["model_loaded"] = os.path.exists(os.path.join(DATA_DIR, "model.pkl"))
        try:
            with open(os.path.join(DATA_DIR, "online_samples.pkl"), "rb") as f:
                d = pickle.load(f)
                result["online_samples"] = len(d.get("online_x", []))
        except: pass
        STATUS = result
        time.sleep(5)

PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LGBM Trading</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;font-size:14px}
h1{color:#58a6ff;font-size:22px;margin-bottom:4px}
h2{color:#8b949e;font-weight:400;font-size:13px;margin-bottom:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:14px}
.card h3{color:#58a6ff;font-size:14px;margin-bottom:10px}
.stat{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d;font-size:14px}
.label{color:#8b949e}.value{font-weight:600}.green{color:#3fb950}.red{color:#f85149}.yellow{color:#d2991d}
.badge{display:inline-block;padding:2px 8px;border-radius:8px;font-size:12px;font-weight:600}
.long{background:rgba(248,81,73,0.15);color:#f85149}.short{background:rgba(63,185,80,0.15);color:#3fb950}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:6px 8px;color:#8b949e;border-bottom:1px solid #30363d;font-weight:500}
td{padding:6px 8px;border-bottom:1px solid #21262d}.num{text-align:right}
.row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px}
.stat-box{flex:1;min-width:120px}
.stat-num{font-size:24px;font-weight:700}.stat-label{font-size:11px;color:#8b949e}
</style></head><body>
<h1>📊 LGBM 多币种择时</h1>
<h2 id="subtitle">⏳ 加载中...</h2>
<div class="row" id="stats"></div>
<div class="card"><h3>📈 实时持仓 <span id="pos-count"></span></h3><div id="positions">—</div></div>
<div class="card"><h3>🤖 AI 信号 <span id="sig-count"></span></h3><div id="signals">—</div></div>
<div class="card"><h3>📋 交易记录 <span id="trade-count"></span></h3><div id="trades">—</div></div>
<div style="text-align:center;color:#8b949e;font-size:11px;margin-top:10px" id="footer">⏳</div>
<script>
const F={n:(v,d)=>v==null||isNaN(v)?"—":v.toFixed(d||2),p:v=>v==null||isNaN(v)?"—":(v>=0?"+":"")+v.toFixed(2)+"%",t:s=>(s||"").slice(11,19)};
async function tick(){
  try{
    const r=await(await fetch("/api/status")).json();
    if(!r.ok) return;

    // Stats
    const a=r.account||{};
    const upl=a.unrealizedPnl||0;
    document.getElementById("stats").innerHTML=
      '<div class="stat-box"><div class="stat-num">$'+F.n(a.totalEquity,0)+'</div><div class="stat-label">总权益</div></div>'+
      '<div class="stat-box"><div class="stat-num">$'+F.n(a.availableBalance,0)+'</div><div class="stat-label">可用余额</div></div>'+
      '<div class="stat-box"><div class="stat-num '+(upl>=0?"green":"red")+'">'+(upl>=0?"+":"")+'$'+F.n(Math.abs(upl),0)+'</div><div class="stat-label">未实现盈亏</div></div>'+
      '<div class="stat-box"><div class="stat-num">$'+F.n(a.marginUsed,0)+'</div><div class="stat-label">已用保证金</div></div>'+
      '<div class="stat-box"><div class="stat-num">'+r.online_samples+'</div><div class="stat-label">在线样本</div></div>';

    document.getElementById("subtitle").textContent=
      '模型:'+(r.model_loaded?'已加载':'需训练')+' | 止盈+12% | 止损-5% | 强平-15%';

    // Positions
    const pos=r.positions||[];
    document.getElementById("pos-count").textContent='('+pos.length+'个)';
    if(pos.length===0){
      document.getElementById("positions").innerHTML='<span style="color:#8b949e">无持仓</span>';
    }else{
      let ph='<table><tr><th class="num">开仓时间</th><th>币种</th><th>方向</th><th class="num">杠杆</th><th class="num">开仓价</th><th class="num">标记价</th><th class="num">数量</th><th class="num">盈亏$</th><th class="num">盈亏%</th><th class="num">保证金</th></tr>';
      for(const p of pos){
        const c=p.pnl>=0?"green":"red";
        const s=p.side==="long"?"多":"空";
        ph+='<tr><td class="num" style="color:#8b949e">'+(p.open_time||"").slice(11,19)+'</td><td><b>'+p.symbol+'</b></td><td><span class="badge '+(p.side)+'">'+s+'</span></td><td class="num">'+p.leverage+'x</td><td class="num">$'+F.n(p.entry,4)+'</td><td class="num">$'+F.n(p.mark,4)+'</td><td class="num">'+p.qty+'</td><td class="num '+c+'">'+(p.pnl>=0?"+":"")+'$'+F.n(p.pnl,2)+'</td><td class="num '+c+'">'+F.p(p.pnl_pct)+'</td><td class="num">$'+F.n(p.margin,2)+'</td></tr>';
      }
      ph+='</table>';
      document.getElementById("positions").innerHTML=ph;
    }

    // Signals
    const sigs=r.signals||[];
    document.getElementById("sig-count").textContent='('+sigs.length+'个)';
    if(sigs.length===0){
      document.getElementById("signals").innerHTML='<span style="color:#8b949e">暂无</span>';
    }else{
      let sh='<table><tr><th>币种</th><th>方向</th><th class="num">预测分</th><th class="num">RSI</th><th>共振</th><th class="num">价格</th></tr>';
      for(const s of sigs.slice(0,10)){
        sh+='<tr><td><b>'+s.symbol+'</b></td><td><span class="badge '+(s.action)+'">'+(s.action==="long"?"多":"空")+'</span></td><td class="num">'+(s.score*100).toFixed(1)+'%</td><td class="num">'+(s.rsi||0).toFixed(0)+'</td><td class="yellow">'+(s.resonance||"")+'</td><td class="num">$'+F.n(s.price,4)+'</td></tr>';
      }
      sh+='</table>';
      document.getElementById("signals").innerHTML=sh;
    }

    // Trades
    const trades=r.trades||[];
    const wins=trades.filter(t=>t.pnl_pct>0).length;
    document.getElementById("trade-count").textContent='('+trades.length+'笔, '+(trades.length?(wins/trades.length*100).toFixed(0):"—")+'%胜率)';
    if(trades.length===0){
      document.getElementById("trades").innerHTML='<span style="color:#8b949e">暂无结单</span>';
    }else{
      let th='<table><tr><th>时间</th><th>币种</th><th>方向</th><th class="num">杠杆</th><th class="num">开仓价</th><th class="num">平仓价</th><th class="num">数量</th><th class="num">盈亏$</th><th class="num">盈亏%</th></tr>';
      for(let i=trades.length-1;i>=Math.max(0,trades.length-30);i--){
        const t=trades[i]; const c=t.pnl_pct>=0?"green":"red";
        const sideBadge=t.side==="long"?'<span class="badge long">多</span>':'<span class="badge short">空</span>';
        const pnlUsd=t.pnl_usd||0;
        th+='<tr><td>'+F.t(t.ts)+'</td><td><b>'+t.symbol+'</b></td><td>'+sideBadge+'</td><td class="num">'+(t.leverage||0)+'x</td><td class="num">$'+F.n(t.entry,4)+'</td><td class="num">$'+F.n(t.exit,4)+'</td><td class="num">'+(t.qty||0)+'</td><td class="num '+c+'">'+(pnlUsd>=0?"+":"")+'$'+F.n(Math.abs(pnlUsd),2)+'</td><td class="num '+c+'">'+(t.pnl_pct>=0?"+":"")+t.pnl_pct.toFixed(2)+'%</td></tr>';
      }
      th+='</table>';
      document.getElementById("trades").innerHTML=th;
    }

    document.getElementById("footer").textContent='🔄 '+new Date().toLocaleTimeString("zh-CN")+' | 每5秒刷新';
  }catch(e){
    document.getElementById("footer").textContent='ERROR: '+e.message;
  }
}
tick();setInterval(tick,5000);
</script></body></html>"""

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        if p.path == "/api/status":
            self.send_header("Content-Type","application/json;charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(STATUS, ensure_ascii=False, default=str).encode())
        else:
            self.send_header("Content-Type","text/html;charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode("utf-8"))
    def log_message(self,*a): pass

if __name__ == "__main__":
    t = threading.Thread(target=refresh, daemon=True); t.start()
    print("🚀 LGBM Dashboard → http://localhost:3101")
    http.server.HTTPServer(("0.0.0.0", 3101), Handler).serve_forever()
