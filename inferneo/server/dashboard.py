"""Self-contained live dashboard served at ``/dashboard``.

A single HTML page (inline CSS + JS, no external deps) that polls ``/stats`` once
a second and renders GPU, KV-cache, throughput, and latency. Deliberately zero
infrastructure — no Prometheus, no Grafana — so the server is its own monitor.
The Prometheus ``/metrics`` endpoint is available alongside for anyone who wants
to wire up Grafana instead.
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>inferneo · live</title>
<style>
  :root {
    --bg:#0b0e14; --panel:#141922; --panel2:#1c2330; --line:#263041;
    --text:#e6edf3; --muted:#8b98a9; --accent:#4ea1ff; --good:#3fb950;
    --warn:#d29922; --hot:#f85149; --track:#222b39;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.4 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { display:flex; align-items:center; gap:14px; padding:16px 22px;
    border-bottom:1px solid var(--line); background:var(--panel); }
  .logo { font-weight:700; font-size:18px; letter-spacing:.5px; }
  .logo span { color:var(--accent); }
  .model { color:var(--muted); font-size:13px; }
  .live { margin-left:auto; display:flex; align-items:center; gap:7px;
    color:var(--muted); font-size:12px; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--good);
    box-shadow:0 0 0 0 rgba(63,185,80,.6); animation:pulse 2s infinite; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(63,185,80,.5)}
    70%{box-shadow:0 0 0 7px rgba(63,185,80,0)} 100%{box-shadow:0 0 0 0 rgba(63,185,80,0)} }
  main { padding:22px; max-width:1200px; margin:0 auto; }
  .grid { display:grid; gap:16px; grid-template-columns:repeat(4,1fr); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:16px 18px; }
  .card h3 { margin:0 0 8px; font-size:12px; font-weight:600; color:var(--muted);
    text-transform:uppercase; letter-spacing:.6px; }
  .big { font-size:30px; font-weight:700; font-variant-numeric:tabular-nums; }
  .unit { font-size:13px; color:var(--muted); font-weight:500; margin-left:5px; }
  .span2 { grid-column:span 2; } .span4 { grid-column:span 4; }
  .row { display:flex; justify-content:space-between; align-items:baseline;
    padding:6px 0; border-top:1px solid var(--line); }
  .row:first-of-type { border-top:none; }
  .row .k { color:var(--muted); } .row .v { font-variant-numeric:tabular-nums; font-weight:600; }
  .bar { height:9px; border-radius:5px; background:var(--track); overflow:hidden; margin-top:8px; }
  .bar > i { display:block; height:100%; border-radius:5px; background:var(--accent);
    transition:width .4s ease,background .4s; }
  canvas { width:100%; height:46px; display:block; margin-top:10px; }
  .sub { color:var(--muted); font-size:12px; margin-top:4px; }
  @media (max-width:820px){ .grid{grid-template-columns:repeat(2,1fr)} .span2,.span4{grid-column:span 2} }
</style>
</head>
<body>
<header>
  <div class="logo">infer<span>neo</span></div>
  <div class="model" id="model">loading…</div>
  <div class="live"><span class="dot"></span><span id="uptime">—</span></div>
</header>
<main>
  <div class="grid">
    <div class="card span2">
      <h3>Generation throughput</h3>
      <div><span class="big" id="gtps">0</span><span class="unit">tok/s</span></div>
      <canvas id="spark_tps"></canvas>
    </div>
    <div class="card"><h3>Running</h3><div class="big" id="running">0</div><div class="sub">decoding now</div></div>
    <div class="card"><h3>Waiting</h3><div class="big" id="waiting">0</div><div class="sub">queued</div></div>

    <div class="card span2">
      <h3>GPU memory</h3>
      <div><span class="big" id="gmem">0</span><span class="unit" id="gmemtot">/ 0 GB</span></div>
      <div class="bar"><i id="gmembar" style="width:0%"></i></div>
      <canvas id="spark_mem"></canvas>
    </div>
    <div class="card">
      <h3>SM utilization</h3>
      <div><span class="big" id="smutil">—</span><span class="unit">%</span></div>
      <div class="bar"><i id="smbar" style="width:0%"></i></div>
    </div>
    <div class="card">
      <h3>KV cache</h3>
      <div><span class="big" id="kv">0</span><span class="unit">%</span></div>
      <div class="bar"><i id="kvbar" style="width:0%"></i></div>
    </div>

    <div class="card span2">
      <h3>Latency (rolling)</h3>
      <div class="row"><span class="k">TTFT p50 / p99</span><span class="v"><span id="ttft50">0</span> / <span id="ttft99">0</span> ms</span></div>
      <div class="row"><span class="k">TPOT p50 / p99</span><span class="v"><span id="tpot50">0</span> / <span id="tpot99">0</span> ms</span></div>
      <div class="row"><span class="k">End-to-end p50 / p99</span><span class="v"><span id="e2e50">0</span> / <span id="e2e99">0</span> ms</span></div>
    </div>
    <div class="card span2">
      <h3>Totals since start</h3>
      <div class="row"><span class="k">Requests finished</span><span class="v" id="t_fin">0</span></div>
      <div class="row"><span class="k">Tokens generated</span><span class="v" id="t_gen">0</span></div>
      <div class="row"><span class="k">Prompt tokens</span><span class="v" id="t_prm">0</span></div>
      <div class="row"><span class="k">Preemptions</span><span class="v" id="t_pre">0</span></div>
    </div>
  </div>
  <div class="sub" id="gpuline" style="margin-top:14px"></div>
</main>
<script>
const $ = id => document.getElementById(id);
const HIST = 90;
const tps_hist = [], mem_hist = [];

function fmtNum(n){ return n>=1e6 ? (n/1e6).toFixed(1)+'M' : n>=1e3 ? (n/1e3).toFixed(1)+'k' : Math.round(n).toString(); }
function fmtGB(b){ return (b/1073741824).toFixed(1); }
function fmtDur(s){ s=Math.floor(s); const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=s%60;
  return (h?h+'h ':'')+(m?m+'m ':'')+ss+'s'; }

function spark(cv, data, color, max){
  const dpr = window.devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr; const c=cv.getContext('2d'); c.scale(dpr,dpr);
  c.clearRect(0,0,w,h); if(data.length<2) return;
  const mx = max || Math.max(...data, 1e-6);
  const step = w/(HIST-1);
  c.beginPath();
  data.forEach((v,i)=>{ const x=i*step, y=h-4-(v/mx)*(h-8); i?c.lineTo(x,y):c.moveTo(x,y); });
  c.strokeStyle=color; c.lineWidth=2; c.lineJoin='round'; c.stroke();
  c.lineTo((data.length-1)*step, h); c.lineTo(0,h); c.closePath();
  c.fillStyle=color+'22'; c.fill();
}

function utilColor(f){ return f<0.6?'var(--good)':f<0.85?'var(--warn)':'var(--hot)'; }

async function tick(){
  let s; try{ s = await (await fetch('/stats')).json(); }catch(e){ return; }
  $('uptime').textContent = 'up '+fmtDur(s.uptime_s);
  $('gtps').textContent = s.generation_tps.toFixed(0);
  $('running').textContent = s.running;
  $('waiting').textContent = s.waiting;
  const kv = s.kv_cache_usage*100;
  $('kv').textContent = kv.toFixed(0); $('kvbar').style.width=kv+'%';
  $('kvbar').style.background = utilColor(s.kv_cache_usage);

  $('ttft50').textContent=s.ttft_ms.p50.toFixed(0); $('ttft99').textContent=s.ttft_ms.p99.toFixed(0);
  $('tpot50').textContent=s.tpot_ms.p50.toFixed(1); $('tpot99').textContent=s.tpot_ms.p99.toFixed(1);
  $('e2e50').textContent=s.e2e_ms.p50.toFixed(0); $('e2e99').textContent=s.e2e_ms.p99.toFixed(0);

  $('t_fin').textContent=fmtNum(s.totals.finished_requests);
  $('t_gen').textContent=fmtNum(s.totals.generation_tokens);
  $('t_prm').textContent=fmtNum(s.totals.prompt_tokens);
  $('t_pre').textContent=fmtNum(s.totals.preemptions);

  const g = s.gpu;
  if(g){
    const frac=g.mem_used_frac;
    $('gmem').textContent=fmtGB(g.mem_used_bytes);
    $('gmemtot').textContent='/ '+fmtGB(g.mem_total_bytes)+' GB';
    $('gmembar').style.width=(frac*100)+'%'; $('gmembar').style.background=utilColor(frac);
    mem_hist.push(frac); if(mem_hist.length>HIST) mem_hist.shift();
    if('sm_util' in g){ const u=g.sm_util*100; $('smutil').textContent=u.toFixed(0);
      $('smbar').style.width=u+'%'; $('smbar').style.background=utilColor(g.sm_util); }
    const bits=[];
    if('power_w' in g) bits.push(g.power_w.toFixed(0)+' W');
    if('temp_c' in g) bits.push(g.temp_c+' °C');
    if('torch_reserved_bytes' in g) bits.push('torch reserved '+fmtGB(g.torch_reserved_bytes)+' GB');
    $('gpuline').textContent = bits.join('   ·   ');
  } else {
    $('gmem').textContent='—'; $('gmemtot').textContent=''; $('smutil').textContent='n/a';
    $('gpuline').textContent='GPU metrics unavailable (running on CPU/MPS, or pynvml not installed)';
  }
  tps_hist.push(s.generation_tps); if(tps_hist.length>HIST) tps_hist.shift();
  spark($('spark_tps'), tps_hist, getComputedStyle(document.documentElement).getPropertyValue('--accent').trim());
  spark($('spark_mem'), mem_hist, getComputedStyle(document.documentElement).getPropertyValue('--good').trim(), 1.0);
}

fetch('/v1/models').then(r=>r.json()).then(d=>{ if(d.data&&d.data[0]) $('model').textContent=d.data[0].id; }).catch(()=>{});
tick(); setInterval(tick, 1000);
</script>
</body>
</html>
"""
