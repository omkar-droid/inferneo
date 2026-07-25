"""Self-contained live dashboard served at ``/dashboard``.

A single HTML page (inline CSS + JS, no external deps) that polls ``/stats`` once
a second and renders GPU, KV-cache, throughput, and latency. Deliberately zero
infrastructure — no Prometheus, no Grafana — so the server is its own monitor.
The Prometheus ``/metrics`` endpoint is available alongside for anyone who wants
to wire up Grafana instead.

Committed to a single dark "control-room" theme on purpose: it's an instrument
panel for a GPU under load, not a document.

The JS is split into a pure ``render(snapshot)`` and a data source below the
``DATA SOURCE`` marker, so a static preview can replay a captured trace through
the identical render path.
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>inferneo · live</title>
<style>
  :root{
    --bg:#0a0d13; --panel:#111722; --panel2:#161d2b; --line:#222c3d;
    --text:#e8eef6; --muted:#7d8ba0; --faint:#4a5a70;
    --accent:#5cc8ff; --good:#3fb950; --warn:#e3b341; --hot:#ff6a5a;
    --track:#1b2331;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,"Cascadia Mono",Consolas,monospace;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  html{-webkit-text-size-adjust:100%}
  body{margin:0;background:
      radial-gradient(1200px 600px at 80% -10%, #12203050, transparent),
      var(--bg);
    color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.4;
    min-height:100vh}
  header{display:flex;align-items:center;gap:16px;padding:15px 24px;
    border-bottom:1px solid var(--line);background:linear-gradient(180deg,#0e141d,#0b1016)}
  .logo{font-weight:700;font-size:17px;letter-spacing:.4px}
  .logo b{color:var(--accent);font-weight:700}
  .model{color:var(--muted);font-size:12.5px;font-family:var(--mono);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:44vw}
  .pill{margin-left:auto;font-family:var(--mono);font-size:11px;font-weight:600;
    letter-spacing:.8px;padding:5px 11px;border-radius:999px;border:1px solid transparent;
    display:flex;align-items:center;gap:7px}
  .pill .dot{width:8px;height:8px;border-radius:50%;background:currentColor}
  .pill.live{color:var(--good);background:#0f2417;border-color:#1c4a2c}
  .pill.live .dot{animation:pulse 1.6s infinite}
  .pill.sat{color:var(--warn);background:#241f0d;border-color:#4a3d16}
  .pill.idle{color:var(--muted);background:#161d2b;border-color:var(--line)}
  @keyframes pulse{0%{box-shadow:0 0 0 0 currentColor}70%{box-shadow:0 0 0 6px transparent}
    100%{box-shadow:0 0 0 0 transparent}}
  .up{color:var(--muted);font-size:12px;font-family:var(--mono);white-space:nowrap}
  .devbar{display:flex;gap:20px;flex-wrap:wrap;padding:9px 24px;
    border-bottom:1px solid var(--line);background:#0c1219;color:var(--muted);
    font-family:var(--mono);font-size:11.5px;letter-spacing:.3px}
  .devbar b{color:var(--text);font-weight:600}
  .devbar .using{color:var(--good)}
  main{padding:22px;max-width:1180px;margin:0 auto}
  .grid{display:grid;gap:15px;grid-template-columns:repeat(4,1fr)}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:13px;
    padding:15px 17px;position:relative;overflow:hidden}
  .card h3{margin:0 0 9px;font-size:11px;font-weight:600;color:var(--muted);
    text-transform:uppercase;letter-spacing:.7px}
  .val{font-family:var(--mono);font-size:30px;font-weight:600;letter-spacing:-.5px;
    font-variant-numeric:tabular-nums;line-height:1}
  .unit{font-family:var(--mono);font-size:13px;color:var(--muted);margin-left:6px}
  .sub{color:var(--muted);font-size:12px;margin-top:5px}
  .span2{grid-column:span 2}
  .row{display:flex;justify-content:space-between;align-items:baseline;
    padding:7px 0;border-top:1px solid var(--line)}
  .row:first-of-type{border-top:none}
  .row .k{color:var(--muted)}
  .row .v{font-family:var(--mono);font-variant-numeric:tabular-nums;font-weight:600}
  .bar{height:8px;border-radius:5px;background:var(--track);overflow:hidden;margin-top:10px}
  .bar>i{display:block;height:100%;border-radius:5px;background:var(--accent);
    transition:width .5s ease,background .5s}
  canvas{width:100%;height:44px;display:block;margin-top:10px}
  .gpuline{color:var(--muted);font-size:12px;font-family:var(--mono);margin-top:16px;
    letter-spacing:.2px}
  @media (max-width:840px){.grid{grid-template-columns:repeat(2,1fr)}.span2{grid-column:span 2}
    .model{max-width:30vw}}
  @media (prefers-reduced-motion:reduce){.pill.live .dot{animation:none}}
</style>
</head>
<body>
<header>
  <div class="logo">infer<b>neo</b></div>
  <div class="model" id="model">loading…</div>
  <div class="pill idle" id="status"><span class="dot"></span><span id="status_t">—</span></div>
  <div class="up" id="uptime">—</div>
</header>
<div class="devbar" id="devbar" style="display:none">
  <span><b id="d_gpu">—</b></span>
  <span id="d_vram">—</span>
  <span id="d_cuda">—</span>
  <span id="d_cc">—</span>
  <span id="d_drv">—</span>
  <span class="using" id="d_cnt">—</span>
</div>
<main>
  <div class="grid">
    <div class="card span2">
      <h3>Generation throughput</h3>
      <div><span class="val" id="gtps">0</span><span class="unit">tok/s</span></div>
      <canvas id="spark_tps"></canvas>
    </div>
    <div class="card"><h3>Running</h3><div class="val" id="running">0</div><div class="sub">decoding now</div></div>
    <div class="card"><h3>Waiting</h3><div class="val" id="waiting">0</div><div class="sub">queued</div></div>

    <div class="card span2">
      <h3>GPU memory</h3>
      <div><span class="val" id="gmem">0</span><span class="unit" id="gmemtot">/ 0 GB</span></div>
      <div class="bar"><i id="gmembar" style="width:0%"></i></div>
      <canvas id="spark_mem"></canvas>
    </div>
    <div class="card">
      <h3>SM utilization</h3>
      <div><span class="val" id="smutil">—</span><span class="unit">%</span></div>
      <div class="bar"><i id="smbar" style="width:0%"></i></div>
    </div>
    <div class="card">
      <h3>HBM bandwidth</h3>
      <div><span class="val" id="hbm">—</span><span class="unit">%</span></div>
      <div class="bar"><i id="hbmbar" style="width:0%"></i></div>
      <div class="sub">the decode ceiling</div>
    </div>

    <div class="card">
      <h3>Effective batch</h3>
      <div><span class="val" id="ebatch">0</span><span class="unit">tok/step</span></div>
      <div class="sub" id="prefill">prefill —</div>
    </div>
    <div class="card">
      <h3>MFU</h3>
      <div><span class="val" id="mfu">—</span><span class="unit" id="mfu_u">%</span></div>
      <div class="sub" id="mfu_sub">bf16 peak</div>
    </div>
    <div class="card">
      <h3>KV cache</h3>
      <div><span class="val" id="kv">0</span><span class="unit">%</span></div>
      <div class="bar"><i id="kvbar" style="width:0%"></i></div>
    </div>
    <div class="card">
      <h3>Preemptions</h3>
      <div><span class="val" id="preempt">0</span><span class="unit">/s</span></div>
      <div class="sub">KV pressure</div>
    </div>

    <div class="card span2">
      <h3>Latency · rolling window</h3>
      <div class="row"><span class="k">TTFT&nbsp; p50 / p99</span><span class="v"><span id="ttft50">0</span> / <span id="ttft99">0</span> ms</span></div>
      <div class="row"><span class="k">TPOT&nbsp; p50 / p99</span><span class="v"><span id="tpot50">0</span> / <span id="tpot99">0</span> ms</span></div>
      <div class="row"><span class="k">End-to-end&nbsp; p50 / p99</span><span class="v"><span id="e2e50">0</span> / <span id="e2e99">0</span> ms</span></div>
    </div>
    <div class="card span2">
      <h3>Totals since start</h3>
      <div class="row"><span class="k">Requests finished</span><span class="v" id="t_fin">0</span></div>
      <div class="row"><span class="k">Tokens generated</span><span class="v" id="t_gen">0</span></div>
      <div class="row"><span class="k">Prompt tokens processed</span><span class="v" id="t_prm">0</span></div>
      <div class="row"><span class="k">Preemptions</span><span class="v" id="t_pre">0</span></div>
    </div>
  </div>
  <div class="gpuline" id="gpuline"></div>
</main>
<script>
const $ = id => document.getElementById(id);
const HIST = 90, tps_hist = [], mem_hist = [];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const fmtNum = n => n>=1e6 ? (n/1e6).toFixed(1)+'M' : n>=1e3 ? (n/1e3).toFixed(1)+'k' : Math.round(n).toString();
const fmtGB = b => (b/1073741824).toFixed(1);
const fmtDur = s => { s=Math.floor(s); const h=(s/3600)|0,m=((s%3600)/60)|0,ss=s%60;
  return (h?h+'h ':'')+(m||h?m+'m ':'')+ss+'s'; };
const utilColor = f => f<0.6 ? css('--good') : f<0.85 ? css('--warn') : css('--hot');

function spark(cv, data, color, max){
  const dpr=window.devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr; const c=cv.getContext('2d'); c.scale(dpr,dpr);
  c.clearRect(0,0,w,h);
  c.strokeStyle='rgba(255,255,255,.05)'; c.lineWidth=1;
  c.beginPath(); c.moveTo(0,h-3.5); c.lineTo(w,h-3.5); c.stroke();
  if(data.length<2) return;
  const mx=max||Math.max(...data,1e-6), step=w/(HIST-1);
  const xy=i=>[i*step, h-3.5-(data[i]/mx)*(h-8)];
  c.beginPath(); data.forEach((v,i)=>{const[x,y]=xy(i); i?c.lineTo(x,y):c.moveTo(x,y);});
  const[lx,ly]=xy(data.length-1);
  c.lineTo(lx,h); c.lineTo(0,h); c.closePath(); c.fillStyle=color+'22'; c.fill();
  c.beginPath(); data.forEach((v,i)=>{const[x,y]=xy(i); i?c.lineTo(x,y):c.moveTo(x,y);});
  c.strokeStyle=color; c.lineWidth=2; c.lineJoin='round'; c.stroke();
  c.beginPath(); c.arc(lx,ly,2.6,0,6.2832); c.fillStyle=color; c.fill();
}

function render(s){
  $('uptime').textContent = 'up '+fmtDur(s.uptime_s);
  const st = s.running>0 ? (s.waiting>0 ? ['SATURATED','sat'] : ['LIVE','live']) : ['IDLE','idle'];
  $('status_t').textContent = st[0]; $('status').className = 'pill '+st[1];

  $('gtps').textContent = s.generation_tps.toFixed(0);
  $('running').textContent = s.running;
  $('waiting').textContent = s.waiting;
  const kv = s.kv_cache_usage*100;
  $('kv').textContent = kv.toFixed(kv<10?1:0); $('kvbar').style.width=kv+'%';
  $('kvbar').style.background = utilColor(s.kv_cache_usage);

  $('ebatch').textContent = s.effective_batch.toFixed(0);
  $('prefill').textContent = 'prefill '+(s.prefill_fraction*100).toFixed(0)+'% of tokens';
  $('preempt').textContent = s.preemptions_per_s.toFixed(2);
  if(s.mfu != null){
    $('mfu').textContent = (s.mfu*100).toFixed(1); $('mfu_u').textContent = '%';
    $('mfu_sub').textContent = s.achieved_tflops.toFixed(0)+' TFLOP/s · memory-bound in decode';
  } else if(s.achieved_tflops != null){
    $('mfu').textContent = s.achieved_tflops.toFixed(0); $('mfu_u').textContent = 'TFLOP/s';
    $('mfu_sub').textContent = 'peak unknown for this GPU';
  } else { $('mfu').textContent='—'; $('mfu_sub').textContent='needs a CUDA fp16/bf16 run'; }

  const inf = s.info || {};
  if(inf.gpu_name){
    $('devbar').style.display='flex';
    $('d_gpu').textContent = inf.gpu_name;
    $('d_vram').textContent = fmtGB(inf.vram_bytes)+' GB';
    $('d_cuda').textContent = 'CUDA '+inf.cuda_version;
    $('d_cc').textContent = 'sm_'+String(inf.compute_capability||'').replace('.','');
    $('d_drv').textContent = inf.driver_version ? 'driver '+inf.driver_version : '';
    const n = inf.gpu_count||1;
    $('d_cnt').textContent = n+' GPU'+(n>1?'s':'')+' present · using 1';
  }

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
    if('mem_bw_util' in g){ const b=g.mem_bw_util*100; $('hbm').textContent=b.toFixed(0);
      $('hbmbar').style.width=b+'%'; $('hbmbar').style.background=utilColor(g.mem_bw_util); }
    const bits=[];
    if('sm_util' in g) bits.push('SM '+(g.sm_util*100).toFixed(0)+'%');
    if('power_w' in g) bits.push(g.power_w.toFixed(0)+' W');
    if('temp_c' in g) bits.push(g.temp_c+' °C');
    if('torch_reserved_bytes' in g) bits.push('torch reserved '+fmtGB(g.torch_reserved_bytes)+' GB');
    $('gpuline').textContent = bits.join('   ·   ');
  } else {
    $('gmem').textContent='—'; $('gmemtot').textContent=''; $('smutil').textContent='n/a';
    $('gpuline').textContent='GPU metrics unavailable — running on CPU/MPS, or pynvml not installed.';
  }
  tps_hist.push(s.generation_tps); if(tps_hist.length>HIST) tps_hist.shift();
  spark($('spark_tps'), tps_hist, css('--accent'));
  spark($('spark_mem'), mem_hist, css('--good'), 1.0);
}

// --- DATA SOURCE (a static preview replaces everything below) ---
async function tick(){ let s; try{ s=await (await fetch('/stats')).json(); }catch(e){ return; } render(s); }
fetch('/v1/models').then(r=>r.json()).then(d=>{ if(d.data&&d.data[0]) $('model').textContent=d.data[0].id; }).catch(()=>{});
tick(); setInterval(tick, 1000);
</script>
</body>
</html>
"""
