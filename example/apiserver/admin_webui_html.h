// 管理页 HTML(单文件, 无外部依赖)。由 apiserver 在 GET /admin 时直接吐出。
// JS 侧: token 存 localStorage, 每次 API 调用带 Authorization: Bearer。
// 3s 轮询 /admin/api/state; 日志 5s 轮询(可暂停); 切换后连接会断,
// 页面持续重试直到新实例起来。
#pragma once

namespace fastllm {
    namespace apiserver {

        static const char *kAdminPageHtml = R"HTML(<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>FastLLM Admin</title>
<style>
:root { --bg:#111418; --panel:#1a1f26; --line:#2a323d; --fg:#d6dde6;
        --dim:#7d8a99; --ok:#5fb878; --warn:#e0a94f; --bad:#e06c60; --acc:#61afef; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--fg);
       font:14px/1.45 ui-monospace,Menlo,Consolas,monospace; padding:16px; }
h1 { font-size:17px; margin-bottom:12px; color:var(--acc); }
h2 { font-size:13px; color:var(--dim); margin:0 0 8px; letter-spacing:.08em; }
.panel { background:var(--panel); border:1px solid var(--line);
         border-radius:8px; padding:14px; margin-bottom:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
        gap:10px; }
.kv { font-size:13px; } .kv b { color:var(--dim); font-weight:normal; display:block; font-size:11px; }
.bar { height:6px; background:var(--line); border-radius:3px; margin-top:5px; overflow:hidden; }
.bar i { display:block; height:100%; background:var(--acc); border-radius:3px; }
button { background:#232a33; color:var(--fg); border:1px solid var(--line);
         border-radius:6px; padding:6px 12px; cursor:pointer; font:inherit; }
button:hover { border-color:var(--acc); }
button.danger:hover { border-color:var(--bad); color:var(--bad); }
button.primary { background:#1d3a55; border-color:#2b567e; }
input { background:#12161b; border:1px solid var(--line); color:var(--fg);
        border-radius:6px; padding:6px 10px; font:inherit; width:100%; }
table { width:100%; border-collapse:collapse; font-size:13px; }
td, th { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); }
th { color:var(--dim); font-weight:normal; font-size:11px; }
.active { color:var(--ok); }
.logbox { background:#0d1013; border:1px solid var(--line); border-radius:6px;
          height:260px; overflow:auto; white-space:pre-wrap; word-break:break-all;
          font-size:11.5px; line-height:1.35; padding:8px; color:#9aa7b4; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
.spacer { flex:1; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px;
         border:1px solid var(--line); color:var(--dim); }
.badge.ok { color:var(--ok); border-color:var(--ok); }
#loginOverlay { position:fixed; inset:0; background:rgba(0,0,0,.75);
  display:flex; align-items:center; justify-content:center; z-index:9; }
#loginBox { background:var(--panel); border:1px solid var(--line);
  border-radius:10px; padding:24px; width:340px; }
.muted { color:var(--dim); font-size:11px; }
.err { color:var(--bad); }
.flag { display:flex; flex-direction:column; gap:4px; padding:8px;
        border:1px solid var(--line); border-radius:6px; background:#151a21; }
.flag .key { font-size:12px; color:var(--acc); word-break:break-all; }
.flag .cmt { font-size:10.5px; color:var(--dim); }
.flag.row2 { flex-direction:row; align-items:center; justify-content:space-between; }
.switch { position:relative; width:38px; height:20px; flex:none; cursor:pointer; }
.switch input { opacity:0; width:0; height:0; position:absolute; }
.switch i { position:absolute; inset:0; background:var(--line); border-radius:10px;
            transition:.15s; }
.switch i:before { content:''; position:absolute; width:14px; height:14px;
  border-radius:50%; background:#8895a3; top:3px; left:3px; transition:.15s; }
.switch input:checked + i { background:#1d4a32; }
.switch input:checked + i:before { transform:translateX(18px); background:var(--ok); }
</style>
</head>
<body>
<div id="loginOverlay" style="display:none">
  <div id="loginBox">
    <h2>AUTH TOKEN</h2>
    <p class="muted" style="margin-bottom:10px">输入 .env 里的 AUTH_TOKEN 以访问管理面。</p>
    <div class="row"><input id="tokenInput" type="password" style="flex:1">
      <button class="primary" onclick="saveToken()">连接</button></div>
    <div id="loginErr" class="err"></div>
  </div>
</div>

<h1>FastLLM <span class="muted">admin @ :8002</span>
    <span id="stateBadge" class="badge">connecting…</span></h1>

<div class="panel">
  <h2>引擎状态</h2>
  <div class="grid" id="statusGrid">…</div>
</div>

<div class="panel">
  <h2>缓存层级</h2>
  <div class="grid" id="cacheGrid">…</div>
</div>

<div class="panel" id="editPanel" style="display:none">
  <div class="row"><h2 style="margin:0">编辑 profile:
    <span id="editName" style="color:var(--acc)"></span></h2>
    <span class="spacer"></span>
    <span class="muted">布尔值用开关; 其余为文本输入。保存后需重启生效。</span>
    <button class="primary" onclick="saveProfile()">保存</button>
    <button onclick="closeEdit()">取消</button></div>
  <div class="grid" id="editGrid"></div>
</div>

<div class="panel">
  <h2>模型切换 (profile)</h2>
  <div class="row">
    <span class="muted">目录:</span><span id="profilesDir" class="muted"></span>
    <span class="spacer"></span>
    <span class="muted">切换 = 停当前全套 + 按新 profile 重启(约 1~5 分钟)</span>
  </div>
  <table id="profileTable"><thead><tr>
    <th>profile</th><th>mtime</th><th>状态</th><th></th></tr></thead>
    <tbody></tbody></table>
  <div class="row" style="margin-top:10px">
    <button onclick="doSuspend('memory')">暂停模型到 RAM</button>
    <button onclick="doSuspend('disk')">暂停模型到磁盘</button>
    <button class="primary" onclick="doResume()">恢复模型</button>
    <button class="danger" onclick="doStop()">■ 彻底停止 proxy+backend</button>
    <span class="muted">暂停态管理页保持在线；彻底停止后需在服务器运行 start_prod.sh。</span>
  </div>
</div>

<div class="panel">
  <div class="row"><h2 style="margin:0">backend 日志</h2>
    <span class="spacer"></span>
    <label class="muted"><input type="checkbox" id="autoLog" checked> 自动刷新</label>
    <button onclick="refreshLogs()">手动刷新</button></div>
  <div class="logbox" id="logBackend">…</div>
</div>
<div class="panel">
  <div class="row"><h2 style="margin:0">thinking_proxy 日志</h2>
    <span class="spacer"></span><button onclick="refreshLogs()">刷新</button></div>
  <div class="logbox" id="logProxy">…</div>
</div>

<script>
const $ = s => document.querySelector(s);
const token = () => localStorage.getItem('fastllm_admin_token') || '';
function authHeaders() { return {'Authorization': 'Bearer ' + token()}; }

function showLogin(show) {
  $('#loginOverlay').style.display = show ? 'flex' : 'none';
  if (show) $('#tokenInput').focus();
}
async function saveToken() {
  localStorage.setItem('fastllm_admin_token', $('#tokenInput').value.trim());
  $('#loginErr').textContent = '';
  const ok = await pollState(true);
  if (ok) showLogin(false); else $('#loginErr').textContent = 'token 无效或服务未就绪';
}
function fmtBytes(b) {
  if (!b && b !== 0) return '?';
  const u = ['B','KB','MB','GB']; let i = 0;
  while (b >= 1024 && i < 3) { b /= 1024; i++; }
  return b.toFixed(i ? 1 : 0) + u[i];
}
function bar(pct) {
  return `<div class="bar"><i style="width:${Math.max(0,Math.min(100,pct)).toFixed(1)}%"></i></div>`;
}

let lastStateOk = false;
async function pollState(isLogin=false) {
  try {
    const r = await fetch('/admin/api/state', {headers: authHeaders()});
    if (r.status === 401) { showLogin(true); return false; }
    const j = await r.json();
    renderState(j); lastStateOk = true;
    return true;
  } catch(e) {
    $('#stateBadge').textContent = 'backend offline — 若刚切换请等待重启完成';
    $('#stateBadge').className = 'badge';
    return isLogin ? false : lastStateOk;   // 断连时保留旧画面
  }
}

function renderState(s) {
  const badge = $('#stateBadge');
  badge.textContent = `${s.model} · ${s.ready ? 'READY' : s.state}`;
  badge.className = 'badge' + (s.ready ? ' ok' : '');
  const g = (k,v,pct) => `<div class="kv"><b>${k}</b>${v}${pct!==undefined?bar(pct):''}</div>`;
  let h = '';
  h += g('活跃 / 排队', `${s.active_requests} / ${s.queued_requests}`);
  h += g('上下文容量', `${s.token_pool.toLocaleString()} tok · batch ${s.max_batch} · kv=${s.kv_dtype} · act=${s.atype}`);
  h += g('页池 (物理页)', `${s.pool.used} / ${s.pool.total} pg (${s.pool.pct}%) · budget ${s.pool.budget}`, s.pool.pct);
  h += g('显存对账', `${s.vram.used_gb.toFixed(1)} / ${s.vram.total_gb.toFixed(1)} GB` +
        ` <span class="muted">(pool ${s.vram.pool_mb.toFixed(0)} · busy ${s.vram.busy_mb.toFixed(0)} · free ${s.vram.free_mb.toFixed(0)} · pin ${s.vram.pin_mb.toFixed(0)} · other ${s.vram.other_mb.toFixed(0)} MB)</span>`,
        s.vram.pct);
  $('#statusGrid').innerHTML = h;

  const p = s.prefix_cache;
  let c = '';
  c += g('L1 GPU trie 页', `${s.pool_trie.pages} pg (~${s.pool_trie.tokens.toLocaleString()} tok)` );
  c += g('L1 命中', `${p.hit_tokens_gpu.toLocaleString()} tok · ${p.requests} 请求`);
  c += g('L2 RAM 层', `${fmtBytes(p.cpu_tier_bytes)} resident · 命中 ${p.hit_tokens_cpu.toLocaleString()} tok`);
  c += g('L3 磁盘层', `${fmtBytes(p.disk_bytes)} resident · gen ${s.disk_persist.generation} · ckpt ${s.disk_persist.checkpoints} · 命中 ${p.hit_tokens_disk.toLocaleString()} tok`);
  c += g('命中率', `${p.hit_ratio.toFixed(1)}% (mem ${fmtBytes(p.mem_resident)} · break: noChild ${p.q_no_child} mat. ${p.q_materialize} gen. ${p.q_gen})`);
  c += g('工具调用约束', `steps ${p_tc(s)} · forced ${p_tc(s,1)} · override ${p_tc(s,2)} · loopBreak ${p_tc(s,3)} · malformed ${p_tc(s,4)}`);
  $('#cacheGrid').innerHTML = c;

  $('#profilesDir').textContent = s.profiles_dir || '';
  const tb = document.querySelector('#profileTable tbody');
  tb.innerHTML = (s.profiles || []).map(pf =>
    `<tr><td>${pf.name}</td><td class="muted">${new Date(pf.mtime*1000).toLocaleString()}</td>` +
    `<td>${pf.active ? '<span class="active">● 运行中</span>' : ''}</td>` +
    `<td><button onclick="openEdit('${pf.name}')">编辑</button> ` +
       (pf.active ? '' :
       `<button class="primary" onclick="doSwitch('${pf.name}')">切换到此</button>`) +
    `</td></tr>`
  ).join('');
}
const p_tc = (s, i) => {
  const t = s.toolcall;
  return [t.steps, t.forced, t.override, t.loopbreak, t.malformed][i||0];
};

// ---- profile 编辑 ----
let editingName = '';
let editingKeys = [];
async function openEdit(name) {
  const r = await fetch(`/admin/api/profile?name=${encodeURIComponent(name)}`,
                        {headers: authHeaders()});
  if (r.status === 401) { showLogin(true); return; }
  const j = await r.json();
  editingName = name; editingKeys = j.keys || [];
  $('#editName').textContent = name;
  $('#editGrid').innerHTML = editingKeys.map(k => {
    if (k.boolean) {
      const on = ['1','true','yes','on'].includes(k.value.toLowerCase());
      return `<div class="flag row2"><div><div class="key">${k.key}</div>` +
             `<div class="cmt">${k.comment||''}</div></div>` +
             `<label class="switch"><input type="checkbox" data-key="${k.key}" ${on?'checked':''}><i></i></label></div>`;
    }
    return `<div class="flag"><div class="key">${k.key}</div>` +
           `<input data-key="${k.key}" value="${k.value.replace(/"/g,'&quot;')}">` +
           `<div class="cmt">${k.comment||''}</div></div>`;
  }).join('');
  $('#editPanel').style.display = 'block';
  $('#editPanel').scrollIntoView({behavior:'smooth'});
}
function closeEdit() { $('#editPanel').style.display = 'none'; }
async function saveProfile() {
  const updates = [];
  $('#editGrid').querySelectorAll('[data-key]').forEach(el => {
    const key = el.dataset.key;
    if (el.type === 'checkbox') updates.push({key, value: el.checked ? '1' : '0'});
    else updates.push({key, value: el.value});
  });
  const r = await fetch('/admin/api/profile', {
    method: 'POST', headers: {...authHeaders(), 'Content-Type': 'application/json'},
    body: JSON.stringify({name: editingName, updates})});
  if (r.status === 401) { showLogin(true); return; }
  const j = await r.json();
  if (!r.ok) { alert('保存失败: ' + JSON.stringify(j)); return; }
  closeEdit();
  alert('已保存 ' + editingName + '。重启后生效(可在切换页面选择该 profile)。');
  pollState();
}

async function doSwitch(name) {
  if (!confirm(`确认切换到 ${name}? 当前服务会被停掉并用新 profile 重启。`)) return;
  await fetch('/admin/api/switch', {method:'POST', headers:{...authHeaders(), 'Content-Type':'application/json'},
                                    body: JSON.stringify({profile: name})});
  $('#stateBadge').textContent = 'switching… 服务重启中, 页面将自动重连';
}
async function postControl(path, body={}) {
  const r = await fetch(path, {
    method:'POST', headers:{...authHeaders(),'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) throw new Error(j?.error?.message || JSON.stringify(j));
  return j;
}
async function doSuspend(tier) {
  if (!confirm(`确认暂停模型到 ${tier}? 管理页会保持在线。`)) return;
  try { await postControl('/admin/suspend',{tier}); pollState(); }
  catch(e) { alert('暂停失败: '+e.message); }
}
async function doResume() {
  try { await postControl('/admin/resume'); pollState(); }
  catch(e) { alert('恢复失败: '+e.message); }
}
async function doStop() {
  if (!confirm('确认彻底停止 proxy + backend? 停止后 WebUI 也会离线。')) return;
  await fetch('/admin/api/stop', {method:'POST', headers:authHeaders()});
}
async function refreshLogs() {
  for (const [which, sel] of [['backend','#logBackend'], ['proxy','#logProxy']]) {
    try {
      const r = await fetch(`/admin/api/logs?which=${which}&lines=120`, {headers: authHeaders()});
      if (r.status === 401) { showLogin(true); return; }
      const j = await r.json();
      const el = $(sel);
      const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      el.textContent = j.text || '(empty)';
      if (stick) el.scrollTop = el.scrollHeight;
    } catch(e) {}
  }
}

(async function init() {
  if (!token()) showLogin(true);
  else { const ok = await pollState(true); if (!ok) showLogin(true); }
  setInterval(() => { if (token()) pollState(); }, 3000);
  setInterval(() => {
    if (token() && $('#autoLog').checked &&
        !$('#loginOverlay').style.display.includes('flex')) refreshLogs();
  }, 5000);
})();
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.activeElement === $('#tokenInput')) saveToken();
});
</script>
</body>
</html>)HTML";

    }
}
