const $ = (s, r) => (r || document).querySelector(s);
const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content; };
function esc(v){ return String(v == null ? "" : v).replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

const NA = '<span class="dim">&mdash;</span>';
function has(v){ return v !== null && v !== undefined && v !== ""; }
function num(v, d){ if(!has(v) || typeof v !== "number" || !isFinite(v)) return NA;
  return `<span class="mono">${v.toFixed(d === undefined ? 4 : d)}</span>`; }
function int(v){ if(!has(v) || typeof v !== "number") return NA;
  return `<span class="mono">${v.toLocaleString()}</span>`; }
function pct(v, d){ if(!has(v) || typeof v !== "number") return NA;
  return `<span class="mono">${(v * 100).toFixed(d === undefined ? 1 : d)}%</span>`; }
function ms(v){ if(!has(v) || typeof v !== "number") return NA;
  return `<span class="mono">${v.toFixed(1)} ms</span>`; }
function usd(v){ if(!has(v) || typeof v !== "number") return NA;
  return `<span class="mono">$${v.toFixed(v < 1 ? 6 : 2)}</span>`; }
function when(v){ if(!has(v)) return NA;
  const d = new Date(v); if(isNaN(d)) return `<span class="mono">${esc(v)}</span>`;
  return `<span class="mono" title="${esc(v)}">${d.toISOString().slice(0,19).replace("T"," ")}</span>`; }
function ago(v){ if(!has(v)) return ""; const d = new Date(v); if(isNaN(d)) return "";
  const s = (Date.now() - d.getTime())/1000;
  if(s < 60) return "just now"; if(s < 3600) return Math.floor(s/60)+"m ago";
  if(s < 86400) return Math.floor(s/3600)+"h ago"; return Math.floor(s/86400)+"d ago"; }

function badge(text, kind, dot){
  return `<span class="badge ${kind||"mute"}">${dot?'<i class="dot"></i>':""}${esc(text)}</span>`; }
function boolBadge(v, tTrue, tFalse, invert){
  if(!has(v)) return badge("unknown","mute");
  const good = invert ? !v : !!v;
  return badge(v ? (tTrue||"yes") : (tFalse||"no"), good ? "ok" : "bad", true); }

function card(title, bodyHtml, opts){
  const o = opts || {};
  return `<section class="card">
    <header><h3>${esc(title)}</h3>${o.sub?`<span class="sub">${esc(o.sub)}</span>`:""}
      <span class="spacer"></span>${o.right||""}</header>
    <div class="body ${o.flush?"flush":""}">${bodyHtml}</div></section>`; }

function kpi(label, value, meta){
  return `<div class="kpi"><div class="k">${esc(label)}</div>
    <div class="v">${value}</div><div class="m">${meta||"&nbsp;"}</div></div>`; }

/* Sorting and filtering happen here rather than in each page, so every table
   in the console behaves the same way. A column opts in with `sort` (a value
   accessor); `opts.filter` adds a text box that matches across the accessors.
   Both are client-side over rows already fetched -- no endpoint gains a query
   parameter it does not have. */
let TABLE_SEQ = 0;
const TABLE_STATE = {};

function table(cols, rows, opts){
  const o = opts || {};
  const all = rows || [];
  const sortable = cols.some(c => c.sort);
  const id = (o.id || `t${++TABLE_SEQ}`);
  const st = TABLE_STATE[id] || (TABLE_STATE[id] = { key:o.sortKey || null,
                                                     dir:o.sortDir || "desc", q:"" });
  TABLE_STATE[id].cols = cols;
  TABLE_STATE[id].rows = all;
  TABLE_STATE[id].opts = o;

  let view = all;
  if(o.filter && st.q){
    const q = st.q.toLowerCase();
    view = view.filter(r => cols.some(c => {
      const v = c.sort ? c.sort(r) : (c.text ? c.text(r) : null);
      return v != null && String(v).toLowerCase().includes(q);
    }));
  }
  if(st.key){
    const col = cols.find(c => c.label === st.key);
    if(col && col.sort){
      const mul = st.dir === "asc" ? 1 : -1;
      view = view.slice().sort((a, b) => {
        const x = col.sort(a), y = col.sort(b);
        if(x == null && y == null) return 0;
        if(x == null) return 1;          // absent values sort last either way
        if(y == null) return -1;
        if(typeof x === "number" && typeof y === "number") return (x - y) * mul;
        return String(x).localeCompare(String(y), undefined, { numeric:true }) * mul;
      });
    }
  }

  const controls = o.filter ? `<div class="tctl">
      <label class="tsearch">${icon("search", 14)}
        <input type="search" data-tfilter="${esc(id)}" value="${esc(st.q)}"
          placeholder="${esc(o.filter === true ? "Filter" : o.filter)}"
          aria-label="${esc(o.filter === true ? "Filter rows" : o.filter)}"></label>
      <span class="tcount" role="status">${view.length} of ${all.length}</span>
    </div>` : "";

  if(!all.length) return controls + emptyState(o.empty || "No records.");
  if(!view.length) return controls + emptyState(
    `Nothing matches "${st.q}". Clear the filter to see all ${all.length} rows.`);

  const head = cols.map(c => {
    const on = st.key === c.label;
    const cls = [c.num ? "num" : "", c.sort ? "sortable" : ""].filter(Boolean).join(" ");
    const aria = c.sort ? ` aria-sort="${on ? (st.dir === "asc" ? "ascending" : "descending") : "none"}"` : "";
    const inner = c.sort
      ? `<button class="thsort" data-tsort="${esc(id)}" data-col="${esc(c.label)}">
           ${esc(c.label)}${icon(on ? (st.dir === "asc" ? "sortAsc" : "sortDesc") : "sortNone", 13)}</button>`
      : esc(c.label);
    return `<th scope="col"${cls ? ` class="${cls}"` : ""}${aria}>${inner}</th>`;
  }).join("");

  const body = view.map(r => `<tr class="${o.rowClass ? esc(o.rowClass(r)) : ""}">` + cols.map(c =>
    `<td${c.num?' class="num"':""}>${c.render(r)}</td>`).join("") + "</tr>").join("");
  return controls + `<div class="scroll"><table><thead><tr>${head}</tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

/* Re-render just the table that changed, in place. Delegated once so tables
   rendered after this point still respond. */
function wireTables(){
  if(window.__tablesWired) return;
  window.__tablesWired = true;
  document.addEventListener("click", e => {
    const b = e.target.closest("[data-tsort]");
    if(!b) return;
    const id = b.getAttribute("data-tsort"), col = b.getAttribute("data-col");
    const st = TABLE_STATE[id];
    if(!st) return;
    if(st.key === col) st.dir = st.dir === "asc" ? "desc" : "asc";
    else { st.key = col; st.dir = "desc"; }
    redrawTable(id, b);
    announce(`Sorted by ${col}, ${st.dir === "asc" ? "ascending" : "descending"}`);
  });
  document.addEventListener("input", e => {
    const f = e.target.closest("[data-tfilter]");
    if(!f) return;
    const id = f.getAttribute("data-tfilter");
    const st = TABLE_STATE[id];
    if(!st) return;
    st.q = f.value;
    const host = redrawTable(id, f);
    if(host){
      const again = host.querySelector(`[data-tfilter="${id}"]`);
      if(again){ again.focus(); again.setSelectionRange(st.q.length, st.q.length); }
    }
  });
}

function redrawTable(id, fromEl){
  const st = TABLE_STATE[id];
  if(!st) return null;
  /* The rendered table and its controls share a parent; replacing that
     parent's contents keeps the surrounding card untouched. */
  const host = fromEl.closest(".body") || fromEl.parentElement.parentElement;
  if(!host) return null;
  const opts = Object.assign({}, st.opts, { id });
  host.innerHTML = table(st.cols, st.rows, opts);
  return host;
}

function emptyState(msg){ return `<div class="state"><div class="big">Nothing to show</div>${esc(msg)}</div>`; }
function unavailable(reason){
  return `<div class="state"><div class="big">Data unavailable</div>${esc(reason || "The backend did not return this section.")}</div>`; }
function errorState(msg, retryId){
  return `<div class="state"><div class="big">Unable to load data</div>
    <div style="margin-bottom:10px">${esc(msg)}</div>
    <button class="btn" data-retry="${esc(retryId||"")}">Retry</button></div>`; }
function skeleton(n){ let s = ""; for(let i=0;i<(n||5);i++)
  s += `<div class="sk" style="margin:9px 14px;width:${55+((i*17)%40)}%"></div>`; return `<div style="padding:8px 0">${s}</div>`; }

/* horizontal bar list -- used for feature drift and token/model splits */
function barList(items, opts){
  const o = opts || {};
  if(!items.length) return emptyState(o.empty || "No values.");
  const max = o.max || Math.max(...items.map(i => Math.abs(i.value) || 0), o.threshold || 0) || 1;
  return `<div style="display:grid;gap:8px">` + items.map(i => {
    const w = Math.max(1, Math.min(100, (Math.abs(i.value)/max)*100));
    const cls = i.bad ? "bad" : (i.good ? "ok" : "");
    /* Feature names are long -- `num_late_payments_12m` needs 145px and was
       being cut off in a 120px column. The label track now sizes to content up
       to a cap, and anything past that ellipsises with the full name on hover
       rather than silently losing characters. */
    return `<div style="display:grid;grid-template-columns:minmax(120px,max-content) 3fr minmax(64px,auto) auto;
      gap:10px;align-items:center">
      <span class="mono blabel" style="font-size:11.5px" title="${esc(i.label)}">${esc(i.label)}</span>
      <div class="bar"><i class="${cls}" style="width:${w}%"></i></div>
      <span class="num">${typeof i.value === "number" ? i.value.toFixed(4) : esc(i.value)}</span>
      <span>${i.tag||""}</span></div>`; }).join("") + `</div>`; }

/* minimal SVG line chart -- points only, no library */
function lineChart(series, opts){
  const o = opts || {}, W = o.width || 640, H = o.height || 130, P = 26;
  const pts = series.filter(p => typeof p.y === "number" && isFinite(p.y));
  if(pts.length < 2) return `<div class="state">${esc(o.empty || "Not enough data points to plot.")}</div>`;
  const ys = pts.map(p => p.y), min = Math.min(...ys, 0), max = Math.max(...ys);
  const span = (max - min) || 1;
  const x = i => P + (i/(pts.length-1)) * (W - P - 8);
  const y = v => H - P - ((v - min)/span) * (H - P - 12);
  const d = pts.map((p,i) => `${i?"L":"M"}${x(i).toFixed(1)},${y(p.y).toFixed(1)}`).join(" ");
  const area = d + ` L${x(pts.length-1).toFixed(1)},${H-P} L${x(0).toFixed(1)},${H-P} Z`;
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" role="img">
    <line x1="${P}" y1="${H-P}" x2="${W-8}" y2="${H-P}" stroke="var(--line)"/>
    <text x="2" y="14" font-size="9" fill="var(--ink-3)">${max.toFixed(o.dp===undefined?2:o.dp)}</text>
    <text x="2" y="${H-P}" font-size="9" fill="var(--ink-3)">${min.toFixed(o.dp===undefined?2:o.dp)}</text>
    <path d="${area}" fill="var(--accent)" opacity=".10"/>
    <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>
    ${pts.map((p,i)=>`<circle cx="${x(i).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="2.2"
       fill="var(--accent)"><title>${esc(p.x)}: ${p.y}</title></circle>`).join("")}
  </svg>`; }

/* -------------------------------------------------------------- api ----- */
const api = (() => {
  const cache = new Map();
  async function raw(path, init){
    const res = await fetch(path, Object.assign({ headers: { "Accept": "application/json" } }, init||{}));
    const text = await res.text();
    let body = null; try { body = text ? JSON.parse(text) : null; } catch(e){ body = text; }
    if(!res.ok){
      const msg = (body && body.error && body.error.message) || (typeof body === "string" ? body : "") ||
        `HTTP ${res.status}`;
      const err = new Error(msg); err.status = res.status; err.body = body; throw err;
    }
    return body;
  }
  return {
    /* GET with a short TTL so switching pages does not re-hit the same
       endpoint repeatedly. Stable data gets a longer TTL than live data. */
    async get(path, ttlMs){
      const ttl = ttlMs === undefined ? 8000 : ttlMs;
      const hit = cache.get(path);
      if(hit && ttl > 0 && (Date.now() - hit.t) < ttl) return hit.v;
      const v = await raw(path);
      cache.set(path, { t: Date.now(), v });
      return v;
    },
    post(path, body, apiKey){
      const headers = { "Content-Type": "application/json", "Accept": "application/json" };
      if(apiKey) headers["X-API-Key"] = apiKey;
      return raw(path, { method: "POST", headers, body: body ? JSON.stringify(body) : "{}" });
    },
    text(path){ return fetch(path).then(r => r.text()); },
    bust(){ cache.clear(); }
  };
})();

/* Load several endpoints at once; a failure on one does not sink the page. */
async function loadAll(spec, ttl){
  const keys = Object.keys(spec);
  const out = {};
  await Promise.all(keys.map(async k => {
    try { out[k] = { ok: true, data: await api.get(spec[k], ttl) }; }
    catch(e){ out[k] = { ok: false, error: e.message || String(e) }; }
  }));
  return out;
}
function sect(res, render, label){
  if(!res) return unavailable();
  if(!res.ok) return errorState(`${label||"Endpoint"}: ${res.error}`, location.hash);
  return render(res.data);
}

/* ------------------------------------------------------------ toasts ---- */
function toast(msg, kind){
  const el = document.createElement("div");
  el.className = "toast " + (kind || "");
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .25s";
    setTimeout(() => el.remove(), 260); }, 4200);
}

/* --------------------------------------------------- guarded actions ---- */
/* Write endpoints require an API key. The key is asked for per action, held
   in a local variable for the duration of that one request, and never
   written to localStorage, sessionStorage, a cookie or the URL. */
let AUTH_REQUIRED = null;   // null = not yet known
async function authRequired(){
  if(AUTH_REQUIRED !== null) return AUTH_REQUIRED;
  try {
    const cfg = await api.get("/api/v1/config", 60000);
    AUTH_REQUIRED = ((cfg.security || {}).auth_backend || "none") !== "none";
  } catch(e){
    // If the backend cannot be asked, ask for a key rather than send a write
    // that is about to be rejected.
    AUTH_REQUIRED = true;
  }
  return AUTH_REQUIRED;
}

/* Keyboard containment for anything that overlays the page.
 *
 * Returns a release function. Without this, Tab from an open dialog walks into
 * the page behind it -- the user is "inside" a modal that the keyboard has
 * already left, which is worse than having no dialog at all.
 */
function trapFocus(container, onEscape){
  const previous = document.activeElement;
  const sel = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),'
            + 'textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
  const focusables = () => [...container.querySelectorAll(sel)]
    .filter(el => el.offsetParent !== null || el === document.activeElement);

  function onKey(e){
    if(e.key === "Escape"){ e.preventDefault(); if(onEscape) onEscape(); return; }
    if(e.key !== "Tab") return;
    const list = focusables();
    if(!list.length) return;
    const first = list[0], last = list[list.length - 1];
    if(e.shiftKey && document.activeElement === first){ e.preventDefault(); last.focus(); }
    else if(!e.shiftKey && document.activeElement === last){ e.preventDefault(); first.focus(); }
  }
  container.addEventListener("keydown", onKey);
  const first = focusables()[0];
  if(first) first.focus();

  return function release(){
    container.removeEventListener("keydown", onKey);
    /* Return the caret to whatever opened the dialog, so keyboard position is
       not lost when it closes. */
    if(previous && typeof previous.focus === "function" && document.contains(previous)){
      previous.focus();
    }
  };
}

function confirmAction(opts){
  return new Promise(resolve => {
    const m = document.createElement("div");
    m.className = "modal";
    const titleId = `acT${++COPY_SEQ}`;
    m.innerHTML = `<div class="box" role="dialog" aria-modal="true" aria-labelledby="${titleId}">
      <div class="body" style="padding:16px">
        <h3 id="${titleId}">${esc(opts.title)}</h3>
        <p style="color:var(--ink-2);margin:8px 0 12px;font-size:12.5px">${esc(opts.body)}</p>
        ${opts.needsKey ? `<label style="font-size:11.5px;color:var(--ink-3)">API key (X-API-Key)</label>
          <input type="password" id="ackey" autocomplete="off" placeholder="required for write actions">
          <p style="font-size:11px;color:var(--ink-3);margin:6px 0 0">
            Used for this request only. Not stored anywhere in the browser.</p>` : ""}
        <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:16px">
          <button class="btn" id="acno">Cancel</button>
          <button class="btn ${opts.danger?"danger":"pri"}" id="acyes">${esc(opts.confirm||"Confirm")}</button>
        </div>
      </div></div>`;
    document.body.appendChild(m);
    const release = trapFocus(m, () => done(null));
    const done = v => { release(); m.remove(); resolve(v); };
    $("#acno", m).onclick = () => done(null);
    $("#acyes", m).onclick = () => done({ key: opts.needsKey ? ($("#ackey", m).value || "") : "" });
    m.onclick = e => { if(e.target === m) done(null); };
  });
}
async function runAction(opts){
  if(opts.needsKey) opts = { ...opts, needsKey: await authRequired() };
  const c = await confirmAction(opts);
  if(!c) return;
  if(opts.needsKey && !c.key){ toast("An API key is required for this action.", "bad"); return; }
  try {
    await api.post(opts.path, opts.payload, c.key);
    api.bust();
    toast(opts.success || "Action completed.", "ok");
    render();
  } catch(e){
    toast((e.status === 401 || e.status === 403)
      ? "Rejected: the API key was missing or not accepted."
      : `Failed: ${e.message}`, "bad");
  }
}

/* ------------------------------------------------------------- pages ---- */
const PAGES = {};

/* ---- Overview ---------------------------------------------------------- */
/* ---- Datasets ---------------------------------------------------------- */
/* ---- AutoML ------------------------------------------------------------ */
/* ---- Wizard steps 2-4 -------------------------------------------------- */
/* ---- Run detail / leaderboard ------------------------------------------ */
/* ---- Training ---------------------------------------------------------- */
/* ---- Evaluation -------------------------------------------------------- */
/* ---- Models ------------------------------------------------------------ */
/* ---- Experiments ------------------------------------------------------- */
/* ---- Deployments ------------------------------------------------------- */
/* ---- Monitoring -------------------------------------------------------- */
/* ---- Drift ------------------------------------------------------------- */
/* ---- Retraining -------------------------------------------------------- */
/* ---- Champion / Challenger --------------------------------------------- */
/* ---- LLMOps ------------------------------------------------------------ */
/* ---- Audit ------------------------------------------------------------- */
/* ---- System ------------------------------------------------------------ */


/* A named loading state beats an anonymous shimmer: it tells the reader what
   is being fetched, which is the difference between "working" and "stuck". */
function loadingPanel(what){
  return `<div class="card"><div class="body">
    <div class="loadrow"><span class="spin" aria-hidden="true"></span>
      <span>Loading ${esc(what || "data")}…</span></div>
    ${skeleton(5)}</div></div>`;
}

/* Sidebar footer: API reachability and the environment the backend reports.
   Both are observed, never assumed -- if the API cannot be reached it says so
   rather than showing a reassuring green dot. */
async function renderSideFoot(){
  const el = $("#sidefoot");
  if(!el) return;
  let cfg = null;
  try { cfg = await api.get("/api/v1/config", 60000); } catch(e){ cfg = null; }
  const aws = (cfg && cfg.aws) || {};
  const env = cfg ? String(cfg.environment || "unknown") : null;
  el.innerHTML = `
    <div class="footrow"><span class="k">API</span>
      ${cfg ? badge("Operational","ok",true) : badge("Unreachable","bad",true)}</div>
    <div class="footrow"><span class="k">Environment</span>
      <span class="mono">${env ? esc(env) : "—"}</span></div>
    <div class="footrow"><span class="k">AWS services</span>
      ${cfg ? (aws.enabled ? badge("enabled","info") : badge("not used","mute")) : "—"}</div>
    <p class="foothint">Hosted on ECS/Fargate. AWS service integration is reported
      from the running configuration, not assumed from where it runs.</p>`;
}

/* ==========================================================================
 * Signature components
 *
 * Each takes an API response and renders it. None of them holds an idea of
 * progress the backend has not confirmed: where a source says nothing, the
 * component renders "unknown" rather than a hopeful default, and a dash is
 * never a zero.
 * ========================================================================== */

/* ---- Production Health Strip --------------------------------------------- *
 * Six units over one /api/v1/dashboard response. The distinction that matters
 * most is measurable vs not-measurable, so every unit that cannot be computed
 * carries the reason underneath it instead of a number.
 */
function healthStrip(d){
  const model = d.model || {}, sys = d.system || {}, dep = d.deployment || {},
        drift = d.drift || {}, lp = sys.live_performance || {};
  const m = model.metrics || {};
  const u = (k, v, meta, kind) =>
    `<div class="u"><div class="k">${esc(k)}</div>
      <div class="v ${kind || ""}">${v}</div>
      <div class="m">${meta || "&nbsp;"}</div></div>`;

  const serving = dep.available && dep.current_version != null
    ? `v${dep.current_version}` : (model.available && model.current_version != null
      ? `v${model.current_version}` : NA);
  const servingMeta = dep.available && dep.state
    ? esc(String(dep.state)) : esc(model.current_stage || "not deployed");

  const latOk = sys.latency_slo_met !== false;
  const errOk = sys.error_slo_met !== false;
  const driftKnown = drift.available && drift.drift_detected != null;

  return `<div class="hstrip">
    ${u("Serving", serving, servingMeta)}
    ${u("ROC-AUC", num(m.roc_auc, 4), "offline, at registration")}
    ${u("p95 latency", sys.latency_p95_ms != null ? ms(sys.latency_p95_ms) : NA,
        sys.slo_latency_ms != null ? `SLO ${sys.slo_latency_ms} ms` : "no SLO set",
        sys.latency_p95_ms == null ? "na" : (latOk ? "ok" : "bad"))}
    ${u("Error rate", sys.error_rate != null ? pct(sys.error_rate, 2) : NA,
        sys.slo_error_rate != null ? `SLO ${(sys.slo_error_rate*100).toFixed(2)}%` : "no SLO set",
        sys.error_rate == null ? "na" : (errOk ? "ok" : "bad"))}
    ${u("Drift", driftKnown ? (drift.drift_detected ? "DETECTED" : "STABLE") : NA,
        driftKnown ? `${(drift.drifted_features||[]).length} feature(s)` : "no scan yet",
        driftKnown ? (drift.drift_detected ? "warn" : "ok") : "na")}
    ${u("Live F1", lp.available ? num(lp.f1, 4) : NA,
        lp.available ? "from returned labels" : "needs labels",
        lp.available ? "" : "na")}
  </div>`;
}

/* ---- Lifecycle Rail ------------------------------------------------------ *
 * Ten stops, each derived from a named source. The caller builds `stops` from
 * real responses; this only renders. A stop whose source did not answer is
 * todo, never done.
 */
const LIFECYCLE_STOPS = ["Dataset","Train","Evaluate","Gate","Approve",
                         "Promote","Deploy","Monitor","Drift","Retrain"];

function lifecycleRail(stops){
  return `<div class="lcrail">` + LIFECYCLE_STOPS.map((label, i) => {
    const s = (stops && stops[i]) || {};
    const state = ["done","now","fail","block","todo"].includes(s.state) ? s.state : "todo";
    const glyph = state === "done" ? "✓" : state === "fail" ? "✕"
      : state === "block" ? "!" : state === "now" ? "●" : String(i+1);
    const joined = i < LIFECYCLE_STOPS.length - 1
      ? `<div class="join ${state === "done" ? "done" : ""}"></div>` : "";
    return `<div class="st ${state}" title="${esc(s.detail || label)}">
        <span class="pip">${glyph}</span>
        <span class="lbl2">${esc(label)}</span>
        <span class="val">${s.value != null ? esc(String(s.value)) : "&mdash;"}</span>
      </div>${joined}`;
  }).join("") + `</div>`;
}

/* ---- Promotion Rail ------------------------------------------------------ *
 * DEV to VAL to STAGING to PROD. Occupants come from the version list, and the
 * last transition into each stage from /models/{name}/history.
 */
const STAGES = ["Development","Validation","Staging","Production"];

function promotionRail(versions, history){
  const vs = versions || [], hist = history || [];
  return `<div class="prail">` + STAGES.map(stage => {
    const here = vs.filter(v => v.stage === stage);
    const last = hist.find(h => h.to_stage === stage);
    const cls = (stage === "Production" && here.length) ? "on" : "past";
    const occupants = here.length ? here.map(v => `v${v.version}`).join(" · ") : "&mdash;";
    let note;
    if(here.length && last) note = `${last.actor || "?"} · ${last.reason || ""}`;
    else if(here.length) note = `${here.length} version(s) held here`;
    else if(last) note = `last transit ${String(last.created_at || "").slice(0,19).replace("T"," ")}`;
    else note = "never occupied";
    return `<div class="seg ${cls}">
      <div class="s">${esc(stage)}</div>
      <div class="v">${occupants}</div>
      <div class="w" title="${esc(note)}">${esc(note)}</div></div>`;
  }).join("") + `</div>`;
}

/* ---- Model Version Rail -------------------------------------------------- *
 * Lineage as navigation. A rejected version stays visibly rejected: the point
 * of this strip is that the registry must not look artificially green.
 */
function versionRail(versions, current, metric){
  const vs = (versions || []).slice().sort((a,b) => b.version - a.version);
  if(!vs.length) return emptyState("No versions registered.");
  const key = metric || "roc_auc";
  return `<div class="vrail">` + vs.map(v => {
    const st = String(v.stage || "");
    const rejected = String(v.status || "").toLowerCase() === "rejected";
    const cls = rejected ? "rejected" : st === "Production" ? "prod"
      : (st === "Staging" || st === "Validation") ? "stage" : "";
    const on = String(v.version) === String(current) ? "on" : "";
    const score = (v.metrics || {})[key];
    return `<a class="${cls} ${on}" href="#/models/${encodeURIComponent(v.version)}"
        title="${esc(v.algorithm || "")}">
      <div class="vn">v${esc(String(v.version))}</div>
      <div class="vs">${esc(rejected ? "rejected" : st || "unknown")}</div>
      <div class="vm">${typeof score === "number" ? score.toFixed(4) : "&mdash;"}</div></a>`;
  }).join("") + `</div>`;
}

/* ---- Evidence Table ------------------------------------------------------ *
 * observed / required / verdict, one row per gate check. Colour appears only
 * in the verdict column, so a failing row is findable in a long list.
 */
function evidenceTable(checks){
  const cs = checks || [];
  if(!cs.length) return emptyState("No gate checks recorded.");
  const fmt = v => v == null ? NA : (typeof v === "number" ? v.toFixed(4) : esc(String(v)));
  return `<div class="scroll"><table class="evid">
    <thead><tr><th>Check</th><th class="num">Observed</th><th class="num">Required</th>
      <th>Blocking</th><th>Verdict</th></tr></thead>
    <tbody>${cs.map(c => `<tr class="${c.passed ? "" : "failed"}">
      <td class="mono">${esc(c.name)}</td>
      <td class="obs">${fmt(c.observed)}</td>
      <td class="req">${fmt(c.threshold)}</td>
      <td>${c.blocking ? badge("blocking","warn") : badge("advisory","mute")}</td>
      <td>${c.passed ? badge("pass","ok") : badge("fail","bad")}</td>
    </tr>`).join("")}</tbody></table></div>`;
}

/* ---- Attention Queue ----------------------------------------------------- *
 * Open alerts, newest first. The empty state names what is being watched
 * rather than declaring everything fine.
 */
function attentionQueue(alerts, opts){
  const o = opts || {};
  const list = (alerts || []).filter(a => o.includeAcknowledged || !a.acknowledged);
  if(!list.length) return `<div class="state">
    <div class="big">Nothing needs attention</div>
    Latency and error-rate SLOs, drift scans and retraining triggers all raise alerts
    here. None are currently open.</div>`;
  return `<div class="aq">` + list.slice(0, o.limit || 8).map(a => {
    const sev = String(a.severity || "info").toLowerCase();
    return `<div class="row">
      <span class="sev ${esc(sev)}"></span>
      <div>
        <div class="ttl">${esc(a.title || a.name || a.kind || "Alert")}</div>
        <div class="msg">${esc(a.message || "")}</div>
        <div class="meta">${esc(sev)} · ${when(a.created_at)}${
          a.source ? " · " + esc(a.source) : ""}</div>
      </div>
      ${a.acknowledged ? badge("acknowledged","mute")
        : `<button class="btn" data-act="ack" data-id="${esc(a.id)}">Acknowledge</button>`}
    </div>`;
  }).join("") + `</div>`;
}

/* ---- Section heading ----------------------------------------------------- *
 * The eyebrow is a real ordinal in a real sequence, not decoration: these
 * number the reading order of a page that has one.
 */
function sect2(n, title, sub){
  return `<div class="shead">
    ${n ? `<span class="n">${esc(n)}</span>` : ""}
    <h3>${esc(title)}</h3>
    ${sub ? `<span class="sub">${esc(sub)}</span>` : ""}</div>`;
}

/* ==========================================================================
 * Run status -- one map, used everywhere
 *
 * There were three of these: automlStatusBadge, runStatusBadge and
 * statusBadge, each with its own idea of what a state looks like. The same
 * `rejected` run rendered grey on AutoML, amber on Training and red in the
 * guided workflow, which makes state colour meaningless the moment a reader
 * moves between pages.
 *
 * One map. A terminal negative outcome is red whether the pipeline called it
 * "failed" or "rejected" -- both mean the model cannot proceed.
 * ========================================================================== */
const RUN_STATE = {
  // terminal, good
  completed:"ok", succeeded:"ok", approved:"ok", passed:"ok", live:"ok",
  healthy:"ok", promoted:"ok", ready:"ok",
  // terminal, good with caveats
  completed_with_warnings:"warn", degraded:"warn", warning:"warn",
  pending_manual:"warn", skipped:"warn",
  // terminal, bad
  failed:"bad", rejected:"bad", error:"bad", unhealthy:"bad", cancelled:"bad",
  rolled_back:"bad", blocked:"bad",
  // in flight
  running:"info", training:"info", profiling:"info", ranking:"info",
  deploying:"info", in_progress:"info", pending:"info",
  // not started
  queued:"mute", not_started:"mute", unknown:"mute",
};
const RUN_IN_FLIGHT = new Set(["running","training","profiling","ranking",
                               "deploying","in_progress","queued","pending"]);

function runStatusBadge(status){
  const s = String(status || "unknown").toLowerCase();
  const label = s.replace(/_/g, " ");
  return badge(label, RUN_STATE[s] || "mute", RUN_IN_FLIGHT.has(s));
}

/* ==========================================================================
 * Icons
 *
 * One set, drawn on a 24-unit grid at a single stroke weight, inheriting
 * currentColor so a nav item and a button never disagree. Inline rather than
 * a font or a sprite request: the console has no guaranteed egress and this
 * costs nothing to ship.
 *
 * Glyph characters were doing this job before. They came from whatever the
 * viewer's font stack happened to resolve, sat on different baselines, and
 * one of them -- a heart, for Runtime -- was a leftover from when the page
 * was called System Health.
 * ========================================================================== */
const ICON_PATHS = {
  grid:      "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  layers:    "M12 3 3 8l9 5 9-5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5",
  deploy:    "M12 20V6M12 6 6 12M12 6l6 6M5 3h14",
  bell:      "M18 9a6 6 0 1 0-12 0c0 5-2 6-2 6h16s-2-1-2-6M10.5 20a2 2 0 0 0 3 0",
  database:  "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  sliders:   "M4 6h10M18 6h2M4 12h4M12 12h8M4 18h10M18 18h2M16 4v4M10 10v4M16 16v4",
  chip:      "M7 7h10v10H7zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4",
  beaker:    "M9 3v6.5L4.5 18A2 2 0 0 0 6.3 21h11.4a2 2 0 0 0 1.8-3L15 9.5V3M8 3h8M7.5 14h9",
  activity:  "M3 12h4l3 8 4-16 3 8h4",
  wave:      "M3 12c2.5-5 4.5 5 7 0s4.5 5 7 0 2-3 4-3",
  refresh:   "M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5",
  shieldOk:  "M12 3 5 6v6c0 4.4 3 8.2 7 9 4-.8 7-4.6 7-9V6l-7-3zM9 12l2 2 4-4",
  list:      "M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01",
  server:    "M3 5h18v6H3zM3 13h18v6H3zM7 8h.01M7 16h.01",
  message:   "M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z",
  fileText:  "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5zM14 3v5h5M9 13h6M9 17h4",
  checkSq:   "M4 4h16v16H4zM8.5 12l2.5 2.5 4.5-5",
  coins:     "M9 4c3.9 0 7 1.1 7 2.5S12.9 9 9 9 2 7.9 2 6.5 5.1 4 9 4zM2 6.5v5c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-5M15 11.5c3.9 0 7 1.1 7 2.5s-3.1 2.5-7 2.5M8 14v3.5c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V14",
  shield:    "M12 3 5 6v6c0 4.4 3 8.2 7 9 4-.8 7-4.6 7-9V6l-7-3z",
  external:  "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  search:    "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4",
  copy:      "M9 9h10v10H9zM5 15V5h10",
  check:     "M5 12.5 10 17 19 7",
  plus:      "M12 5v14M5 12h14",
  sortAsc:   "M7 15l5-5 5 5",
  sortDesc:  "M7 9l5 5 5-5",
  sortNone:  "M8 10l4-4 4 4M8 14l4 4 4-4",
  close:     "M6 6l12 12M18 6 6 18",
};

/* 16px default: it sits on the cap height of 13px UI text without optical
   correction, which is why nav labels and icons line up. */
function icon(name, size){
  const d = ICON_PATHS[name];
  if(!d) return "";
  const s = size || 16;
  return `<svg class="ic" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="1.75" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="${d}"/></svg>`;
}

/* ==========================================================================
 * Copy to clipboard
 *
 * Hashes, run ids and commits are the values most likely to be pasted into a
 * terminal or a ticket, and until now they could only be selected by hand.
 * ========================================================================== */
let COPY_SEQ = 0;

/* Renders a value with a copy control. `value` is the full string to copy;
   `shown` is what the reader sees, so a 64-character hash can be truncated on
   screen while the whole thing reaches the clipboard. */
function copyable(value, shown, opts){
  if(value == null || value === "") return NA;
  const o = opts || {};
  const id = `cp${++COPY_SEQ}`;
  const text = String(value);
  return `<span class="copyable">
    <span class="cv ${o.mono === false ? "" : "mono"}">${esc(shown == null ? text : String(shown))}</span>
    <button class="cbtn" data-copy="${esc(text)}" id="${id}"
      title="Copy ${esc(o.label || "value")}"
      aria-label="Copy ${esc(o.label || "value")}: ${esc(text)}">${icon("copy", 13)}</button>
  </span>`;
}

/* Delegated once at the document level, so markup rendered later still works
   without every page re-binding handlers. */
function wireCopyButtons(){
  if(window.__copyWired) return;
  window.__copyWired = true;
  document.addEventListener("click", async e => {
    const b = e.target.closest("[data-copy]");
    if(!b) return;
    const text = b.getAttribute("data-copy");
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch(err){
      /* clipboard API needs a secure context; fall back to a selection copy so
         this still works over plain HTTP, which is how the demo is served. */
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.cssText = "position:fixed;top:-1000px;opacity:0";
      document.body.appendChild(ta);
      ta.select();
      try { ok = document.execCommand("copy"); } catch(e2){ ok = false; }
      ta.remove();
    }
    if(ok){
      const original = b.innerHTML;
      b.innerHTML = icon("check", 13);
      b.classList.add("done");
      setTimeout(() => { b.innerHTML = original; b.classList.remove("done"); }, 1100);
      announce("Copied to clipboard");
    } else {
      toast("Could not copy. Select the value and copy manually.", "bad");
    }
  });
}

/* A single polite live region. Status changes that are obvious visually are
   otherwise silent to a screen reader. */
function announce(message){
  let el = $("#a11y-status");
  if(!el){
    el = document.createElement("div");
    el.id = "a11y-status";
    el.className = "sr-only";
    el.setAttribute("role", "status");
    el.setAttribute("aria-live", "polite");
    document.body.appendChild(el);
  }
  el.textContent = "";
  setTimeout(() => { el.textContent = message; }, 30);
}
