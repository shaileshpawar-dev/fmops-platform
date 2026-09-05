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

function table(cols, rows, opts){
  const o = opts || {};
  if(!rows || !rows.length) return emptyState(o.empty || "No records.");
  const head = cols.map(c => `<th${c.num?' class="num"':""}>${esc(c.label)}</th>`).join("");
  const body = rows.map(r => `<tr class="${o.rowClass ? esc(o.rowClass(r)) : ""}">` + cols.map(c =>
    `<td${c.num?' class="num"':""}>${c.render(r)}</td>`).join("") + "</tr>").join("");
  return `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`; }

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
    return `<div style="display:grid;grid-template-columns:minmax(120px,1.1fr) 3fr minmax(64px,auto) auto;
      gap:10px;align-items:center">
      <span class="mono" style="font-size:11.5px">${esc(i.label)}</span>
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

function confirmAction(opts){
  return new Promise(resolve => {
    const m = document.createElement("div");
    m.className = "modal";
    m.innerHTML = `<div class="box">
      <div class="body" style="padding:16px">
        <h3>${esc(opts.title)}</h3>
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
    const done = v => { m.remove(); resolve(v); };
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
