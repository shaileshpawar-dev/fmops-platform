/* Sidebar structure. Every entry resolves to a page backed by a real API --
   there are no placeholder destinations, because a nav item that goes nowhere
   is worse than one that does not exist. Environments, Rollouts, Performance
   and Settings are deliberately absent: the backend exposes no distinct data
   for them, and the information they would show already lives in Deployments,
   Monitoring and System Health. */
/* Navigation is organised around what an operator is doing, not around which
   backend module answers. Every entry resolves to a page backed by a real API.

   Three scope notes, because the labels could otherwise overclaim:
     - Incidents is the alerts API. There is no incident lifecycle in this
       backend -- raise and acknowledge, nothing else -- so the page is written
       as an attention queue.
     - Quality Gates has no collection endpoint. It shows the configured policy
       and evaluates any version against it on demand.
     - Runtime reads the process, not the cloud. It is deliberately not called
       Infrastructure: there is no AWS introspection here. */
const NAV = [
  { group:"Control", items:[
      ["overview","Command Center","grid"], ["models","Models","layers"],
      ["deployments","Deployments","deploy"], ["incidents","Incidents","bell"] ] },
  { group:"Build", items:[
      ["datasets","Datasets","database"], ["training","Training","sliders"],
      ["automl","AutoML","chip"], ["experiments","Experiments","beaker"] ] },
  { group:"Operate", items:[
      ["monitoring","Observability","activity"], ["drift","Drift","wave"],
      ["retraining","Retraining","refresh"] ] },
  { group:"Govern", items:[
      ["gates","Quality Gates","shieldOk"], ["audit","Audit","list"],
      ["runtime","Runtime","server"] ] },
  { group:"LLMOps", items:[
      ["llm-overview","Overview","message"], ["llm-prompts","Prompts","fileText"],
      ["llm-evals","Evaluations","checkSq"], ["llm-cost","Tokens & Cost","coins"],
      ["llm-safety","Safety","shield"] ] },
  { group:"", items:[ ["__docs","API Docs","external"] ] },
];

/* Counts shown in the rail come from the dashboard payload the shell already
   fetches for the header. No navigation chrome adds a request. */
let NAV_COUNTS = {};

function buildNav(){
  const cur = route();
  const cta = `<div class="navcta">
    <a class="btn pri" href="#/newproject">+ New ML Project</a></div>`;
  $("#nav").innerHTML = cta + NAV.map(sec =>
    (sec.group ? `<div class="navgrp">${esc(sec.group)}</div>` : "") +
    sec.items.map(([id,label,glyph]) => {
      if(id === "__docs") return `<a class="navlink" href="/docs" target="_blank" rel="noopener">
        <span class="ico">${icon(glyph)}</span>${esc(label)}</a>`;
      const c = NAV_COUNTS[id];
      const cnt = (c && c.n) ? `<span class="cnt ${c.alert?"alert":""}">${esc(String(c.n))}</span>` : "";
      return `<a class="navlink ${id===cur?"on":""}" href="#/${id}"
        ${id===cur?'aria-current="page"':""}>
        <span class="ico">${icon(glyph)}</span>${esc(label)}${cnt}</a>`;
    }).join("")
  ).join("");
}

/* Routes are `#/page` or `#/page/param`. The param exists so a model version
   can be addressable -- #/models/3 -- without a router rewrite. */
function routeParts(){
  const raw = (location.hash || "#/overview").replace(/^#\//,"").split("?")[0];
  const seg = raw.split("/").filter(Boolean);
  return { id: seg[0] || "overview", param: seg[1] ? decodeURIComponent(seg[1]) : null };
}
function route(){ const p = routeParts(); return PAGES[p.id] ? p.id : "overview"; }
function routeParam(){ return routeParts().param; }

let refreshTimer = null;
async function render(){
  const id = route(), page = PAGES[id];
  buildNav();
  $("#crumb").textContent = page.title;
  const sub = $("#crumbsub");
  if(sub) sub.textContent = page.intro || "";
  document.title = `${page.title} · FMOps Platform`;
  const view = $("#main");
  view.setAttribute("aria-busy", "true");
  view.innerHTML = loadingPanel(page.title);

  if(refreshTimer){ clearInterval(refreshTimer); refreshTimer = null; }

  let html;
  try { html = await page.render(); }
  catch(e){ html = errorState(e.message || String(e), id); }
  if(route() !== id) return;                       // navigated away mid-load
  view.innerHTML = html;
  view.setAttribute("aria-busy", "false");
  announce(`${page.title} loaded`);
  stampRefresh();

  /* Auto-refresh only where it is genuinely useful, only while the tab is
     visible, and never faster than the backend sampling interval. */
  if(page.refresh){
    refreshTimer = setInterval(() => {
      if(document.visibilityState === "visible" && route() === id){ api.bust(); render(); }
    }, page.refresh);
  }
  wirePage();
}


/* A run in flight is the one thing worth following without a manual refresh.
   Polls until the run reaches a terminal state, then stops -- there is nothing
   further to learn, so continuing would be pure load. */
function wirePage(){
  document.querySelectorAll("[data-retry]").forEach(b =>
    b.onclick = () => { api.bust(); render(); });

  /* A page may own its own wiring. Pages predating this hook are still wired
     below; new ones should declare wire() and keep their handlers next to the
     markup that needs them. */
  const own = PAGES[route()];
  if(own && typeof own.wire === "function") own.wire();

  const q = $("#auditq"), sel = $("#auditact");
  if(q){
    const apply = () => {
      const term = (q.value||"").toLowerCase(), act = sel.value;
      const rows = (window.__audit||[]).filter(e => {
        if(act && e.action !== act) return false;
        if(!term) return true;
        return JSON.stringify(e).toLowerCase().includes(term);
      });
      $("#audittbl").innerHTML = auditTable(rows);
      $("#auditcount").textContent = `${rows.length} of ${(window.__audit||[]).length} entries`;
    };
    q.oninput = apply; sel.onchange = apply;
  }

  /* ---- Datasets: upload, validate, preview ----------------------------- */
  const up = $("#dsupload");
  if(up) up.onclick = async () => {
    const input = $("#dsfile");
    const file = input && input.files && input.files[0];
    if(!file){ toast("Choose a CSV file first.", "bad"); return; }
    if(!/\.csv$/i.test(file.name)){ toast("Only .csv files are accepted.", "bad"); return; }

    const needsKey = await authRequired();
    const proceed = await confirmAction({
      title: "Upload dataset",
      body: `Register ${file.name} (${(file.size/1024).toFixed(0)} KB) as a new dataset version and validate it.`,
      confirm: "Upload", needsKey,
    });
    if(!proceed) return;
    if(needsKey && !proceed.key){ toast("An API key is required to upload.", "bad"); return; }

    up.disabled = true; up.textContent = "Uploading...";
    const desc = ($("#dsdesc") && $("#dsdesc").value) || "";
    const qs = `?filename=${encodeURIComponent(file.name)}&description=${encodeURIComponent(desc)}`;
    try {
      const res = await fetch("/api/v1/datasets/upload" + qs, {
        method: "POST",
        headers: Object.assign({ "Content-Type": "text/csv" },
                               proceed.key ? { "X-API-Key": proceed.key } : {}),
        body: file,
      });
      const body = await res.json();
      if(!res.ok){
        const msg = (body.error && body.error.message) || `HTTP ${res.status}`;
        $("#dsresult").innerHTML = `<div class="note" style="border-color:var(--bad);
          background:var(--bad-bg)"><b>Upload rejected.</b><br>${esc(msg)}</div>`;
        toast("Upload rejected: " + msg, "bad");
        return;
      }
      api.bust();
      toast(`Registered ${body.version} (${body.rows} rows).`, "ok");
      $("#dsresult").innerHTML =
        `<div class="note"><b>Registered ${esc(body.version)}</b> &mdash; ${body.rows} rows,
          ${body.columns} columns.</div>` +
        (body.validation ? renderValidation(body.validation) : "");
    } catch(e){
      toast("Upload failed: " + e.message, "bad");
    } finally {
      up.disabled = false; up.textContent = "Upload and validate";
    }
  };

  document.querySelectorAll('[data-act="ds-validate"]').forEach(b => b.onclick = async () => {
    const target = $("#dsdetail");
    b.disabled = true; b.textContent = "Validating...";
    target.innerHTML = card("Validation", skeleton(4), { flush:true });
    try {
      const v = await api.get(`/api/v1/datasets/${encodeURIComponent(b.dataset.v)}/validation`, 0);
      target.innerHTML = card(`Validation - ${esc(b.dataset.v)}`, renderValidation(v));
      target.scrollIntoView({ behavior:"smooth", block:"nearest" });
    } catch(e){
      target.innerHTML = card("Validation", errorState(e.message, location.hash));
    } finally { b.disabled = false; b.textContent = "Validate"; }
  });

  document.querySelectorAll('[data-act="ds-preview"]').forEach(b => b.onclick = async () => {
    const target = $("#dsdetail");
    b.disabled = true; b.textContent = "Loading...";
    target.innerHTML = card("Preview", skeleton(4), { flush:true });
    try {
      const p = await api.get(`/api/v1/datasets/${encodeURIComponent(b.dataset.v)}/preview?rows=15`, 0);
      target.innerHTML = renderPreview(p);
      target.scrollIntoView({ behavior:"smooth", block:"nearest" });
    } catch(e){
      target.innerHTML = card("Preview", errorState(e.message, location.hash));
    } finally { b.disabled = false; b.textContent = "Preview"; }
  });

  /* ---- Training: start a run, then poll it ----------------------------- */
  const start = $("#trstart");
  if(start) start.onclick = async () => {
    const stage = $("#trstage").value;
    const payload = {
      dataset_version: $("#trds").value || null,
      algorithm: $("#tralgo").value || null,
      tune: !!$("#trtune").checked,
      promote: !!stage,
    };
    if(stage) payload.target_stage = stage;

    const needsKey = await authRequired();
    const proceed = await confirmAction({
      title: "Start training run",
      body: payload.promote
        ? `Train on ${payload.dataset_version || "the latest dataset"} and, if the approval gate passes and it beats the incumbent, promote to ${stage}.`
        : `Train on ${payload.dataset_version || "the latest dataset"}. The model will not be registered.`,
      confirm: "Start training", needsKey,
    });
    if(!proceed) return;
    if(needsKey && !proceed.key){ toast("An API key is required to start training.", "bad"); return; }

    start.disabled = true; start.textContent = "Starting...";
    try {
      const res = await api.post("/api/v1/training/runs", payload, proceed.key);
      api.bust();
      toast("Training started.", "ok");
      $("#trresult").innerHTML = `<div class="note"><b>Run ${esc(res.run_id)} accepted.</b>
        Polling for progress.</div>`;
      pollRun(res.run_id);
    } catch(e){
      const msg = (e.status === 401 || e.status === 403)
        ? "Rejected: the API key was missing or not accepted." : e.message;
      $("#trresult").innerHTML = `<div class="note" style="border-color:var(--bad);
        background:var(--bad-bg)"><b>Could not start.</b><br>${esc(msg)}</div>`;
      toast(msg, "bad");
    } finally {
      start.disabled = false; start.textContent = "Start training";
    }
  };

  document.querySelectorAll('[data-act="tr-detail"]').forEach(b => b.onclick = async () => {
    const target = $("#trdetail");
    target.innerHTML = card("Run detail", skeleton(4), { flush:true });
    try {
      const run = await api.get(`/api/v1/training/runs/${encodeURIComponent(b.dataset.id)}`, 0);
      target.innerHTML = renderRunDetail(run);
      target.scrollIntoView({ behavior:"smooth", block:"nearest" });
    } catch(e){
      target.innerHTML = card("Run detail", errorState(e.message, location.hash));
    }
  });

  /* ---- AutoML wizard --------------------------------------------------- */
  const amProfile = $("#amprofile");
  if(amProfile) amProfile.onclick = async () => {
    const version = $("#amds").value;
    AUTOML.version = version;
    AUTOML.target = null; AUTOML.selection = [];
    amProfile.disabled = true; amProfile.textContent = "Profiling...";
    $("#amwizard").innerHTML = card("Profiling dataset", skeleton(5), { flush:true });
    try {
      const d = await api.get(`/api/v1/automl/profile/${encodeURIComponent(version)}`, 0);
      $("#amwizard").innerHTML = renderWizard(d);
      wireWizard();
    } catch(e){
      $("#amwizard").innerHTML = card("Profiling failed", errorState(e.message, location.hash));
    } finally {
      amProfile.disabled = false; amProfile.textContent = "Profile dataset";
    }
  };

  document.querySelectorAll('[data-act="am-detail"]').forEach(b => b.onclick = async () => {
    const target = $("#amdetail");
    target.innerHTML = card("AutoML run", skeleton(5), { flush:true });
    try {
      const run = await api.get(`/api/v1/automl/runs/${encodeURIComponent(b.dataset.id)}`, 0);
      target.innerHTML = renderAutoMLRun(run);
      target.scrollIntoView({ behavior:"smooth", block:"nearest" });
    } catch(e){
      target.innerHTML = card("AutoML run", errorState(e.message, location.hash));
    }
  });

  wireWizard();

  const rb = document.querySelector('[data-act="rollback"]');
  if(rb) rb.onclick = () => runAction({
    title:"Roll back deployment",
    body:"This restores the previous model version as the serving version. It changes what production traffic is scored by.",
    confirm:"Roll back", danger:true, needsKey:true,
    path:"/api/v1/deployments/rollback", payload:{ reason:"manual rollback from console" },
    success:"Rollback requested." });

  document.querySelectorAll('[data-act="promote"]').forEach(b => b.onclick = () => runAction({
    title:`Promote v${b.dataset.ver} to Production`,
    body:"The registry stage machine and the approval gate still apply — this request can be rejected by the backend.",
    confirm:"Promote", needsKey:true,
    path:`/api/v1/models/${encodeURIComponent(b.dataset.model)}/versions/${encodeURIComponent(b.dataset.ver)}/stage`,
    payload:{ stage:"Production" }, success:"Stage transition requested." }));

  document.querySelectorAll('[data-act="ack"]').forEach(b => b.onclick = () => runAction({
    title:"Acknowledge alert",
    body:"Marks this alert as acknowledged. It stays in the history.",
    confirm:"Acknowledge", needsKey:true,
    path:`/api/v1/alerts/${encodeURIComponent(b.dataset.id)}/acknowledge`,
    success:"Alert acknowledged." }));
}

/* ------------------------------------------------------------- header --- */

/* Re-profiling on a target change is what makes the problem type honest: a
   different column can mean a different problem, and showing the old answer
   next to the new target would be worse than showing nothing. */
function stampRefresh(){
  const el = $("#lastrefresh");
  if(el) el.textContent = "Updated " + new Date().toLocaleTimeString([], {hour12:false});
}

async function header(){
  try {
    const d = await api.get("/api/v1/dashboard", 20000);
    const svc = d.service || {}, sys = d.system || {};
    $("#envbadge").innerHTML = badge(String(svc.environment||"unknown").toUpperCase(), "info");
    const ok = sys.latency_slo_met !== false && sys.error_slo_met !== false;
    $("#healthbadge").innerHTML = badge(ok ? "Healthy" : "Degraded", ok ? "ok" : "warn", true);

    /* Rail counts, from the payload just fetched. Only counts the backend
       actually reports -- an absent section leaves its entry unnumbered
       rather than showing a zero it did not confirm. */
    const model = d.model || {}, alerts = d.alerts || {};
    window.__dashModel = model;   // the palette reads versions from here
    const next = {};
    if(model.available && model.total_versions != null)
      next.models = { n: model.total_versions };
    if(alerts.available && alerts.open_count)
      next.incidents = { n: alerts.open_count, alert: true };
    if(JSON.stringify(next) !== JSON.stringify(NAV_COUNTS)){ NAV_COUNTS = next; buildNav(); }
  } catch(e){
    $("#envbadge").innerHTML = "";
    $("#healthbadge").innerHTML = badge("API unreachable","bad",true);
    if(Object.keys(NAV_COUNTS).length){ NAV_COUNTS = {}; buildNav(); }
  }
}

/* --------------------------------------------------------------- boot --- */
(function boot(){
  const saved = localStorage.getItem("fmops-theme");   // a UI preference, not data
  if(saved) document.documentElement.setAttribute("data-theme", saved);
  const syncTheme = t => {
    const btn = $("#theme");
    btn.setAttribute("aria-pressed", t === "dark" ? "true" : "false");
    btn.setAttribute("aria-label", t === "dark"
      ? "Switch to light theme" : "Switch to dark theme");
  };
  syncTheme(document.documentElement.getAttribute("data-theme") || "light");
  $("#theme").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    syncTheme(next);
    announce(`${next === "dark" ? "Dark" : "Light"} theme`);
    try { localStorage.setItem("fmops-theme", next); } catch(e){ /* private mode */ }
  };
  $("#refresh").onclick = () => { api.bust(); render(); header(); toast("Reloaded."); };
  $("#cmdk").onclick = () => openPalette();
  $("#burger").onclick = () => {
    const open = $("#side").classList.toggle("open");
    $("#burger").setAttribute("aria-expanded", open ? "true" : "false");
    if(open){ const first = $("#nav .navlink"); if(first) first.focus(); }
  };
  $("#nav").addEventListener("click", e => {
    if(e.target.closest(".navlink")) $("#side").classList.remove("open"); });
  window.addEventListener("hashchange", render);
  if(!location.hash) location.hash = "#/overview";
  wireCopyButtons();
  wireTables();
  wirePalette();
  render(); header(); renderSideFoot();
  setInterval(header, 30000);
})();

/* ==========================================================================
 * Command palette
 *
 * Cmd/Ctrl-K. Navigation only: every entry resolves to a route that already
 * exists, plus the model versions the registry actually reports. It calls no
 * endpoint of its own -- the version list is whatever the shell fetched for
 * the sidebar count, so opening the palette costs nothing.
 *
 * Deliberately not a search API. There is no search endpoint in this backend,
 * and a palette that pretended to search would be inventing one.
 * ========================================================================== */
let PALETTE_ITEMS = null;
let PALETTE_SEL = 0;

function paletteItems(){
  const nav = [];
  NAV.forEach(sec => sec.items.forEach(([id, label, glyph]) => {
    if(id === "__docs"){
      nav.push({ label:"API Docs", group:"Open", icon:glyph, href:"/docs", external:true });
    } else {
      nav.push({ label, group:sec.group || "Navigate", icon:glyph, hash:`#/${id}` });
    }
  }));
  nav.push({ label:"New ML Project", group:"Navigate", icon:"plus", hash:"#/newproject" });
  return nav;
}

/* Model versions come from the dashboard payload the header already holds, so
   the palette adds no request. If it has not loaded yet the palette simply
   shows routes -- it never blocks on a fetch. */
function paletteVersions(){
  const m = (window.__dashModel || {});
  const name = m.model_name;
  if(!name || !Array.isArray(m.versions)) return [];
  return m.versions.slice()
    .sort((a, b) => b.version - a.version)
    .slice(0, 25)
    .map(v => ({
      label: `v${v.version}`,
      hint: `${name} · ${v.stage || "unknown"}${v.algorithm ? " · " + v.algorithm : ""}`,
      group: "Model versions",
      icon: "layers",
      hash: `#/models/${v.version}`,
    }));
}

function openPalette(){
  if($("#palette")) return;
  PALETTE_ITEMS = paletteItems().concat(paletteVersions());
  PALETTE_SEL = 0;
  const el = document.createElement("div");
  el.className = "pal-wrap";
  el.id = "palette";
  el.innerHTML = `
    <div class="pal-scrim" data-pal-close></div>
    <div class="pal" role="dialog" aria-modal="true" aria-label="Command palette">
      <div class="pal-in">${icon("search", 16)}
        <input id="pal-q" type="text" autocomplete="off" spellcheck="false"
          placeholder="Go to a page or a model version…"
          aria-label="Search pages and model versions"
          aria-controls="pal-list" aria-activedescendant="pal-0" role="combobox"
          aria-expanded="true">
        <kbd>Esc</kbd></div>
      <div class="pal-list" id="pal-list" role="listbox" aria-label="Results"></div>
    </div>`;
  document.body.appendChild(el);
  renderPalette("");

  const q = $("#pal-q");
  q.focus();
  q.oninput = () => { PALETTE_SEL = 0; renderPalette(q.value); };
  q.onkeydown = e => {
    const rows = [...document.querySelectorAll(".pal-item")];
    if(e.key === "ArrowDown"){ e.preventDefault(); PALETTE_SEL = Math.min(PALETTE_SEL + 1, rows.length - 1); }
    else if(e.key === "ArrowUp"){ e.preventDefault(); PALETTE_SEL = Math.max(PALETTE_SEL - 1, 0); }
    else if(e.key === "Enter"){ e.preventDefault(); if(rows[PALETTE_SEL]) rows[PALETTE_SEL].click(); return; }
    else if(e.key === "Escape"){ e.preventDefault(); closePalette(); return; }
    else return;
    highlightPalette(rows);
  };
  el.addEventListener("click", e => {
    if(e.target.closest("[data-pal-close]")) closePalette();
    const item = e.target.closest(".pal-item");
    if(!item) return;
    const href = item.getAttribute("data-href"), hash = item.getAttribute("data-hash");
    closePalette();
    if(href) window.open(href, "_blank", "noopener");
    else if(hash) location.hash = hash;
  });
}

function highlightPalette(rows){
  rows.forEach((r, i) => {
    const on = i === PALETTE_SEL;
    r.classList.toggle("on", on);
    r.setAttribute("aria-selected", on ? "true" : "false");
    if(on){
      r.scrollIntoView({ block:"nearest" });
      const q = $("#pal-q");
      if(q) q.setAttribute("aria-activedescendant", r.id);
    }
  });
}

function renderPalette(query){
  const q = String(query || "").trim().toLowerCase();
  const hits = !q ? PALETTE_ITEMS : PALETTE_ITEMS.filter(it =>
    it.label.toLowerCase().includes(q) ||
    (it.hint || "").toLowerCase().includes(q) ||
    (it.group || "").toLowerCase().includes(q));

  const list = $("#pal-list");
  if(!hits.length){
    list.innerHTML = `<div class="pal-empty">Nothing matches &ldquo;${esc(query)}&rdquo;.
      The palette navigates pages and model versions &mdash; it does not search
      datasets or runs, because this platform has no search endpoint.</div>`;
    return;
  }
  let html = "", lastGroup = null, i = 0;
  hits.forEach(it => {
    if(it.group !== lastGroup){
      html += `<div class="pal-group">${esc(it.group)}</div>`;
      lastGroup = it.group;
    }
    html += `<div class="pal-item ${i === PALETTE_SEL ? "on" : ""}" id="pal-${i}"
      role="option" aria-selected="${i === PALETTE_SEL}"
      ${it.hash ? `data-hash="${esc(it.hash)}"` : ""}
      ${it.href ? `data-href="${esc(it.href)}"` : ""}>
      <span class="pi">${icon(it.icon || "grid", 15)}</span>
      <span class="pl">${esc(it.label)}</span>
      ${it.hint ? `<span class="ph">${esc(it.hint)}</span>` : ""}
      ${it.external ? `<span class="ph">opens a new tab</span>` : ""}
    </div>`;
    i++;
  });
  list.innerHTML = html;
}

function closePalette(){
  const el = $("#palette");
  if(el) el.remove();
  const t = $("#cmdk");
  if(t) t.focus();
}

function wirePalette(){
  if(window.__palWired) return;
  window.__palWired = true;
  document.addEventListener("keydown", e => {
    const key = (e.key || "").toLowerCase();
    if((e.metaKey || e.ctrlKey) && key === "k"){
      e.preventDefault();
      $("#palette") ? closePalette() : openPalette();
    }
    /* "/" is the other convention, but only when the caret is not already in
       a field -- otherwise it swallows a legitimate slash. */
    if(key === "/" && !$("#palette")){
      const t = e.target;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA"
        || t.tagName === "SELECT" || t.isContentEditable);
      if(!typing){ e.preventDefault(); openPalette(); }
    }
  });
}
