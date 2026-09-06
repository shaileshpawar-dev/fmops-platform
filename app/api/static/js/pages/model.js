/* Models -- lineage entry point and model-version detail.
 *
 * The primary object is the model VERSION, not the model name. This platform
 * registers every model it has ever trained under one configured name
 * (tracking.registered_model_name), so there is no fleet to browse: the index
 * is a lineage entry point, and the depth is in the version.
 *
 * Routes:
 *   #/models              lineage: version rail, promotion rail, all versions
 *   #/models/3            version detail, six tabs
 *   #/models/3?tab=audit  a specific tab
 *
 * Every panel names the endpoint it came from. Nothing is computed in the
 * browser that the backend already computes.
 */

const MODEL_TABS = ["overview","versions","evaluation","deployments","observability","audit"];

function modelTab(){
  const q = (location.hash.split("?")[1] || "");
  const t = new URLSearchParams(q).get("tab");
  return MODEL_TABS.includes(t) ? t : "overview";
}

/* The registered name is read from the registry, never assumed: one global
   name today, but the page should not hardcode which. */
async function primaryModelName(){
  const d = await api.get("/api/v1/models", 30000);
  const names = (d.models || []).map(m => typeof m === "string" ? m : (m && m.name));
  return names.filter(Boolean)[0] || null;
}

function stageBadge(v){
  const st = String(v.stage || "");
  if(String(v.status || "").toLowerCase() === "rejected") return badge("rejected","bad");
  return badge(st || "unknown",
    st === "Production" ? "ok" : st === "Staging" ? "info"
      : st === "Validation" ? "info" : "mute");
}

/* ---------------------------------------------------------------- index -- */
async function modelsIndex(){
  const name = await primaryModelName();
  if(!name) return card("Model registry",
    unavailable("The registry holds no models yet. Train one from "
      + "New ML Project or the Training page."), { flush:true });

  const r = await loadAll({
    versions: `/api/v1/models/${encodeURIComponent(name)}/versions`,
    history:  `/api/v1/models/${encodeURIComponent(name)}/history`,
    prod:     `/api/v1/models/${encodeURIComponent(name)}/production`,
  }, 5000);
  const versions = r.versions.ok ? r.versions.data : [];
  const history = r.history.ok ? r.history.data : [];
  const prod = versions.find(v => v.stage === "Production");

  const head = `<div class="mhead">
    <div class="top">
      <div style="min-width:0">
        <h1>${esc(name)}</h1>
        <div class="idl">
          <span><b>${versions.length}</b> version(s)</span>
          <span>production <b>${prod ? "v" + prod.version : "none"}</b></span>
          <span>registry <b>${esc(r.versions.ok ? "reachable" : "unavailable")}</b></span>
        </div>
      </div>
      <span class="spacer"></span>
      ${prod ? `<a class="btn pri" href="#/models/${prod.version}">Open v${prod.version}</a>` : ""}
    </div>
    <p class="dim" style="margin:13px 0 0;font-size:12.5px;max-width:78ch">
      Every model this platform trains is registered under one configured name and
      distinguished by version, so this is a single lineage rather than a fleet. The
      version is the object worth opening.</p>
  </div>`;

  const rail = card("Version lineage", versionRail(versions, null, "roc_auc")
    + `<p class="dim" style="margin:10px 0 0;font-size:12px">Colour is stage; a rejected
       version stays visibly rejected. Click a version to open it.</p>`,
    { sub:"newest first" });

  const promo = card("Stage occupancy", promotionRail(versions, history),
    { sub:`${history.length} transition(s) recorded` });

  const tbl = card("All versions", table([
    { label:"Version", sort:v => v.version,
      render:v => `<a class="mono" href="#/models/${v.version}"><b>v${int(v.version)}</b></a>` },
    { label:"Stage", sort:v => v.stage, render:v => stageBadge(v) },
    { label:"Algorithm", sort:v => v.algorithm,
      render:v => `<span class="mono">${esc(v.algorithm || "-")}</span>` },
    { label:"ROC-AUC", num:true, sort:v => (v.metrics||{}).roc_auc,
      render:v => num((v.metrics||{}).roc_auc, 4) },
    { label:"F1", num:true, sort:v => (v.metrics||{}).f1,
      render:v => num((v.metrics||{}).f1, 4) },
    { label:"Dataset", sort:v => v.dataset_version, render:v => v.dataset_version
        ? `<span class="mono dim">${esc(v.dataset_version)}</span>` : NA },
    { label:"Registered", sort:v => v.created_at, render:v => when(v.created_at) },
  ], versions, {
    id:"models-all", sortKey:"Version", sortDir:"desc",
    filter:"Filter by version, stage, algorithm or dataset",
    empty:"No versions registered.", rowClass:v => v.stage === "Production" ? "win" : "" }),
    { flush:true, sub:`${versions.length} total` });

  return head + rail + promo + tbl;
}

/* --------------------------------------------------------------- detail -- */
async function modelDetail(version){
  const name = await primaryModelName();
  if(!name) return card("Model", unavailable("The registry holds no models."), { flush:true });

  const enc = encodeURIComponent(name);
  const r = await loadAll({
    v:        `/api/v1/models/${enc}/versions/${encodeURIComponent(version)}`,
    versions: `/api/v1/models/${enc}/versions`,
    history:  `/api/v1/models/${enc}/history`,
    deps:     "/api/v1/deployments",
    current:  "/api/v1/deployments/current",
  }, 4000);

  if(!r.v.ok) return `<div class="mhead"><h1>${esc(name)} v${esc(version)}</h1></div>`
    + errorState(r.v.error, "models");

  const v = r.v.data;
  const versions = r.versions.ok ? r.versions.data : [];
  const history = (r.history.ok ? r.history.data : [])
    .filter(h => String(h.version) === String(version));
  const deps = (r.deps.ok ? r.deps.data : []);
  const live = r.current.ok ? (r.current.data.deployment || null) : null;
  const serving = live && String(live.current_version) === String(version);
  const tab = modelTab();

  const head = `<div class="mhead">
    <div class="top">
      <div style="min-width:0">
        <h1>${esc(name)}</h1>
        <div style="display:flex;gap:10px;align-items:center;margin-top:7px;flex-wrap:wrap">
          <span class="mono" style="font-size:17px;font-weight:600">v${esc(String(v.version))}</span>
          ${stageBadge(v)}
          ${serving ? badge("serving","acc",true) : ""}
        </div>
        <div class="idl">
          <span>algorithm <b>${esc(v.algorithm || "-")}</b></span>
          <span>dataset <b>${esc(v.dataset_version || "-")}</b></span>
          <span>run <b>${esc(String(v.run_id || "-").slice(0,16))}</b></span>
          <span>commit <b>${esc(String(v.git_commit || "-").slice(0,12))}</b></span>
          <span>registered <b>${esc(String(v.created_at || "").slice(0,19).replace("T"," "))}</b></span>
        </div>
      </div>
      <span class="spacer"></span>
      <a class="btn" href="#/models">All versions</a>
      <a class="btn" href="#/deployments">Deployment Room</a>
    </div>
  </div>`;

  const rail = versionRail(versions, v.version, "roc_auc");

  const tabs = `<div class="tabs">${MODEL_TABS.map(t =>
    `<a class="tab ${t === tab ? "on" : ""}"
       href="#/models/${encodeURIComponent(version)}?tab=${t}">${
      t.charAt(0).toUpperCase() + t.slice(1)}</a>`).join("")}</div>`;

  let body = "";
  if(tab === "overview")            body = await mvOverview(name, v, history);
  else if(tab === "versions")       body = mvVersions(versions, v);
  else if(tab === "evaluation")     body = await mvEvaluation(name, v);
  else if(tab === "deployments")    body = mvDeployments(deps, v, live);
  else if(tab === "observability")  body = await mvObservability(v, serving, live);
  else if(tab === "audit")          body = await mvAudit(v);

  return head + rail + tabs + body;
}

/* -- Overview: lineage, promotion timeline, metrics ------------------------ */
async function mvOverview(name, v, history){
  const met = v.metrics || {};
  const lineage = card("Lineage", `<div class="kv">
      <div><dt>Dataset version</dt><dd>${copyable(v.dataset_version, null, { label:"dataset version" })}
        ${v.dataset_version ? `<a href="#/datasets" style="margin-left:8px">Datasets</a>` : ""}</dd></div>
      <div><dt>Dataset hash</dt><dd>${copyable(v.dataset_hash,
        String(v.dataset_hash || "").slice(0,24) + (String(v.dataset_hash||"").length > 24 ? "…" : ""),
        { label:"dataset hash" })}</dd></div>
      <div><dt>Training run</dt><dd>${copyable(v.run_id, null, { label:"run id" })}
        <a href="#/experiments" style="margin-left:8px">Experiments</a></dd></div>
      <div><dt>Algorithm</dt><dd><span class="mono">${esc(v.algorithm || "-")}</span></dd></div>
      <div><dt>Git commit</dt><dd>${copyable(v.git_commit, null, { label:"git commit" })}</dd></div>
      <div><dt>Registered by</dt><dd>${esc(v.created_by || "-")}</dd></div>
      <div><dt>Artifact</dt><dd>${copyable(v.artifact_uri,
        String(v.artifact_uri || "").slice(0,44) + (String(v.artifact_uri||"").length > 44 ? "…" : ""),
        { label:"artifact URI" })}</dd></div>
    </div>`, { sub:"dataset → run → model" });

  const tl = history.length ? `<div class="tl">${history.slice().reverse().map(h => `
      <div class="ev done"><span class="pip"></span>
        <div><div class="when">${when(h.created_at)}</div>
          <div class="what">${esc(h.from_stage || "registered")} → ${esc(h.to_stage)}</div>
          <div class="det">${esc(h.actor || "?")}${h.reason ? " · " + esc(h.reason) : ""}</div>
        </div></div>`).join("")}</div>`
    : emptyState("No stage transitions recorded for this version.");

  const params = v.params && Object.keys(v.params).length
    ? card("Hyperparameters", `<div class="kv">${Object.entries(v.params).map(([k,val]) =>
        `<div><dt>${esc(k)}</dt><dd><span class="mono">${esc(String(val))}</span></dd></div>`
      ).join("")}</div>`, { sub:"recorded with the run" }) : "";

  const metrics = card("Offline metrics", Object.keys(met).length
    ? `<div class="grid g4">${Object.keys(met).slice(0,12).map(k =>
        kpi(k.replace(/_/g," "), metricCell(met[k]))).join("")}</div>
      <p class="dim" style="margin:13px 0 0;font-size:12px">Measured on the held-out split at
        registration. These are not live production numbers — see the Observability tab.</p>`
    : unavailable("No metrics were recorded for this version."), { sub:"at registration" });

  return `<div class="grid g2">${lineage}${card("Promotion timeline", tl,
    { flush:history.length > 0, sub:`${history.length} transition(s)` })}</div>`
    + metrics + params;
}

function metricCell(v){
  if(typeof v !== "number" || !isFinite(v)) return NA;
  return Number.isInteger(v) ? int(v) : num(v, 4);
}

/* -- Versions -------------------------------------------------------------- */
function mvVersions(versions, current){
  const prod = versions.find(x => x.stage === "Production");
  const base = (prod && prod.metrics) || {};
  return card("All versions", table([
    { label:"Version", render:x => `<a class="mono" href="#/models/${x.version}"><b>v${int(x.version)}</b></a>` },
    { label:"Stage", render:x => stageBadge(x) },
    { label:"Algorithm", render:x => `<span class="mono">${esc(x.algorithm || "-")}</span>` },
    { label:"ROC-AUC", num:true, render:x => num((x.metrics||{}).roc_auc, 4) },
    { label:"Δ vs prod", num:true, render:x => {
        const a = (x.metrics||{}).roc_auc, b = base.roc_auc;
        if(typeof a !== "number" || typeof b !== "number" || x === prod) return NA;
        const d = a - b;
        return `<span class="mono" style="color:var(--${d >= 0 ? "ok" : "bad"})">${
          d >= 0 ? "+" : ""}${d.toFixed(4)}</span>`;
      } },
    { label:"F1", num:true, render:x => num((x.metrics||{}).f1, 4) },
    { label:"Dataset", render:x => x.dataset_version
        ? `<span class="mono dim">${esc(x.dataset_version)}</span>` : NA },
  ], versions.slice().sort((a,b) => b.version - a.version), {
    empty:"No versions.",
    rowClass:x => String(x.version) === String(current.version) ? "win" : "",
  }), { flush:true, sub:"Δ is measured against the production version" });
}

/* -- Evaluation: the gate, and the champion comparison --------------------- */
async function mvEvaluation(name, v){
  const met = v.metrics || {};
  const all = card("Full metric set", Object.keys(met).length
    ? `<div class="scroll"><table><thead><tr><th>Metric</th><th class="num">Value</th></tr></thead>
        <tbody>${Object.entries(met).map(([k,val]) =>
          `<tr><td class="mono">${esc(k)}</td><td class="num">${metricCell(val)}</td></tr>`
        ).join("")}</tbody></table></div>`
    : unavailable("No metrics recorded."), { flush:true, sub:"held-out split" });

  /* The gate endpoint is a POST. On a key-protected deployment an
     unauthenticated probe can only 401, so ask first and offer the action
     instead of firing a request that is known to fail. */
  const needsKey = await authRequired();
  if(needsKey) return all + card("Approval gate", `
    <p class="dim" style="margin:0 0 11px;font-size:12.5px">Re-running the gate is a
      write-method call and this API requires a key for it. The key is used for this request
      and is not stored.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input type="password" id="mvkey" placeholder="X-API-Key" style="flex:1 1 240px" autocomplete="off">
      <button class="btn" id="mvgate" data-name="${esc(name)}" data-v="${esc(String(v.version))}">
        Evaluate against current thresholds</button></div>
    <div id="mvgateout" style="margin-top:13px"></div>`, { sub:"POST evaluate-gate" });

  let gate = null, err = null;
  try {
    gate = await api.post(`/api/v1/models/${encodeURIComponent(name)}`
      + `/versions/${encodeURIComponent(v.version)}/evaluate-gate`);
  } catch(e){ err = e.message; }

  if(!gate) return all + card("Approval gate",
    errorState(err || "The gate endpoint returned nothing usable.", "models"), { sub:"" });

  const ap = gate.approval || {}, cmp = gate.comparison || null;
  const dec = String(ap.decision || "unknown");
  const cls = dec === "approved" ? "pass" : dec === "pending_manual" ? "warn" : "fail";
  const title = dec === "approved" ? "APPROVED FOR PRODUCTION"
    : dec === "pending_manual" ? "AWAITING MANUAL APPROVAL" : "NOT ELIGIBLE FOR PRODUCTION";
  const blurb = dec === "approved" ? "Every blocking check passed."
    : dec === "pending_manual"
      ? "Every automated check passed. This environment requires a human to promote."
      : "At least one blocking check failed.";

  const gateCard = card("Approval gate", `
    <div class="verdict ${cls}">
      <span class="mark" aria-hidden="true">${dec === "approved" ? "✓"
        : dec === "pending_manual" ? "⏳" : "✕"}</span>
      <span class="txt"><b>${title}</b><span>${blurb}</span></span></div>
    ${evidenceTable(ap.checks || [])}
    <p class="dim" style="margin:12px 0 0;font-size:12px">Thresholds come from platform
      configuration, not from this page, and are never relaxed to make a candidate pass.</p>`,
    { sub:`${(ap.checks||[]).filter(c => !c.passed).length} failed` });

  /* When there is no incumbent the backend returns baseline_score 0 and says so
     in `reason`. That zero is a sentinel, not a measurement: rendering it as
     0.0000 would invent a champion that scored nothing, and the improvement
     figure derived from it is equally meaningless. Read the backend's own
     statement rather than inferring from the magic zero. */
  const noIncumbent = cmp && /no production model exists/i.test(String(cmp.reason || ""));
  const champ = cmp ? card("Champion / challenger", noIncumbent ? `
    <div class="note"><b>No incumbent to compare against.</b><br>
      <span style="font-size:12.5px">${esc(cmp.reason)}. The comparison reports a baseline of
      zero to signal absence — it is not a score the champion achieved, so no improvement figure
      is shown here.</span></div>
    <div class="grid g3" style="margin-top:13px">
      ${kpi("Candidate", `<span class="mono">v${esc(String(cmp.candidate_version))}</span>`)}
      ${kpi(String(cmp.metric || "metric").replace(/_/g," "), num(cmp.candidate_score,4))}
      ${kpi("Minimum improvement", num(cmp.min_improvement,4), "applies once a champion exists")}
    </div>` : `
    ${table([
      { label:"", render:x => `<b>${esc(x.k)}</b>` },
      { label:`Champion${cmp.baseline_version != null ? " v" + cmp.baseline_version : ""}`,
        num:true, render:x => x.a },
      { label:`Candidate${cmp.candidate_version != null ? " v" + cmp.candidate_version : ""}`,
        num:true, render:x => x.b },
    ], [
      { k:String(cmp.metric || "metric").replace(/_/g," "),
        a:num(cmp.baseline_score,4), b:num(cmp.candidate_score,4) },
      { k:"Improvement", a:NA,
        b:`${cmp.improvement >= 0 ? "+" : ""}${num(cmp.improvement,4)}` },
      { k:"Minimum required", a:NA, b:num(cmp.min_improvement,4) },
    ], { empty:"" })}
    <div style="margin-top:12px;display:flex;gap:9px;flex-wrap:wrap">
      ${cmp.candidate_is_better ? badge("candidate is better","ok") : badge("not an improvement","warn")}
      ${cmp.decision ? badge(String(cmp.decision).replace(/_/g," "),
        cmp.decision === "promote" ? "ok" : "mute") : ""}</div>
    ${cmp.reason ? `<p class="dim" style="margin:10px 0 0;font-size:12.5px">${esc(cmp.reason)}</p>` : ""}
    <p class="dim" style="margin:10px 0 0;font-size:12px">Beating the incumbent is necessary but
      not sufficient: a candidate must also clear the absolute thresholds above.</p>`,
    { flush:false, sub:noIncumbent ? "no baseline" : "promotion comparison" })
    : card("Champion / challenger",
        unavailable("No incumbent to compare against, or the comparison was not recorded."),
        { flush:true });

  return gateCard + champ + all;
}

/* -- Deployments ----------------------------------------------------------- */
function mvDeployments(deps, v, live){
  const mine = (deps || []).filter(d =>
    String(d.current_version) === String(v.version) ||
    String(d.previous_version) === String(v.version) ||
    String(d.candidate_version) === String(v.version) ||
    String(d.shadow_version) === String(v.version));
  if(!mine.length) return card("Deployments",
    emptyState(`No deployment has named v${v.version}.`), { flush:true });

  return mine.map(d => {
    const role = String(d.current_version) === String(v.version) ? "serving"
      : String(d.previous_version) === String(v.version) ? "previous"
      : String(d.candidate_version) === String(v.version) ? "candidate" : "shadow";
    const events = d.events || [];
    return card(`Deployment ${d.id || ""}`, `
      <div class="grid g4">
        ${kpi("Role for v" + v.version, badge(role, role === "serving" ? "ok" : "mute"))}
        ${kpi("State", badge(d.state, d.state === "live" ? "ok" : "info"))}
        ${kpi("Strategy", esc(d.strategy || "-"))}
        ${kpi("Endpoint", `<span class="mono">${esc(d.endpoint_name || "-")}</span>`)}
      </div>
      ${events.length ? `<div class="tl" style="margin-top:15px">${events.map(e => `
        <div class="ev done"><span class="pip"></span>
          <div><div class="when">${when(e.created_at)}</div>
            <div class="what">${esc(String(e.event || "").replace(/_/g," "))}</div>
            <div class="det">${esc(JSON.stringify(e.detail || {}).slice(0,160))}</div>
          </div></div>`).join("")}</div>` : ""}`,
      { sub:`created ${String(d.created_at || "").slice(0,19).replace("T"," ")}` });
  }).join("");
}

/* -- Observability --------------------------------------------------------- */
/* Live metrics are endpoint-wide, not per-version. Attributing them to a
   version that has never served traffic would be a straight misreading, so the
   tab refuses rather than rendering numbers under the wrong model. */
async function mvObservability(v, serving, live){
  if(!serving) return card("Live performance", `
    <div class="verdict warn">
      <span class="mark" aria-hidden="true">◌</span>
      <span class="txt"><b>V${esc(String(v.version))} IS NOT SERVING TRAFFIC</b>
        <span>${live ? `The endpoint is currently serving v${esc(String(live.current_version))}.`
          : "No deployment exists on this endpoint."}</span></span></div>
    <p style="margin:0;font-size:13px;line-height:1.6">Live latency, error rate and throughput
      are measured per <em>endpoint</em>, not per model version. Showing them here would
      attribute production behaviour to a model that never handled a request.
      ${live ? `<a href="#/models/${esc(String(live.current_version))}?tab=observability">Open
        the serving version instead.</a>` : ""}</p>`, { sub:"not applicable" });

  const r = await loadAll({ sum:"/api/v1/monitoring/summary",
                            perf:"/api/v1/monitoring/performance" }, 4000);
  return sect(r.sum, d => {
    const s = d.service || {}, lat = s.latency || {}, lp = d.live_performance || {};
    return card("Live performance", `
      <div class="grid g4">
        ${kpi("Requests", int(s.request_count), `last ${int(s.window_minutes)} min`)}
        ${kpi("p95 latency", ms(lat.p95_ms), s.latency_slo_met === false ? "SLO breached" : "within SLO")}
        ${kpi("Error rate", pct(s.error_rate, 2), s.error_slo_met === false ? "SLO breached" : "within SLO")}
        ${kpi("Positive rate", pct(d.prediction_positive_rate, 2))}
      </div>
      <div class="grid g4" style="margin-top:14px">
        ${kpi("Labelled samples", int(lp.labelled_samples))}
        ${kpi("Live F1", lp.available ? num(lp.f1,4) : NA, lp.available ? "" : "needs labels")}
        ${kpi("Live ROC-AUC", lp.available ? num(lp.roc_auc,4) : NA, lp.available ? "" : "needs labels")}
        ${kpi("Open alerts", int(d.open_alerts))}
      </div>
      <p class="dim" style="margin:14px 0 0;font-size:12.5px">${lp.available
        ? "Live performance is computed only over predictions whose true outcome came back."
        : esc(lp.detail || "Live accuracy cannot be computed without ground-truth labels.")}</p>`,
      { sub:"endpoint-wide, this version is serving" });
  }, "monitoring");
}

/* -- Audit ----------------------------------------------------------------- */
async function mvAudit(v){
  let entries = null;
  try { entries = (await api.get("/api/v1/audit?limit=400", 4000)).entries || []; }
  catch(e){ return card("Audit", errorState(e.message, "models"), { flush:true }); }
  const needle = String(v.version);
  const rows = entries.filter(e =>
    String(e.resource_id || "") === needle ||
    String(e.resource_id || "").endsWith("/" + needle) ||
    JSON.stringify(e.detail || {}).includes(`"version": ${needle}`) ||
    JSON.stringify(e.detail || {}).includes(`"model_version": ${needle}`));
  return card("Audit entries naming this version", table([
    { label:"When", render:e => when(e.created_at) },
    { label:"Action", render:e => `<span class="mono">${esc(e.action)}</span>` },
    { label:"Actor", render:e => esc(e.actor || "-") },
    { label:"Outcome", render:e => e.outcome
        ? badge(e.outcome, e.outcome === "success" ? "ok" : "bad") : NA },
    { label:"Resource", render:e => `<span class="mono dim">${esc(e.resource_id || "-")}</span>` },
  ], rows, { empty:`No audit entries name v${v.version}. The audit log records actions by `
    + `resource id; entries for this version appear once it is transitioned or deployed.` }),
    { flush:true, sub:`${rows.length} of ${entries.length} entries` });
}

PAGES.models = {
  title: "Models",
  intro: "One registered model name, many versions. The version is the object: open one to "
       + "see its dataset, run, metrics, gate outcome, promotions and deployments.",
  async render(){
    const v = routeParam();
    return v ? await modelDetail(v) : await modelsIndex();
  },
  wire(){
    const b = $("#mvgate");
    if(b) b.onclick = async () => {
      const key = ($("#mvkey") && $("#mvkey").value) || "";
      if(!key){ toast("Enter an API key.", "bad"); return; }
      b.disabled = true;
      try {
        const g = await api.post(`/api/v1/models/${encodeURIComponent(b.dataset.name)}`
          + `/versions/${encodeURIComponent(b.dataset.v)}/evaluate-gate`, null, key);
        const ap = g.approval || {};
        $("#mvgateout").innerHTML = `<div class="verdict ${
          ap.decision === "approved" ? "pass" : ap.decision === "pending_manual" ? "warn" : "fail"}">
          <span class="txt"><b>${esc(String(ap.decision || "unknown").toUpperCase()
            .replace(/_/g," "))}</b><span>${esc(ap.reason || "")}</span></span></div>`
          + evidenceTable(ap.checks || []);
      } catch(e){
        toast(e.status === 401 || e.status === 403
          ? "The API key was not accepted." : e.message, "bad");
      } finally { b.disabled = false; }
    };
  }
};
