/* Models -- every registered model, its versions, and each version's story.
 *
 * A model is a name with a lineage of versions, its own serving endpoint and
 * its own input contract (signature). Three levels:
 *
 *   #/models                          every model: what serves, where, what waits
 *   #/models/churn_model              one model: lineage, stage occupancy, actions
 *   #/models/churn_model/3?tab=gate   one version, seven tabs
 *
 * Every panel names the endpoint it came from. Nothing is computed in the
 * browser that the backend already computes.
 */

const MODEL_TABS = ["overview","versions","evaluation","lineage","deployments","observability","audit"];

function modelTab(){
  const t = hashQuery().get("tab");
  return MODEL_TABS.includes(t) ? t : "overview";
}

function mhref(name, version, tab){
  return `#/models/${encodeURIComponent(name)}${version != null ? "/" + version : ""}`
    + (tab ? `?tab=${tab}` : "");
}

function stageBadge(v){
  const st = String(v.stage || "");
  const status = String(v.status || "").toLowerCase();
  if(status === "rejected") return badge("rejected","bad");
  if(status === "pending" && (st === "Development" || st === "Validation"))
    return badge("awaiting approval","warn");
  return badge(st || "unknown",
    st === "Production" ? "ok" : st === "Staging" ? "info"
      : st === "Validation" ? "info" : "mute");
}


/* ---------------------------------------------------------------- fleet -- */
async function modelsIndex(){
  const r = await loadAll({ dash:"/api/v1/dashboard", models:"/api/v1/models" }, 15000);
  const fleet = r.dash.ok ? ((r.dash.data.models || {}).models || []) : null;
  const meta = r.models.ok ? Object.fromEntries((r.models.data.models || []).map(m => [m.name, m])) : {};
  if(fleet === null) return errorState(r.dash.error, "models");
  if(!fleet.length) return card("Model registry", `<div class="state"><div class="big">No models yet</div>
    A model appears here once a training or AutoML run registers a version.
    <div style="margin-top:12px"><a class="btn pri" href="#/newproject">New ML Project</a></div></div>`);

  return card("Registered models", table([
    { label:"Model", sort:m => m.name, render:m =>
        `<a href="${mhref(m.name)}"><b>${esc(m.name)}</b></a>` },
    { label:"Predicts", sort:m => m.target, render:m => m.target
        ? `<span class="mono">${esc(m.target)}</span>` : NA },
    { label:"Serving", sort:m => m.serving_version, render:m => m.serving_version != null
        ? `<a class="mono" href="${mhref(m.name, m.serving_version)}">v${int(m.serving_version)}</a> ${
          badge(m.serving_stage || "", m.serving_stage === "Production" ? "ok" : "info")}`
        : `<span class="dim">not serving</span>` },
    { label:"Deployment", sort:m => m.deployment_state, render:m => m.deployment_state
        ? runStatusBadge(m.deployment_state) : `<span class="dim">none</span>` },
    { label:"ROC-AUC", num:true, sort:m => m.roc_auc, render:m => num(m.roc_auc, 4) },
    { label:"Versions", num:true, sort:m => m.versions, render:m => int(m.versions) },
    { label:"Awaiting approval", num:true, sort:m => m.awaiting_approval, render:m =>
        m.awaiting_approval ? badge(String(m.awaiting_approval), "warn") : `<span class="dim">0</span>` },
    { label:"Last drift scan", sort:m => m.last_drift_at, render:m => m.last_drift_at == null
        ? `<span class="dim">never</span>`
        : `${m.last_drift_detected ? badge("drift","warn") : badge("stable","ok")} ${
          `<span class="dim">${esc(ago(m.last_drift_at))}</span>`}` },
    { label:"Endpoint", render:m => `<span class="mono dim">${esc(m.endpoint || (meta[m.name]||{}).endpoint || "-")}</span>` },
  ], fleet, { id:"models-fleet", sortKey:"Model", filter:"Filter models",
    empty:"No models." }), { flush:true, sub:`${fleet.length} model(s)` })
  + `<p class="dim" style="margin:12px 2px 0;font-size:12.5px;max-width:80ch">Each model has its own
     serving endpoint and input contract, recorded when its versions were trained. A version is
     compared only with other versions of the same model, and a model name keeps one target for
     life.</p>`;
}

/* ------------------------------------------------------------ one model -- */
async function modelPage(name){
  const enc = encodeURIComponent(name);
  const r = await loadAll({
    versions: `/api/v1/models/${enc}/versions`,
    history:  `/api/v1/models/${enc}/history`,
    sig:      `/api/v1/models/${enc}/signature`,
    current:  `/api/v1/deployments/current?model=${enc}`,
    decisions:`/api/v1/models/${enc}/decisions?limit=12`,
  }, 4000);
  if(!r.versions.ok) return errorState(r.versions.error, "models");
  const versions = r.versions.data;
  if(!versions.length) return card(name, unavailable(`No versions of ${name} are registered.`));
  const history = r.history.ok ? r.history.data : [];
  const sig = r.sig.ok ? r.sig.data.signature : null;
  const live = r.current.ok ? r.current.data.deployment : null;
  // Only a Production version answers callers; Staging is for shadow evaluation.
  const serving = versions.find(v => v.stage === "Production");
  const waiting = versions.filter(v => String(v.status) === "pending");

  const head = `<div class="mhead"><div class="top">
      <div style="min-width:0">
        <h1>${esc(name)}</h1>
        <div class="idl">
          <span>predicts <b class="mono">${esc(sig ? sig.target : "?")}</b>${sig ? ` (positive: <b>${
            esc((sig.display_labels || sig.class_labels)[1])}</b>)` : ""}</span>
          <span><b>${versions.length}</b> version(s)</span>
          <span>serving <b>${serving ? "v" + serving.version + " " + serving.stage : "nothing"}</b></span>
          <span>endpoint <b class="mono">${esc(r.sig.ok ? r.sig.data.endpoint : "-")}</b></span>
          <span>live <b>${live ? "v" + live.current_version + " · " + live.state : "not deployed"}</b></span>
        </div>
      </div>
      <span class="spacer"></span>
      ${serving ? `<a class="btn" href="#/predict?model=${enc}">${icon("target",14)} Predict</a>` : ""}
      <button class="btn" id="mretrain" data-name="${esc(name)}">${icon("refresh",14)} Retrain</button>
      ${serving && !(live && String(live.current_version) === String(serving.version) && live.state === "live")
        ? `<button class="btn pri" data-deploy="${esc(String(serving.version))}" data-stage="Production"
         data-name="${esc(name)}">${icon("deploy",14)} Deploy v${serving.version}</button>` : ""}
    </div>
    ${waiting.length ? `<div class="note warnnote" style="margin-top:13px">${waiting.length} version(s)
      passed every automated check and wait for a human decision:
      ${waiting.map(v => `<a href="${mhref(name, v.version, "evaluation")}">v${v.version}</a>`).join(", ")}.
      </div>` : ""}
    <div id="actionout"></div>
  </div>`;

  const rail = card("Version lineage", versionRail(versions, null, "roc_auc", name)
    + `<p class="dim" style="margin:10px 0 0;font-size:12px">Colour is stage; a rejected version
       stays visibly rejected. Click a version to open it.</p>`, { sub:"newest first" });
  const promo = card("Stage occupancy", promotionRail(versions, history),
    { sub:`${history.length} transition(s) recorded` });

  const tbl = card("All versions", table([
    { label:"Version", sort:v => v.version,
      render:v => `<a class="mono" href="${mhref(name, v.version)}"><b>v${int(v.version)}</b></a>` },
    { label:"Stage", sort:v => v.stage, render:v => stageBadge(v) },
    { label:"Algorithm", sort:v => v.algorithm,
      render:v => `<span class="mono">${esc(v.algorithm || "-")}</span>` },
    { label:"ROC-AUC", num:true, sort:v => (v.metrics||{}).roc_auc,
      render:v => num((v.metrics||{}).roc_auc, 4) },
    { label:"F1", num:true, sort:v => (v.metrics||{}).f1, render:v => num((v.metrics||{}).f1, 4) },
    { label:"Dataset", sort:v => v.dataset_version, render:v => v.dataset_version
        ? `<a class="mono dim" href="#/datasets/${encodeURIComponent(v.dataset_version)}">${
          esc(v.dataset_version)}</a>` : NA },
    { label:"Registered", sort:v => v.created_at, render:v => when(v.created_at) },
  ], versions, { id:"model-versions", sortKey:"Version", sortDir:"desc",
    filter:"Filter by version, stage, algorithm or dataset", empty:"No versions.",
    rowClass:v => v.stage === "Production" ? "win" : "" }), { flush:true, sub:`${versions.length} total` });

  const decisions = r.decisions.ok ? r.decisions.data.decisions : [];
  const dec = card("Gate decisions", decisions.length ? decisionTimeline(name, decisions)
    : emptyState("No gate decision recorded yet."), { flush:decisions.length > 0,
      sub:"every automated verdict and human sign-off" });

  const contract = sig ? card("Input contract", signatureTable(sig),
      { flush:true, sub:`${sig.features.length} feature(s), recorded at training` }) : "";

  return head + rail + `<div class="grid g2">${promo}${dec}</div>` + tbl + contract;
}

function decisionTimeline(name, decisions){
  return `<div class="tl">${decisions.map(d => {
    const kind = d.decision === "approved" ? "done" : d.decision === "rejected" ? "bad" : "now";
    const cmp = d.comparison;
    return `<div class="ev ${kind}"><span class="pip"></span><div>
      <div class="when">${when(d.created_at)} · ${esc(d.source)} · ${esc(d.actor || "system")}</div>
      <div class="what"><a href="${mhref(name, d.model_version, "evaluation")}">v${d.model_version}</a>
        ${runStatusBadge(d.decision)} ${d.final_stage ? `<span class="dim">→ ${esc(d.final_stage)}</span>` : ""}</div>
      <div class="det">${esc(String(d.reason || "").slice(0, 260))}${
        cmp && cmp.basis ? ` <span class="dim">[${esc(String(cmp.basis).replace(/_/g," "))}${
          cmp.holdout_rows ? ", " + cmp.holdout_rows + " rows" : ""}]</span>` : ""}</div>
    </div></div>`;
  }).join("")}</div>`;
}

function signatureTable(sig){
  return table([
    { label:"Feature", render:f => `<span class="mono">${esc(f.name)}</span>` },
    { label:"Kind", render:f => badge(f.kind, f.kind === "numeric" ? "info" : "mute") },
    { label:"Training range / categories", render:f => f.kind === "numeric"
        ? `<span class="mono">${num(f.minimum, 2)} … ${num(f.maximum, 2)}</span> <span class="dim">median ${num(f.median, 2)}</span>`
        : `<span class="mono">${esc((f.categories || []).slice(0, 8).join(", "))}${
          (f.n_categories || 0) > 8 ? ` <span class="dim">+${f.n_categories - 8} more</span>` : ""}</span>` },
    { label:"Missing in training", num:true, render:f => pct(f.missing_fraction, 1) },
  ], sig.features, { empty:"No features recorded." });
}

/* --------------------------------------------------------------- detail -- */
async function modelDetail(name, version){
  const enc = encodeURIComponent(name);
  const r = await loadAll({
    v:        `/api/v1/models/${enc}/versions/${encodeURIComponent(version)}`,
    versions: `/api/v1/models/${enc}/versions`,
    history:  `/api/v1/models/${enc}/history`,
    deps:     `/api/v1/deployments?model=${enc}`,
    current:  `/api/v1/deployments/current?model=${enc}`,
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
        <h1><a href="${mhref(name)}" style="color:inherit">${esc(name)}</a></h1>
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
      ${v.stage === "Staging" ? `<button class="btn" data-approve="${esc(String(v.version))}"
         data-name="${esc(name)}" data-target="Production">Promote to Production…</button>` : ""}
      ${v.stage === "Production" && !serving ? `<button class="btn pri" data-deploy="${esc(String(v.version))}"
         data-stage="Production" data-name="${esc(name)}">${icon("deploy",14)} Deploy v${esc(String(v.version))}</button>` : ""}
      ${v.stage === "Staging" && live && live.state === "live" && !serving && String(live.shadow_version) !== String(v.version)
        ? `<button class="btn" data-deploy="${esc(String(v.version))}" data-stage="Staging"
         data-name="${esc(name)}">${icon("deploy",14)} Shadow v${esc(String(v.version))}</button>` : ""}
      <a class="btn" href="${mhref(name)}">All versions</a>
    </div>
    <div id="actionout"></div>
  </div>`;

  const rail = versionRail(versions, v.version, "roc_auc", name);

  const tabs = `<div class="tabs" role="tablist">${MODEL_TABS.map(t =>
    `<a class="tab ${t === tab ? "on" : ""}" role="tab" aria-selected="${t === tab}"
       href="${mhref(name, version, t)}">${t.charAt(0).toUpperCase() + t.slice(1)}</a>`).join("")}</div>`;

  let body = "";
  if(tab === "overview")            body = await mvOverview(name, v, history);
  else if(tab === "versions")       body = mvVersions(name, versions, v);
  else if(tab === "evaluation")     body = await mvEvaluation(name, v);
  else if(tab === "lineage")        body = await mvLineage(name, v);
  else if(tab === "deployments")    body = mvDeployments(name, deps, v, live);
  else if(tab === "observability")  body = await mvObservability(name, v, serving, live);
  else if(tab === "audit")          body = await mvAudit(name, v);

  return head + rail + tabs + body;
}

/* -- Overview: provenance, promotion timeline, metrics ---------------------- */
async function mvOverview(name, v, history){
  const met = v.metrics || {};
  const lineage = card("Provenance", `<div class="kv">
      <div><dt>Dataset version</dt><dd>${copyable(v.dataset_version, null, { label:"dataset version" })}
        ${v.dataset_version ? `<a href="#/datasets/${encodeURIComponent(v.dataset_version)}"
          style="margin-left:8px">Open</a>` : ""}</dd></div>
      <div><dt>Dataset hash</dt><dd>${copyable(v.dataset_hash,
        String(v.dataset_hash || "").slice(0,24) + (String(v.dataset_hash||"").length > 24 ? "…" : ""),
        { label:"dataset hash" })}</dd></div>
      <div><dt>Tracking run</dt><dd>${copyable(v.run_id, null, { label:"run id" })}</dd></div>
      <div><dt>Algorithm</dt><dd><span class="mono">${esc(v.algorithm || "-")}</span></dd></div>
      <div><dt>Git commit</dt><dd>${copyable(v.git_commit, null, { label:"git commit" })}</dd></div>
      <div><dt>Registered by</dt><dd>${esc(v.created_by || "-")}</dd></div>
      <div><dt>Artifact</dt><dd>${copyable(v.artifact_uri,
        String(v.artifact_uri || "").slice(0,44) + (String(v.artifact_uri||"").length > 44 ? "…" : ""),
        { label:"artifact URI" })}</dd></div>
    </div><p class="dim" style="margin:12px 0 0;font-size:12px">The full chain — run, job, gate,
      deployments, drift — is on the <a href="${mhref(name, v.version, "lineage")}">Lineage</a> tab.</p>`,
    { sub:"dataset → run → model" });

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
function mvVersions(name, versions, current){
  const prod = versions.find(x => x.stage === "Production");
  const base = (prod && prod.metrics) || {};
  return card("All versions", table([
    { label:"Version", render:x => `<a class="mono" href="${mhref(name, x.version)}"><b>v${int(x.version)}</b></a>` },
    { label:"Stage", render:x => stageBadge(x) },
    { label:"Algorithm", render:x => `<span class="mono">${esc(x.algorithm || "-")}</span>` },
    { label:"ROC-AUC", num:true, render:x => num((x.metrics||{}).roc_auc, 4) },
    { label:"Δ vs prod (recorded)", num:true, render:x => {
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
  }), { flush:true, sub:"recorded metrics come from each version's own test split" })
  + `<p class="dim" style="margin:10px 2px 0;font-size:12px;max-width:80ch">The Δ column compares
     recorded test metrics, which were measured on different splits. Promotion decisions do not
     use it: they re-score both models on the same held-out rows (see Evaluation).</p>`;
}

/* -- Evaluation: the gate's record, and the human decision ------------------ */
async function mvEvaluation(name, v){
  const enc = encodeURIComponent(name);
  const met = v.metrics || {};
  let decisions = [];
  try {
    decisions = (await api.get(`/api/v1/models/${enc}/decisions?version=${v.version}`, 3000)).decisions || [];
  } catch(e){ decisions = null; }

  const all = card("Full metric set", Object.keys(met).length
    ? `<div class="scroll"><table><thead><tr><th scope="col">Metric</th><th scope="col" class="num">Value</th></tr></thead>
        <tbody>${Object.entries(met).map(([k,val]) =>
          `<tr><td class="mono">${esc(k)}</td><td class="num">${metricCell(val)}</td></tr>`
        ).join("")}</tbody></table></div>`
    : unavailable("No metrics recorded."), { flush:true, sub:"held-out split" });

  if(decisions === null) return card("Approval gate",
    errorState("The gate decision record could not be read.", "models")) + all;

  const latest = decisions[0] || null;
  const automated = decisions.find(d => d.source === "pipeline") || null;
  const status = String(v.status || "");
  const canDecide = !["Production","Archived"].includes(v.stage) && status !== "rejected";

  const verdict = latest ? verdictFor(latest, v) : `<div class="verdict warn"><span class="mark"
      aria-hidden="true">◌</span><span class="txt"><b>NO DECISION RECORDED</b>
      <span>This version predates recorded gate decisions. Preview the gate below.</span></span></div>`;

  const actions = canDecide ? `<div class="actions" style="margin-top:14px">
      <button class="btn pri" data-approve="${esc(String(v.version))}" data-name="${esc(name)}">
        ${icon("check2",14)} Approve…</button>
      ${["Development","Validation"].includes(v.stage) ? `<button class="btn danger" data-reject="${
        esc(String(v.version))}" data-name="${esc(name)}">Reject…</button>` : ""}
      <button class="btn" data-preview="${esc(String(v.version))}" data-name="${esc(name)}">
        Preview against current thresholds</button>
    </div>
    <p class="dim" style="margin:10px 0 0;font-size:12px">Approval re-runs the gate on the recorded
      metrics under today's thresholds and, for Production, re-scores this version and the live one
      on the same held-out rows. A failing check cannot be approved past.</p>` : "";

  const gateCard = card("Approval gate", verdict
    + (automated && automated.checks && automated.checks.length ? evidenceTable(automated.checks) : "")
    + actions + `<div id="gatepreview" style="margin-top:13px"></div>`,
    { sub: latest ? `${latest.source} decision · ${esc(ago(latest.created_at))}` : "no record" });

  const cmp = (automated && automated.comparison) || (latest && latest.comparison) || null;
  const champ = cmp ? championCard(cmp) : "";

  const history = decisions.length > 1 ? card("Decision history", decisionTimeline(name, decisions),
    { flush:true, sub:`${decisions.length} decision(s)` }) : "";

  return gateCard + champ + history + all;
}

function verdictFor(d, v){
  const dec = String(d.decision || "unknown");
  const waiting = dec === "pending_manual" && String(v.status) === "pending";
  const cls = dec === "approved" ? "pass" : dec === "pending_manual" ? "warn" : "fail";
  const title = dec === "approved" ? "APPROVED"
    : waiting ? "AWAITING MANUAL APPROVAL" : dec === "pending_manual" ? "AWAITING MANUAL APPROVAL"
    : "REJECTED";
  return `<div class="verdict ${cls}">
    <span class="mark" aria-hidden="true">${dec === "approved" ? "✓" : dec === "pending_manual" ? "⏳" : "✕"}</span>
    <span class="txt"><b>${title}</b><span>${esc(d.reason || "")}${
      d.actor && !String(d.reason || "").includes(d.actor) ? ` — ${esc(d.actor)}` : ""}${
      d.comment && !String(d.reason || "").includes(d.comment) ? ` · “${esc(d.comment)}”` : ""}</span></span></div>`;
}

/* When there is no incumbent the backend reports a baseline of zero and says
   so in `reason`. That zero is a sentinel, not a measurement: rendering it as
   0.0000 would invent a champion that scored nothing. */
function championCard(cmp){
  const noIncumbent = cmp.basis === "no_incumbent"
    || /no production model exists/i.test(String(cmp.reason || ""));
  const basis = cmp.basis === "shared_holdout"
    ? `Both models scored on the same ${int(cmp.holdout_rows)} held-out rows, none of which either trained on.`
    : cmp.basis === "recorded_metrics"
      ? "Compared on each version's own recorded test metric — too few shared held-out rows existed."
      : "";
  return card("Champion / challenger", noIncumbent ? `
    <div class="note"><b>No incumbent to compare against.</b><br>
      <span style="font-size:12.5px">${esc(cmp.reason)}. The comparison reports a baseline of zero
      to signal absence — it is not a score a champion achieved, so no improvement is shown.</span></div>
    <div class="grid g3" style="margin-top:13px">
      ${kpi("Candidate", `<span class="mono">v${esc(String(cmp.candidate_version))}</span>`)}
      ${kpi(String(cmp.metric || "metric").replace(/_/g," "), num(cmp.candidate_score,4))}
      ${kpi("Minimum improvement", num(cmp.min_improvement,4), "applies once a champion exists")}
    </div>` : `
    ${table([
      { label:"", render:x => `<b>${esc(x.k)}</b>` },
      { label:`Champion${cmp.baseline_version != null ? " v" + cmp.baseline_version : ""}`, num:true, render:x => x.a },
      { label:`Candidate${cmp.candidate_version != null ? " v" + cmp.candidate_version : ""}`, num:true, render:x => x.b },
    ], [
      { k:String(cmp.metric || "metric").replace(/_/g," "), a:num(cmp.baseline_score,4), b:num(cmp.candidate_score,4) },
      { k:"Improvement", a:NA, b:`${cmp.improvement >= 0 ? "+" : ""}${num(cmp.improvement,4)}` },
      { k:"Minimum required", a:NA, b:num(cmp.min_improvement,4) },
    ], { empty:"" })}
    <div style="margin-top:12px;display:flex;gap:9px;flex-wrap:wrap">
      ${cmp.candidate_is_better ? badge("candidate is better","ok") : badge("not an improvement","warn")}
      ${cmp.basis ? badge(String(cmp.basis).replace(/_/g," "), cmp.basis === "shared_holdout" ? "info" : "mute") : ""}</div>
    ${basis ? `<p class="dim" style="margin:10px 0 0;font-size:12.5px">${esc(basis)}</p>` : ""}
    <p class="dim" style="margin:6px 0 0;font-size:12px">Beating the incumbent is necessary but not
      sufficient: a candidate must also clear the absolute thresholds above.</p>`,
    { sub: noIncumbent ? "no baseline" : "promotion comparison" });
}

/* -- Lineage: the whole recorded story -------------------------------------- */
async function mvLineage(name, v){
  let L;
  try { L = await api.get(`/api/v1/models/${encodeURIComponent(name)}/versions/${v.version}/lineage`, 3000); }
  catch(e){ return card("Lineage", errorState(e.message, "models"), { flush:true }); }
  const ds = L.dataset, tr = L.training, sv = L.serving || {};
  const step = (title, state, body) => `<div class="ln ${state}"><div class="lnpip"></div>
    <div class="lnbody"><div class="lnt">${title}</div>${body}</div></div>`;
  const kv = pairs => `<div class="lnkv">${pairs.filter(p => p[1] != null && p[1] !== "")
    .map(([k, val]) => `<span><em>${esc(k)}</em> ${val}</span>`).join("")}</div>`;

  const steps = [
    step("Dataset", ds ? (ds.available ? "done" : "bad") : "none", ds ? kv([
      ["version", `<a class="mono" href="#/datasets/${encodeURIComponent(ds.version)}">${esc(ds.version)}</a>`],
      ["rows", ds.rows != null ? int(ds.rows) : null],
      ["hash", ds.content_hash ? `<span class="mono">${esc(String(ds.content_hash).slice(0,12))}</span>` : null],
      ["integrity", ds.available ? (ds.hash_matches_model ? badge("hash matches","ok") : badge("hash mismatch","bad"))
        : badge("dataset missing","bad")],
      ["derived from", ds.parent_version ? `<span class="mono">${esc(ds.parent_version)}</span>` : null],
    ]) : `<span class="dim">No dataset recorded.</span>`),
    step("Training", tr ? "done" : "none", tr ? kv([
      ["kind", esc(tr.kind)],
      ["run", tr.id ? `<span class="mono">${esc(tr.id)}</span>` : null],
      ["algorithm", tr.algorithm ? `<span class="mono">${esc(tr.algorithm)}</span>` : null],
      ["rank", tr.rank != null ? `#${tr.rank} of ${tr.candidates}${tr.winner ? " (winner)" : ""}` : null],
      ["job", tr.job ? `<a class="mono" href="#/jobs/${encodeURIComponent(tr.job.id)}">${esc(tr.job.id)}</a> ${runStatusBadge(tr.job.status)}` : null],
      ["tracking run", tr.tracking_run_id ? `<span class="mono">${esc(String(tr.tracking_run_id).slice(0,16))}</span>` : null],
    ]) : `<span class="dim">Registered outside a recorded run.</span>`),
    step("Evaluation", Object.keys((L.evaluation||{}).metrics || {}).length ? "done" : "none", kv([
      ["ROC-AUC", num(((L.evaluation||{}).metrics||{}).roc_auc, 4)],
      ["F1", num(((L.evaluation||{}).metrics||{}).f1, 4)],
      ["threshold", (L.evaluation||{}).threshold != null ? num(L.evaluation.threshold, 3) : null],
    ])),
    step("Gate", L.gate_decisions.length ? (L.gate_decisions.some(d => d.decision === "approved") ? "done"
        : L.gate_decisions.some(d => d.decision === "rejected") ? "bad" : "now") : "none",
      L.gate_decisions.length ? L.gate_decisions.map(d => `<div class="lnrow">${runStatusBadge(d.decision)}
        <span class="dim">${esc(d.source)} · ${esc(d.actor || "system")} · ${esc(ago(d.created_at))}</span>
        <div class="dim" style="font-size:12px">${esc(String(d.reason || "").slice(0, 200))}</div></div>`).join("")
        : `<span class="dim">No decision recorded.</span>`),
    step("Stages", L.stage_history.length ? "done" : "none", L.stage_history.length
      ? `<div class="lnkv">${L.stage_history.map(h => `<span>${esc(h.from_stage || "·")} → <b>${esc(h.to_stage)}</b></span>`).join("")}</div>`
      : `<span class="dim">None.</span>`),
    step("Deployments", L.deployments.length ? (L.deployments.some(d => d.role === "serving" && d.state === "live") ? "now" : "done") : "none",
      L.deployments.length ? L.deployments.map(d => `<div class="lnrow"><span class="mono">${esc(d.endpoint)}</span>
        ${badge(d.role, d.role === "serving" ? "ok" : "mute")} ${runStatusBadge(d.state)}
        <span class="dim">${esc(d.strategy)} · ${esc(ago(d.created_at))}</span></div>`).join("")
        : `<span class="dim">Never deployed.</span>`),
    step("Serving", sv.predictions ? "done" : "none", kv([
      ["predictions", int(sv.predictions)], ["errors", int(sv.errors)],
      ["labelled", int(sv.labelled)], ["last", sv.last_prediction_at ? esc(ago(sv.last_prediction_at)) : null],
    ])),
    step("Drift", L.drift.length ? (L.drift[0].drift_detected ? "bad" : "done") : "none", L.drift.length
      ? L.drift.slice(0, 4).map(d => `<div class="lnrow">${d.drift_detected ? badge("drift","warn") : badge("stable","ok")}
          <span class="dim">score ${num(d.dataset_drift_score, 3)} · ${esc(ago(d.created_at))}</span>
          ${d.drifted_features.length ? `<span class="mono dim">${esc(d.drifted_features.slice(0,4).join(", "))}</span>` : ""}</div>`).join("")
      : `<span class="dim">No drift scan of this version.</span>`),
    step("Retraining", (L.retraining.produced_by || L.retraining.triggered_from.length) ? "done" : "none",
      (L.retraining.produced_by ? `<div class="lnrow">produced by ${esc(L.retraining.produced_by.trigger)} retraining
        ${runStatusBadge(L.retraining.produced_by.status)} <span class="dim">${esc(ago(L.retraining.produced_by.created_at))}</span></div>` : "")
      + L.retraining.triggered_from.map(e => `<div class="lnrow">${esc(e.trigger)} → candidate
        ${e.candidate_version ? `<a href="${mhref(name, e.candidate_version)}">v${e.candidate_version}</a>` : "?"}
        ${runStatusBadge(e.decision || e.status)}${e.data_sources ? ` <span class="dim">${int(e.data_sources.labelled_production_rows)} labelled rows</span>` : ""}</div>`).join("")
      || `<span class="dim">None.</span>`),
  ];
  return card("Lineage", `<div class="lineage">${steps.join("")}</div>`,
    { sub:"read from the records each stage wrote — nothing inferred" });
}

/* -- Deployments ----------------------------------------------------------- */
function mvDeployments(name, deps, v, live){
  const mine = (deps || []).filter(d =>
    [d.current_version, d.previous_version, d.candidate_version, d.shadow_version]
      .some(x => String(x) === String(v.version)));
  if(!mine.length) return card("Deployments",
    emptyState(`No deployment of ${name} has named v${v.version}.`), { flush:true });

  return mine.map(d => {
    const role = String(d.current_version) === String(v.version) ? "serving"
      : String(d.previous_version) === String(v.version) ? "previous"
      : String(d.candidate_version) === String(v.version) ? "candidate" : "shadow";
    const events = d.events || [];
    return card(`Deployment ${d.id || ""}`, `
      <div class="grid g4">
        ${kpi("Role for v" + v.version, badge(role, role === "serving" ? "ok" : "mute"))}
        ${kpi("State", runStatusBadge(d.state))}
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
/* Live metrics are per endpoint, not per model version. Attributing them to a
   version that has never served traffic would be a straight misreading, so the
   tab refuses rather than rendering numbers under the wrong version. */
async function mvObservability(name, v, serving, live){
  if(!serving) return card("Live performance", `
    <div class="verdict warn">
      <span class="mark" aria-hidden="true">◌</span>
      <span class="txt"><b>V${esc(String(v.version))} IS NOT SERVING TRAFFIC</b>
        <span>${live ? `The endpoint is currently serving v${esc(String(live.current_version))}.`
          : "No deployment exists on this model's endpoint."}</span></span></div>
    <p style="margin:0;font-size:13px;line-height:1.6">Live latency, error rate and throughput
      are measured per <em>endpoint</em>, not per model version. Showing them here would
      attribute production behaviour to a version that never handled a request.
      ${live ? `<a href="${mhref(name, live.current_version, "observability")}">Open
        the serving version instead.</a>` : ""}</p>`, { sub:"not applicable" });

  const enc = encodeURIComponent(name);
  const r = await loadAll({ sum:`/api/v1/monitoring/summary?model=${enc}` }, 4000);
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
      { sub:"this model's endpoint; not per model version" });
  }, "monitoring");
}

/* -- Audit ----------------------------------------------------------------- */
async function mvAudit(name, v){
  let entries = null;
  try { entries = (await api.get("/api/v1/audit?limit=500", 4000)).entries || []; }
  catch(e){ return card("Audit", errorState(e.message, "models"), { flush:true }); }
  const key = `${name}:${v.version}`;
  const rows = entries.filter(e => String(e.resource_id || "") === key
    || (JSON.stringify(e.detail || {}).includes(`"${name}"`)
        && JSON.stringify(e.detail || {}).includes(`${v.version}`)));
  return card("Audit entries naming this version", table([
    { label:"When", render:e => when(e.created_at) },
    { label:"Action", render:e => `<span class="mono">${esc(e.action)}</span>` },
    { label:"Actor", render:e => esc(e.actor || "-") },
    { label:"Outcome", render:e => e.outcome
        ? badge(e.outcome, e.outcome === "success" ? "ok" : "bad") : NA },
    { label:"Resource", render:e => `<span class="mono dim">${esc(e.resource_id || "-")}</span>` },
  ], rows, { empty:`No audit entries name ${key}.` }),
    { flush:true, sub:`${rows.length} of ${entries.length} entries` });
}

/* ------------------------------------------------------------- actions -- */
function followInto(job, onDone){
  const out = $("#actionout");
  if(!job || !out) return;
  out.innerHTML = jobPanel(job);
  followJob(job.id, async done => {
    toast(`${done.kind} ${done.status}`, done.status === "succeeded" ? "ok" : "bad");
    if(onDone) await onDone(done);
  });
}

function wireModelActions(){
  document.querySelectorAll("[data-deploy]").forEach(b => b.onclick = () => runAction({
    title: `${b.dataset.stage === "Staging" ? "Shadow" : "Deploy"} ${b.dataset.name} v${b.dataset.deploy}`,
    body: b.dataset.stage === "Staging"
      ? "A Staging version cleared the thresholds but has not been compared with the live version, "
        + "so it may only be shadowed: it scores a mirror of live traffic and answers nobody. "
        + "Promote it to Production to let it serve."
      : "Rolls this Production version onto the model's own endpoint. Canary and shadow need the "
        + "live version to take the same inputs; the platform refuses them otherwise.",
    extra: `<label for="depstrat" style="font-size:11.5px;color:var(--ink-3)">Strategy</label>
      <select id="depstrat" data-field="strategy">${b.dataset.stage === "Staging"
        ? `<option value="shadow">shadow — mirror traffic, serve nothing</option>`
        : `<option value="blue_green">blue/green — switch all traffic after a health check</option>
        <option value="direct">direct — switch immediately</option>
        <option value="canary">canary — shift traffic in steps, watching errors and latency</option>
        <option value="shadow">shadow — mirror traffic, serve nothing</option>`}</select>`,
    confirm: "Deploy", needsKey: true,
    path: "/api/v1/deployments",
    payload: f => ({ model_name: b.dataset.name, model_version: Number(b.dataset.deploy),
                     strategy: f.strategy, reason: "deployed from the console" }),
    success: "Deployment queued.",
    after: res => followInto(res.job, () => { api.bust(); render(); }),
  }));

  const rt = $("#mretrain");
  if(rt) rt.onclick = async () => {
    let versions = [];
    try { versions = (await api.get("/api/v1/datasets", 10000)).versions || []; } catch(e){ versions = []; }
    runAction({
      title: `Retrain ${rt.dataset.name}`,
      body: "Trains a candidate on the serving version's data plus every labelled production row, "
          + "optionally with a new dataset version. It replaces nothing unless it clears the gate and "
          + "beats the live version on held-out rows neither has seen.",
      extra: `<label for="rtds" style="font-size:11.5px;color:var(--ink-3)">Add a new dataset version (optional)</label>
        <select id="rtds" data-field="dataset_version"><option value="">none — base data and labels only</option>
        ${versions.slice().reverse().map(v => `<option value="${esc(v.version)}">${esc(v.version)} · ${
          esc(v.dataset_name)} · ${int(v.n_rows)} rows</option>`).join("")}</select>`,
      confirm: "Retrain", needsKey: true,
      path: "/api/v1/retraining/run",
      payload: f => ({ model_name: rt.dataset.name, force: true,
                       dataset_version: f.dataset_version || null }),
      success: "Retraining queued.",
      after: res => followInto(res.job, () => { api.bust(); render(); }),
    });
  };

  document.querySelectorAll("[data-approve]").forEach(b => b.onclick = () => runAction({
    title: `Approve ${b.dataset.name} v${b.dataset.approve}`,
    body: "Re-runs the gate under today's thresholds. Approval is refused if any automated check "
        + "fails, and for Production if this version does not beat the live one on shared held-out rows.",
    extra: `<label for="aptarget" style="font-size:11.5px;color:var(--ink-3)">Promote to</label>
      <select id="aptarget" data-field="target_stage">
        <option value="Staging" ${b.dataset.target === "Production" ? "" : "selected"}>Staging — approved for shadow evaluation</option>
        <option value="Production" ${b.dataset.target === "Production" ? "selected" : ""}>Production — may take live traffic (compared with the live version first)</option></select>
      <label for="apnote" style="font-size:11.5px;color:var(--ink-3);margin-top:8px;display:block">Comment (recorded)</label>
      <input id="apnote" data-field="comment" maxlength="500" placeholder="why this is ready">`,
    confirm: "Approve", needsKey: true,
    path: `/api/v1/models/${encodeURIComponent(b.dataset.name)}/versions/${b.dataset.approve}/approve`,
    payload: f => ({ target_stage: f.target_stage, comment: f.comment || "" }),
    success: "Approved.",
  }));

  document.querySelectorAll("[data-reject]").forEach(b => b.onclick = () => runAction({
    title: `Reject ${b.dataset.name} v${b.dataset.reject}`,
    body: "Records a human rejection. The version stays registered, for the record.",
    extra: `<label for="rjnote" style="font-size:11.5px;color:var(--ink-3)">Reason (required, recorded)</label>
      <input id="rjnote" data-field="comment" maxlength="500" placeholder="why it is not fit to ship">`,
    confirm: "Reject", danger: true, needsKey: true,
    path: `/api/v1/models/${encodeURIComponent(b.dataset.name)}/versions/${b.dataset.reject}/reject`,
    payload: f => ({ comment: f.comment || "" }),
    success: "Rejected.",
  }));

  document.querySelectorAll("[data-preview]").forEach(b => b.onclick = async () => {
    const res = await runAction({
      title: "Preview the gate", body: "Runs the gate against today's thresholds. Records nothing.",
      confirm: "Preview", needsKey: true,
      path: `/api/v1/models/${encodeURIComponent(b.dataset.name)}/versions/${b.dataset.preview}/evaluate-gate`,
      payload: null, success: "Gate evaluated.", after: () => {},
    });
    if(!res) return;
    const ap = res.approval || {};
    $("#gatepreview").innerHTML = `<div class="verdict ${
      ap.decision === "approved" ? "pass" : ap.decision === "pending_manual" ? "warn" : "fail"}">
      <span class="txt"><b>PREVIEW: ${esc(String(ap.decision || "unknown").toUpperCase().replace(/_/g," "))}</b>
      <span>${esc(ap.reason || "")}</span></span></div>` + evidenceTable(ap.checks || [])
      + (res.comparison ? championCard(res.comparison) : "");
  });
}

PAGES.models = {
  title: "Models",
  intro: "Every registered model, its versions, and each version's recorded story — from the "
       + "data it learned from to where it serves today.",
  async render(){
    const [name, version] = routeParams();
    if(!name) return await modelsIndex();
    return version ? await modelDetail(name, version) : await modelPage(name);
  },
  wire(){ wireModelActions(); },
};
