/* Command Center -- the operations control tower, for every model.
 *
 * One question, in a fixed reading order: what is happening to my models right
 * now? Health first, because it is the only thing that might need you in the
 * next minute. The lifecycle rail second, because it is the orientation
 * device. Attention and promotion next -- the two "should I act?" panels.
 * Operations stream last, because history is context, not alarm.
 *
 * Deliberately not a KPI grid. Every figure is read from the API, and where a
 * section reports available:false it says so rather than rendering a zero.
 */

/* Lifecycle state, derived stop by stop from responses that actually answered.
 * Nothing here infers completion from the absence of an error: a section that
 * did not answer leaves its stop `todo`. */
function buildLifecycle(d, extra){
  const model = d.model || {}, dep = d.deployment || {}, drift = d.drift || {},
        sys = d.system || {}, rt = d.retraining || {};
  const met = model.metrics || {};
  const nDatasets = extra.ds && extra.ds.ok ? (extra.ds.data.versions || []).length : null;
  const nAutoml = extra.automl && extra.automl.ok ? extra.automl.data.count : null;
  const nTrain = extra.runs && extra.runs.ok ? extra.runs.data.count : null;
  const thresholds = model.thresholds || {};

  // The gate stop reads the registered version's own metrics against the
  // configured thresholds -- the same comparison the backend makes.
  let gate = { state:"todo", value:null, detail:"no registered metrics" };
  if(model.available && Object.keys(met).length && Object.keys(thresholds).length){
    // A threshold of 0 gates nothing, so counting it as a passed check would
    // inflate the ratio -- "4/4" where only two requirements exist.
    const pairs = [["f1","min_f1"],["roc_auc","min_roc_auc"],
                   ["precision","min_precision"],["recall","min_recall"]]
      .filter(([m2,t]) => met[m2] != null
        && typeof thresholds[t] === "number" && thresholds[t] > 0);
    const failed = pairs.filter(([m2,t]) => met[m2] < thresholds[t]);
    if(pairs.length){
      gate = failed.length
        ? { state:"fail", value:`${pairs.length - failed.length}/${pairs.length}`,
            detail:`below threshold: ${failed.map(p => p[0]).join(", ")}` }
        : { state:"done", value:`${pairs.length}/${pairs.length}`,
            detail:"every configured threshold met" };
    }
  }

  const stage = model.current_stage || null;
  // The same rule the deployment API enforces: only a Production version takes
  // live traffic. Staging cleared the thresholds and may only be shadowed.
  const deployable = stage === "Production";
  const staged = stage === "Staging";
  const deployed = dep.available && dep.current_version != null;

  return [
    nDatasets == null
      ? { state:"todo", detail:"datasets API did not answer" }
      : { state: nDatasets ? "done" : "todo", value: nDatasets ? `${nDatasets} version(s)` : null,
          detail:"registered dataset versions" },
    (nAutoml == null && nTrain == null)
      ? { state:"todo", detail:"run APIs did not answer" }
      : { state:((nAutoml||0)+(nTrain||0)) ? "done" : "todo",
          value:((nAutoml||0)+(nTrain||0)) ? `${(nAutoml||0)+(nTrain||0)} run(s)` : null,
          detail:"AutoML and training runs" },
    { state: Object.keys(met).length ? "done" : "todo",
      value: met.roc_auc != null ? Number(met.roc_auc).toFixed(4) : null,
      detail:"primary metric at registration" },
    gate,
    { state: stage ? (deployable || staged ? "done" : "block") : "todo",
      value: stage || null,
      detail: deployable ? "approved for live traffic"
        : staged ? "approved for shadow evaluation only"
        : (stage ? "held in " + stage : "not registered") },
    { state: deployable ? "done" : "todo", value: deployable ? stage : null,
      detail: staged ? "promote to Production to serve" : "stage machine position" },
    { state: deployed ? "now" : "todo",
      value: deployed ? `v${dep.current_version}` : null,
      detail: deployed ? `${dep.state || "deployed"} via ${dep.strategy || "?"}` : "not deployed" },
    { state: sys.available ? ((sys.latency_slo_met !== false && sys.error_slo_met !== false)
        ? "done" : "block") : "todo",
      value: sys.available ? ((sys.latency_slo_met !== false && sys.error_slo_met !== false)
        ? "SLO ok" : "breach") : null,
      detail:"latency and error-rate SLOs" },
    { state: drift.available ? (drift.drift_detected ? "block" : "done") : "todo",
      value: drift.available ? ((drift.drifted_features || []).length + " drifted") : null,
      detail: drift.available ? "last drift scan" : "no drift report yet" },
    { state: rt.available ? (rt.would_trigger ? "block" : "done") : "todo",
      value: rt.available ? (rt.would_trigger ? "due" : "not due") : null,
      detail: rt.reason || "retraining trigger evaluation" },
  ];
}

PAGES.overview = {
  title: "Command Center",
  intro: "What the platform is doing right now: every model, the decisions waiting on a human, the "
       + "jobs in flight, and what just happened. Every figure is read from the live API.",
  refresh: 30000,
  async render(){
    const models = await listModels();
    const focus = currentModel(models || []);
    const q = focus ? `?model=${encodeURIComponent(focus)}` : "";
    const r = await loadAll({
      dash:    `/api/v1/dashboard${q}`,
      ds:      "/api/v1/datasets",
      runs:    "/api/v1/training/runs?limit=1",
      automl:  "/api/v1/automl/runs?limit=1",
      alerts:  "/api/v1/alerts",
      pending: "/api/v1/models/pending",
    }, 5000);
    if(!r.dash.ok) return errorState(r.dash.error, "overview");
    const d = r.dash.data;
    const P = d.platform || {}, fleet = (d.models || {}).models || [], jobs = d.jobs || {};
    const counts = jobs.counts || {};

    /* -- 1. the platform, in counts ----------------------------------------- */
    const stat = (n, label, href, meta) => `<a class="stat" href="${href}">
      <span class="n">${n}</span><span class="l">${esc(label)}</span>${meta ? `<span class="m">${meta}</span>` : ""}</a>`;
    const platform = P.available === false ? unavailable("Platform counts are unavailable.")
      : `<div class="grid g6 statstrip">
        ${stat(int(P.models), "models", "#/models", `${int(P.model_versions)} versions`)}
        ${stat(int(P.datasets), "dataset versions", "#/datasets")}
        ${stat(int(P.training_runs), "training runs", "#/training")}
        ${stat(int(P.active_deployments), "live endpoints", "#/deployments")}
        ${stat(int(P.predictions_7d), "predictions · 7d", "#/monitoring")}
        ${stat(int((counts.running || 0) + (counts.queued || 0)), "jobs in flight", "#/jobs",
               counts.failed ? `<span class="bad">${int(counts.failed)} failed</span>` : "")}
      </div>`;

    if(!fleet.length) return sect2("01", "Platform", "counts, not estimates") + platform
      + card("Get started", `<ol class="steps">
          <li><a href="#/datasets"><b>Upload a CSV</b></a><span>It is versioned by content and validated
            against its own columns.</span></li>
          <li><a href="#/newproject"><b>Train a model</b></a><span>Pick the column to predict; AutoML
            trains candidates as a background job and registers the winner under your model name.</span></li>
          <li><a href="#/gates"><b>Approve it</b></a><span>The gate checks the thresholds; a human signs
            off where the environment requires it.</span></li>
          <li><a href="#/deployments"><b>Deploy it</b></a><span>Onto the model's own endpoint, blue/green,
            canary, shadow or direct.</span></li>
          <li><a href="#/predict"><b>Score records</b></a><span>Against the input contract the model was
            trained on — then record what really happened.</span></li>
        </ol>`, { sub:"nothing is registered yet" });

    /* -- 2. every model ------------------------------------------------------- */
    const fleetCard = card("Models", table([
      { label:"Model", render:m => `<a href="#/models/${encodeURIComponent(m.name)}"><b>${esc(m.name)}</b></a>` },
      { label:"Predicts", render:m => m.target ? `<span class="mono">${esc(m.target)}</span>` : NA },
      { label:"Serving", render:m => m.serving_version != null
          ? `<span class="mono">v${int(m.serving_version)}</span> ${badge(m.serving_stage || "", m.serving_stage === "Production" ? "ok" : "info")}`
          : `<span class="dim">not serving</span>` },
      { label:"Endpoint", render:m => m.deployment_state ? runStatusBadge(m.deployment_state) : `<span class="dim">not deployed</span>` },
      { label:"ROC-AUC", num:true, render:m => num(m.roc_auc, 4) },
      { label:"Drift", render:m => m.last_drift_at == null ? `<span class="dim">no scan</span>`
          : m.last_drift_detected ? badge("detected","warn") : badge("stable","ok") },
      { label:"Waiting", num:true, render:m => m.awaiting_approval ? badge(String(m.awaiting_approval),"warn") : "" },
    ], fleet, { empty:"" }), { flush:true, sub:`${fleet.length} model(s)`,
      right:`<a class="btn sm" href="#/models">All models</a>` });

    /* -- 3. what needs a decision -------------------------------------------- */
    const pending = r.pending.ok ? r.pending.data.versions : [];
    const failed = (jobs.recent || []).filter(j => j.status === "failed");
    const alerts = r.alerts.ok ? r.alerts.data : null;
    const decisions = card("Needs a decision", `
      ${pending.length ? `<div class="aq">${pending.slice(0, 5).map(v => `<div class="row">
          <span class="sev warning"></span><div>
            <div class="ttl">${esc(v.name)} v${int(v.version)} is waiting for approval</div>
            <div class="msg">Passed every automated check · ROC-AUC ${num((v.metrics||{}).roc_auc, 4)}</div>
            <div class="meta">registered ${when(v.created_at)}</div></div>
          <a class="btn sm" href="#/models/${encodeURIComponent(v.name)}/${v.version}?tab=evaluation">Review</a></div>`).join("")}</div>` : ""}
      ${failed.length ? `<div class="aq">${failed.slice(0, 4).map(j => `<div class="row">
          <span class="sev critical"></span><div>
            <div class="ttl">${esc(String(j.kind).replace("_"," "))} job failed${j.model_name ? " · " + esc(j.model_name) : ""}</div>
            <div class="msg">${esc(String(j.error || "").slice(0, 160))}</div>
            <div class="meta">${when(j.finished_at || j.created_at)}</div></div>
          <a class="btn sm" href="#/jobs/${encodeURIComponent(j.id)}">Open</a></div>`).join("")}</div>` : ""}
      ${alerts === null ? unavailable("The alerts API could not be reached.")
        : (pending.length || failed.length) && !alerts.filter(a => !a.acknowledged).length ? ""
        : attentionQueue(alerts, { limit: 5 })}`,
      { flush:true, sub:`${pending.length} approval(s) · ${failed.length} failed job(s) · ${
        alerts ? alerts.filter(a => !a.acknowledged).length : "?"} open alert(s)` });

    /* -- 4. the focus model ---------------------------------------------------- */
    const model = d.model || {}, dep = d.deployment || {}, svc = d.service || {};
    const focusHead = `<div class="pagebar">${modelPicker(models, focus)}
      <span class="dim" style="font-size:12.5px">${esc(svc.environment || "")} · endpoint
        <span class="mono">${esc(dep.endpoint || "—")}</span></span></div>`;
    const rail = `<div class="card"><div class="body">${lifecycleRail(buildLifecycle(d, r))}
        <p class="dim" style="margin:12px 0 0;font-size:12px">Each stop is derived from the endpoint
          that owns it, for this model. A stop whose source did not answer stays unstarted —
          completion is never inferred from the absence of an error.</p></div></div>`;

    /* -- 5. what just happened ------------------------------------------------- */
    const events = ((d.activity || {}).events || []).slice(0, 10);
    const activity = card("Recent activity", events.length ? `<div class="tl">${events.map(e => {
        const bad = e.outcome === "failure" || e.outcome === "denied";
        return `<div class="ev ${bad ? "bad" : "done"}"><span class="pip"></span>
          <div><div class="when">${when(e.created_at)}</div>
            <div class="what">${esc(String(e.action || "").replace(/[._]/g," "))}</div>
            <div class="det">${esc(e.resource_id || "")}${e.actor ? " · " + esc(e.actor) : ""}${
              bad ? " · " + esc(e.outcome) : ""}</div></div></div>`; }).join("")}</div>`
      : emptyState("No activity recorded yet."), { flush:events.length > 0, sub:"from the audit log" });

    const recentJobs = card("Jobs", (jobs.recent || []).length ? table([
      { label:"Job", render:j => `<a class="mono" href="#/jobs/${encodeURIComponent(j.id)}">${esc(String(j.id).slice(4, 14))}</a>` },
      { label:"Kind", render:j => esc(String(j.kind).replace("_"," ")) },
      { label:"Model", render:j => esc(j.model_name || "—") },
      { label:"Status", render:j => runStatusBadge(j.status) },
      { label:"Queued", render:j => `<span class="dim">${esc(ago(j.created_at))}</span>` },
    ], jobs.recent, { empty:"" }) : emptyState("No jobs have run yet."),
      { flush:true, sub:"training · automl · retraining · deployment · drift",
        right:`<a class="btn sm" href="#/jobs">All jobs</a>` });

    return sect2("01", "Platform", "counts, not estimates") + platform
      + sect2("02", "Models and decisions", "what serves, and what waits on you")
      + `<div class="grid g2">${fleetCard}${decisions}</div>`
      + sect2("03", `Focus: ${esc(focus || "—")}`, "one model's lifecycle, end to end")
      + focusHead + healthStrip(d) + rail
      + sect2("04", "Recent activity", "audit log and jobs")
      + `<div class="grid g2">${activity}${recentJobs}</div>`;
  }
};
