/* Command Center -- the operations control tower.
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
  const deployable = ["Staging","Production","Validation"].includes(stage);
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
    { state: stage ? (deployable ? "done" : "block") : "todo",
      value: stage || null,
      detail: deployable ? "reached a deployable stage"
        : (stage ? "held in " + stage : "not registered") },
    { state: deployable ? "done" : "todo", value: deployable ? stage : null,
      detail:"stage machine position" },
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
  intro: "What is happening to the models right now. Every figure is read from "
       + "the live API; sections that cannot report say so rather than showing a zero.",
  refresh: 30000,
  async render(){
    const r = await loadAll({
      dash:    "/api/v1/dashboard",
      ds:      "/api/v1/datasets",
      runs:    "/api/v1/training/runs?limit=1",
      automl:  "/api/v1/automl/runs?limit=1",
      alerts:  "/api/v1/alerts",
      audit:   "/api/v1/audit?limit=200",
    }, 5000);
    if(!r.dash.ok) return errorState(r.dash.error, "overview");
    const d = r.dash.data;
    const model = d.model || {}, dep = d.deployment || {}, svc = d.service || {};

    /* -- 1. production health ------------------------------------------- */
    const health = sect2("01", "Production health",
      `${esc(model.model_name || "no model")} · ${esc(svc.environment || "?")}`)
      + healthStrip(d);

    /* -- 2. lifecycle --------------------------------------------------- */
    const rail = sect2("02", "Model lifecycle", "derived from live API state")
      + `<div class="card"><div class="body">${lifecycleRail(buildLifecycle(d, r))}
        <p class="dim" style="margin:12px 0 0;font-size:12px">Each stop is derived from the
          endpoint that owns it. A stop whose source did not answer stays unstarted —
          completion is never inferred from the absence of an error.</p></div></div>`;

    /* -- 3. attention + promotion --------------------------------------- */
    const alerts = r.alerts.ok ? r.alerts.data : null;
    const attention = card("Attention queue",
      alerts === null ? unavailable("The alerts API could not be reached.")
        : attentionQueue(alerts, { limit:6 }),
      { flush:true, sub: alerts ? `${alerts.filter(a => !a.acknowledged).length} open` : "" });

    const versions = model.versions || [];
    /* Stage transitions are a separate call; without them the rail can show an
       occupant but not who moved it, so fetch them when a model name exists. */
    let history = [];
    if(model.model_name){
      try {
        history = await api.get(
          `/api/v1/models/${encodeURIComponent(model.model_name)}/history`, 15000);
      } catch(e){ history = []; }
    }
    const promo = card("Promotion rail",
      model.available
        ? promotionRail(versions, history)
          + `<p class="dim" style="margin:12px 0 0;font-size:12px">Occupancy of each stage in
             the registry. Open a version to see who moved it and why.</p>`
        : unavailable("The registry reported no model."),
      { sub: model.total_versions != null ? `${model.total_versions} version(s)` : "" });

    /* -- 4. operations + deployment ------------------------------------- */
    const MODEL_ACTIONS = ["model.registered","model.transitioned","model.promoted",
      "deployment.created","deployment.rolled_back","automl.run_completed",
      "training.completed","dataset.uploaded"];
    const audit = r.audit.ok ? (r.audit.data.entries || r.audit.data.audit || []) : null;
    const ops = audit === null ? unavailable("The audit API could not be reached.")
      : (() => {
          const rows = audit.filter(e => MODEL_ACTIONS.some(a =>
            String(e.action || "").startsWith(a.split(".")[0] + "."))).slice(0, 8);
          if(!rows.length) return emptyState("No model operations recorded yet.");
          return `<div class="tl">` + rows.map(e => {
            const bad = String(e.outcome || "") === "failure";
            return `<div class="ev ${bad ? "bad" : "done"}">
              <span class="pip"></span>
              <div><div class="when">${when(e.created_at)}</div>
                <div class="what">${esc(String(e.action || "").replace(/[._]/g," "))}</div>
                <div class="det">${esc(e.resource_id || "")}${
                  e.actor ? " · " + esc(e.actor) : ""}</div></div></div>`;
          }).join("") + `</div>`;
        })();

    const depCard = card("Deployment state", dep.available === false
      ? unavailable("No deployment has been created for this endpoint.")
      : `<div class="grid g4">
          ${kpi("Endpoint", `<span class="mono">${esc(dep.endpoint || "-")}</span>`)}
          ${kpi("Serving", dep.current_version != null
            ? `<span class="mono">v${int(dep.current_version)}</span>` : NA,
            dep.previous_version != null ? `previous v${dep.previous_version}` : "no previous")}
          ${kpi("Strategy", dep.strategy ? badge(dep.strategy, "mute") : NA)}
          ${kpi("Provider", esc(dep.provider || "-"),
            svc.aws_enabled ? "AWS integration on" : "in-process, not an AWS ML service")}
        </div>
        ${dep.checks ? `<div class="checks" style="margin-top:14px">${
          Object.entries(dep.checks).map(([k,v]) =>
            `<span class="c ${v ? "" : "no"}">${v ? "✓" : "✕"} ${esc(k)}</span>`).join("")}</div>` : ""}`,
      { right:`<a class="btn" href="#/deployments">Deployment Room</a>` });

    return health + rail
      + sect2("03", "Needs a decision", "alerts and stage occupancy")
      + `<div class="grid g2">${attention}${promo}</div>`
      + sect2("04", "Recent activity", "model-affecting operations")
      + `<div class="grid g2">${card("Recent model operations", ops,
          { flush:true, sub:"from the audit log" })}${depCard}</div>`;
  }
};
