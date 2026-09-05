PAGES.overview = {
  title: "Command Center",
  intro: "Production AI lifecycle, model health and deployment intelligence. Every figure on this page is read from the FMOps API — nothing here is illustrative.",
  async render(){
    const extra = await loadAll({ dash:"/api/v1/dashboard", ds:"/api/v1/datasets",
      runs:"/api/v1/training/runs?limit=1", automl:"/api/v1/automl/runs?limit=1" }, 5000);
    if(!extra.dash.ok) return errorState(extra.dash.error, "#/overview");
    const d = extra.dash.data;
    const m = d.model || {}, dep = d.deployment || {}, dr = d.drift || {},
          sys = d.system || {}, rt = d.retraining || {}, al = d.alerts || {}, svc = d.service || {};
    const met = m.metrics || {};
    const datasets = extra.ds.ok ? (extra.ds.data.versions || []) : null;
    const runCount = extra.runs.ok ? extra.runs.data.count : null;
    const automlCount = extra.automl.ok ? extra.automl.data.count : null;

    const driftBadge = dr.available
      ? (dr.drift_detected ? badge("Detected","bad",true) : badge("Stable","ok",true))
      : badge("Unavailable","mute");
    const health = (sys.latency_slo_met && sys.error_slo_met) ? badge("Within SLO","ok",true)
      : badge("SLO breach","bad",true);

    const lifecycle = card("Lifecycle", `<div class="flow">${[
        ["Dataset", datasets === null ? null : datasets.length > 0, "#/datasets"],
        ["Profile", datasets === null ? null : datasets.length > 0, "#/automl"],
        ["Validation", datasets === null ? null : datasets.length > 0, "#/datasets"],
        ["AutoML", automlCount === null ? null : automlCount > 0, "#/automl"],
        ["Training", runCount === null ? null : runCount > 0, "#/training"],
        ["Evaluation", m.available && Object.keys(met).length > 0, "#/evaluation"],
        ["Registry", !!m.available, "#/models"],
        ["Approval", !!m.current_stage, "#/champion"],
        ["Deployment", (dep.health === "healthy") || !!dep.state, "#/deployments"],
        ["Monitoring", !!sys.available, "#/monitoring"],
        ["Drift", dr.available ? !dr.drift_detected : null, "#/drift"],
        ["Retraining", rt.available ? !rt.would_trigger : null, "#/retraining"],
      ].map(([label, ok, href], i, arr) => {
        const cls = ok === null ? "" : (ok ? "done" : "on");
        const mark = ok === null ? "" : (ok ? " &#10003;" : " &#9888;");
        return `<a class="step ${cls}" href="${href}" style="text-decoration:none">${label}${mark}</a>` +
          (i < arr.length - 1 ? '<span class="arw">&rarr;</span>' : "");
      }).join("")}</div>
      <p class="dim" style="margin:12px 0 0;font-size:11.5px">A tick means the stage has produced
        real state in this deployment. A warning marks a stage asking for attention &mdash; drift
        detected, or a retraining trigger that would fire. Click a stage to open it.</p>`);

    const kpis = `<div class="grid g4" style="margin-bottom:14px">
      ${kpi("Dataset versions", datasets === null ? NA : int(datasets.length),
            datasets && datasets.length ? `latest ${esc(datasets[datasets.length-1].version)}` : "")}
      ${kpi("Training runs", runCount === null ? NA : int(runCount),
            automlCount ? `${automlCount} AutoML run(s)` : "started from the API")}
      ${kpi("Registered versions", int(m.total_versions), m.model_name ? esc(m.model_name) : "")}
      ${kpi("Serving version", m.available ? `v${esc(m.current_version)}` : NA,
            m.current_stage ? badge(m.current_stage, "info") : "")}
    </div>
    <div class="grid g4" style="margin-bottom:14px">
      ${kpi("Deployment", dep.strategy ? esc(dep.strategy) : (dep.provider ? esc(dep.provider) : NA),
            dep.health ? badge(dep.health, dep.health === "healthy" ? "ok" : "bad", true) : "")}
      ${kpi("Predictions logged", int(sys.requests), "since process start")}
      ${kpi("Drift", driftBadge, dr.available ? `score ${(dr.dataset_drift_score||0).toFixed(4)}` : "")}
      ${kpi("Open alerts", int(al.open_count), health)}
    </div>`;

    const modelCard = card("Serving model", m.available ? `
      <dl class="kv">
        <dt>Model</dt><dd class="mono">${esc(m.model_name)}</dd>
        <dt>Version / stage</dt><dd>v${esc(m.current_version)} &nbsp;${badge(m.current_stage,"info")}</dd>
        <dt>Algorithm</dt><dd class="mono">${esc(m.algorithm)}</dd>
        <dt>Dataset</dt><dd class="mono">${esc(m.dataset_version)}</dd>
      </dl>
      <div class="grid g3" style="margin-top:14px">
        ${kpi("ROC-AUC", num(met.roc_auc))}
        ${kpi("F1", num(met.f1))}
        ${kpi("Accuracy", num(met.accuracy))}
      </div>` : unavailable("No model is registered."), { right: `<a class="btn" href="#/models">Open registry</a>` });

    const deployCard = card("Deployment", `
      <dl class="kv">
        <dt>Endpoint</dt><dd class="mono">${esc(dep.endpoint || "—")}</dd>
        <dt>Provider</dt><dd>${badge(dep.provider || "unknown","info")}</dd>
        <dt>Strategy</dt><dd>${dep.strategy ? badge(dep.strategy,"info") : NA}</dd>
        <dt>State</dt><dd>${dep.state ? badge(dep.state,"info") : NA}</dd>
        <dt>Health</dt><dd>${dep.health ? badge(dep.health, dep.health==="healthy"?"ok":"bad", true) : NA}</dd>
      </dl>
      ${Object.keys(dep.checks||{}).length ? `<div style="margin-top:12px">
        <div class="k" style="font-size:10.5px;font-weight:700;letter-spacing:.06em;
          text-transform:uppercase;color:var(--ink-3);margin-bottom:7px">Health checks</div>
        <div style="display:grid;gap:5px">${Object.entries(dep.checks).map(([k,v]) =>
          `<div style="display:flex;justify-content:space-between;align-items:center">
            <span class="mono">${esc(k)}</span>${boolBadge(v,"pass","fail")}</div>`).join("")}</div>
      </div>` : ""}`, { right: `<a class="btn" href="#/deployments">Details</a>` });

    const driftCard = card("Drift", dr.available ? `
      <div style="display:grid;gap:9px">
        ${driftRow("Data drift", dr.dataset_drift_score, dr.threshold, dr.drift_detected)}
        ${driftRow("Feature drift", (dr.drifted_features||[]).length,
            (dr.feature_drift||[]).length, (dr.drifted_features||[]).length > 0, true)}
        ${driftRow("Prediction drift", dr.prediction_drift_score, dr.threshold,
            has(dr.prediction_drift_score) && dr.prediction_drift_score > dr.threshold)}
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span>Concept drift</span>${badge(dr.concept_drift_status || "unavailable",
            dr.concept_drift_status === "measured" ? "info" : "mute")}</div>
      </div>
      ${dr.concept_drift_status !== "measured" ? `<div class="note" style="margin-top:12px">
        <b>Concept drift is not computed from unlabelled data.</b><br>${esc(dr.concept_drift_detail||"")}</div>` : ""}`
      : unavailable(dr.reason || "No drift report yet."), { right: `<a class="btn" href="#/drift">Details</a>` });

    const rtCard = card("Retraining trigger", rt.available ? `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
        ${rt.would_trigger ? badge("Would trigger","warn",true) : badge("No trigger","ok",true)}
        ${rt.trigger ? badge(rt.trigger,"info") : ""}
        ${rt.suppressed_by_cooldown ? badge("cooldown","mute") : ""}
      </div>
      <p style="margin:0;color:var(--ink-2);font-size:12.5px">${esc(rt.reason || "")}</p>`
      : unavailable(), { right: `<a class="btn" href="#/retraining">Details</a>` });

    const alerts = (al.alerts || []);
    const alertCard = card("Open alerts", table([
      { label:"Severity", render:a => badge(a.severity, a.severity==="critical"?"bad":a.severity==="warning"?"warn":"info") },
      { label:"Title", render:a => esc(a.title) },
      { label:"Category", render:a => `<span class="mono">${esc(a.category)}</span>` },
      { label:"Raised", render:a => `${when(a.created_at)} <span class="dim">${ago(a.created_at)}</span>` },
    ], alerts, { empty:"No open alerts." }), { flush:true, right:`<a class="btn" href="#/monitoring">Monitoring</a>` });

    // Capability chips name only what this build actually implements. AWS is
    // listed because the app runs on ECS/Fargate behind an ALB; it is not a
    // claim that the application calls AWS services, which the sidebar footer
    // reports separately and honestly.
    const chips = ["MLOps","AutoML","Model registry","Approval gates","Blue/green · Canary · Shadow",
                   "Monitoring","Drift detection","Retraining","LLMOps","AWS ECS/Fargate"]
      .map(c => `<span class="chip">${esc(c)}</span>`).join("");

    const hero = `<section class="hero">
      <h1>Foundation Model Operations</h1>
      <p>An end-to-end MLOps and LLMOps platform for training, evaluating, governing,
         deploying and monitoring machine-learning models. Every number below is read
         from this deployment's own API.</p>
      <div class="chips">${chips}</div>
    </section>`;

    // Platform status is derived, not asserted: it reflects the health of the
    // sections that actually answered.
    const failed = ["model","deployment","drift","system"].filter(k => !(d[k] || {}).available &&
                    (d[k] || {}).available !== undefined && k !== "drift").length;
    const sloOk = sys.latency_slo_met !== false && sys.error_slo_met !== false;
    const banner = (sloOk && !failed)
      ? `<div class="banner"><i class="dot" aria-hidden="true"></i>
           <b>Platform operational</b>
           <span class="sub">All core services responding; latency and error rate within SLO.</span></div>`
      : `<div class="banner warn"><i class="dot" aria-hidden="true"></i>
           <b>Attention required</b>
           <span class="sub">${sloOk ? "One or more sections did not report."
             : "An SLO is currently breached — see Monitoring."}</span></div>`;

    // Entry point for someone who has not built anything here yet. The
    // Command Center answers "what is happening"; this answers "how do I
    // start", which is a different question and deserves its own affordance.
    const cta = `<div class="cta">
      <div class="txt">
        <b>Build and deploy a machine-learning model</b>
        <span>Upload a dataset, train candidate models, compare them, and deploy one that
          clears the approval gate — guided step by step, driving the same APIs as the
          pages below.</span>
      </div>
      <a class="btn pri" href="#/newproject">+ Create ML Project</a>
    </div>`;

    return hero + cta + banner + lifecycle + kpis
      + `<div class="grid g2">${modelCard}${deployCard}</div>`
      + `<div class="grid g2">${driftCard}${rtCard}</div>`
      + alertCard
      + card("Platform configuration", `<dl class="kv">
          <dt>Environment</dt><dd>${badge(svc.environment||"?","info")}</dd>
          <dt>Version</dt><dd class="mono">${esc(svc.version)}</dd>
          <dt>Deployment provider</dt><dd class="mono">${esc(svc.deployment_provider)}</dd>
          <dt>Registry backend</dt><dd class="mono">${esc(svc.registry_backend)}</dd>
          <dt>LLM provider</dt><dd class="mono">${esc(svc.llm_provider)}${
            svc.llm_provider === "mock" ? " " + badge("offline mock — not a language model","warn") : ""}</dd>
          <dt>AWS integration</dt><dd>${boolBadge(svc.aws_enabled,"enabled","disabled")}</dd>
        </dl>`);
  }
};
function driftRow(label, value, ref, bad, isCount){
  const v = isCount ? `${value} / ${ref} features` :
    (has(value) ? `<span class="mono">${Number(value).toFixed(4)}</span> <span class="dim">/ ${ref}</span>` : NA);
  return `<div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
    <span>${esc(label)}</span><span style="display:flex;gap:8px;align-items:center">${v}
    ${bad ? badge("above threshold","bad") : badge("normal","ok")}</span></div>`;
}
