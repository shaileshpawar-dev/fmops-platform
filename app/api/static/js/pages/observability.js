PAGES.monitoring = {
  title: "Monitoring",
  intro: "Service throughput, latency against SLO, resource usage, live model quality, and open alerts.",
  refresh: 15000,
  async render(){
    const r = await loadAll({ sum:"/api/v1/monitoring/summary", perf:"/api/v1/monitoring/performance",
      res:"/api/v1/monitoring/resources", alerts:"/api/v1/alerts" }, 4000);

    const svc = (r.sum.ok ? r.sum.data.service : {}) || {};
    const inference = card("Inference", `<div class="grid g4">
      ${kpi("Requests", int(svc.requests))}
      ${kpi("Errors", int(svc.errors), has(svc.error_rate)?`rate ${(svc.error_rate*100).toFixed(2)}%`:"")}
      ${kpi("Throughput", has(svc.throughput_rpm)?`<span class="mono">${svc.throughput_rpm.toFixed(1)}</span>`:NA,"req/min")}
      ${kpi("Error SLO", has(svc.error_slo_met)?boolBadge(svc.error_slo_met,"met","breached"):NA,
            has(svc.slo_error_rate)?`target ≤ ${(svc.slo_error_rate*100).toFixed(2)}%`:"")}
    </div>`);

    const latency = card("Latency", `<div class="grid g4">
      ${kpi("p50", ms(svc.latency_p50_ms))}${kpi("p95", ms(svc.latency_p95_ms))}
      ${kpi("p99", ms(svc.latency_p99_ms))}
      ${kpi("Latency SLO", has(svc.latency_slo_met)?boolBadge(svc.latency_slo_met,"met","breached"):NA,
            has(svc.slo_latency_ms)?`target ≤ ${svc.slo_latency_ms} ms`:"")}
    </div>`);

    const resources = sect(r.res, res => card("Resources", `<div class="grid g4">
      ${kpi("CPU", has(res.cpu_percent)?`<span class="mono">${res.cpu_percent.toFixed(1)}%</span>`:NA)}
      ${kpi("Memory", has(res.memory_percent)?`<span class="mono">${res.memory_percent.toFixed(1)}%</span>`:NA,
            has(res.memory_used_mb)?`${res.memory_used_mb.toFixed(0)} / ${(res.memory_total_mb||0).toFixed(0)} MB`:"")}
      ${kpi("Process RSS", has(res.process_rss_mb)?`<span class="mono">${res.process_rss_mb.toFixed(0)} MB</span>`:NA)}
      ${kpi("GPU", res.gpu_available?badge("present","ok"):badge("none detected","mute"),
            res.gpu_available?"":"reported as absent, not as 0%")}
    </div>
    <p class="dim" style="margin:12px 0 0;font-size:11.5px">Sampled on a background interval, not on
      the request path.</p>`), "resources");

    const live = sect(r.perf, p => card("Live model quality", p.available ? `<div class="grid g4">
        ${kpi("Accuracy", num(p.accuracy))}${kpi("Precision", num(p.precision))}
        ${kpi("Recall", num(p.recall))}${kpi("ROC-AUC", num(p.roc_auc))}
      </div>
      <p class="dim" style="margin:12px 0 0">Computed from ${int(p.labelled_samples)} labelled production rows.</p>`
      : `<div class="state"><div class="big">Requires ground-truth labels</div>
         ${esc(p.detail || "No labels submitted.")}</div>`), "performance");

    const alerts = sect(r.alerts, list => card("Alerts", table([
      { label:"Severity", render:a => badge(a.severity, a.severity==="critical"?"bad":a.severity==="warning"?"warn":"info") },
      { label:"Title", render:a => esc(a.title) },
      { label:"Category", render:a => `<span class="mono">${esc(a.category)}</span>` },
      { label:"Message", render:a => `<span class="dim">${esc(String(a.message||"").slice(0,110))}</span>` },
      { label:"Status", render:a => a.acknowledged?badge("acknowledged","mute"):badge("open","warn",true) },
      { label:"Raised", render:a => when(a.created_at) },
      { label:"", render:a => a.acknowledged?"":
          `<button class="btn" data-act="ack" data-id="${esc(a.id)}">Acknowledge</button>` },
    ], list, { empty:"No alerts raised." }), { flush:true }), "alerts");

    return `<div class="grid g2">${inference}${latency}</div>` + resources + live + alerts;
  }
};

PAGES.drift = {
  title: "Drift Detection",
  intro: "Data, feature and prediction drift measured against the training distribution. Concept drift is reported only when labels exist.",
  async render(){
    const r = await loadAll({ dash:"/api/v1/dashboard", reports:"/api/v1/drift" }, 6000);
    const dr = (r.dash.ok ? r.dash.data.drift : {}) || {};
    if(!dr.available) return unavailable(dr.reason || "No drift report has been produced yet.")
      + `<div class="note" style="margin-top:14px">Run a scan from the API
         (<span class="mono">POST /api/v1/drift/scan</span>) once enough production traffic has
         been logged.</div>`;

    const feats = (dr.feature_drift || []).slice().sort((a,b) =>
      (b.drifted - a.drifted) || (b.score - a.score));

    const summary = `<div class="grid g4" style="margin-bottom:14px">
      ${kpi("Dataset drift", num(dr.dataset_drift_score),
            dr.drift_detected?badge("detected","bad",true):badge("stable","ok",true))}
      ${kpi("Prediction drift", num(dr.prediction_drift_score),
            dr.prediction_drift_detected?badge("detected","bad",true):badge("normal","ok",true))}
      ${kpi("Drifted features", `${(dr.drifted_features||[]).length} / ${feats.length}`,
            `threshold ${dr.threshold}`)}
      ${kpi("Concept drift", badge(dr.concept_drift_status||"unavailable",
            dr.concept_drift_status==="measured"?"info":"mute"), `engine: ${esc(dr.engine||"—")}`)}
    </div>`;

    const conceptCard = card("Concept drift", `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
        <b>Status</b> ${badge(dr.concept_drift_status || "unavailable",
          dr.concept_drift_status === "measured" ? "info" : "mute")}</div>
      <p style="margin:0;color:var(--ink-2)">${esc(dr.concept_drift_detail || "")}</p>
      ${dr.concept_drift_status !== "measured" ? `<div class="note" style="margin-top:12px">
        Concept drift is a change in P(y|x) — what the inputs <i>mean</i>. It cannot be derived from
        unlabelled inputs, because the inputs themselves need not change at all. This console shows
        <b>no number</b> here rather than a proxy dressed up as a measurement.</div>`:""}`);

    const featCard = card("Feature drift", barList(feats.map(f => ({
      label: f.feature, value: f.score, bad: f.drifted,
      tag: f.drifted ? badge("drifted","bad") : badge("stable","ok")
    })), { threshold: dr.threshold, empty:"No per-feature results." }),
    { sub:`PSI primary, ${esc(feats[0] ? feats[0].test : "—")} supporting · threshold ${dr.threshold}`,
      flush:false });

    const hist = (dr.history || []).slice().reverse();
    const histCard = card("Drift score history", hist.length > 1
      ? lineChart(hist.map(x => ({ x: x.created_at, y: x.score })), { dp:3 })
      : emptyState("At least two scans are needed to plot a trend."),
      { sub:`${hist.length} scan(s)` });

    const reports = sect(r.reports, d => card("Scan history", table([
      { label:"Scanned", render:x => `${when(x.created_at)} <span class="dim">${ago(x.created_at)}</span>` },
      { label:"Detected", render:x => x.drift_detected?badge("yes","bad",true):badge("no","ok",true) },
      { label:"Dataset score", num:true, render:x => num(x.dataset_drift_score) },
      { label:"Prediction score", num:true, render:x => num(x.prediction_drift_score) },
      { label:"Drifted", num:true, render:x => int((x.drifted_features||[]).length) },
      { label:"Concept", render:x => badge(x.concept_drift_status||"—",
          x.concept_drift_status==="measured"?"info":"mute") },
    ], d.reports || [], { empty:"No scans recorded." }), { flush:true }), "drift reports");

    return summary + `<div class="grid g2">${featCard}${conceptCard}</div>` + histCard + reports;
  }
};

PAGES.retraining = {
  title: "Retraining",
  intro: "The automated retraining loop: what would trigger it now, which checks fired, and the history of runs.",
  async render(){
    const r = await loadAll({ dash:"/api/v1/dashboard", events:"/api/v1/retraining" }, 6000);
    const rt = (r.dash.ok ? r.dash.data.retraining : {}) || {};

    const pipeline = card("Retraining pipeline", `<div class="flow">
      ${["Drift / degradation","Trigger","Validation","Features","Training","Evaluation",
         "Champion / challenger","Approval gate","Deployment"]
        .map((s,i,a) => `<span class="step${i===0&&rt.would_trigger?" on":""}">${s}</span>` +
          (i<a.length-1?'<span class="arw">→</span>':"")).join("")}
    </div>
    <p class="dim" style="margin:12px 0 0">A candidate that fails the approval gate is
      <b>rejected</b> and production is left untouched.</p>`);

    const status = card("Current trigger evaluation", rt.available ? `
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
        ${rt.would_trigger?badge("Would trigger now","warn",true):badge("No trigger","ok",true)}
        ${rt.trigger?badge(rt.trigger,"info"):""}
        ${rt.suppressed_by_cooldown?badge("suppressed by cooldown","mute"):""}
      </div>
      <p style="margin:0 0 12px;color:var(--ink-2)">${esc(rt.reason||"")}</p>
      ${(rt.checks||[]).length ? `<div style="display:grid;gap:6px">${(rt.checks||[]).map(c =>
        `<div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start">
          <span class="mono">${esc(c.name)}</span>
          <span style="display:flex;gap:8px;align-items:center;text-align:right">
            <span class="dim" style="font-size:11.5px">${esc(c.detail||"")}</span>
            ${c.fired?badge("fired","warn"):badge("quiet","mute")}</span></div>`).join("")}</div>`:""}`
      : unavailable());

    const events = sect(r.events, list => card("Retraining runs", table([
      { label:"Run", render:x => `<span class="mono">${esc(String(x.id||"").slice(0,12))}</span>` },
      { label:"Trigger", render:x => badge(x.trigger||"—","info") },
      { label:"Status", render:x => badge(x.status||"—", x.status==="completed"?"ok":"mute") },
      { label:"Decision", render:x => x.decision ?
          badge(x.decision, x.decision==="promoted"?"ok":x.decision==="rejected"?"bad":"mute") : NA },
      { label:"Model", render:x => `<span class="mono">${esc(x.model_name||"—")}</span>` },
      { label:"Started", render:x => when(x.created_at) },
    ], list, { empty:"No retraining runs recorded yet." }), { flush:true }), "retraining");

    return status + pipeline + events;
  }
};
