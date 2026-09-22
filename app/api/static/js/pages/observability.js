/* Operate: observability, drift and retraining -- for one model at a time.
 *
 * Every model has its own endpoint, traffic, drift reference and retraining
 * history, so every panel here is scoped to the model in the picker. The
 * figures are read from the model-scoped endpoints; nothing is aggregated in
 * the browser across models that the backend keeps apart.
 */

/* The ranges the metrics endpoints actually accept. `window_minutes` is the
   only time parameter this API takes -- there is no from/to, so an arbitrary
   date picker would be inventing one. */
const OBS_WINDOWS = [
  { label:"15m", minutes:15 }, { label:"1h", minutes:60 },
  { label:"6h", minutes:360 }, { label:"24h", minutes:1440 },
  { label:"7d", minutes:10080 },
];
let OBS_WINDOW = 60;

PAGES.monitoring = {
  title: "Observability",
  intro: "For one model: throughput, latency against its SLO, errors, live quality from labelled "
       + "traffic, and the process resources it shares with every other model.",
  refresh: 15000,
  async render(){
    return withModel(async (name, models) => {
      const w = OBS_WINDOW, enc = encodeURIComponent(name);
      const r = await loadAll({
        sum:`/api/v1/monitoring/summary?window_minutes=${w}&model=${enc}`,
        res:"/api/v1/monitoring/resources", alerts:"/api/v1/alerts" }, 4000);

      const bar = `<div class="pagebar">${modelPicker(models, name)}
        <div class="rangebar"><span class="rl">Window</span>
          <div class="rseg" role="group" aria-label="Metrics window">
            ${OBS_WINDOWS.map(o => `<button class="rb ${o.minutes === w ? "on" : ""}"
              data-window="${o.minutes}" aria-pressed="${o.minutes === w}">${o.label}</button>`).join("")}
          </div></div></div>`;

      if(!r.sum.ok) return bar + errorState(r.sum.error, "monitoring");
      const d = r.sum.data, svc = d.service || {}, lat = svc.latency || {}, lp = d.live_performance || {};

      const inference = card("Inference", `<div class="grid g4">
        ${kpi("Requests", int(svc.request_count), `last ${int(svc.window_minutes)} min`)}
        ${kpi("Errors", int(svc.error_count), `rate ${pct(svc.error_rate, 2)}`)}
        ${kpi("Throughput", has(svc.throughput_rpm) ? `<span class="mono">${svc.throughput_rpm.toFixed(2)}</span>` : NA, "req/min")}
        ${kpi("Error SLO", svc.request_count ? boolBadge(svc.error_slo_met, "met", "breached") : badge("no traffic","mute"),
              `target ≤ ${pct(svc.slo_error_rate, 2)}`)}
      </div>
      <p class="dim" style="margin:12px 0 0;font-size:12px">Serving v${esc(String(d.model_version ?? "—"))}
        ${esc(d.model_stage || "")}. Prediction positive rate ${pct(d.prediction_positive_rate, 2)}.</p>`,
        { sub:"this model's endpoint" });

      const latency = card("Latency", `<div class="grid g4">
        ${kpi("p50", ms(lat.p50_ms))}${kpi("p95", ms(lat.p95_ms))}${kpi("p99", ms(lat.p99_ms))}
        ${kpi("Latency SLO", svc.request_count ? boolBadge(svc.latency_slo_met, "met", "breached") : badge("no traffic","mute"),
              `p95 target ≤ ${num(svc.slo_latency_ms, 0)} ms`)}
      </div>`, { sub:`${int(lat.count)} request(s) measured` });

      const live = card("Live model quality", lp.available ? `<div class="grid g4">
          ${kpi("Accuracy", num(lp.accuracy, 4))}${kpi("Precision", num(lp.precision, 4))}
          ${kpi("Recall", num(lp.recall, 4))}${kpi("ROC-AUC", lp.roc_auc != null ? num(lp.roc_auc, 4) : NA)}
        </div>
        <p class="dim" style="margin:12px 0 0">${esc(lp.detail || "")}</p>`
        : `<div class="state"><div class="big">Requires ground-truth labels</div>
           ${esc(lp.detail || "No labels submitted.")} Record outcomes on the
           <a href="#/predict?model=${enc}">Predict</a> page or via <span class="mono">POST /api/v1/feedback</span>.</div>`,
        { sub:"from labelled production predictions only" });

      const resources = sect(r.res, res => card("Process resources", `<div class="grid g4">
        ${kpi("CPU", has(res.cpu_percent) ? `<span class="mono">${res.cpu_percent.toFixed(1)}%</span>` : NA)}
        ${kpi("Memory", has(res.memory_percent) ? `<span class="mono">${res.memory_percent.toFixed(1)}%</span>` : NA,
              has(res.memory_used_mb) ? `${res.memory_used_mb.toFixed(0)} / ${(res.memory_total_mb||0).toFixed(0)} MB` : "")}
        ${kpi("Process RSS", has(res.process_rss_mb) ? `<span class="mono">${res.process_rss_mb.toFixed(0)} MB</span>` : NA)}
        ${kpi("GPU", res.gpu_available ? badge("present","ok") : badge("none detected","mute"),
              res.gpu_available ? "" : "reported as absent, not as 0%")}
      </div>
      <p class="dim" style="margin:12px 0 0;font-size:11.5px">Shared by every model this process serves.
        Sampled on a background interval, not on the request path.</p>`), "resources");

      const mine = r.alerts.ok ? r.alerts.data.filter(a =>
        JSON.stringify(a.context || {}).includes(`"${name}"`) || !a.context || !a.context.model) : null;
      const alerts = mine === null ? card("Alerts", unavailable("Alerts could not be read."), { flush:true })
        : card("Alerts", table([
          { label:"Severity", render:a => badge(a.severity, a.severity==="critical"?"bad":a.severity==="warning"?"warn":"info") },
          { label:"Title", render:a => esc(a.title) },
          { label:"Category", render:a => `<span class="mono">${esc(a.category)}</span>` },
          { label:"Message", render:a => `<span class="dim">${esc(String(a.message||"").slice(0,120))}</span>` },
          { label:"Status", render:a => a.acknowledged ? badge("acknowledged","mute") : badge("open","warn",true) },
          { label:"Raised", render:a => when(a.created_at) },
          { label:"", render:a => a.acknowledged ? "" :
              `<button class="btn sm" data-act="ack" data-id="${esc(a.id)}">Acknowledge</button>` },
        ], mine, { empty:"No alerts for this model." }), { flush:true, sub:"this model and platform-wide" });

      return bar + `<div class="grid g2">${inference}${latency}</div>` + live + resources + alerts;
    });
  },
  wire(){
    /* The window is a view preference, kept in a module variable rather than
       the URL because it is not an addressable location. */
    document.querySelectorAll("[data-window]").forEach(b => b.onclick = () => {
      const next = Number(b.dataset.window);
      if(next === OBS_WINDOW) return;
      OBS_WINDOW = next;
      api.bust();
      const w = OBS_WINDOWS.find(o => o.minutes === next);
      announce(`Metrics window ${w ? w.label : next + " minutes"}`);
      render();
    });
  },
};

PAGES.drift = {
  title: "Drift Detection",
  intro: "For one model: how production inputs and predictions differ from the data its serving "
       + "version was trained on. Concept drift is reported only when labels exist.",
  async render(){
    return withModel(async (name, models) => {
      const enc = encodeURIComponent(name);
      const r = await loadAll({ latest:`/api/v1/drift/latest?model=${enc}`,
        reports:`/api/v1/drift?model=${enc}&limit=30` }, 5000);
      const bar = `<div class="pagebar">${modelPicker(models, name)}
        <button class="btn" id="driftscan" data-name="${esc(name)}">${icon("wave",14)} Scan now</button></div>`;
      if(!r.latest.ok) return bar + errorState(r.latest.error, "drift");
      if(!r.latest.data.found) return bar + card("No drift report yet", `<div class="state">
        <div class="big">Not scanned</div>A scan compares this model's logged production traffic with the
        data its serving version was trained on. It needs at least the configured minimum of predictions;
        the scheduler runs one automatically once enough have arrived.</div>`);

      const dr = r.latest.data.report;
      const feats = (dr.feature_drift || []).slice().sort((a,b) => (b.drifted - a.drifted) || (b.score - a.score));
      const summary = `<div class="grid g4" style="margin-bottom:14px">
        ${kpi("Dataset drift", num(dr.dataset_drift_score, 4),
              dr.drift_detected ? badge("detected","bad",true) : badge("stable","ok",true))}
        ${kpi("Prediction drift", dr.prediction_drift_score != null ? num(dr.prediction_drift_score, 4) : NA,
              dr.prediction_drift_score == null ? "not measured"
              : dr.prediction_drift_detected ? badge("shifted","warn",true) : badge("normal","ok",true))}
        ${kpi("Drifted features", `${(dr.drifted_features||[]).length} / ${feats.length}`, `threshold ${num(dr.threshold, 2)}`)}
        ${kpi("Concept drift", runStatusBadge(dr.concept_drift_status || "unavailable"),
              `${int(dr.n_reference)} reference · ${int(dr.n_current)} production rows`)}
      </div>`;

      const conceptCard = card("Concept drift", `
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
          <b>Status</b> ${badge(dr.concept_drift_status || "unavailable",
            dr.concept_drift_status === "measured" ? "info" : "mute")}</div>
        <p style="margin:0;color:var(--ink-2)">${esc(dr.concept_drift_detail || "")}</p>
        ${dr.concept_drift_status !== "measured" ? `<div class="note" style="margin-top:12px">
          Concept drift is a change in P(y|x) — what the inputs <i>mean</i>. It cannot be derived from
          unlabelled inputs, because the inputs themselves need not change at all. This console shows
          <b>no number</b> here rather than a proxy dressed up as a measurement.</div>` : ""}`);

      const featCard = card("Feature drift", barList(feats.map(f => ({
        label: f.feature, value: f.score, bad: f.drifted,
        tag: f.drifted ? badge("drifted","bad") : badge("stable","ok")
      })), { threshold: dr.threshold, empty:"No per-feature results." }),
      { sub:`PSI primary, KS / χ² supporting · threshold ${num(dr.threshold, 2)}` });

      const reports = r.reports.ok ? r.reports.data.reports : [];
      const hist = reports.slice().reverse();
      const histCard = card("Drift score history", hist.length > 1
        ? lineChart(hist.map(x => ({ x: x.created_at, y: x.dataset_drift_score })), { dp:3 })
        : emptyState("At least two scans are needed to plot a trend."), { sub:`${hist.length} scan(s)` });

      const table_ = card("Scan history", table([
        { label:"Scanned", render:x => `${when(x.created_at)} <span class="dim">${ago(x.created_at)}</span>` },
        { label:"Version", render:x => `<span class="mono">v${esc(String(x.model_version ?? "?"))}</span>` },
        { label:"Detected", render:x => x.drift_detected ? badge("yes","bad",true) : badge("no","ok",true) },
        { label:"Dataset score", num:true, render:x => num(x.dataset_drift_score, 4) },
        { label:"Prediction score", num:true, render:x => num(x.prediction_drift_score, 4) },
        { label:"Drifted", num:true, render:x => int((x.drifted_features||[]).length) },
        { label:"Concept", render:x => badge(x.concept_drift_status || "—", x.concept_drift_status === "measured" ? "info" : "mute") },
      ], reports, { empty:"No scans recorded." }), { flush:true });

      return bar + summary + `<div class="grid g2">${featCard}${conceptCard}</div>` + histCard + table_
        + `<div id="actionout"></div>`;
    });
  },
  wire(){
    const b = $("#driftscan");
    if(b) b.onclick = () => runAction({
      title: `Scan ${b.dataset.name} for drift`,
      body: "Compares recent production traffic for this model with its serving version's training "
          + "data, and records the report.",
      confirm: "Scan", needsKey: true,
      path: `/api/v1/drift/scan?model=${encodeURIComponent(b.dataset.name)}`, payload: null,
      success: "Drift scan recorded.",
    });
  },
};

PAGES.retraining = {
  title: "Retraining",
  intro: "For one model: what would trigger retraining now, and what every retraining run learned "
       + "from, produced and decided.",
  async render(){
    return withModel(async (name, models) => {
      const enc = encodeURIComponent(name);
      const r = await loadAll({ trig:`/api/v1/retraining/trigger/evaluate?model=${enc}`,
        events:`/api/v1/retraining?model=${enc}&limit=30` }, 5000);
      const bar = `<div class="pagebar">${modelPicker(models, name)}
        <button class="btn" id="mretrain" data-name="${esc(name)}">${icon("refresh",14)} Retrain now…</button></div>`;

      const t = r.trig.ok ? r.trig.data : null;
      const status = card("Would retraining fire now?", t ? `
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap">
          ${t.should_retrain ? badge("would retrain","warn",true) : badge("no","ok",true)}
          ${t.trigger ? badge(t.trigger,"info") : ""}
          ${t.suppressed_by_cooldown ? badge("in cooldown","mute") : ""}
          ${(t.evidence || {}).blocked_by === "no_new_labelled_data" ? badge("waiting for labels","warn") : ""}
        </div>
        <p style="margin:0 0 12px;color:var(--ink-2)">${esc(t.reason || "")}</p>
        ${(t.checks||[]).length ? `<div class="checklist">${t.checks.map(c =>
          `<div class="crow"><span class="mono">${esc(c.name)}</span>
            <span class="dim">${esc(c.detail || "")}</span>
            ${c.fired ? badge("fired","warn") : badge("quiet","mute")}</div>`).join("")}</div>` : ""}`
        : unavailable("The trigger evaluation could not be read."), { sub:"evaluated now, no side effects" });

      const how = card("How a retraining run decides", `<div class="flow">
        ${["New labelled data","Trigger","Validation","Training","Shared-holdout comparison",
           "Approval gate","Approval or deployment"].map((s,i,a) =>
          `<span class="step">${s}</span>` + (i<a.length-1 ? '<span class="arw">→</span>' : "")).join("")}
        </div>
        <ul class="plain" style="margin:12px 0 0;font-size:12.5px">
          <li>Trains on the serving version's data, every labelled production row, and optionally a new
            dataset version. Nothing is generated.</li>
          <li>Does not run on a trigger without new labelled data — retraining on the same data cannot
            fix drift — and never inside the cooldown.</li>
          <li>The candidate must clear the gate <b>and</b> beat the live version on held-out rows neither
            trained on; otherwise it is rejected and production is untouched.</li>
        </ul>`);

      const events = r.events.ok ? r.events.data : null;
      const list = card("Retraining runs", events === null ? unavailable("History unavailable.") : table([
        { label:"Started", render:x => when(x.created_at) },
        { label:"Trigger", render:x => badge(x.trigger || "—","info") },
        { label:"Status", render:x => runStatusBadge(x.status) },
        { label:"Decision", render:x => x.decision ? runStatusBadge(x.decision) : NA },
        { label:"From", render:x => x.baseline_version != null
            ? `<a class="mono" href="#/models/${enc}/${x.baseline_version}">v${x.baseline_version}</a>` : NA },
        { label:"Candidate", render:x => x.candidate_version != null
            ? `<a class="mono" href="#/models/${enc}/${x.candidate_version}?tab=evaluation">v${x.candidate_version}</a>` : NA },
        { label:"Learned from", render:x => {
            const s = (x.detail || {}).data_sources;
            return s ? `<span class="dim">${int(s.base_rows)} base${s.new_dataset_rows ? ` + ${int(s.new_dataset_rows)} new`
              : ""} + ${int(s.labelled_production_rows)} labelled</span>` : NA; } },
        { label:"Reason", render:x => `<span class="dim">${esc(String(x.reason || "").slice(0, 110))}</span>` },
      ], events, { empty:"No retraining runs for this model yet." }), { flush:true });

      return bar + `<div id="actionout"></div>` + status + how + list;
    });
  },
  wire(){ wireModelActions(); },
};
