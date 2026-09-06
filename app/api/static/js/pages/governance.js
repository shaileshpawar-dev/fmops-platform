PAGES.audit = {
  title: "Audit Log",
  intro: "Recorded platform actions: who did what, to which resource, and whether it succeeded.",
  async render(){
    const d = await api.get("/api/v1/audit", 6000);
    const entries = d.entries || [];
    const actions = [...new Set(entries.map(e => e.action).filter(Boolean))].sort();
    const filter = `<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px">
      <input type="text" id="auditq" placeholder="Filter by actor, action, resource or detail…"
        style="max-width:340px">
      <select id="auditact" style="max-width:200px"><option value="">All actions</option>
        ${actions.map(a => `<option>${esc(a)}</option>`).join("")}</select>
      <span class="dim" id="auditcount">${entries.length} entr${entries.length===1?"y":"ies"}</span></div>`;
    window.__audit = entries;
    return filter + `<div id="audittbl">${auditTable(entries)}</div>`;
  }
};
/* The Resource and Status columns read `e.resource` and `e.status`, which this
   API has never returned -- both rendered empty on every row since the page was
   written. The audit payload carries resource_type/resource_id and `outcome`. */
function auditTable(rows){
  return card("Audit trail", table([
    { label:"Timestamp", sort:e => e.created_at, render:e => when(e.created_at) },
    { label:"Actor", sort:e => e.actor,
      render:e => `<span class="mono">${esc(e.actor || "system")}</span>` },
    { label:"Action", sort:e => e.action,
      render:e => `<span class="mono">${esc(String(e.action || "—"))}</span>` },
    { label:"Resource", sort:e => e.resource_type, render:e => {
        if(!e.resource_type && !e.resource_id) return NA;
        const id = String(e.resource_id || "");
        return `<span class="dim">${esc(e.resource_type || "")}</span>
          ${id ? " " + copyable(id, id.length > 22 ? id.slice(0,22) + "…" : id,
                                { label:"resource id" }) : ""}`;
      } },
    { label:"Outcome", sort:e => e.outcome, render:e => e.outcome
        ? badge(e.outcome, /success|ok/i.test(String(e.outcome)) ? "ok" : "bad") : NA },
    { label:"Detail", render:e => {
        const d = e.detail;
        if(d == null || d === "") return NA;
        const t = typeof d === "string" ? d : JSON.stringify(d);
        return `<span class="dim" title="${esc(t)}">${esc(t.slice(0,120))}${
          t.length > 120 ? "…" : ""}</span>`;
      } },
  /* No in-table filter here: the page already has a text search and an action
     dropdown above it, and two search boxes on one screen is worse than one. */
  ], rows, { id:"audit", sortKey:"Timestamp", sortDir:"desc",
             empty:"No audit entries recorded." }), { flush:true });
}

/* Runtime -- deliberately not called Infrastructure.
 *
 * This page reads the PROCESS, not the cloud. There is no AWS introspection
 * API in this platform: what follows is the running service's own health,
 * readiness, resource usage and effective configuration. Where deployment
 * topology appears it is labelled as documented architecture, because it is
 * written down rather than queried.
 */
PAGES.runtime = {
  title: "Runtime",
  intro: "Health, readiness, resource usage and effective configuration of the running "
       + "process. This reads the service, not the cloud — there is no AWS introspection here.",
  async render(){
    const r = await loadAll({ health:"/health", ready:"/health/ready", cfg:"/api/v1/config",
      res:"/api/v1/monitoring/resources" }, 5000);

    const health = sect(r.health, d => card("API health", `
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:12px">
        ${badge(d.status||"?", d.status==="healthy"?"ok":d.status==="degraded"?"warn":"bad", true)}
        <span class="dim mono">${esc(d.service||"")} ${esc(d.version||"")}</span></div>
      <div style="display:grid;gap:6px">${Object.entries(d.components||{}).map(([k,v]) =>
        `<div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
          <span class="mono">${esc(k)}</span>
          <span style="display:flex;gap:8px;align-items:center">
            <span class="dim" style="font-size:11.5px">${esc(
              Object.entries(v||{}).filter(([kk])=>kk!=="status")
                .map(([kk,vv])=>`${kk}=${typeof vv==="object"?"…":vv}`).join(" · ").slice(0,90))}</span>
            ${badge((v&&v.status)||"?", (v&&v.status)==="ok"?"ok":"bad")}</span></div>`).join("")}</div>`),
      "health");

    const ready = sect(r.ready, d => card("Readiness", `
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:12px">
        ${badge(d.status||"?", d.status==="ready"?"ok":"warn", true)}</div>
      <div style="display:grid;gap:6px">${Object.entries(d.checks||{}).map(([k,v]) =>
        `<div style="display:flex;justify-content:space-between"><span class="mono">${esc(k)}</span>
         ${boolBadge(v,"ok","not ready")}</div>`).join("")}</div>
      ${d.detail?`<div class="dim" style="margin-top:10px;font-size:11.5px">${esc(
        Object.entries(d.detail).map(([k,v])=>`${k}: ${v}`).join(" · "))}</div>`:""}`), "readiness");

    /* Only a safe, explicit allowlist of configuration is rendered. */
    const cfg = sect(r.cfg, d => {
      const safe = [
        ["Environment", d.environment], ["Service", d.service_name], ["Version", d.version],
        ["Log level", d.log_level], ["Log format", d.log_format], ["Git commit", d.git_commit],
        ["Deployment provider", (d.deployment||{}).provider],
        ["Deployment strategy", (d.deployment||{}).strategy],
        ["Registry backend", (d.tracking||{}).registry_backend],
        ["Tracking backend", (d.tracking||{}).backend],
        ["Drift engine", (d.drift||{}).engine],
        ["LLM provider", (d.llm||{}).provider], ["LLM model", (d.llm||{}).model],
        ["AWS enabled", String((d.aws||{}).enabled)], ["AWS region", (d.aws||{}).region],
        ["S3 bucket", (d.aws||{}).s3_bucket],
        ["Auth backend", (d.security||{}).auth_backend],
        ["CloudWatch", String((d.monitoring||{}).cloudwatch_enabled)],
      ].filter(([,v]) => has(v));
      return card("Platform configuration", `<dl class="kv">${safe.map(([k,v]) =>
        `<dt>${esc(k)}</dt><dd class="mono">${esc(v)}</dd>`).join("")}</dl>
        <div class="note" style="margin-top:12px">This view renders an explicit allowlist of
        non-sensitive settings. API keys, AWS credentials and connection strings are never sent to
        the browser.</div>`);
    }, "config");

    const infra = sect(r.cfg, d => {
      const aws = d.aws || {};
      const rows = [
        ["AWS integration", aws.enabled ? badge("enabled","ok",true) : badge("disabled","mute"),
          aws.enabled ? `region ${esc(aws.region||"—")}` : "adapters present, not active"],
        ["S3 artifact bucket", aws.s3_bucket ? badge("configured","info") : badge("not configured","mute"),
          aws.s3_bucket ? `<span class="mono">${esc(aws.s3_bucket)}</span>` : ""],
        ["SageMaker endpoint", badge("not deployed","mute"),
          "the deployment provider in this environment is the in-process router"],
        ["CloudWatch metrics", (d.monitoring||{}).cloudwatch_enabled ?
          badge("enabled","ok") : badge("disabled","mute"),
          "container logs ship via the ECS awslogs driver regardless"],
      ];
      return card("Documented architecture", `<div style="display:grid;gap:9px">${rows.map(([k,b,note]) =>
        `<div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
          <span>${esc(k)}</span><span style="display:flex;gap:9px;align-items:center;text-align:right">
          <span class="dim" style="font-size:11.5px">${note}</span>${b}</span></div>`).join("")}</div>
        <div class="note warn" style="margin-top:12px"><b>Documented, not probed.</b>
        Every row above is read from the running service&rsquo;s own configuration: it says what
        the application is <i>set to use</i>. It is not a live read of AWS. Nothing on this page
        queries an AWS control-plane API, and the console has no cloud introspection.</div>`);
    }, "config");

    const res = sect(r.res, d => card("Process resources", `<div class="grid g4">
      ${kpi("CPU", has(d.cpu_percent)?`<span class="mono">${d.cpu_percent.toFixed(1)}%</span>`:NA)}
      ${kpi("Memory", has(d.memory_percent)?`<span class="mono">${d.memory_percent.toFixed(1)}%</span>`:NA)}
      ${kpi("RSS", has(d.process_rss_mb)?`<span class="mono">${d.process_rss_mb.toFixed(0)} MB</span>`:NA)}
      ${kpi("GPU", d.gpu_available?badge("present","ok"):badge("none","mute"))}
    </div>`), "resources");

    return `<div class="grid g2">${health}${ready}</div>` + infra + cfg + res;
  }
};

/* ------------------------------------------------------------- routes --- */

/* Incidents -- an attention queue, not an incident-management system.
 *
 * This is GET /api/v1/alerts. The backend can raise an alert and acknowledge
 * it, and that is the whole lifecycle: there is no assignment, no severity
 * workflow, no resolution note, no on-call routing. The page is written to
 * that shape rather than implying a product that does not exist here.
 */
PAGES.incidents = {
  title: "Incidents",
  intro: "Open alerts raised by the platform's own watchdogs — SLO breaches, drift scans and "
       + "retraining triggers. Raise and acknowledge is the full lifecycle this backend supports.",
  refresh: 30000,
  async render(){
    const r = await loadAll({
      alerts: "/api/v1/alerts?limit=100",
      dash:   "/api/v1/dashboard",
      rt:     "/api/v1/retraining/trigger/evaluate",
    }, 5000);

    if(!r.alerts.ok) return errorState(r.alerts.error, "incidents");
    const alerts = r.alerts.data || [];
    const open = alerts.filter(a => !a.acknowledged);
    const ack = alerts.filter(a => a.acknowledged);
    const d = r.dash.ok ? r.dash.data : {};
    const sys = d.system || {}, drift = d.drift || {};

    /* What is being watched, and what each watcher currently says. This is the
       useful half of an empty queue: "nothing is open" means little without
       naming the things that would open one. */
    const watchers = [
      ["Latency SLO", sys.available,
        sys.latency_slo_met === false ? "breached" : "within SLO",
        sys.latency_slo_met === false ? "bad" : "ok",
        sys.slo_latency_ms != null ? `p95 ≤ ${sys.slo_latency_ms} ms` : ""],
      ["Error-rate SLO", sys.available,
        sys.error_slo_met === false ? "breached" : "within SLO",
        sys.error_slo_met === false ? "bad" : "ok",
        sys.slo_error_rate != null ? `≤ ${(sys.slo_error_rate*100).toFixed(2)}%` : ""],
      ["Drift", drift.available,
        drift.drift_detected ? "drift detected" : "stable",
        drift.drift_detected ? "warn" : "ok",
        drift.available ? `${(drift.drifted_features||[]).length} feature(s) drifted` : "no scan yet"],
      ["Retraining trigger", r.rt.ok,
        r.rt.ok ? (r.rt.data.should_retrain ? "would fire" : "not due") : "unknown",
        r.rt.ok ? (r.rt.data.should_retrain ? "warn" : "ok") : "mute",
        r.rt.ok ? esc(String(r.rt.data.reason || "").slice(0, 70)) : ""],
    ];

    const watch = card("What is being watched", `
      <div style="display:grid;gap:9px">${watchers.map(([name, avail, state, kind, note]) =>
        `<div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
          <span style="font-weight:500">${esc(name)}</span>
          <span style="display:flex;gap:10px;align-items:center;text-align:right">
            <span class="dim" style="font-size:11.5px">${note}</span>
            ${avail ? badge(state, kind) : badge("unavailable","mute")}</span></div>`).join("")}</div>
      <p class="dim" style="margin:13px 0 0;font-size:12px">Each of these can raise an alert into
        the queue. A watcher reporting <span class="mono">unavailable</span> is not reporting
        healthy — it means the endpoint that owns it did not answer.</p>`,
      { sub:"alert sources" });

    const queue = card("Open", attentionQueue(open, { limit:50 }),
      { flush:true, sub:`${open.length} open`,
        right: open.length
          ? `<button class="btn" data-act="ackall">Acknowledge all</button>` : "" });

    const history = card("Acknowledged", ack.length
      ? table([
          { label:"Severity", render:a => badge(a.severity || "info",
              /crit|error/i.test(a.severity || "") ? "bad"
              : /warn/i.test(a.severity || "") ? "warn" : "info") },
          { label:"Alert", render:a => `<b>${esc(a.title || "Alert")}</b>` },
          { label:"Message", render:a => esc(String(a.message || "").slice(0, 120)) },
          { label:"Category", render:a => `<span class="mono dim">${esc(a.category || "-")}</span>` },
          { label:"Raised", render:a => when(a.created_at) },
        ], ack, { empty:"" })
      : emptyState("No alerts have been acknowledged."),
      { flush:true, sub:`${ack.length} acknowledged` });

    const scope = `<div class="note"><b>Scope.</b>
      <span style="font-size:12.5px">This is the alerts API. The platform can raise an alert and
      record that someone acknowledged it — there is no assignment, no severity escalation, no
      resolution note and no paging integration. Calling the page Incidents describes what the
      queue is <i>for</i>, not a workflow the backend implements.</span></div>`;

    return sect2("01", "Attention queue", `${open.length} open · ${ack.length} acknowledged`)
      + queue
      + sect2("02", "Sources", "what can raise an alert")
      + watch + history + scope;
  },
  wire(){
    const all = document.querySelector('[data-act="ackall"]');
    if(all) all.onclick = () => runAction({
      title: "Acknowledge every open alert",
      body: "Marks all open alerts as acknowledged. They stay in the history.",
      confirm: "Acknowledge all", needsKey: true,
      path: "/api/v1/alerts/acknowledge-all",
      success: "All alerts acknowledged.",
    });
  }
};
