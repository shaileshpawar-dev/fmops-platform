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
function auditTable(rows){
  return card("Audit trail", table([
    { label:"Timestamp", render:e => when(e.created_at || e.timestamp) },
    { label:"Actor", render:e => `<span class="mono">${esc(e.actor||"system")}</span>` },
    { label:"Action", render:e => badge(e.action||"—","info") },
    { label:"Resource", render:e => `<span class="mono">${esc(e.resource||"—")}</span>` },
    { label:"Status", render:e => e.status ?
        badge(e.status, /ok|success|200/i.test(String(e.status))?"ok":"bad") : NA },
    { label:"Detail", render:e => `<span class="dim">${esc(String(e.detail||e.details||"").slice(0,140))}</span>` },
  ], rows, { empty:"No audit entries recorded." }), { flush:true });
}

PAGES.system = {
  title: "System Health",
  intro: "Process health, readiness, and the effective platform configuration. Secrets are never rendered here.",
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
      return card("Infrastructure", `<div style="display:grid;gap:9px">${rows.map(([k,b,note]) =>
        `<div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
          <span>${esc(k)}</span><span style="display:flex;gap:9px;align-items:center;text-align:right">
          <span class="dim" style="font-size:11.5px">${note}</span>${b}</span></div>`).join("")}</div>
        <div class="note" style="margin-top:12px">Status is read from the running service's own
        configuration. It reflects what the application is set to use — it is not an independent
        probe of each AWS service.</div>`);
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
