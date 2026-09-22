/* Deployment and experiment surfaces.
 *
 * The model registry that used to live here is now pages/model.js: the version
 * is the primary object and it earned its own module. What remains is the
 * experiment tracker view, the Deployment Room, and the Quality Gates policy
 * page that absorbed the old Champion / Challenger comparison.
 */

PAGES.experiments = {
  title: "Experiments",
  intro: "Training runs recorded by the experiment tracker, with the parameters and metrics logged for each.",
  async render(){
    const r = await loadAll({ ex:"/api/v1/experiments", runs:"/api/v1/experiments/runs" }, 10000);
    const exp = sect(r.ex, d => {
      const list = d.experiments || [];
      return card("Experiments", table([
        { label:"Name", render:e => `<b class="mono">${esc(e.name || e.experiment_id || "—")}</b>` },
        { label:"ID", render:e => `<span class="mono dim">${esc(e.experiment_id || "—")}</span>` },
        { label:"Runs", num:true, render:e => int(e.run_count) },
        { label:"Backend", render:() => badge(d.backend || "—","info") },
      ], list, { empty:"No experiments recorded." }), { flush:true });
    }, "experiments");

    const runs = sect(r.runs, d => {
      const list = d.runs || [];
      return card("Runs", table([
        { label:"Run", render:x => `<span class="mono">${esc(String(x.run_id||"").slice(0,16))}</span>` },
        { label:"Status", render:x => badge(x.status || "—", x.status==="FINISHED"?"ok":"mute") },
        { label:"Algorithm", render:x => `<span class="mono">${esc((x.params||{}).algorithm || "—")}</span>` },
        { label:"ROC-AUC", num:true, render:x => num((x.metrics||{}).roc_auc) },
        { label:"F1", num:true, render:x => num((x.metrics||{}).f1) },
        { label:"Started", render:x => when(x.start_time) },
      ], list, { empty:"No runs recorded yet. Runs appear here after the training pipeline executes." }),
      { flush:true, sub: `${d.count || list.length} run(s) · experiment ${esc(d.experiment||"—")}` });
    }, "runs");

    return exp + runs + `<div class="note">The tracker backend in this deployment is
      <b>local (SQLite)</b>. A shared MLflow tracking server is supported by configuration but is
      not part of this deployment.</div>`;
  }
};

/* Deployment Room -- deployment engineering made visible.
 *
 * Traffic routing, health evidence and rollback lineage in one place rather
 * than a table of deployment ids. Every strategy the backend supports gets its
 * own traffic rendering, because "how traffic moves" is the thing a rollout
 * view exists to show.
 */

/* Traffic lanes. The weights are the backend's, not a guess: deployment.traffic
   is the routing table the API actually enforces. A shadow version is drawn in
   outline because it mirrors traffic without serving any of it. */
function trafficLanes(dep){
  const traffic = dep.traffic || {};
  const keys = Object.keys(traffic);
  const lanes = [];
  keys.sort((a,b) => (Number(traffic[b])||0) - (Number(traffic[a])||0)).forEach(v => {
    const pctv = Number(traffic[v]) || 0;
    const isPrev = String(v) === String(dep.previous_version);
    lanes.push(`<div class="lane">
      <span class="vn">v${esc(v)}</span>
      <div class="track"><div class="fill ${isPrev ? "prev" : ""}" style="width:${pctv}%"></div></div>
      <span class="pctv">${pctv.toFixed(0)}%</span></div>`);
  });
  if(dep.shadow_version != null) lanes.push(`<div class="lane shadow">
      <span class="vn">v${esc(dep.shadow_version)}</span>
      <div class="track"><div class="fill" style="width:100%"></div></div>
      <span class="pctv">mirror</span></div>`);
  if(!lanes.length) return unavailable("No routing table has been published for this endpoint.");
  return `<div class="traffic">${lanes.join("")}</div>`;
}

function strategyNote(strat, dep){
  const s = String(strat || "").toLowerCase();
  if(s === "canary") return "Traffic moves in configured steps, and each step must clear its "
    + "evidence thresholds from the persisted inference log before the next one begins.";
  if(s === "blue_green") return "One version serves all traffic; the previous version stays "
    + "registered so a rollback is a routing change rather than a redeploy.";
  if(s === "shadow") return "The shadow version receives mirrored traffic and its responses are "
    + "discarded. It never serves a user request.";
  if(s === "direct") return "Traffic is switched in one step with no intermediate state.";
  return "No strategy recorded for this deployment.";
}

PAGES.deployments = {
  title: "Deployment Room",
  intro: "For one model: what is serving, how traffic is routed, what the endpoint's health "
       + "checks say, and what a rollback would target.",
  refresh: 30000,
  async render(){
    return withModel(async (name, models) => {
      const enc = encodeURIComponent(name);
      const r = await loadAll({
        list:    `/api/v1/deployments?model=${enc}`,
        current: `/api/v1/deployments/current?model=${enc}`,
        health:  `/api/v1/deployments/health?model=${enc}`,
        jobs:    `/api/v1/jobs?kind=deployment&model=${enc}&limit=5`,
      }, 5000);
      const picker = `<div class="pagebar">${modelPicker(models, name)}</div>`;
      const cur = r.current.ok ? r.current.data : null;
      const dep = cur && cur.deployment ? cur.deployment : null;
      const cols = deployCols(name);
      const running = r.jobs.ok ? r.jobs.data.jobs.filter(j => !JOB_TERMINAL.has(j.status)) : [];
      const inflight = running.length ? card("Deployments in progress", running.map(j =>
        `<div class="lnrow">${runStatusBadge(j.status)} <a class="mono" href="#/jobs/${esc(j.id)}">${esc(j.id)}</a>
          <span class="dim">v${esc(String((j.payload.request || {}).model_version || "?"))} · ${
          esc((j.payload.request || {}).strategy || "")} · queued ${esc(ago(j.created_at))}</span></div>`).join(""))
        : "";

      if(!dep) return picker + inflight + card(`${name} endpoint`,
        unavailable(cur ? (cur.detail || "No deployment has been created for this endpoint.")
          + " Approve a version into Production, then deploy it from its model page. (A Staging"
          + " version may only be shadowed once something is live.)"
          : "The deployments API could not be reached."))
        + card("Deployment history", r.list.ok
          ? table(cols, r.list.data, { empty:"No deployments recorded." })
          : unavailable("History unavailable."), { flush:true });

      const head = `<div class="mhead"><div class="top">
          <div style="min-width:0">
            <h1>${esc(dep.endpoint_name || "endpoint")}</h1>
            <div style="display:flex;gap:9px;align-items:center;margin-top:8px;flex-wrap:wrap">
              ${runStatusBadge(dep.state)}
              ${badge(dep.strategy || "no strategy", "mute")}
              ${badge((dep.provider || "provider") + " provider", "mute")}
            </div>
            <div class="idl">
              <span>model <b><a href="#/models/${enc}">${esc(dep.model_name || name)}</a></b></span>
              <span>serving <b>v${esc(String(dep.current_version ?? "—"))}</b></span>
              <span>previous <b>${dep.previous_version != null
                ? "v" + esc(String(dep.previous_version)) : "none"}</b></span>
              <span>updated <b>${esc(String(dep.updated_at || dep.created_at || "")
                .slice(0,19).replace("T"," "))}</b></span>
            </div>
          </div>
          <span class="spacer"></span>
          ${dep.current_version != null ? `<a class="btn" href="#/models/${enc}/${esc(String(dep.current_version))}">
            Open v${esc(String(dep.current_version))}</a>` : ""}
          <a class="btn" href="#/predict?model=${enc}">${icon("target",14)} Predict</a>
        </div><div id="actionout"></div></div>`;

      const traffic = card("Traffic", trafficLanes(dep)
        + `<p class="dim" style="margin:13px 0 0;font-size:12.5px">${esc(strategyNote(dep.strategy, dep))}</p>`,
        { sub:`${esc(dep.strategy || "—")}` });

      const h = r.health.ok ? r.health.data : null;
      const health = card("Health evidence", h ? `
        <div class="checks">${Object.entries(h.checks || {}).map(([k,v]) =>
          `<span class="c ${v ? "" : "no"}">${v ? "✓" : "✕"} ${esc(k)}</span>`).join("")
          || `<span class="dim">No named checks reported.</span>`}</div>
        <div class="grid g4" style="margin-top:15px">
          ${kpi("Status", badge(h.status || "unknown",
            h.status === "healthy" ? "ok" : h.status === "degraded" ? "warn" : "bad"))}
          ${kpi("p95 latency", ms(h.latency_p95_ms))}
          ${kpi("Error rate", pct(h.error_rate, 2))}
          ${kpi("Requests", int(h.request_count), "in the health window")}
        </div>
        ${h.detail ? `<p class="dim" style="margin:12px 0 0;font-size:12.5px">${esc(h.detail)}</p>` : ""}`
        : unavailable("The endpoint health API could not be reached."),
        { sub: h ? `checked ${String(h.checked_at||"").slice(11,19)}` : "" });

      const events = dep.events || [];
      const timeline = card("Deployment timeline", events.length
        ? `<div class="tl">${events.map(e => {
            const bad = /fail|error|rollback|interrupt/i.test(String(e.event || ""));
            return `<div class="ev ${bad ? "bad" : "done"}"><span class="pip"></span>
              <div><div class="when">${when(e.created_at)}</div>
                <div class="what">${esc(String(e.event || "").replace(/[._]/g," "))}</div>
                <div class="det">${esc(JSON.stringify(e.detail || {}).slice(0,200))}</div>
              </div></div>`;
          }).join("")}</div>`
        : emptyState("This deployment recorded no events."),
        { sub:`${events.length} event(s)` });

      /* The API resolves a target: explicit version, else the recorded previous,
         else the most recently archived production version. Naming which applies
         beats failing after the click. */
      const target = dep.previous_version;
      const rollback = card("Rollback", target != null ? `
        <div class="grid g3">
          ${kpi("Would restore", `<span class="mono">v${int(target)}</span>`, "recorded previous version")}
          ${kpi("From", `<span class="mono">v${int(dep.current_version)}</span>`, "currently serving")}
          ${kpi("Already rolled back", boolBadge(dep.rolled_back, "yes", "no", true))}
        </div>
        <div style="margin-top:14px">
          <button class="btn danger" id="rollback" data-name="${esc(name)}" data-to="${esc(String(target))}">
            Roll back to v${int(target)}</button></div>`
        : `<div class="note warn"><b>No rollback target.</b><br>
          <span style="font-size:12.5px">This endpoint has no recorded previous version. The API
          would fall back to the most recently archived production version, and fail if none
          exists — so the control stays disabled rather than failing after the click.</span></div>
        <div style="margin-top:12px"><button class="btn" disabled>Roll back</button></div>`,
        { sub:"POST /api/v1/deployments/rollback" });

      const hist = sect(r.list, list => card("All deployments of " + name,
        table(cols, list, { empty:"No deployments recorded.",
          rowClass:x => x.state === "live" ? "win" : "" }),
        { flush:true, sub:`${list.length} record(s)` }), "deployments");

      const provider = `<div class="note"><b>Provider is
        <span class="mono">${esc(dep.provider || "local")}</span>.</b>
        <span style="font-size:12.5px">Routing is enforced in-process by the API: a canary split
        genuinely routes that share of predictions. It does not provision infrastructure — that is
        the SageMaker provider, which is not what runs here.</span></div>`;

      return picker + head + inflight
        + sect2("01", "Routing", "what is serving and how")
        + `<div class="grid g2">${traffic}${health}</div>`
        + sect2("02", "History", "events and rollback")
        + `<div class="grid g2">${timeline}${rollback}</div>`
        + hist + provider;
    });
  },
  wire(){
    const b = $("#rollback");
    if(b) b.onclick = () => runAction({
      title: `Roll back ${b.dataset.name} to v${b.dataset.to}`,
      body: "Restores the previous version on this model's endpoint and records why. "
          + "The version rolled back from stays registered.",
      extra: `<label for="rbwhy" style="font-size:11.5px;color:var(--ink-3)">Reason (recorded)</label>
        <input id="rbwhy" data-field="reason" maxlength="300" value="manual rollback from the console">`,
      confirm: "Roll back", danger: true, needsKey: true,
      path: "/api/v1/deployments/rollback",
      payload: f => ({ model_name: b.dataset.name, reason: f.reason || "manual rollback" }),
      success: "Rolled back.",
    });
  },
};

function deployCols(name){
  const enc = encodeURIComponent(name);
  return [
    { label:"ID", render:x => `<span class="mono">${esc(String(x.id).slice(0,14))}</span>` },
    { label:"Version", render:x => x.current_version != null
        ? `<a class="mono" href="#/models/${enc}/${esc(String(x.current_version))}"><b>v${esc(String(x.current_version))}</b></a>` : NA },
    { label:"Strategy", render:x => badge(x.strategy || "—","mute") },
    { label:"State", render:x => runStatusBadge(x.state) },
    { label:"Previous", render:x => x.previous_version != null
        ? `<span class="mono dim">v${esc(String(x.previous_version))}</span>` : NA },
    { label:"Message", render:x => `<span class="dim">${esc(String(x.message || "").slice(0, 90))}</span>` },
    { label:"Started", render:x => when(x.created_at) },
  ];
}

/* ==========================================================================
 * Approvals & Gates
 *
 * The configured policy, the versions waiting for a human, and every gate
 * decision the platform has recorded -- automated verdicts and sign-offs
 * alike, with the checks, the comparison basis and who decided.
 * ========================================================================== */
PAGES.gates = {
  title: "Approvals & Gates",
  intro: "The policy every model must clear, the versions waiting for a human decision, "
       + "and the recorded history of every gate decision.",
  async render(){
    const r = await loadAll({
      cfg: "/api/v1/config",
      pending: "/api/v1/models/pending",
      decisions: "/api/v1/models/decisions?limit=60",
    }, 8000);
    if(!r.cfg.ok) return errorState(r.cfg.error, "gates");
    const ap = r.cfg.data.approval || {};
    const gated = t => typeof ap[t] === "number" && ap[t] > 0;
    const rows = [["F1","min_f1"],["ROC-AUC","min_roc_auc"],["Precision","min_precision"],["Recall","min_recall"]];

    const policy = card("Policy", `
      <div class="grid g4">${rows.map(([label, t]) =>
        kpi(label, gated(t) ? `≥ ${num(ap[t], 2)}` : `<span class="dim">no minimum</span>`)).join("")}</div>
      <div class="grid g4" style="margin-top:14px">
        ${kpi("p95 latency", ap.max_inference_latency_ms != null ? `≤ ${num(ap.max_inference_latency_ms, 0)} ms` : NA)}
        ${kpi("Manual approval", boolBadge(ap.require_manual_approval, "required", "not required"),
          ap.require_manual_approval ? "a human signs every promotion" : "")}
        ${kpi("Beat production by", `<span class="mono">${esc(String(ap.min_improvement ?? "—"))}</span>`,
          `on ${esc(ap.comparison_metric || "the comparison metric")}`)}
        ${kpi("Clean validation", boolBadge(ap.require_clean_validation, "required", "optional"))}
      </div>
      <p class="dim" style="margin:14px 0 0;font-size:12.5px;max-width:84ch">A threshold of zero gates
        nothing, so it is shown as "no minimum" rather than a passed check. Thresholds come from
        configuration and are never relaxed to let a candidate through. Champion and challenger are
        re-scored on the same held-out rows before a promotion is allowed.</p>`,
      { sub:"from platform configuration" });

    const pending = r.pending.ok ? r.pending.data.versions : null;
    const waiting = card("Waiting for a decision", pending === null
      ? unavailable("Pending approvals could not be read.")
      : pending.length ? table([
          { label:"Model", render:v => `<a href="#/models/${encodeURIComponent(v.name)}"><b>${esc(v.name)}</b></a>` },
          { label:"Version", render:v => `<a class="mono" href="#/models/${encodeURIComponent(v.name)}/${v.version}?tab=evaluation">v${int(v.version)}</a>` },
          { label:"Algorithm", render:v => `<span class="mono">${esc(v.algorithm || "-")}</span>` },
          { label:"ROC-AUC", num:true, render:v => num((v.metrics||{}).roc_auc, 4) },
          { label:"F1", num:true, render:v => num((v.metrics||{}).f1, 4) },
          { label:"Registered", render:v => when(v.created_at) },
          { label:"", render:v => `<a class="btn sm" href="#/models/${encodeURIComponent(v.name)}/${v.version}?tab=evaluation">Review</a>` },
        ], pending, { empty:"" })
        : emptyState("Nothing is waiting. Versions that clear every automated check appear here "
          + "when this environment requires a human to approve them."),
      { flush:!!(pending && pending.length), sub: pending ? `${pending.length} version(s)` : "" });

    const decisions = r.decisions.ok ? r.decisions.data.decisions : null;
    const history = card("Gate decisions", decisions === null
      ? unavailable("The decision record could not be read.")
      : decisions.length ? table([
          { label:"When", render:d => when(d.created_at) },
          { label:"Model", render:d => `<a href="#/models/${encodeURIComponent(d.model_name)}">${esc(d.model_name)}</a>` },
          { label:"Version", render:d => `<a class="mono" href="#/models/${encodeURIComponent(d.model_name)}/${d.model_version}?tab=evaluation">v${int(d.model_version)}</a>` },
          { label:"Decision", render:d => runStatusBadge(d.decision) },
          { label:"By", render:d => `${esc(d.source)} <span class="dim">${esc(d.actor || "")}</span>` },
          { label:"Compared on", render:d => d.comparison && d.comparison.basis
              ? badge(String(d.comparison.basis).replace(/_/g," "), d.comparison.basis === "shared_holdout" ? "info" : "mute") : NA },
          { label:"Reason", render:d => `<span class="dim">${esc(String(d.reason || "").slice(0, 140))}</span>` },
        ], decisions, { id:"gate-decisions", filter:"Filter decisions", empty:"" })
        : emptyState("No gate decision has been recorded yet."),
      { flush:!!(decisions && decisions.length), sub:"recorded by /api/v1/models/decisions" });

    return policy + waiting + history;
  }
};
