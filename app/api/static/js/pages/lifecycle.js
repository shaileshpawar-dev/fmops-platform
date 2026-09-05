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
  intro: "What is serving, how traffic is routed, what the endpoint's health checks say, "
       + "and what a rollback would target.",
  refresh: 30000,
  async render(){
    const r = await loadAll({
      list:    "/api/v1/deployments",
      current: "/api/v1/deployments/current",
      health:  "/api/v1/deployments/health",
      dash:    "/api/v1/dashboard",
    }, 5000);

    const cur = r.current.ok ? r.current.data : null;
    const dep = cur && cur.deployment ? cur.deployment : null;
    const svc = (r.dash.ok ? r.dash.data.service : {}) || {};

    if(!dep) return card("Endpoint",
      unavailable(cur ? (cur.detail || "No deployment has been created for this endpoint.")
        : "The deployments API could not be reached."), { flush:true })
      + card("Deployment history", r.list.ok
        ? table(DEPLOY_COLS, r.list.data, { empty:"No deployments recorded." })
        : unavailable("History unavailable."), { flush:true });

    /* -- header ------------------------------------------------------- */
    const head = `<div class="mhead">
      <div class="top">
        <div style="min-width:0">
          <h1>${esc(dep.endpoint_name || "endpoint")}</h1>
          <div style="display:flex;gap:9px;align-items:center;margin-top:8px;flex-wrap:wrap">
            ${badge(dep.state || "unknown", dep.state === "live" ? "ok" : "info", dep.state === "live")}
            ${badge(dep.strategy || "no strategy", "mute")}
            ${badge((dep.provider || "provider") + " provider", "mute")}
          </div>
          <div class="idl">
            <span>serving <b>v${esc(String(dep.current_version ?? "—"))}</b></span>
            <span>previous <b>${dep.previous_version != null
              ? "v" + esc(String(dep.previous_version)) : "none"}</b></span>
            <span>model <b>${esc(dep.model_name || "-")}</b></span>
            <span>updated <b>${esc(String(dep.updated_at || dep.created_at || "")
              .slice(0,19).replace("T"," "))}</b></span>
          </div>
        </div>
        <span class="spacer"></span>
        ${dep.current_version != null
          ? `<a class="btn" href="#/models/${esc(String(dep.current_version))}">Open v${esc(String(dep.current_version))}</a>` : ""}
      </div>
    </div>`;

    /* -- traffic ------------------------------------------------------ */
    const traffic = card("Traffic", trafficLanes(dep)
      + `<p class="dim" style="margin:13px 0 0;font-size:12.5px">${esc(strategyNote(dep.strategy, dep))}</p>`,
      { sub:`${esc(dep.strategy || "—")}` });

    /* -- health evidence ---------------------------------------------- */
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

    /* -- timeline ----------------------------------------------------- */
    const events = dep.events || [];
    const timeline = card("Deployment timeline", events.length
      ? `<div class="tl">${events.map(e => {
          const bad = /fail|error|rollback/i.test(String(e.event || ""));
          return `<div class="ev ${bad ? "bad" : "done"}"><span class="pip"></span>
            <div><div class="when">${when(e.created_at)}</div>
              <div class="what">${esc(String(e.event || "").replace(/_/g," "))}</div>
              <div class="det">${esc(JSON.stringify(e.detail || {}).slice(0,200))}</div>
            </div></div>`;
        }).join("")}</div>`
      : emptyState("This deployment recorded no events."),
      { sub:`${events.length} event(s)` });

    /* -- rollback ------------------------------------------------------ */
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
        <button class="btn danger" data-act="rollback">Roll back to v${int(target)}</button></div>`
      : `<div class="note warn"><b>No rollback target.</b><br>
        <span style="font-size:12.5px">This endpoint has no recorded previous version. The API
        would fall back to the most recently archived production version, and fail if none
        exists — so the control stays disabled rather than failing after the click.</span></div>
      <div style="margin-top:12px"><button class="btn" disabled>Roll back</button></div>`,
      { sub:"POST /api/v1/deployments/rollback" });

    /* -- history ------------------------------------------------------- */
    const hist = sect(r.list, list => card("All deployments",
      table(DEPLOY_COLS, list, { empty:"No deployments recorded.",
        rowClass:x => x.state === "live" ? "win" : "" }),
      { flush:true, sub:`${list.length} record(s)` }), "deployments");

    const provider = `<div class="note"><b>Provider is
      <span class="mono">${esc(dep.provider || "local")}</span>.</b>
      <span style="font-size:12.5px">Routing is enforced in-process by the API: a canary split
      genuinely routes that share of predictions. It does not provision infrastructure — that is
      the SageMaker provider, which is not what runs here.
      ${svc.aws_enabled ? "" : "AWS integration is disabled on this deployment."}</span></div>`;

    return head
      + sect2("01", "Routing", "what is serving and how")
      + `<div class="grid g2">${traffic}${health}</div>`
      + sect2("02", "History", "events and rollback")
      + `<div class="grid g2">${timeline}${rollback}</div>`
      + hist + provider;
  }
};

const DEPLOY_COLS = [
  { label:"ID", render:x => `<span class="mono">${esc(String(x.id).slice(0,14))}</span>` },
  { label:"Version", render:x => x.current_version != null
      ? `<a class="mono" href="#/models/${esc(String(x.current_version))}"><b>v${esc(String(x.current_version))}</b></a>` : NA },
  { label:"Strategy", render:x => badge(x.strategy || "—","mute") },
  { label:"State", render:x => badge(x.state || "—",
      x.state === "live" || x.state === "succeeded" ? "ok"
      : x.state === "rolled_back" ? "bad" : "mute") },
  { label:"Previous", render:x => x.previous_version != null
      ? `<span class="mono dim">v${esc(String(x.previous_version))}</span>` : NA },
  { label:"Endpoint", render:x => `<span class="mono dim">${esc(x.endpoint_name || "—")}</span>` },
  { label:"Started", render:x => when(x.created_at) },
];

/* ==========================================================================
 * Quality Gates
 *
 * There is no gate-collection endpoint in this backend. What exists is the
 * configured policy (/api/v1/config -> approval) and on-demand evaluation of
 * one version. The page is therefore the policy, plus the champion/challenger
 * comparison that decides a promotion -- it cannot show a history of gate
 * decisions, and says so rather than implying one.
 * ========================================================================== */
PAGES.gates = {
  title: "Quality Gates",
  intro: "The policy a model must clear before it reaches production, and the "
       + "champion/challenger comparison that decides a promotion.",
  async render(){
    const r = await loadAll({ cfg:"/api/v1/config", dash:"/api/v1/dashboard" }, 15000);
    if(!r.cfg.ok) return errorState(r.cfg.error, "gates");
    const ap = r.cfg.data.approval || {};
    const model = (r.dash.ok ? r.dash.data.model : {}) || {};
    const met = model.metrics || {};
    const versions = model.versions || [];

    /* A configured minimum of 0 is not a requirement -- every possible value
       clears it. Rendering it as a passed check would invent a gate and pad
       the evidence table with meaningless green rows, so it is shown as
       "not gated" instead. */
    const gated = t => typeof ap[t] === "number" && ap[t] > 0;
    const absolute = [
      ["f1", "min_f1", "F1"],
      ["roc_auc", "min_roc_auc", "ROC-AUC"],
      ["precision", "min_precision", "Precision"],
      ["recall", "min_recall", "Recall"],
    ].filter(([, t]) => ap[t] != null);

    const policy = card("Production policy", `
      <div class="scroll"><table class="evid">
        <thead><tr><th>Requirement</th><th class="num">Threshold</th>
          <th class="num">Production v${esc(String(model.current_version ?? "—"))}</th>
          <th>Verdict</th></tr></thead>
        <tbody>${absolute.map(([m, t, label]) => {
          const val = met[m];
          const on = gated(t);
          const pass = typeof val === "number" && val >= ap[t];
          return `<tr class="${on && typeof val === "number" && !pass ? "failed" : ""}">
            <td>${esc(label)}</td>
            <td class="req">${on ? num(ap[t], 4)
              : `<span class="dim">no minimum</span>`}</td>
            <td class="obs">${num(val, 4)}</td>
            <td>${!on ? badge("not gated","mute")
              : typeof val !== "number" ? badge("no data","mute")
              : pass ? badge("pass","ok") : badge("fail","bad")}</td></tr>`;
        }).join("")}
        ${ap.max_inference_latency_ms != null ? `<tr>
          <td>Inference latency p95</td>
          <td class="req">&le; ${num(ap.max_inference_latency_ms, 1)} ms</td>
          <td class="obs">${num(met.inference_latency_p95_ms, 2)}</td>
          <td>${typeof met.inference_latency_p95_ms !== "number" ? badge("no data","mute")
            : met.inference_latency_p95_ms <= ap.max_inference_latency_ms
              ? badge("pass","ok") : badge("fail","bad")}</td></tr>` : ""}
        </tbody></table></div>
      <div class="grid g4" style="margin-top:15px">
        ${kpi("Gate enabled", boolBadge(ap.enabled, "yes", "no"))}
        ${kpi("Clean validation", boolBadge(ap.require_clean_validation, "required", "optional"))}
        ${kpi("Manual approval", boolBadge(ap.require_manual_approval, "required", "not required"),
          ap.require_manual_approval ? "a human must promote" : "")}
        ${kpi("Comparison metric", `<span class="mono">${esc(ap.comparison_metric || "-")}</span>`,
          ap.min_improvement != null ? `min improvement ${ap.min_improvement}` : "")}
      </div>`, { sub:"from platform configuration" });

    const improve = card("Beating the incumbent", `
      <p style="margin:0 0 12px;font-size:13px;line-height:1.6">Clearing the absolute thresholds
        is necessary but not sufficient. A candidate must also improve
        <span class="mono">${esc(ap.comparison_metric || "the comparison metric")}</span> over the
        current production version by at least
        <span class="mono">${esc(String(ap.min_improvement ?? "—"))}</span>.</p>
      <p class="dim" style="margin:0;font-size:12.5px">A margin of zero would let run-to-run noise
        churn the production model, which is why it is not zero. Thresholds are read from
        configuration and are never relaxed to make a candidate pass.</p>`,
      { sub:"champion / challenger" });

    const ladder = card("Versions against the policy", versions.length ? table([
      { label:"Version", render:v => `<a class="mono" href="#/models/${v.version}"><b>v${int(v.version)}</b></a>` },
      { label:"Stage", render:v => badge(v.stage || "-", v.stage === "Production" ? "ok"
          : v.stage === "Staging" ? "info" : "mute") },
      ...absolute.filter(([, t]) => gated(t)).map(([m, t, label]) => ({
        label, num:true, render:v => {
          const val = (v.metrics || {})[m];
          if(typeof val !== "number") return NA;
          const pass = val >= ap[t];
          return `<span class="mono" style="color:var(--${pass ? "ok" : "bad"})">${val.toFixed(4)}</span>`;
        }
      })),
      { label:"Meets policy", render:v => {
          const vals = absolute.filter(([, t]) => gated(t))
            .map(([m, t]) => [(v.metrics||{})[m], ap[t]]);
          if(!vals.length) return badge("no thresholds","mute");
          if(vals.some(([a]) => typeof a !== "number")) return badge("no data","mute");
          return vals.every(([a,b]) => a >= b) ? badge("yes","ok") : badge("no","bad");
        } },
    ], versions.slice().sort((a,b) => b.version - a.version), { empty:"No versions." })
      : unavailable("The registry reported no versions."),
      { flush:true, sub:"absolute thresholds only" });

    return policy + `<div class="grid g2">${improve}${card("What this page cannot show", `
      <p style="margin:0 0 11px;font-size:13px;line-height:1.6">There is no gate-decision history
        in this platform. Gate outcomes are recorded on the run and on the model version that
        produced them, not as a separate collection.</p>
      <p class="dim" style="margin:0;font-size:12.5px">To see a specific decision — the checks, the
        observed values and the comparison — open a version and use its
        <b>Evaluation</b> tab, which re-runs the gate against the thresholds in force now.</p>`,
      { sub:"an honest gap" })}</div>` + ladder;
  }
};
