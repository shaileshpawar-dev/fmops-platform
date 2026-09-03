PAGES.models = {
  title: "Model Registry",
  intro: "Model registry. Versions, stages, offline evaluation metrics and the approval gate each version was measured against.",
  async render(){
    const r = await loadAll({ models:"/api/v1/models", dash:"/api/v1/dashboard" }, 8000);
    const dash = r.dash.ok ? r.dash.data : {};
    const m = dash.model || {}, th = m.thresholds || {}, met = m.metrics || {};
    const versions = m.versions || [];

    const gate = th.enabled ? card("Approval gate", `
      <p class="dim" style="margin:0 0 10px">A candidate must clear every absolute threshold
        <b>and</b> beat the incumbent on <span class="mono">${esc(th.comparison_metric)}</span>
        by at least <span class="mono">${th.min_improvement}</span> before it can be promoted.</p>
      <div class="grid g4">
        ${kpi("Min F1", num(th.min_f1,2), gateMet(met.f1, th.min_f1))}
        ${kpi("Min ROC-AUC", num(th.min_roc_auc,2), gateMet(met.roc_auc, th.min_roc_auc))}
        ${kpi("Min precision", num(th.min_precision,2), gateMet(met.precision, th.min_precision))}
        ${kpi("Max latency p95", ms(th.max_inference_latency_ms),
              gateMet(th.max_inference_latency_ms, met.inference_latency_p95_ms))}
      </div>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
        ${boolBadge(th.require_clean_validation,"clean validation required","validation optional")}
        ${boolBadge(th.require_manual_approval,"manual approval required","auto-promote allowed", true)}
      </div>`) : "";

    const detail = m.available ? card("Offline evaluation — serving version", `
      <div class="grid g4">
        ${kpi("Accuracy", num(met.accuracy))}${kpi("Precision", num(met.precision))}
        ${kpi("Recall", num(met.recall))}${kpi("F1", num(met.f1))}
      </div>
      <div class="grid g4" style="margin-top:12px">
        ${kpi("ROC-AUC", num(met.roc_auc))}${kpi("PR-AUC", num(met.pr_auc))}
        ${kpi("Log loss", num(met.log_loss))}${kpi("Brier", num(met.brier_score))}
      </div>
      <div class="grid g3" style="margin-top:12px">
        ${kpi("Latency p50", ms(met.inference_latency_p50_ms))}
        ${kpi("Latency p95", ms(met.inference_latency_p95_ms))}
        ${kpi("Eval samples", int(met.n_samples), `positive rate ${(met.positive_rate*100||0).toFixed(1)}%`)}
      </div>
      <div class="note" style="margin-top:14px">These are <b>offline</b> metrics from the held-out
        evaluation set. Live quality requires ground-truth labels — see Monitoring.</div>`) : "";

    const lifecycle = card("Model lifecycle", `<div class="flow">
      <span class="step done">Registered</span><span class="arw">→</span>
      <span class="step done">Evaluated</span><span class="arw">→</span>
      <span class="step ${m.current_stage?"done":""}">Approved</span><span class="arw">→</span>
      <span class="step ${m.current_stage==="Production"?"on":""}">Production</span>
    </div>`);

    const tbl = card("Registered versions", table([
      { label:"Version", render:v => `<b class="mono">v${esc(v.version)}</b>` },
      { label:"Stage", render:v => badge(v.stage || "None", v.stage==="Production"?"ok":v.stage==="Staging"?"info":"mute") },
      { label:"Status", render:v => badge(v.status || "—","mute") },
      { label:"ROC-AUC", num:true, render:v => num(v.roc_auc) },
      { label:"F1", num:true, render:v => num(v.f1) },
      { label:"Created", render:v => when(v.created_at) },
      { label:"Actions", render:v => v.stage === "Production" ? `<span class="dim">serving</span>` :
          `<button class="btn" data-act="promote" data-model="${esc(m.model_name)}" data-ver="${esc(v.version)}">Promote</button>` },
    ], versions, { empty:"No versions registered." }), { flush:true,
      sub: `${versions.length} version${versions.length===1?"":"s"}` });

    return sect(r.models, () => "", "models") + tbl + detail + gate + lifecycle;
  }
};
function gateMet(actual, threshold){
  if(!has(actual) || !has(threshold)) return "";
  return actual >= threshold ? badge("met","ok") : badge("not met","bad");
}

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

PAGES.deployments = {
  title: "Deployments",
  intro: "Deployment history and the current routing state, including the strategy in force and its traffic split.",
  async render(){
    const r = await loadAll({ list:"/api/v1/deployments", dash:"/api/v1/dashboard" }, 6000);
    const dep = (r.dash.ok ? r.dash.data.deployment : {}) || {};
    const traffic = dep.traffic || {};
    const strat = (dep.strategy || "").toLowerCase();

    let diagram = "";
    if(strat === "canary" || Object.keys(traffic).length > 1){
      diagram = Object.entries(traffic).map(([v,p]) =>
        `<div style="display:grid;grid-template-columns:70px 1fr 56px;gap:10px;align-items:center;margin-bottom:7px">
          <span class="mono">v${esc(v)}</span>
          <div class="bar"><i style="width:${Number(p)||0}%"></i></div>
          <span class="num">${Number(p)||0}%</span></div>`).join("");
    } else if(strat === "blue_green"){
      diagram = `<div class="flow" style="gap:14px">
        <div><div class="dim" style="font-size:11px;margin-bottom:4px">Live (blue)</div>
          <span class="step on">v${esc(dep.current_version ?? "—")}</span></div>
        <span class="arw">↔</span>
        <div><div class="dim" style="font-size:11px;margin-bottom:4px">Standby (green)</div>
          <span class="step">v${esc(dep.candidate_version ?? dep.previous_version ?? "—")}</span></div></div>`;
    } else if(strat === "shadow"){
      diagram = `<div class="flow" style="gap:10px">
        <span class="step">Production traffic</span><span class="arw">→</span>
        <span class="step on">champion v${esc(dep.current_version ?? "—")}</span>
        <span class="arw">⇢</span>
        <span class="step">shadow v${esc(dep.shadow_version ?? "—")} <span class="dim">(not served)</span></span></div>`;
    }

    const current = card("Current routing", `
      <dl class="kv">
        <dt>Endpoint</dt><dd class="mono">${esc(dep.endpoint||"—")}</dd>
        <dt>Provider</dt><dd>${badge(dep.provider||"—","info")}</dd>
        <dt>Strategy</dt><dd>${dep.strategy?badge(dep.strategy,"info"):NA}</dd>
        <dt>State</dt><dd>${dep.state?badge(dep.state,"info"):NA}</dd>
        <dt>Serving</dt><dd>${has(dep.current_version)?`<b class="mono">v${esc(dep.current_version)}</b>`:NA}</dd>
        <dt>Previous</dt><dd>${has(dep.previous_version)?`<span class="mono">v${esc(dep.previous_version)}</span>`:NA}</dd>
        <dt>Rolled back</dt><dd>${boolBadge(dep.rolled_back,"yes","no",true)}</dd>
      </dl>
      ${diagram?`<div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--line-2)">${diagram}</div>`:""}`,
      { right: has(dep.previous_version) ?
        `<button class="btn danger" data-act="rollback">Roll back</button>` :
        `<button class="btn" disabled title="No previous version to roll back to">Roll back</button>` });

    const hist = sect(r.list, list => card("Deployment history", table([
      { label:"ID", render:x => `<span class="mono">${esc(String(x.id).slice(0,12))}</span>` },
      { label:"Model", render:x => `<span class="mono">${esc(x.model_name||"—")}</span>` },
      { label:"Version", render:x => has(x.current_version)?`<b class="mono">v${esc(x.current_version)}</b>`:NA },
      { label:"Strategy", render:x => badge(x.strategy||"—","info") },
      { label:"State", render:x => badge(x.state||"—", x.state==="live"||x.state==="succeeded"?"ok":
          x.state==="rolled_back"?"bad":"mute") },
      { label:"Endpoint", render:x => `<span class="mono dim">${esc(x.endpoint_name||"—")}</span>` },
      { label:"Started", render:x => when(x.created_at) },
    ], list, { empty:"No deployments recorded." }), { flush:true, sub:`${list.length} record(s)` }), "deployments");

    return current + hist + `<div class="note"><b>Local provider.</b> Routing is enforced
      in-process by the API: a canary split genuinely routes that share of predictions. It does not
      provision infrastructure — that is the SageMaker provider, which is not deployed here.</div>`;
  }
};

PAGES.champion = {
  title: "Champion / Challenger",
  intro: "Every candidate is compared against the incumbent. It is promoted only if it clears the absolute gate and beats the champion by the configured margin.",
  async render(){
    const d = await api.get("/api/v1/dashboard", 8000);
    const m = d.model || {}, th = m.thresholds || {};
    const vs = (m.versions || []).slice().sort((a,b) => Number(b.version) - Number(a.version));
    const champ = vs.find(v => v.stage === "Production") || vs[vs.length-1];
    const chall = vs.find(v => v !== champ && v.stage !== "Production");

    const metric = th.comparison_metric || "roc_auc";
    const minImp = has(th.min_improvement) ? th.min_improvement : null;

    if(!champ) return unavailable("No registered versions to compare.");

    if(!chall) return card("Comparison", `<div class="state">
      <div class="big">No challenger registered</div>
      Only one version exists (<span class="mono">v${esc(champ.version)}</span>, ${esc(champ.stage||"—")}).
      A challenger appears here after the training or retraining pipeline registers a new candidate.</div>`)
      + gateExplainer(th, metric, minImp)
      + card("Champion", versionPanel(champ, "CHAMPION"));

    const delta = (a,b) => (has(a) && has(b)) ? a - b : null;
    const rows = [["ROC-AUC","roc_auc"],["F1","f1"]].map(([label,key]) => {
      const c = champ[key], k = chall[key], dl = delta(k,c);
      return { label, c, k, dl, decisive: key === metric };
    });

    const dMetric = delta(chall[metric], champ[metric]);
    let verdict;
    if(dMetric === null) verdict = badge("Not comparable — metric missing on one side","mute");
    else if(minImp !== null && dMetric >= minImp)
      verdict = badge(`PASS — challenger exceeds champion by ${dMetric.toFixed(4)} (≥ ${minImp})`,"ok",true);
    else verdict = badge(`REJECTED — improvement ${dMetric.toFixed(4)} is below the required ${minImp}`,"bad",true);

    const compare = card("Head to head", `
      <div class="grid g2" style="margin-bottom:14px">
        ${versionPanel(champ,"CHAMPION")}${versionPanel(chall,"CHALLENGER")}</div>
      <div class="scroll"><table><thead><tr>
        <th>Metric</th><th class="num">Champion v${esc(champ.version)}</th>
        <th class="num">Challenger v${esc(chall.version)}</th><th class="num">Difference</th><th></th>
      </tr></thead><tbody>${rows.map(r => `<tr>
        <td>${esc(r.label)}${r.decisive?" "+badge("decisive","info"):""}</td>
        <td class="num">${num(r.c)}</td><td class="num">${num(r.k)}</td>
        <td class="num">${r.dl===null?NA:`<span class="mono" style="color:${r.dl>=0?"var(--ok)":"var(--bad)"}">
          ${r.dl>=0?"+":""}${r.dl.toFixed(4)}</span>`}</td>
        <td>${r.dl===null?"":(r.dl>=0?badge("better","ok"):badge("worse","bad"))}</td></tr>`).join("")}
      </tbody></table></div>
      <div style="margin-top:14px;padding-top:14px;border-top:1px solid var(--line-2)">
        <div class="dim" style="font-size:11px;font-weight:700;letter-spacing:.06em;
          text-transform:uppercase;margin-bottom:7px">Gate decision</div>${verdict}</div>`);

    return compare + gateExplainer(th, metric, minImp);
  }
};
function versionPanel(v, role){
  return `<div class="kpi"><div class="k">${role}</div>
    <div class="v">v${esc(v.version)}</div>
    <div class="m">${badge(v.stage||"None", v.stage==="Production"?"ok":"mute")}
      &nbsp;<span class="dim">${esc(v.status||"")}</span></div>
    <div class="grid g2" style="margin-top:10px;gap:8px">
      <div><div class="dim" style="font-size:10.5px">ROC-AUC</div>${num(v.roc_auc)}</div>
      <div><div class="dim" style="font-size:10.5px">F1</div>${num(v.f1)}</div></div></div>`;
}
function gateExplainer(th, metric, minImp){
  return card("How the gate decides", `<p style="margin:0 0 10px;color:var(--ink-2)">
    Promotion requires <b>both</b> conditions. Clearing the absolute thresholds is not enough:
    a candidate that is merely as good as the incumbent is rejected, which is what stops run-to-run
    noise from churning the production model.</p>
    <div class="flow" style="margin-bottom:12px">
      <span class="step">Absolute thresholds</span><span class="arw">AND</span>
      <span class="step">Beats champion on <span class="mono">${esc(metric)}</span>
        by ≥ <span class="mono">${minImp === null ? "—" : minImp}</span></span>
      <span class="arw">→</span><span class="step done">Promote</span></div>
    <dl class="kv">
      <dt>Comparison metric</dt><dd class="mono">${esc(metric)}</dd>
      <dt>Minimum improvement</dt><dd class="mono">${minImp === null ? "—" : minImp}</dd>
      <dt>Manual approval</dt><dd>${boolBadge(th.require_manual_approval,"required","not required",true)}</dd>
    </dl>`);
}
