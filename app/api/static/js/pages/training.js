PAGES.training = {
  title: "Training",
  intro: "Start a training run and follow it. A run executes the platform's existing pipeline: validation, feature engineering, training, optional tuning, evaluation, and the approval gate.",
  async render(){
    const r = await loadAll({ algos:"/api/v1/models/algorithms", ds:"/api/v1/datasets",
      runs:"/api/v1/training/runs?limit=25" }, 4000);

    const algos = r.algos.ok ? (r.algos.data.algorithms || {}) : {};
    const available = Object.entries(algos).filter(([, ok]) => ok).map(([k]) => k);
    const unavailable = Object.entries(algos).filter(([, ok]) => !ok).map(([k]) => k);
    const versions = r.ds.ok ? (r.ds.data.versions || []) : [];

    const form = card("Start a training run", `
      <div class="grid g3" style="gap:12px">
        <div><label class="dim" style="font-size:11.5px">Dataset version</label>
          <select id="trds">${versions.length
            ? `<option value="">Latest registered</option>` + versions.map(v =>
                `<option value="${esc(v.version)}">${esc(v.version)} (${v.n_rows} rows)</option>`).join("")
            : `<option value="">No datasets registered</option>`}</select></div>
        <div><label class="dim" style="font-size:11.5px">Algorithm</label>
          <select id="tralgo">${available.map(a =>
            `<option value="${esc(a)}">${esc(a)}</option>`).join("")}</select></div>
        <div><label class="dim" style="font-size:11.5px">After evaluation</label>
          <select id="trstage">
            <option value="">Train only - do not register</option>
            <option value="Staging">Register and attempt Staging</option>
            <option value="Production">Register and attempt Production</option>
          </select></div>
      </div>
      <div style="display:flex;gap:14px;align-items:center;margin-top:12px;flex-wrap:wrap">
        <label style="display:flex;gap:7px;align-items:center;cursor:pointer">
          <input type="checkbox" id="trtune"> Run hyperparameter search</label>
        <span class="spacer"></span>
        <button class="btn pri" id="trstart" ${versions.length?"":"disabled"}>Start training</button>
      </div>
      ${unavailable.length ? `<p class="dim" style="margin:12px 0 0">Not installed in this
        environment: ${unavailable.map(a=>`<span class="mono">${esc(a)}</span>`).join(", ")}</p>`:""}
      <div class="note" style="margin-top:12px">Promotion is never automatic. The approval gate still
        has to pass <b>and</b> the candidate has to beat the incumbent by the configured margin -
        this only asks for the attempt.</div>
      <div id="trresult" style="margin-top:12px"></div>`);

    const runs = sect(r.runs, d => card("Training runs", table([
      { label:"Run", render:x => `<span class="mono">${esc(String(x.run_id).replace("train-","").slice(0,12))}</span>` },
      { label:"Status", render:x => runStatusBadge(x.status) },
      { label:"Dataset", render:x => x.dataset_version ?
          `<span class="mono">${esc(x.dataset_version)}</span>` : `<span class="dim">latest</span>` },
      { label:"Algorithm", render:x => `<span class="mono">${esc(x.algorithm||"-")}</span>` },
      { label:"Tuned", render:x => x.tune ? badge("yes","info") : `<span class="dim">no</span>` },
      { label:"ROC-AUC", num:true, render:x => num((x.metrics||{}).roc_auc) },
      { label:"F1", num:true, render:x => num((x.metrics||{}).f1) },
      { label:"Model", render:x => has(x.model_version) ? `<b class="mono">v${esc(x.model_version)}</b>` : NA },
      { label:"Duration", num:true, render:x => has(x.duration_seconds) ?
          `<span class="mono">${x.duration_seconds.toFixed(1)}s</span>` : NA },
      { label:"Started", render:x => when(x.created_at) },
      { label:"", render:x => `<button class="btn" data-act="tr-detail" data-id="${esc(x.run_id)}">Detail</button>` },
    ], d.runs || [], { empty:"No training runs yet. Start one above." }),
    { flush:true, sub:`${d.count || 0} run(s)`,
      right:`<button class="btn" data-retry="${esc(location.hash)}">Refresh</button>` }), "training runs");

    return form + runs + `<div id="trdetail"></div>`;
  }
};

function runStatusBadge(s){
  const map = { completed:"ok", rejected:"warn", failed:"bad", running:"info", queued:"mute" };
  return badge(s || "unknown", map[s] || "mute", s === "running" || s === "queued");
}

function renderRunDetail(run){
  const rep = run.report || {}, m = run.metrics || {};
  const stages = ["Dataset","Validation","Features","Training"];
  if(run.tune) stages.push("Tuning");
  stages.push("Evaluation");
  if(run.promote) stages.push("Approval gate");
  if(rep.promoted) stages.push("Promoted");
  const done = run.status === "completed";
  const pipeline = `<div class="flow">${stages.map((s,i,a) =>
    `<span class="step ${done?"done":""}">${esc(s)}</span>` +
    (i<a.length-1?'<span class="arw">-&gt;</span>':"")).join("")}</div>`;

  const metrics = Object.keys(m).length ? `<div class="grid g4" style="margin-top:14px">
      ${kpi("Accuracy", num(m.accuracy))}${kpi("Precision", num(m.precision))}
      ${kpi("Recall", num(m.recall))}${kpi("F1", num(m.f1))}</div>
    <div class="grid g4" style="margin-top:12px">
      ${kpi("ROC-AUC", num(m.roc_auc))}${kpi("PR-AUC", num(m.pr_auc))}
      ${kpi("Log loss", num(m.log_loss))}${kpi("Brier", num(m.brier_score))}</div>`
    : `<div class="state" style="margin-top:12px">No metrics - the run did not reach evaluation.</div>`;

  const tuning = rep.tuning ? card("Hyperparameter search", `<dl class="kv">
      <dt>Backend</dt><dd class="mono">${esc(rep.tuning.backend)}</dd>
      <dt>Trials</dt><dd class="mono">${esc(rep.tuning.trials)}</dd>
      <dt>Best score</dt><dd>${num(rep.tuning.best_score)}</dd>
      <dt>Best params</dt><dd class="mono" style="font-size:11px">${esc(JSON.stringify(rep.tuning.best_params||{}))}</dd>
    </dl>`) : "";

  const cmp = rep.comparison;
  const approval = has(rep.approval_decision) ? card("Approval gate", `
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
        ${badge(rep.approval_decision, rep.approval_decision==="approved"?"ok":"bad", true)}
        ${rep.promoted ? badge("promoted to " + (rep.final_stage||""),"ok")
                       : badge("not promoted","mute")}</div>
      <p style="margin:0;color:var(--ink-2)">${esc(rep.approval_reason||"")}</p>
      ${(rep.failed_checks||[]).length ? `<p style="margin:10px 0 0">Failed checks:
        ${(rep.failed_checks||[]).map(c=>badge(c,"bad")).join(" ")}</p>`:""}
      ${cmp ? `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--line-2)">
        <div class="dim" style="font-size:10.5px;font-weight:700;letter-spacing:.06em;
          text-transform:uppercase;margin-bottom:6px">Champion / challenger</div>
        <span class="mono">${esc(cmp.metric||"")}</span>:
        candidate ${num(cmp.candidate_value)} vs baseline ${num(cmp.baseline_value)}
        &rarr; ${badge(cmp.decision||"-", cmp.decision==="promote"?"ok":"bad")}
      </div>`:""}`) : "";

  const error = run.error ? `<div class="note" style="border-color:var(--bad);
    background:var(--bad-bg);margin-bottom:14px"><b>Run failed.</b><br>${esc(run.error)}</div>` : "";

  return card("Run " + esc(String(run.run_id).replace("train-","")), `
    ${error}
    <dl class="kv">
      <dt>Status</dt><dd>${runStatusBadge(run.status)}</dd>
      <dt>Dataset</dt><dd class="mono">${esc(run.dataset_version || "latest registered")}</dd>
      <dt>Algorithm</dt><dd class="mono">${esc(run.algorithm||"-")}</dd>
      <dt>Tuning</dt><dd>${run.tune?badge("enabled","info"):`<span class="dim">disabled</span>`}</dd>
      <dt>Model</dt><dd>${has(run.model_version)?`<b class="mono">v${esc(run.model_version)}</b>
        <span class="dim">${esc(run.model_name||"")}</span>`:NA}</dd>
      <dt>Duration</dt><dd>${has(run.duration_seconds)?`<span class="mono">${run.duration_seconds.toFixed(2)}s</span>`:NA}</dd>
      <dt>Started</dt><dd>${when(run.started_at || run.created_at)}</dd>
      <dt>Completed</dt><dd>${when(run.completed_at)}</dd>
    </dl>
    ${pipeline}
    ${metrics}`) + tuning + approval;
}

PAGES.evaluation = {
  title: "Evaluation",
  intro: "Offline evaluation for every registered version, and the live quality of the serving model where ground-truth labels exist.",
  async render(){
    const r = await loadAll({ dash:"/api/v1/dashboard", runs:"/api/v1/training/runs?limit=50",
      perf:"/api/v1/monitoring/performance" }, 6000);
    const m = (r.dash.ok ? r.dash.data.model : {}) || {};
    const met = m.metrics || {};
    const versions = m.versions || [];

    const serving = m.available ? card("Serving version - offline evaluation", `
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:12px">
        <b class="mono">${esc(m.model_name)}</b> v${esc(m.current_version)}
        ${badge(m.current_stage,"info")}
        <span class="dim mono">${esc(m.algorithm||"")}</span></div>
      <div class="grid g4">
        ${kpi("Accuracy", num(met.accuracy))}${kpi("Precision", num(met.precision))}
        ${kpi("Recall", num(met.recall))}${kpi("F1", num(met.f1))}</div>
      <div class="grid g4" style="margin-top:12px">
        ${kpi("ROC-AUC", num(met.roc_auc))}${kpi("PR-AUC", num(met.pr_auc))}
        ${kpi("Log loss", num(met.log_loss))}${kpi("Brier", num(met.brier_score))}</div>
      <div class="note" style="margin-top:14px">These are <b>offline</b> metrics from the held-out
        evaluation set, not production accuracy.</div>`)
      : unavailable("No model registered.");

    const live = sect(r.perf, p => card("Live quality (from labels)", p.available ? `
        <div class="grid g4">
          ${kpi("Accuracy", num(p.accuracy))}${kpi("Precision", num(p.precision))}
          ${kpi("Recall", num(p.recall))}${kpi("ROC-AUC", num(p.roc_auc))}</div>
        <p class="dim" style="margin:12px 0 0">From ${int(p.labelled_samples)} labelled production rows.</p>`
      : `<div class="state"><div class="big">Ground-truth labels required</div>
         ${esc(p.detail||"")}</div>`), "performance");

    const byVersion = card("Registered versions", table([
      { label:"Version", render:v => `<b class="mono">v${esc(v.version)}</b>` },
      { label:"Stage", render:v => badge(v.stage||"None", v.stage==="Production"?"ok":v.stage==="Staging"?"info":"mute") },
      { label:"Status", render:v => badge(v.status||"-","mute") },
      { label:"ROC-AUC", num:true, render:v => num(v.roc_auc) },
      { label:"F1", num:true, render:v => num(v.f1) },
      { label:"Created", render:v => when(v.created_at) },
    ], versions, { empty:"No registered versions." }), { flush:true });

    const evalRuns = sect(r.runs, d => card("Evaluation from training runs", table([
      { label:"Run", render:x => `<span class="mono">${esc(String(x.run_id).replace("train-","").slice(0,12))}</span>` },
      { label:"Status", render:x => runStatusBadge(x.status) },
      { label:"Dataset", render:x => `<span class="mono">${esc(x.dataset_version||"latest")}</span>` },
      { label:"Algorithm", render:x => `<span class="mono">${esc(x.algorithm||"-")}</span>` },
      { label:"Accuracy", num:true, render:x => num((x.metrics||{}).accuracy) },
      { label:"F1", num:true, render:x => num((x.metrics||{}).f1) },
      { label:"ROC-AUC", num:true, render:x => num((x.metrics||{}).roc_auc) },
      { label:"Model", render:x => has(x.model_version)?`<span class="mono">v${esc(x.model_version)}</span>`:NA },
    ], (d.runs||[]).filter(x => Object.keys(x.metrics||{}).length),
       { empty:"No evaluated training runs yet." }), { flush:true }), "training runs");

    return `<div class="grid g2">${serving}${live}</div>` + byVersion + evalRuns;
  }
};

async function pollRun(runId, attempt){
  attempt = attempt || 0;
  if(attempt > 200) return;
  let run;
  try { run = await api.get(`/api/v1/training/runs/${encodeURIComponent(runId)}`, 0); }
  catch(e){ return; }

  const box = $("#trresult");
  if(box) box.innerHTML = `<div class="note"><b>Run ${esc(runId)}</b> &mdash;
    ${runStatusBadge(run.status)}${run.duration_seconds != null
      ? ` <span class="dim">${run.duration_seconds.toFixed(1)}s</span>` : ""}</div>`;

  if(["completed","failed","rejected"].includes(run.status)){
    api.bust();
    toast(`Training ${run.status}.`, run.status === "completed" ? "ok" : "bad");
    const detail = $("#trdetail");
    if(detail) detail.innerHTML = renderRunDetail(run);
    if(route() === "training") render();
    return;
  }
  setTimeout(() => pollRun(runId, attempt + 1), 3000);
}
