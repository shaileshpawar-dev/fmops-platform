/* Training -- one algorithm, one run, one model version.
 *
 * A run trains a named model on a registered dataset: pick the dataset, the
 * column to predict and a name, and the platform derives the data contract
 * from the data itself. It executes as a background job; this page follows
 * the job and its log, then shows what the pipeline decided. AutoML is the
 * same pipeline over several candidates.
 */

PAGES.training = {
  title: "Training",
  intro: "Train one algorithm on a registered dataset as a background job, and follow it: "
       + "validation, training, optional tuning, evaluation and the approval gate.",
  async render(){
    const r = await loadAll({ algos:"/api/v1/models/algorithms", ds:"/api/v1/datasets",
      runs:"/api/v1/training/runs?limit=25" }, 4000);

    const algos = r.algos.ok ? (r.algos.data.algorithms || {}) : {};
    const available = Object.entries(algos).filter(([, ok]) => ok).map(([k]) => k);
    const missing = Object.entries(algos).filter(([, ok]) => !ok).map(([k]) => k);
    const versions = (r.ds.ok ? (r.ds.data.versions || []) : []).slice().reverse();

    const form = card("Start a training run", versions.length ? `
      <div class="grid g3" style="gap:12px">
        <div><label for="trds">Dataset version</label>
          <select id="trds">${versions.map(v =>
            `<option value="${esc(v.version)}">${esc(v.version)} · ${esc(v.dataset_name)} · ${
              int(v.n_rows)} rows</option>`).join("")}</select></div>
        <div><label for="trtarget">Column to predict</label>
          <select id="trtarget"><option value="">Loading columns…</option></select></div>
        <div><label for="trname">Model name</label>
          <input id="trname" maxlength="64" placeholder="defaults to &lt;target&gt;_classifier"
            pattern="[a-z][a-z0-9_-]{2,63}" autocomplete="off"></div>
        <div><label for="tralgo">Algorithm</label>
          <select id="tralgo">${available.map(a =>
            `<option value="${esc(a)}">${esc(a)}</option>`).join("")}</select></div>
        <div><label for="trpos">Positive class (optional)</label>
          <input id="trpos" maxlength="128" placeholder="defaults to yes/true/1, else the rarer class"
            autocomplete="off"></div>
        <div><label for="trstage">After evaluation</label>
          <select id="trstage">
            <option value="Staging">Register and run the gate toward Staging (shadow evaluation only)</option>
            <option value="Production">Register and run the gate toward Production (may take live traffic)</option>
            <option value="">Register only — skip the gate (stays in Development)</option>
          </select></div>
      </div>
      <div style="display:flex;gap:14px;align-items:center;margin-top:12px;flex-wrap:wrap">
        <label class="chk"><input type="checkbox" id="trtune"> Hyperparameter search</label>
        <span class="spacer"></span>
        <button class="btn pri" id="trstart">Start training</button>
      </div>
      ${missing.length ? `<p class="dim" style="margin:12px 0 0">Not installed in this
        environment: ${missing.map(a=>`<span class="mono">${esc(a)}</span>`).join(", ")}</p>`:""}
      <p class="dim" style="margin:12px 0 0;font-size:12.5px">Every run registers a version. The gate
        decides whether it moves up: it must clear the absolute thresholds <b>and</b> beat the model's
        current production version on held-out rows neither has seen. In an environment that requires
        sign-off, a passing version waits for approval.</p>
      <div id="trresult" style="margin-top:12px"></div>`
      : `<div class="state"><div class="big">No datasets yet</div>Upload a CSV on the
          <a href="#/datasets">Datasets</a> page first.</div>`);

    const runs = sect(r.runs, d => card("Training runs", table([
      { label:"Run", render:x => `<span class="mono">${esc(String(x.run_id).replace("train-","").slice(0,12))}</span>` },
      { label:"Status", sort:x => x.status, render:x => runStatusBadge(x.status) },
      { label:"Model", sort:x => x.model_name, render:x => x.model_name
          ? `<a href="#/models/${encodeURIComponent(x.model_name)}${has(x.model_version) ? "/" + x.model_version : ""}">${
            esc(x.model_name)}${has(x.model_version) ? ` <b class="mono">v${esc(x.model_version)}</b>` : ""}</a>` : NA },
      { label:"Predicts", render:x => x.target_column ? `<span class="mono">${esc(x.target_column)}</span>`
          : `<span class="dim">reference</span>` },
      { label:"Dataset", render:x => x.dataset_version
          ? `<span class="mono">${esc(x.dataset_version)}</span>` : `<span class="dim">latest</span>` },
      { label:"Algorithm", render:x => `<span class="mono">${esc(x.algorithm||"-")}</span>` },
      { label:"ROC-AUC", num:true, sort:x => (x.metrics||{}).roc_auc, render:x => num((x.metrics||{}).roc_auc) },
      { label:"F1", num:true, render:x => num((x.metrics||{}).f1) },
      { label:"Duration", num:true, render:x => has(x.duration_seconds) ?
          `<span class="mono">${x.duration_seconds.toFixed(1)}s</span>` : NA },
      { label:"Started", sort:x => x.created_at, render:x => when(x.created_at) },
      { label:"", render:x => `<button class="btn sm" data-act="tr-detail" data-id="${esc(x.run_id)}">Detail</button>` },
    ], d.runs || [], { id:"training-runs", empty:"No training runs yet. Start one above." }),
    { flush:true, sub:`${d.count || 0} run(s)`,
      right:`<button class="btn" data-retry="${esc(location.hash)}">Refresh</button>` }), "training runs");

    return form + `<div id="trdetail"></div>` + runs;
  },
  wire(){
    const ds = $("#trds"), target = $("#trtarget");
    if(!ds) return;
    const loadColumns = async () => {
      target.innerHTML = `<option value="">Loading columns…</option>`;
      try {
        const p = await api.get(`/api/v1/automl/profile/${encodeURIComponent(ds.value)}`, 60000);
        const prof = p.profile || {};
        const suggested = (prof.suggested_target || {}).column;
        const cols = (prof.columns || []).map(c => c.name);
        target.innerHTML = cols.map(c => `<option value="${esc(c)}" ${c === suggested ? "selected" : ""}>${
          esc(c)}${c === suggested ? " (suggested)" : ""}</option>`).join("");
      } catch(e){
        target.innerHTML = `<option value="">Could not read columns: ${esc(e.message)}</option>`;
      }
    };
    ds.onchange = loadColumns;
    loadColumns();

    const start = $("#trstart");
    start.onclick = async () => {
      const stage = $("#trstage").value;
      const payload = {
        dataset_version: ds.value,
        target_column: target.value || null,
        model_name: ($("#trname").value || "").trim() || null,
        positive_label: ($("#trpos").value || "").trim() || null,
        algorithm: $("#tralgo").value || null,
        tune: !!$("#trtune").checked,
        promote: !!stage,
      };
      if(stage) payload.target_stage = stage;
      if(!payload.target_column){ toast("Choose the column to predict.", "bad"); return; }
      const res = await runAction({
        title: "Start training",
        body: `Train ${payload.algorithm} on ${payload.dataset_version} to predict `
            + `${payload.target_column}${payload.promote ? `, then run the gate toward ${stage}` : ""}.`,
        confirm: "Start training", needsKey: true,
        path: "/api/v1/training/runs", payload,
        success: "Training queued.", after: () => {},
      });
      if(!res) return;
      $("#trresult").innerHTML = `<div class="note">Run <span class="mono">${esc(res.run_id)}</span>
        will add a version to <b>${esc(res.model_name || "?")}</b>.</div>`
        + jobPanel({ id: res.job_id, kind: "training", status: "queued" });
      followJob(res.job_id, async () => {
        api.bust();
        try {
          const run = await api.get(`/api/v1/training/runs/${encodeURIComponent(res.run_id)}`, 0);
          $("#trdetail").innerHTML = renderRunDetail(run);
        } catch(e){ /* the job panel already shows the outcome */ }
      });
    };
  },
};

function renderRunDetail(run){
  const rep = run.report || {}, m = run.metrics || {};
  const stages = ["Dataset","Validation","Features","Training"];
  if(run.tune) stages.push("Tuning");
  stages.push("Evaluation");
  if(run.promote) stages.push("Approval gate");
  if(rep.promoted) stages.push("Promoted");
  const done = ["completed","rejected"].includes(run.status);
  const pipeline = `<div class="flow">${stages.map((s,i,a) =>
    `<span class="step ${done?"done":""}">${esc(s)}</span>` +
    (i<a.length-1?'<span class="arw">-&gt;</span>':"")).join("")}</div>`;

  const metrics = Object.keys(m).length ? `<div class="grid g4" style="margin-top:14px">
      ${kpi("Accuracy", num(m.accuracy))}${kpi("Precision", num(m.precision))}
      ${kpi("Recall", num(m.recall))}${kpi("F1", num(m.f1))}</div>
    <div class="grid g4" style="margin-top:12px">
      ${kpi("ROC-AUC", num(m.roc_auc))}${kpi("PR-AUC", num(m.pr_auc))}
      ${kpi("Log loss", num(m.log_loss))}${kpi("Brier", num(m.brier_score))}</div>`
    : `<div class="state" style="margin-top:12px">No metrics — the run did not reach evaluation.</div>`;

  const tuning = rep.tuning ? card("Hyperparameter search", `<dl class="kv">
      <div><dt>Backend</dt><dd class="mono">${esc(rep.tuning.backend)}</dd></div>
      <div><dt>Trials</dt><dd class="mono">${esc(rep.tuning.trials)}</dd></div>
      <div><dt>Best score</dt><dd>${num(rep.tuning.best_score)}</dd></div>
      <div><dt>Best params</dt><dd class="mono" style="font-size:11px">${esc(JSON.stringify(rep.tuning.best_params||{}))}</dd></div>
    </dl>`) : "";

  const cmp = rep.comparison;
  const approval = has(rep.approval_decision) ? card("Approval gate", `
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
        ${runStatusBadge(rep.approval_decision)}
        ${rep.promoted ? badge("promoted to " + (rep.final_stage||""),"ok") : badge("not promoted","mute")}</div>
      <p style="margin:0;color:var(--ink-2)">${esc(rep.approval_reason||"")}</p>
      ${(rep.failed_checks||[]).length ? `<p style="margin:10px 0 0">Failed checks:
        ${(rep.failed_checks||[]).map(c=>badge(c,"bad")).join(" ")}</p>`:""}
      ${cmp ? `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--line-2)">
        <div class="eyebrow">Champion / challenger</div>
        <span class="mono">${esc(cmp.metric||"")}</span>:
        candidate ${num(cmp.candidate_score, 4)} vs production ${num(cmp.baseline_score, 4)}
        &rarr; ${badge(cmp.decision||"-", cmp.decision==="promote"?"ok":"bad")}
        ${cmp.basis ? badge(String(cmp.basis).replace(/_/g," "), "mute") : ""}
      </div>`:""}
      ${run.model_name && has(run.model_version) ? `<div style="margin-top:12px"><a class="btn sm"
        href="#/models/${encodeURIComponent(run.model_name)}/${run.model_version}?tab=evaluation">Open the version</a></div>` : ""}`) : "";

  const error = run.error ? `<div class="note bad" style="margin-bottom:14px"><b>Run failed.</b><br>${esc(run.error)}</div>` : "";

  return card("Run " + esc(String(run.run_id).replace("train-","")), `
    ${error}
    <dl class="kv">
      <div><dt>Status</dt><dd>${runStatusBadge(run.status)}</dd></div>
      <div><dt>Model</dt><dd>${run.model_name ? `<b>${esc(run.model_name)}</b>${has(run.model_version) ? ` <span class="mono">v${esc(run.model_version)}</span>` : ""}` : NA}</dd></div>
      <div><dt>Predicts</dt><dd class="mono">${esc(run.target_column || "reference target")}</dd></div>
      <div><dt>Dataset</dt><dd class="mono">${esc(run.dataset_version || "latest registered")}</dd></div>
      <div><dt>Algorithm</dt><dd class="mono">${esc(run.algorithm||"-")}</dd></div>
      <div><dt>Tuning</dt><dd>${run.tune?badge("enabled","info"):`<span class="dim">disabled</span>`}</dd></div>
      <div><dt>Duration</dt><dd>${has(run.duration_seconds)?`<span class="mono">${run.duration_seconds.toFixed(2)}s</span>`:NA}</dd></div>
      <div><dt>Started</dt><dd>${when(run.started_at || run.created_at)}</dd></div>
    </dl>
    ${pipeline}
    ${metrics}`) + tuning + approval;
}
