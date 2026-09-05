/* Guided build workflow: dataset -> deployed, monitored model.
 *
 * Every stage here drives an API that already exists. There is no "project"
 * entity in the backend and this file does not invent one: the workflow is a
 * guided path over the same endpoints the individual console pages use, and
 * its state lives in the browser (sessionStorage) so a refresh mid-run does
 * not lose your place. Nothing shown is stored server-side except what the
 * real APIs store -- dataset versions, runs, model versions, deployments.
 *
 * Two consequences of the actual backend shape are surfaced rather than
 * hidden, because pretending otherwise would make the UI lie:
 *
 *   1. Registration is not a separate button. Both AutoML and a training run
 *      with promotion requested register and gate the winner inside the run.
 *      The registry step therefore reports what the run did; it does not
 *      offer an action the API has no endpoint for.
 *
 *   2. Online prediction is bound to one fixed request schema. A model
 *      trained on differently-shaped data cannot be exercised through
 *      /api/v1/predict, and the predict step says so instead of rendering a
 *      form that would 422.
 */

const STEPS = [
  { n:1,  key:"dataset",    label:"Dataset" },
  { n:2,  key:"profile",    label:"Profile" },
  { n:3,  key:"target",     label:"Target" },
  { n:4,  key:"training",   label:"Training" },
  { n:5,  key:"evaluation", label:"Evaluation" },
  { n:6,  key:"registry",   label:"Registry" },
  { n:7,  key:"approval",   label:"Approval" },
  { n:8,  key:"deploy",     label:"Deploy" },
  { n:9,  key:"predict",    label:"Predict" },
  { n:10, key:"monitor",    label:"Monitor" },
];

const PRJ_KEY = "fmops-project";

/* Session, not local: a half-finished walkthrough is a property of this tab,
   not a preference worth surviving the browser. */
function prjLoad(){
  try { return JSON.parse(sessionStorage.getItem(PRJ_KEY) || "null") || prjBlank(); }
  catch(e){ return prjBlank(); }
}
function prjBlank(){
  return { version:null, datasetName:null, rows:null, columns:null, columnNames:[],
           validation:null, profile:null, target:null, problemType:null,
           problemSupported:null, primaryMetric:null, mode:null, runKind:null,
           runId:null, runStatus:null, modelName:null, modelVersion:null,
           registeredStage:null, gateDecision:null, deployedVersion:null };
}
let PRJ = prjLoad();
function prjSave(){
  try { sessionStorage.setItem(PRJ_KEY, JSON.stringify(PRJ)); } catch(e){ /* private mode */ }
}
function prjReset(){ PRJ = prjBlank(); prjSave(); }

/* --- step reachability -------------------------------------------------- *
 * A step is reachable only once the thing it operates on exists. This is
 * what stops someone landing on "Deploy" with no model: the guard is derived
 * from real state, not from a counter that could drift out of sync with it.
 */
function stepState(n){
  const p = PRJ;
  const failed = p.runStatus === "failed";
  switch(n){
    case 1:  return p.version ? "done" : "todo";
    case 2:  return !p.version ? "locked" : (p.profile ? "done" : "todo");
    case 3:  return !p.profile ? "locked" : (p.target ? "done" : "todo");
    case 4:  return (!p.target || !p.problemSupported) ? "locked" : (p.runId ? "done" : "todo");
    case 5:  if(!p.runId) return "locked";
             if(failed) return "fail";
             return p.runStatus && p.runStatus !== "running" && p.runStatus !== "queued"
               ? "done" : "todo";
    case 6:  if(!p.runId || failed) return "locked";
             return p.modelVersion ? "done" : (stepState(5) === "done" ? "todo" : "locked");
    case 7:  if(!p.modelVersion) return "locked";
             return p.gateDecision ? (p.gateDecision === "approved" ? "done" : "fail") : "todo";
    case 8:  if(!p.modelVersion) return "locked";
             return p.deployedVersion ? "done" : "todo";
    case 9:  return p.deployedVersion ? "todo" : "locked";
    case 10: return p.deployedVersion ? "todo" : "locked";
  }
  return "locked";
}
function currentStep(){
  const q = (location.hash.split("?")[1] || "");
  const asked = parseInt(new URLSearchParams(q).get("step") || "", 10);
  if(asked >= 1 && asked <= 10 && stepState(asked) !== "locked") return asked;
  // Otherwise land on the first step that still needs doing.
  for(const s of STEPS){ const st = stepState(s.n); if(st === "todo" || st === "fail") return s.n; }
  return 1;
}
function goStep(n){ location.hash = `#/newproject?step=${n}`; }

/* --- stepper ------------------------------------------------------------ */
function stepper(cur){
  const mark = { done:"✓", on:"●", todo:"○", locked:"○", fail:"✕" };
  const bits = STEPS.map((s, i) => {
    let st = stepState(s.n);
    const cls = s.n === cur ? "on" : (st === "done" ? "done" : st === "fail" ? "fail" : "");
    const glyph = s.n === cur ? mark.on : mark[st];
    const clickable = st === "done" || st === "fail" || s.n === cur || st === "todo";
    return `<div class="st ${cls}">
      <button class="stbtn" ${clickable ? `data-goto="${s.n}"` : "disabled"}
        ${s.n === cur ? 'aria-current="step"' : ""}
        title="${esc(String(s.n).padStart(2,"0"))} ${esc(s.label)}${st==="locked"?" (not yet available)":""}">
        <span class="dotn">${st === "done" || st === "fail" ? glyph : String(s.n).padStart(2,"0")}</span>
        <span class="stlbl">${esc(s.label)}</span>
      </button>
    </div>${i < STEPS.length - 1 ? '<div class="join"></div>' : ""}`;
  }).join("");
  const now = STEPS[cur-1];
  return `<div class="stepper" role="group" aria-label="Workflow progress">${bits}</div>
    <div class="stepmini"><div class="row">
      <span class="n">${String(cur).padStart(2,"0")}</span>
      <span class="t">${esc(now.label)}</span>
      <span class="of">Step ${cur} of ${STEPS.length}</span></div>
      <div class="bar"><i style="width:${Math.round(cur/STEPS.length*100)}%"></i></div></div>`;
}

function wizFrame(cur, title, sub, body){
  return stepper(cur) + `<section class="card"><div class="body">
    <p class="dim mono" style="margin:0 0 6px;font-size:11px;letter-spacing:.08em">
      STEP ${String(cur).padStart(2,"0")} / ${STEPS.length}</p>
    <h3 class="wizhead">${esc(title)}</h3>
    <p class="wizsub">${sub}</p>
    ${body}</div></section>`;
}

/* Small reusable summary of what the workflow already knows, so later steps
   do not make the user scroll back to remember which dataset this is. */
function prjContext(){
  const p = PRJ; const bits = [];
  if(p.version) bits.push(["Dataset", `${esc(p.datasetName || "-")} <span class="mono dim">${esc(p.version)}</span>`]);
  if(p.rows != null) bits.push(["Shape", `${int(p.rows)} rows · ${int(p.columns)} columns`]);
  if(p.target) bits.push(["Target", `<span class="mono">${esc(p.target)}</span>`]);
  if(p.problemType) bits.push(["Problem", esc(String(p.problemType).replace(/_/g," "))]);
  if(p.modelVersion) bits.push(["Model", `<span class="mono">${esc(p.modelName||"")} v${int(p.modelVersion)}</span>`]);
  if(!bits.length) return "";
  return `<div class="note" style="margin-bottom:16px;display:flex;gap:20px;flex-wrap:wrap">
    ${bits.map(([k,v]) => `<span><span class="dim">${esc(k)}</span> ${v}</span>`).join("")}</div>`;
}

/* ===================================================================== */
/* 01 Dataset                                                            */
/* ===================================================================== */
async function stepDataset(){
  let limits = null;
  try { limits = await api.get("/api/v1/datasets/limits", 300000); } catch(e){ limits = null; }
  const maxMb = limits && limits.max_upload_bytes
    ? Math.floor(limits.max_upload_bytes / (1024*1024)) : null;
  const formats = limits && limits.accepted_formats ? limits.accepted_formats.join(", ") : "CSV";

  const drop = `
    <label class="drop" id="prjdrop" for="prjfile">
      <div class="icn" aria-hidden="true">&uarr;</div>
      <div class="big">Drag &amp; drop a CSV here</div>
      <div class="hint">or click to browse</div>
      <input type="file" id="prjfile" accept=".csv,text/csv">
    </label>
    <div class="grid g2" style="margin-top:14px">
      <div><span class="dim" style="font-size:12px">Maximum file size</span><br>
        ${maxMb ? `<b>${maxMb} MB</b>` : `<span class="dim">Not reported by the API</span>`}</div>
      <div><span class="dim" style="font-size:12px">Supported format</span><br>
        <b>${esc(formats)}</b></div>
    </div>
    <div style="margin-top:14px">
      <input type="text" id="prjdesc" placeholder="Description (optional)"
        style="width:100%;max-width:420px">
    </div>
    <div id="prjupstatus" style="margin-top:14px"></div>`;

  let done = "";
  if(PRJ.version){
    const v = PRJ.validation;
    done = `<div style="margin-top:20px">${card("Uploaded", `
      <div class="verdict pass" style="margin-bottom:14px">
        <span class="mark" aria-hidden="true">&check;</span>
        <span class="txt"><b>UPLOAD SUCCESSFUL</b>
          <span>Registered as an immutable, content-addressed version.</span></span></div>
      <div class="grid g4">
        ${kpi("Filename", esc(PRJ.datasetName || "-"))}
        ${kpi("Version", `<span class="mono">${esc(PRJ.version)}</span>`)}
        ${kpi("Rows", int(PRJ.rows))}
        ${kpi("Columns", int(PRJ.columns))}
      </div>
      ${v ? `<div style="margin-top:14px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
          <b style="font-size:13px">Validation</b>
          ${v.passed ? badge("PASSED","ok",true) : badge("FAILED","bad",true)}
          <span class="dim" style="font-size:12px">${int(v.succeeded)} of
            ${int(v.expectations)} expectations passed</span>
        </div>
        ${!v.passed ? `<p class="dim" style="margin:0;font-size:12.5px">
          Validation failures do not block this walkthrough, but the training pipeline applies
          the same expectations and may reject the run.
          <a href="#/datasets">See the full report on the Datasets page.</a></p>` : ""}
      </div>` : `<p class="dim" style="margin:14px 0 0;font-size:12.5px">
        Validation was not run for this upload.</p>`}`, { flush:true })}
      <div class="wiznav">
        <button class="btn pri" data-goto="2">Continue to profile</button>
        <button class="btn" id="prjreset">Start over with a different dataset</button>
      </div></div>`;
  }

  return wizFrame(1, "Upload your data",
    `Upload the CSV you want FMOps to train on. It is parsed with pandas, registered as a
     content-addressed version, and checked against the platform validation suite &mdash; the
     same expectations the training pipeline gates on. Uploading identical bytes twice returns
     the existing version rather than creating a duplicate.`,
    drop + done);
}

/* ===================================================================== */
/* 02 Profile                                                            */
/* ===================================================================== */
async function stepProfile(){
  if(!PRJ.version) return wizFrame(2, "Profile", "No dataset has been uploaded yet.",
    unavailable("Upload a dataset first.") + `<div class="wiznav">
      <button class="btn pri" data-goto="1">Back to upload</button></div>`);

  let d;
  try { d = await api.get(`/api/v1/automl/profile/${encodeURIComponent(PRJ.version)}`, 60000); }
  catch(e){
    return wizFrame(2, "Profile", "The profiler could not read this dataset.",
      errorState(e.message, "newproject") + `<div class="wiznav">
        <button class="btn" data-goto="1">Back to upload</button></div>`);
  }

  PRJ.profile = d;
  PRJ.primaryMetric = d.primary_metric;
  if(!PRJ.target){ PRJ.problemType = d.problem_type; PRJ.problemSupported = d.problem_supported; }
  prjSave();

  const p = d.profile || {};
  const cols = p.columns || [];
  const dateLike = cols.filter(c => c.kind === "datetime").length;
  // pct() scales a fraction, so keep this a fraction.
  const missingFrac = p.n_rows && p.n_columns
    ? p.missing_cells / (p.n_rows * p.n_columns) : null;

  const overview = `<div class="grid g6">
    ${kpi("Rows", int(p.n_rows))}
    ${kpi("Columns", int(p.n_columns))}
    ${kpi("Numeric", int(p.n_numeric))}
    ${kpi("Categorical", int(p.n_categorical))}
    ${kpi("Date-like", int(dateLike))}
    ${kpi("Missing values", missingFrac == null ? NA : pct(missingFrac, 2),
          `${int(p.missing_cells)} cells`)}
  </div>`;

  const sug = p.suggested_target;
  const targetPanel = sug ? `
    <div class="rectarget">
      <div class="dim mono" style="font-size:11px;letter-spacing:.08em;margin-bottom:6px">
        POTENTIAL TARGET</div>
      <div class="nm">${esc(sug.column)}</div>
      ${confidenceBadge(sug.confidence)}
      <ul class="reasons">${(sug.reasons||[]).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      <p class="dim" style="margin:0;font-size:12.5px">You confirm the target in the next step.</p>
    </div>`
    : `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>No target could be identified.</b><br>
        <span style="font-size:12.5px">Nothing in this dataset looks like an outcome column.
        You will have to choose one yourself in the next step.</span></div>`;

  const warns = p.warnings || [];
  const warnPanel = warns.length ? card("Data quality warnings", `
    <p class="dim" style="margin:0 0 12px;font-size:12.5px">Reported, never acted on
      automatically. Nothing below is resampled, dropped or corrected without you asking.</p>
    ${table([
      { label:"Warning", render:w => badge(String(w.type||"warning").replace(/_/g," "),
          w.severity === "error" ? "bad" : w.severity === "warning" ? "warn" : "mute") },
      { label:"Column", render:w => w.column ? `<span class="mono">${esc(w.column)}</span>` : NA },
      { label:"Detail", render:w => esc(w.detail || "") },
    ], warns, { empty:"" })}`, { flush:true, sub:`${warns.length} warning(s)` }) : "";

  const colTable = card("Columns", table([
    { label:"Column", render:c => `<span class="mono">${esc(c.name)}</span>` },
    { label:"Type", render:c => esc(c.kind) },
    { label:"Role", render:c => c.role === "excluded"
        ? badge("excluded","mute") + (c.exclusion_reason
            ? ` <span class="dim" style="font-size:11.5px">${esc(c.exclusion_reason)}</span>` : "")
        : badge(c.role, c.role === "target" ? "info" : "ok") },
    { label:"Unique", num:true, render:c => int(c.n_unique) },
    { label:"Missing", num:true, render:c => c.missing ? pct(c.missing_pct / 100, 2) : NA },
    { label:"Example", render:c => c.example != null
        ? `<span class="mono dim">${esc(String(c.example).slice(0,32))}</span>` : NA },
  ], cols, { empty:"No columns." }), { flush:true, sub:`${cols.length} column(s)` });

  return wizFrame(2, "Dataset profile",
    `Every column classified, every exclusion given a reason. This is a deterministic pass over
     observable properties &mdash; no model is involved in any recommendation below.`,
    prjContext() + overview + `<div style="margin:18px 0">${targetPanel}</div>` +
    warnPanel + colTable + `<div class="wiznav">
      <button class="btn pri" data-goto="3">Continue to target selection</button>
      <button class="btn" data-goto="1">Back</button></div>`);
}

function confidenceBadge(c){
  const k = String(c || "").toLowerCase();
  return badge(`${k.toUpperCase()} CONFIDENCE`,
    k === "high" ? "ok" : k === "medium" ? "info" : "warn");
}

/* ===================================================================== */
/* 03 Target                                                             */
/* ===================================================================== */
/* Re-profiling on a target change is what keeps the problem type honest: a
   different column can mean a different problem, so the answer is recomputed
   against the chosen column rather than carried over from the suggestion. */
async function stepTarget(){
  if(!PRJ.profile) return wizFrame(3, "Target", "Profile the dataset first.",
    unavailable("No profile available.") + `<div class="wiznav">
      <button class="btn pri" data-goto="2">Back to profile</button></div>`);

  const d = PRJ.target
    ? await profileFor(PRJ.target).catch(() => PRJ.profile)
    : PRJ.profile;
  const p = d.profile || {};
  const sug = p.suggested_target;
  const chosen = PRJ.target;

  const candidates = (p.columns || [])
    .filter(c => c.role !== "excluded" || c.role === "target")
    .map(c => c.name);
  const allCols = (p.columns || []).map(c => c.name);

  const rec = (!chosen && sug) ? `
    <div class="rectarget" style="max-width:460px">
      <div class="dim mono" style="font-size:11px;letter-spacing:.08em;margin-bottom:6px">
        RECOMMENDED TARGET</div>
      <div class="nm">${esc(sug.column)}</div>
      ${confidenceBadge(sug.confidence)}
      <ul class="reasons">${(sug.reasons||[]).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      <button class="btn pri" data-usetarget="${esc(sug.column)}">Use ${esc(sug.column)}</button>
    </div>` : "";

  const lowConfidence = sug && String(sug.confidence).toLowerCase() === "low";
  const warnNoTarget = (!chosen && !sug) ? `
    <div class="note" style="border-color:var(--warn-line);background:var(--warn-bg);max-width:640px">
      <b>We could not confidently identify the target column.</b><br>
      <span style="font-size:12.5px">Nothing in this dataset scored as a plausible outcome
      column, so there is no recommendation to accept. Choose the column you want to predict
      below. Training never starts on a guess.</span></div>` : "";
  const warnLow = (!chosen && lowConfidence) ? `
    <div class="note" style="border-color:var(--warn-line);background:var(--warn-bg);max-width:640px;margin-top:14px">
      <b>Low confidence.</b> <span style="font-size:12.5px">Another column scored almost as
      well, which is ambiguity rather than a recommendation. Check this is the column you
      actually want to predict.</span></div>` : "";

  const picker = `
    <div style="margin-top:${rec ? "20px" : "0"};max-width:460px">
      <label for="prjtarget" style="display:block;font-size:12.5px;font-weight:600;margin-bottom:6px">
        ${rec ? "Or choose a different column" : "Choose the column you want to predict"}</label>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <select id="prjtarget" style="flex:1 1 240px">
          <option value="">Select a column&hellip;</option>
          ${allCols.map(c => `<option value="${esc(c)}" ${c === chosen ? "selected" : ""}>
            ${esc(c)}${candidates.includes(c) ? "" : " (excluded from features)"}</option>`).join("")}
        </select>
        <button class="btn" id="prjsettarget">Use this column</button>
      </div>
    </div>`;

  /* Once a target is chosen the inferred problem type decides whether this
     stack can fit it at all -- and an unsupported one is refused here rather
     than attempted and quietly mangled. */
  let verdict = "";
  if(chosen){
    const supported = d.problem_supported;
    PRJ.problemType = d.problem_type;
    PRJ.problemSupported = supported;
    PRJ.primaryMetric = d.primary_metric;
    prjSave();
    const pretty = String(d.problem_type || "unknown").replace(/_/g, " ");
    verdict = supported ? `
      <div style="margin-top:20px">${card("Target confirmed", `
        <div class="grid g3">
          ${kpi("Target", `<span class="mono">${esc(chosen)}</span>`)}
          ${kpi("Problem type", esc(pretty))}
          ${kpi("Supported", badge("YES","ok",true))}
        </div>
        ${d.metric_note ? `<p class="dim" style="margin:14px 0 0;font-size:12.5px">
          ${esc(d.metric_note)}</p>` : ""}`, { flush:true })}
        <div class="wiznav">
          <button class="btn pri" data-goto="4">Continue to training</button>
          <button class="btn" data-goto="2">Back</button></div></div>`
      : `<div style="margin-top:20px">${card("Unsupported problem type", `
        <div class="verdict fail">
          <span class="mark" aria-hidden="true">&times;</span>
          <span class="txt"><b>FMOPS CANNOT TRAIN THIS TARGET</b>
            <span>Detected: ${esc(pretty)}</span></span></div>
        <p style="margin:0 0 10px;font-size:13px;line-height:1.6">
          This platform currently fits <b>binary classification</b> only. That is a property of
          the training stack, not of the profiler: the model factory builds classifiers, the
          label is cast with <span class="mono">astype(int)</span>, and evaluation reads
          <span class="mono">predict_proba[:, 1]</span>. Training a
          ${esc(pretty)} target here would produce a meaningless score, so the API refuses it.</p>
        <p class="dim" style="margin:0;font-size:12.5px">Supported:
          ${(d.supported_problem_types||[]).map(t =>
            `<span class="mono">${esc(String(t).replace(/_/g," "))}</span>`).join(", ") || "&mdash;"}</p>`,
        { flush:true })}
        <div class="wiznav">
          <button class="btn pri" id="prjcleartarget">Back to target selection</button>
        </div></div>`;
  }

  return wizFrame(3, "What do you want to predict?",
    `FMOps recommends a target from observable evidence, but you confirm it. The recommendation
     is a deterministic score over column name, cardinality, missingness and position &mdash;
     never a guess the platform acts on by itself.`,
    prjContext() + warnNoTarget + rec + warnLow + picker + verdict);
}

function profileFor(target){
  return api.get(`/api/v1/automl/profile/${encodeURIComponent(PRJ.version)}`
    + `?target=${encodeURIComponent(target)}`, 60000);
}

/* ===================================================================== */
/* 04 Training strategy                                                  */
/* ===================================================================== */
async function stepTraining(){
  if(!PRJ.target || !PRJ.problemSupported)
    return wizFrame(4, "Training", "A supported target must be confirmed first.",
      unavailable("No usable target selected.") + `<div class="wiznav">
        <button class="btn pri" data-goto="3">Back to target</button></div>`);

  const d = PRJ.profile || {};
  const cands = (d.candidates || []).filter(c => c.available);
  const preselect = d.default_selection || [];
  const maxModels = d.max_models_limit || 3;

  let algos = {};
  try { algos = (await api.get("/api/v1/models/algorithms", 300000)).algorithms || {}; }
  catch(e){ algos = {}; }
  const installed = Object.keys(algos).filter(k => algos[k]);

  const automl = `
    <div class="choice rec">
      <span class="tag">RECOMMENDED</span>
      <h4>AutoML</h4>
      <p>Let FMOps evaluate several supported algorithms on this dataset and rank them by the
        primary metric.</p>
      <ul>
        <li>Dataset analysis</li><li>Candidate selection</li><li>Multiple model training</li>
        <li>Evaluation</li><li>Deterministic ranking</li><li>Best candidate registered</li>
      </ul>
      <div class="foot">
        <div style="margin-bottom:10px">
          <label style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
            Candidates (up to ${maxModels})</label>
          <div style="display:grid;gap:6px">
            ${cands.map(c => `<label class="cand" style="display:flex;gap:8px;align-items:flex-start">
              <input type="checkbox" class="prjcand" value="${esc(c.algorithm)}"
                ${preselect.includes(c.algorithm) ? "checked" : ""}>
              <span><b style="font-size:12.5px">${esc(c.label)}</b>
                ${badge(c.tier, c.tier === "recommended" ? "ok" : c.tier === "baseline" ? "info" : "mute")}
                <br><span class="dim" style="font-size:11.5px">${esc((c.reasons||[])[0] || "")}</span>
              </span></label>`).join("") || `<span class="dim">No candidates available.</span>`}
          </div>
        </div>
        <p id="prjcandcount" class="dim" style="font-size:12px;margin:0 0 10px"></p>
        <button class="btn pri" id="prjstartautoml">Start AutoML</button>
      </div>
    </div>`;

  const manual = `
    <div class="choice">
      <h4>Manual training</h4>
      <p>Choose the algorithm yourself. Same pipeline, same gate &mdash; you pick the estimator
        instead of comparing several.</p>
      <div class="foot">
        <div style="margin-bottom:10px">
          <label for="prjalgo" style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
            Algorithm</label>
          <select id="prjalgo" style="width:100%">
            ${installed.map(a => `<option value="${esc(a)}">${esc(a.replace(/_/g," "))}</option>`).join("")}
          </select>
          <p class="dim" style="font-size:11.5px;margin:6px 0 0">Only estimators this environment
            can actually build are listed${Object.keys(algos).length > installed.length
              ? `; ${Object.keys(algos).length - installed.length} optional backend(s) are not installed`
              : ""}.</p>
        </div>
        <label style="display:flex;gap:8px;align-items:center;margin-bottom:10px;font-size:12.5px">
          <input type="checkbox" id="prjtune"> Run hyperparameter search first (slower)</label>
        <button class="btn" id="prjstartmanual">Start training</button>
      </div>
    </div>`;

  const shared = card("What happens to the winner", `
    <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end">
      <div><label for="prjstage" style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
        Promote into</label>
        <select id="prjstage">
          <option value="Staging" selected>Staging</option>
          <option value="Production">Production</option>
        </select></div>
      <p class="dim" style="flex:1 1 320px;margin:0;font-size:12.5px;line-height:1.6">
        Both paths register the resulting model and put it through the approval gate
        automatically &mdash; that happens inside the run, which is why there is no separate
        register button later. Asking for a stage only asks for the attempt: promotion still
        requires clearing the absolute thresholds and beating the incumbent.</p>
    </div>`, { flush:true });

  const note = `<div class="note" style="margin-top:16px">
    <b>Where this runs.</b> <span style="font-size:12.5px">Runs execute as background tasks
    inside the API process, so training competes with request handling for the same CPU, and a
    run in flight is lost if the process restarts. Each candidate is a full training run, so
    <i>n</i> candidates cost roughly <i>n</i> times one run.</span></div>`;

  return wizFrame(4, "How should FMOps train?",
    `Both routes use the same training pipeline, the same evaluation and the same approval gate.
     AutoML compares several estimators and ranks them; manual training fits the one you pick.`,
    prjContext() + shared + `<div class="choices" style="margin-top:16px">${automl}${manual}</div>`
    + note + `<div id="prjstartresult" style="margin-top:16px"></div>
    <div class="wiznav"><button class="btn" data-goto="3">Back</button></div>`);
}

/* ===================================================================== */
/* 05 Evaluation                                                         */
/* ===================================================================== */
function runPath(){
  return PRJ.runKind === "automl"
    ? `/api/v1/automl/runs/${encodeURIComponent(PRJ.runId)}`
    : `/api/v1/training/runs/${encodeURIComponent(PRJ.runId)}`;
}
const RUN_TERMINAL = ["completed", "completed_with_warnings", "failed", "rejected"];
function runDone(status){ return RUN_TERMINAL.includes(status); }

/* Record what the run actually produced. Called from both the poller and the
   step renderer so a reload picks up a run that finished while away. */
function absorbRun(run){
  PRJ.runStatus = run.status;
  if(PRJ.runKind === "automl"){
    if(run.best_model_version != null) PRJ.modelVersion = run.best_model_version;
    const promo = run.promotion || null;
    if(promo && promo.decision) PRJ.gateDecision = promo.decision;
  } else {
    if(run.model_version != null) PRJ.modelVersion = run.model_version;
    if(run.model_name) PRJ.modelName = run.model_name;
  }
  prjSave();
}

async function stepEvaluation(){
  if(!PRJ.runId) return wizFrame(5, "Evaluation", "No run has been started.",
    unavailable("Start a training run first.") + `<div class="wiznav">
      <button class="btn pri" data-goto="4">Back to training</button></div>`);

  let run;
  try { run = await api.get(runPath(), 0); }
  catch(e){
    return wizFrame(5, "Evaluation", "The run could not be read.",
      errorState(e.message, "newproject"));
  }
  absorbRun(run);

  const live = !runDone(run.status);
  const body = `<div data-progress>`
    + (PRJ.runKind === "automl" ? automlBody(run, live) : trainingBody(run, live))
    + `</div>`;
  const nav = live ? "" : `<div class="wiznav">
    ${run.status === "failed"
      ? `<button class="btn pri" data-goto="4">Try a different configuration</button>`
      : `<button class="btn pri" data-goto="6">Continue to registry</button>`}
    <button class="btn" data-goto="4">Back</button></div>`;

  return wizFrame(5, live ? "Training candidate models" : "Evaluation",
    live
      ? `The run is executing now. This page follows it until it reaches a terminal state and
         then stops polling &mdash; there is nothing further to learn from a finished run.`
      : `Results as the platform recorded them. Nothing here is recomputed in the browser.`,
    prjContext() + body + `<div id="prjrunbox"></div>` + nav)
    + (live ? `<span id="prjpoll" data-run="${esc(PRJ.runId)}" hidden></span>` : "");
}

function statusBadge(s){
  const kind = s === "completed" ? "ok" : s === "completed_with_warnings" ? "warn"
    : s === "failed" || s === "rejected" ? "bad"
    : s === "running" ? "info" : "mute";
  return badge(String(s || "unknown").replace(/_/g, " "), kind, s === "running" || s === "queued");
}

function automlBody(run, live){
  const cands = run.candidates || [];
  const finished = cands.filter(c => c.status !== "queued" && c.status !== "training").length;
  const pct2 = runDone(run.status) ? 100 : (cands.length ? Math.round(finished/cands.length*100) : 5);
  const stages = ["profiling", "training", "ranking", "completed"];
  const reachedName = run.status === "completed_with_warnings" ? "completed" : run.status;
  const reached = stages.indexOf(reachedName);

  const progress = `${card("Progress", `
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
      ${statusBadge(run.status)}
      <span class="spacer"></span>
      <span class="mono dim">${pct2}%</span>
    </div>
    <div class="bar" style="margin-bottom:12px"><i style="width:${pct2}%"></i></div>
    <div class="flow" style="margin-bottom:14px">${stages.map((s,i) =>
      `<span class="step ${runDone(run.status)||i<reached ? "done" : i===reached ? "on" : ""}">
        ${esc(s)}</span>` + (i<stages.length-1 ? '<span class="arw">&rarr;</span>' : "")).join("")}</div>
    <div style="display:grid;gap:7px">${cands.map(c => `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
        <span class="mono">${esc(c.algorithm)}</span>
        <span style="display:flex;gap:8px;align-items:center">
          ${c.duration_seconds != null
            ? `<span class="mono dim">${Number(c.duration_seconds).toFixed(1)}s</span>` : ""}
          ${statusBadge(c.status)}</span></div>`).join("")
      || `<span class="dim">No candidates recorded yet.</span>`}</div>
    ${run.duration_seconds != null
      ? `<p class="dim" style="margin:12px 0 0;font-size:12px">Elapsed
         ${Number(run.duration_seconds).toFixed(1)}s</p>` : ""}`, { flush:true })}`;

  if(live) return progress;

  if(run.status === "failed") return progress + card("Run failed", `
    <div class="verdict fail"><span class="mark" aria-hidden="true">&times;</span>
      <span class="txt"><b>NO MODEL WAS PRODUCED</b>
        <span>The error below is what the backend recorded.</span></span></div>
    <pre class="scroll" style="margin:0">${esc(run.error || "no error recorded")}</pre>`,
    { flush:true });

  const board = run.leaderboard || [];
  const metric = run.primary_metric || "roc_auc";
  const lb = card("Model leaderboard", table([
    { label:"#", render:c => c.rank === 1
        ? `<span class="ranked">1</span>` : `<span class="ranked other">${int(c.rank)}</span>` },
    { label:"Algorithm", render:c => `<b class="mono">${esc(c.algorithm)}</b>` },
    { label:metric.toUpperCase().replace(/_/g,"-"), num:true,
      render:c => num((c.metrics||{})[metric], 4) },
    { label:"F1", num:true, render:c => num((c.metrics||{}).f1, 4) },
    { label:"Precision", num:true, render:c => num((c.metrics||{}).precision, 4) },
    { label:"Recall", num:true, render:c => num((c.metrics||{}).recall, 4) },
    { label:"Time", num:true, render:c => c.duration_seconds != null
        ? `${Number(c.duration_seconds).toFixed(1)}s` : NA },
    { label:"", render:c => `<button class="btn" data-cand="${esc(c.algorithm)}">Details</button>` },
  ], board, { empty:"No ranked candidates.", rowClass:c => c.rank === 1 ? "win" : "" }),
    { flush:true, sub:`best: ${esc(run.best_algorithm || "-")}` });

  const failedCands = (run.candidates||[]).filter(c => c.status === "failed");
  const partial = failedCands.length ? `<div class="note"
    style="border-color:var(--warn-line);background:var(--warn-bg);margin-top:14px">
    <b>${failedCands.length} candidate(s) failed and were not ranked.</b>
    <div style="font-size:12.5px;margin-top:6px">${failedCands.map(c =>
      `<div><span class="mono">${esc(c.algorithm)}</span> &mdash; ${esc(c.error||"no error recorded")}</div>`
      ).join("")}</div></div>` : "";

  const rule = run.ranking_rule ? card("How ranking works", `
    <p style="margin:0 0 8px;font-size:13px">Candidates are ordered by:</p>
    <p class="mono" style="margin:0 0 10px;font-size:12.5px">${esc(run.ranking_rule)}</p>
    <p class="dim" style="margin:0;font-size:12.5px">Quoted from the run record, so what is
      shown is the rule that actually ordered these candidates.</p>`, { flush:true }) : "";

  return progress + lb + partial + rule;
}

function trainingBody(run, live){
  const m = run.metrics || {};
  const head = card("Run", `
    <div class="grid g4">
      ${kpi("Status", statusBadge(run.status))}
      ${kpi("Algorithm", `<span class="mono">${esc(run.algorithm||"-")}</span>`)}
      ${kpi("Tuning", run.tune ? "enabled" : "off")}
      ${kpi("Duration", run.duration_seconds != null
        ? `${Number(run.duration_seconds).toFixed(1)}s` : NA)}
    </div>`, { flush:true });

  if(live) return head + `<div class="note" style="margin-top:14px">
    <span class="spin" aria-hidden="true"></span> Training is running in the background.
    This page follows it until it finishes.</div>`;

  if(run.status === "failed" || run.status === "rejected") return head + card(
    run.status === "rejected" ? "Rejected by the pipeline" : "Run failed", `
    <div class="verdict fail"><span class="mark" aria-hidden="true">&times;</span>
      <span class="txt"><b>${run.status === "rejected"
        ? "THE PIPELINE REJECTED THIS RUN" : "NO MODEL WAS PRODUCED"}</b>
        <span>The message below is what the backend recorded.</span></span></div>
    <pre class="scroll" style="margin:0">${esc(run.error || "no error recorded")}</pre>`,
    { flush:true });

  const keys = Object.keys(m);
  return head + card("Metrics", keys.length ? `<div class="grid g4">
    ${keys.slice(0,8).map(k => kpi(k.replace(/_/g," "), metricValue(m[k]))).join("")}</div>`
    : unavailable("The run recorded no metrics."), { flush:true });
}

/* ===================================================================== */
/* 06 Registry                                                           */
/* ===================================================================== */
async function stepRegistry(){
  if(!PRJ.runId) return wizFrame(6, "Registry", "No run has been started.",
    unavailable("Nothing has been trained yet."));

  let run = null;
  try { run = await api.get(runPath(), 0); absorbRun(run); } catch(e){ /* fall through */ }

  if(!PRJ.modelVersion){
    const why = PRJ.runKind === "training"
      ? `This run was started without asking for registration, so the model was trained and
         evaluated but never entered the registry. There is no endpoint that registers a
         finished run after the fact &mdash; registration happens inside the run.`
      : `The run did not record a registered model version.`;
    return wizFrame(6, "Model registry", "Nothing was registered by this run.",
      prjContext() + `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>No model version was registered.</b>
        <div style="font-size:12.5px;margin-top:6px">${why}</div></div>
      <div class="wiznav"><button class="btn pri" data-goto="4">Run again with registration</button>
        <button class="btn" data-goto="5">Back to results</button></div>`);
  }

  /* The registered name comes from the registry, never from this page: the
     platform registers every model under one configured name, so inventing a
     per-dataset name here would misrepresent what is actually stored. */
  let versions = [], modelName = PRJ.modelName;
  try {
    const models = await api.get("/api/v1/models", 15000);
    const names = (models.models || []).map(m => typeof m === "string" ? m : m.name);
    modelName = modelName || names[0] || null;
    if(modelName){
      versions = await api.get(`/api/v1/models/${encodeURIComponent(modelName)}/versions`, 5000);
    }
  } catch(e){ /* reported below */ }
  PRJ.modelName = modelName; prjSave();

  const mine = versions.find(v => Number(v.version) === Number(PRJ.modelVersion)) || null;
  if(mine) { PRJ.registeredStage = mine.stage; prjSave(); }

  const detail = mine ? card("Registered", `
    <div class="verdict pass" style="margin-bottom:14px">
      <span class="mark" aria-hidden="true">&check;</span>
      <span class="txt"><b>MODEL REGISTERED</b>
        <span>Recorded by the registry during the run.</span></span></div>
    <div class="grid g4">
      ${kpi("Model", `<span class="mono">${esc(modelName)}</span>`)}
      ${kpi("Version", `<span class="mono">v${int(mine.version)}</span>`)}
      ${kpi("Stage", badge(mine.stage, mine.stage === "Production" ? "ok"
        : mine.stage === "Staging" ? "info" : "mute"))}
      ${kpi("Algorithm", `<span class="mono">${esc(mine.algorithm
        || (PRJ.runKind === "automl" && run && run.best_algorithm) || "-")}</span>`)}
    </div>
    ${mine.metrics ? `<div class="grid g4" style="margin-top:14px">
      ${Object.keys(mine.metrics).slice(0,4).map(k =>
        kpi(k.replace(/_/g," "), num(mine.metrics[k], 4))).join("")}</div>` : ""}
    <p class="dim" style="margin:14px 0 0;font-size:12.5px">
      Every model on this platform is registered under one configured name
      (<span class="mono">${esc(modelName)}</span>) and distinguished by version. There is no
      per-dataset model namespace, so this version sits in the same lineage as every other
      model trained here.</p>`, { flush:true })
    : `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>Version v${int(PRJ.modelVersion)} was recorded by the run but could not be read back
        from the registry.</b></div>`;

  return wizFrame(6, "Model registry",
    `Registration happened inside the run: both AutoML and a training run with promotion
     requested register the winner and put it through the gate before the run reports back.
     This step shows what the registry actually holds.`,
    prjContext() + detail + `<div class="wiznav">
      <button class="btn pri" data-goto="7">Continue to approval</button>
      <button class="btn" data-goto="5">Back to results</button></div>`);
}

/* The platform registers every model under one configured name, so the name is
   read from the registry rather than guessed from the dataset. */
async function ensureModelName(){
  if(PRJ.modelName) return PRJ.modelName;
  try {
    const models = await api.get("/api/v1/models", 15000);
    const names = (models.models || []).map(m => typeof m === "string" ? m : m.name);
    PRJ.modelName = names[0] || null;
    prjSave();
  } catch(e){ /* caller renders the unavailable state */ }
  return PRJ.modelName;
}

/* ===================================================================== */
/* 07 Approval gate                                                      */
/* ===================================================================== */
/* Two sources, deliberately kept apart:
 *
 *   - what the gate decided during the run (recorded, needs no auth), and
 *   - what the gate would decide against the thresholds in force right now.
 *
 * They can differ if configuration changed since the run, and collapsing them
 * into one number would hide exactly the discrepancy an operator cares about.
 */
async function stepApproval(){
  if(!PRJ.modelVersion) return wizFrame(7, "Approval gate", "No registered model version.",
    unavailable("Nothing has been registered yet.") + `<div class="wiznav">
      <button class="btn pri" data-goto="6">Back to registry</button></div>`);

  await ensureModelName();
  if(!PRJ.modelName) return wizFrame(7, "Approval gate", "The registered model could not be named.",
    unavailable("The registry did not report any model to evaluate against."));

  let recorded = null;
  if(PRJ.runKind === "automl"){
    try {
      const run = await api.get(runPath(), 0);
      recorded = run.promotion || null;
      if(recorded && recorded.decision){ PRJ.gateDecision = recorded.decision; prjSave(); }
    } catch(e){ /* recorded outcome simply unavailable */ }
  }

  let gate = null, gateError = null, needsKey = false;
  try {
    gate = await api.post(
      `/api/v1/models/${encodeURIComponent(PRJ.modelName)}/versions/${PRJ.modelVersion}/evaluate-gate`);
  } catch(e){
    if(e.status === 401 || e.status === 403) needsKey = true;
    else gateError = e.message;
  }
  if(gate && gate.approval && gate.approval.decision){
    PRJ.gateDecision = gate.approval.decision; prjSave();
  }

  const decision = (gate && gate.approval && gate.approval.decision)
    || (recorded && recorded.decision) || null;
  const approved = decision === "approved";

  const verdict = decision ? `<div class="verdict ${approved ? "pass" : "fail"}">
      <span class="mark" aria-hidden="true">${approved ? "&check;" : "&times;"}</span>
      <span class="txt"><b>${approved ? "APPROVED FOR PRODUCTION" : "NOT ELIGIBLE FOR PRODUCTION"}</b>
        <span>${approved
          ? "Every blocking check passed. Deployment is unlocked."
          : "At least one blocking check failed. Production deployment stays locked."}</span>
      </span></div>`
    : `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>The gate outcome could not be read.</b>
        <div style="font-size:12.5px;margin-top:6px">${needsKey
          ? "Re-running the gate is a write-method call and this API requires a key for it."
          : esc(gateError || "The endpoint returned nothing usable.")}</div></div>`;

  const keyPanel = needsKey ? card("Re-run the gate", `
    <p class="dim" style="margin:0 0 10px;font-size:12.5px">This API requires a key for
      write-method calls, and the gate endpoint is a POST. Supplying a key here re-runs the gate
      against the thresholds currently in force. The key is used for this request and is not
      stored.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input type="password" id="prjgatekey" placeholder="X-API-Key" style="flex:1 1 240px"
        autocomplete="off">
      <button class="btn" id="prjrungate">Run the gate</button></div>`, { flush:true }) : "";

  const checks = (gate && gate.approval && gate.approval.checks) || [];
  const checkTable = checks.length ? card("Production gate", `
    ${table([
      { label:"Check", render:c => `<span class="mono">${esc(c.name)}</span>` },
      { label:"Observed", num:true, render:c => typeof c.observed === "number"
          ? num(c.observed, 4) : (c.observed != null ? esc(String(c.observed)) : NA) },
      { label:"Required", num:true, render:c => typeof c.threshold === "number"
          ? num(c.threshold, 4) : (c.threshold != null ? esc(String(c.threshold)) : NA) },
      { label:"Blocking", render:c => c.blocking ? badge("blocking","warn") : badge("advisory","mute") },
      { label:"Result", render:c => c.passed ? badge("PASSED","ok") : badge("FAILED","bad") },
    ], checks, { empty:"" })}
    <p class="dim" style="margin:12px 0 0;font-size:12.5px">Thresholds come from the platform
      configuration, not from this page. They are not relaxed to make a candidate pass.</p>`,
    { flush:true, sub:`${checks.filter(c => !c.passed).length} failed` }) : "";

  const cmp = (gate && gate.comparison) || (recorded && recorded.comparison) || null;
  const champion = cmp ? card("Champion vs candidate", `
    ${table([
      { label:"", render:r => `<b>${esc(r.k)}</b>` },
      { label:"Champion" + (cmp.baseline_version != null ? ` (v${cmp.baseline_version})` : ""),
        num:true, render:r => r.a },
      { label:"Candidate" + (cmp.candidate_version != null ? ` (v${cmp.candidate_version})` : ""),
        num:true, render:r => r.b },
    ], [
      { k:String(cmp.metric || "metric").replace(/_/g," "),
        a:num(cmp.baseline_score, 4), b:num(cmp.candidate_score, 4) },
      { k:"Improvement", a:NA,
        b:`${cmp.improvement >= 0 ? "+" : ""}${num(cmp.improvement, 4)}` },
      { k:"Minimum required", a:NA, b:num(cmp.min_improvement, 4) },
    ], { empty:"" })}
    <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      ${cmp.candidate_is_better ? badge("candidate is better","ok") : badge("not an improvement","warn")}
      ${cmp.decision ? badge(String(cmp.decision).replace(/_/g," "),
        cmp.decision === "promote" ? "ok" : "mute") : ""}
    </div>
    ${cmp.reason ? `<p class="dim" style="margin:10px 0 0;font-size:12.5px">${esc(cmp.reason)}</p>` : ""}
    <p class="dim" style="margin:10px 0 0;font-size:12.5px">Beating the incumbent is necessary
      but not sufficient: a candidate must also clear the absolute thresholds above.</p>`,
    { flush:true }) : (PRJ.modelVersion ? card("Champion vs candidate",
      unavailable("No incumbent to compare against, or the comparison was not recorded."),
      { flush:true }) : "");

  const failedNames = (recorded && recorded.failed_checks) || [];
  const recordedPanel = recorded ? card("Recorded at run time", `
    <div class="grid g3">
      ${kpi("Decision", badge(String(recorded.decision||"-").toUpperCase(),
        recorded.decision === "approved" ? "ok" : "bad"))}
      ${kpi("Promoted", recorded.promoted ? badge("YES","ok") : badge("NO","mute"))}
      ${kpi("Final stage", esc(recorded.final_stage || "-"))}
    </div>
    ${recorded.reason ? `<p class="dim" style="margin:12px 0 0;font-size:12.5px">
      ${esc(recorded.reason)}</p>` : ""}
    ${failedNames.length ? `<p style="margin:10px 0 0;font-size:12.5px">Failed checks:
      ${failedNames.map(n => `<span class="mono">${esc(n)}</span>`).join(", ")}</p>` : ""}
    <p class="dim" style="margin:10px 0 0;font-size:12px">This is what the gate decided while the
      run executed. The panels above re-run it against the thresholds in force now.</p>`,
    { flush:true }) : "";

  const nav = `<div class="wiznav">
    ${approved
      ? `<button class="btn pri" data-goto="8">Continue to deployment</button>`
      : `<button class="btn pri" data-goto="4">Train another candidate</button>`}
    <button class="btn" data-goto="6">Back</button>
    ${!approved ? `<span class="dim" style="font-size:12.5px">Deployment stays locked until a
      candidate satisfies the configured policy.</span>` : ""}</div>`;

  return wizFrame(7, "Model readiness",
    `The backend owns this decision. This page reads the gate; it cannot override it, and there
     is no path through this workflow that deploys a version the gate rejected.`,
    prjContext() + verdict + keyPanel + checkTable + champion + recordedPanel + nav);
}

/* ===================================================================== */
/* 08 Deploy                                                             */
/* ===================================================================== */
async function stepDeploy(){
  if(!PRJ.modelVersion) return wizFrame(8, "Deploy", "No registered model version.",
    unavailable("Nothing has been registered yet."));

  await ensureModelName();
  const approved = PRJ.gateDecision === "approved";
  let current = null;
  try { current = await api.get("/api/v1/deployments/current", 5000); } catch(e){ /* below */ }

  const live = current && current.deployment ? current.deployment : null;
  const state = card("Endpoint now", live ? `
    <div class="grid g4">
      ${kpi("Endpoint", `<span class="mono">${esc(live.endpoint_name)}</span>`)}
      ${kpi("Serving", `<span class="mono">v${int(live.current_version)}</span>`)}
      ${kpi("State", badge(live.state, live.state === "live" ? "ok" : "info"))}
      ${kpi("Previous", live.previous_version != null
        ? `<span class="mono">v${int(live.previous_version)}</span>` : NA)}
    </div>` : unavailable(current
      ? "No deployment has been created for this endpoint yet."
      : "The deployments API could not be reached."), { flush:true });

  const form = approved ? `
    <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end">
      <div><label for="prjstrategy" style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
        Strategy</label>
        <select id="prjstrategy">
          <option value="blue_green" selected>Blue / green</option>
          <option value="direct">Direct</option>
          <option value="canary">Canary</option>
        </select></div>
      <div style="flex:1 1 260px"><p class="dim" style="margin:0;font-size:12.5px;line-height:1.6">
        Deploys <span class="mono">${esc(PRJ.modelName||"")} v${int(PRJ.modelVersion)}</span>.
        Shadow deployments are not offered here because they mirror traffic from a live version
        rather than replacing it &mdash; start one from the
        <a href="#/deployments">Deployments page</a>.</p></div>
    </div>
    <div style="margin-top:14px"><button class="btn pri" id="prjdeploy">Deploy this version</button></div>
    <div id="prjdeployresult" style="margin-top:14px"></div>`
    : `<div class="verdict fail">
        <span class="mark" aria-hidden="true">&#128274;</span>
        <span class="txt"><b>DEPLOYMENT LOCKED</b>
          <span>This version has not cleared the approval gate.</span></span></div>
      <p style="margin:0;font-size:13px;line-height:1.6">The deployment API refuses a version
        that is not in a deployable stage, and this workflow does not offer the override that
        would bypass it. Train a candidate that satisfies the policy, or promote an eligible
        version from the <a href="#/models">Model Registry</a>.</p>`;

  return wizFrame(8, "Deploy",
    `Rolling a version onto the serving endpoint. The strategy decides how traffic moves; the
     approval gate decides whether it moves at all.`,
    prjContext() + state + card("Roll out", form, { flush:true }) + `<div class="wiznav">
      ${PRJ.deployedVersion ? `<button class="btn pri" data-goto="9">Continue to predictions</button>` : ""}
      <button class="btn" data-goto="7">Back</button></div>`);
}

/* ===================================================================== */
/* 09 Predict                                                            */
/* ===================================================================== */
/* Online scoring accepts exactly one request shape. Rather than hardcode a
   copy of it here -- which would drift the first time the backend changed --
   the field list is read from the served OpenAPI document, and the step
   refuses to render a form it knows would be rejected. */
async function predictSchema(){
  const doc = await api.get("/openapi.json", 300000);
  const schemas = (doc.components && doc.components.schemas) || {};
  const body = ((((doc.paths || {})["/api/v1/predict"] || {}).post || {}).requestBody || {});
  const ref = (((body.content || {})["application/json"] || {}).schema || {}).$ref || "";
  const reqName = ref.split("/").pop();
  const req = schemas[reqName] || {};
  const featRef = ((req.properties || {}).features || {}).$ref
    || (((req.properties || {}).features || {}).allOf || [{}])[0].$ref || "";
  const feat = schemas[featRef.split("/").pop()] || null;
  if(!feat || !feat.properties) return null;
  return {
    name: featRef.split("/").pop(),
    required: feat.required || Object.keys(feat.properties),
    properties: feat.properties,
  };
}

async function stepPredict(){
  if(!PRJ.deployedVersion) return wizFrame(9, "Predict", "Nothing is deployed from this workflow.",
    unavailable("Deploy a model first.") + `<div class="wiznav">
      <button class="btn pri" data-goto="8">Back to deployment</button></div>`);

  let schema = null, schemaError = null;
  try { schema = await predictSchema(); }
  catch(e){ schemaError = e.message; }

  if(!schema) return wizFrame(9, "Test predictions",
    "The request schema for online scoring could not be read.",
    prjContext() + unavailable(schemaError || "The API description did not describe /api/v1/predict."));

  const need = schema.required;
  const have = new Set((PRJ.columnNames || []).map(c => String(c)));
  const missing = need.filter(f => !have.has(f));

  /* A model trained on differently-shaped data cannot be exercised through
     this endpoint. Saying so beats rendering a form that would 422. */
  if(missing.length) return wizFrame(9, "Test predictions",
    `Online scoring on this platform accepts one fixed request shape.`,
    prjContext() + card("Not available for this dataset", `
      <div class="verdict fail"><span class="mark" aria-hidden="true">&times;</span>
        <span class="txt"><b>THIS DATASET DOES NOT MATCH THE SCORING SCHEMA</b>
          <span>The prediction endpoint would reject every request built from it.</span></span></div>
      <p style="margin:0 0 10px;font-size:13px;line-height:1.6">
        <span class="mono">POST /api/v1/predict</span> validates against a single fixed schema
        (<span class="mono">${esc(schema.name)}</span>), which belongs to this platform's
        reference model. Your dataset does not carry
        ${missing.length} of its ${need.length} required fields, so there is no honest way to
        build a request from it. The model you trained is registered and deployable; it is the
        <i>serving contract</i> that is fixed, not the training path.</p>
      <p class="dim" style="margin:0;font-size:12.5px">Missing:
        ${missing.map(m => `<span class="mono">${esc(m)}</span>`).join(", ")}</p>`, { flush:true })
    + `<div class="wiznav"><button class="btn pri" data-goto="10">Continue to monitoring</button>
       <button class="btn" data-goto="8">Back</button></div>`);

  /* Columns line up, so prefill from a real row of the uploaded data rather
     than inventing plausible-looking values. */
  let sample = null;
  try {
    const pv = await api.get(`/api/v1/datasets/${encodeURIComponent(PRJ.version)}/preview?rows=1`, 60000);
    sample = (pv.sample || [])[0] || null;
  } catch(e){ sample = null; }

  const fields = need.map(f => {
    const p = schema.properties[f] || {};
    const t = p.type || (p.anyOf ? "string" : "string");
    const numeric = t === "number" || t === "integer";
    const val = sample && sample[f] != null ? String(sample[f]) : "";
    return `<label style="display:block">
      <span style="display:block;font-size:11.5px;font-weight:600;margin-bottom:4px">
        ${esc(f)}</span>
      <input class="prjfeat" data-f="${esc(f)}" data-num="${numeric ? "1" : "0"}"
        type="${numeric ? "number" : "text"}" step="any" value="${esc(val)}"
        style="width:100%" ${p.minimum != null ? `min="${esc(p.minimum)}"` : ""}
        ${p.maximum != null ? `max="${esc(p.maximum)}"` : ""}></label>`;
  }).join("");

  return wizFrame(9, "Test predictions",
    `A real request against the live endpoint. The response reports which model version and
     which variant actually served it, so a canary or shadow rollout is visible per request.`,
    prjContext() + card("Score one record", `
      ${sample ? `<p class="dim" style="margin:0 0 12px;font-size:12.5px">Prefilled from the first
        row of <span class="mono">${esc(PRJ.version)}</span>. Edit any value before scoring.</p>`
        : `<p class="dim" style="margin:0 0 12px;font-size:12.5px">Fill in the fields below.</p>`}
      <div class="grid g3" style="gap:12px">${fields}</div>
      <div style="margin-top:14px"><button class="btn pri" id="prjpredict">Score</button></div>
      <div id="prjpredresult" style="margin-top:14px"></div>`, { flush:true })
    + `<div class="wiznav"><button class="btn pri" data-goto="10">Continue to monitoring</button>
       <button class="btn" data-goto="8">Back</button></div>`);
}

/* ===================================================================== */
/* 10 Monitor                                                            */
/* ===================================================================== */
async function stepMonitor(){
  const r = await loadAll({
    summary: "/api/v1/monitoring/summary",
    current: "/api/v1/deployments/current",
    drift:   "/api/v1/drift/latest",
  }, 5000);

  const svc = sect(r.summary, d => {
    const s = d.service || {}, lat = s.latency || {}, lp = d.live_performance || {};
    return card("Serving", `
      <div class="grid g4">
        ${kpi("Model", `<span class="mono">${esc(d.model_name||"-")}${d.model_version != null
          ? ` v${d.model_version}` : ""}</span>`, esc(d.model_stage||""))}
        ${kpi("Requests", int(s.request_count), `last ${int(s.window_minutes)} min`)}
        ${kpi("p95 latency", ms(lat.p95_ms),
          s.latency_slo_met === false ? "SLO breached" : "within SLO")}
        ${kpi("Open alerts", int(d.open_alerts))}
      </div>
      <div class="grid g4" style="margin-top:14px">
        ${kpi("Error rate", pct(s.error_rate, 2))}
        ${kpi("Positive rate", pct(d.prediction_positive_rate, 2))}
        ${kpi("Labelled samples", int(lp.labelled_samples))}
        ${kpi("Live F1", lp.available ? num(lp.f1, 4) : NA,
          lp.available ? "from returned labels" : "needs labels")}
      </div>
      <p class="dim" style="margin:14px 0 0;font-size:12.5px">${lp.available
        ? "Live performance is computed only over predictions whose true outcome came back."
        : esc(lp.detail || "Live accuracy cannot be computed without ground-truth labels.")}</p>`,
      { flush:true });
  }, "monitoring summary");

  /* /drift/latest answers {found, report}: "no scan yet" is a legitimate
     state, and reporting it as stable would be an invented reassurance. */
  const drift = sect(r.drift, d => {
    const rep = d && d.report;
    if(!d || !d.found || !rep) return card("Drift", unavailable(
      "No drift report has been produced yet. Run a scan from the Drift Detection page."),
      { flush:true });
    const drifted = (rep.drifted_features || []).length;
    const examined = (rep.feature_drift || []).length;
    const status = String(rep.concept_drift_status || "unknown");
    const measured = status === "measured";
    return card("Latest drift scan", `
      <div class="grid g4">
        ${kpi("Status", rep.drift_detected ? badge("DRIFT","warn",true) : badge("STABLE","ok",true))}
        ${kpi("Features drifted", int(drifted), `of ${examined} examined`)}
        ${kpi("Share drifted", pct(rep.dataset_drift_share, 1))}
        ${kpi("Scanned", when(rep.created_at))}
      </div>
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--line-2)">
        <div style="display:flex;gap:10px;align-items:center;margin-bottom:6px">
          <b style="font-size:13px">Concept drift</b>
          ${badge(status.replace(/_/g," "), measured ? "info" : "mute")}
        </div>
        <p class="dim" style="margin:0;font-size:12.5px">${esc(rep.concept_drift_detail
          || "No detail recorded.")}</p>
      </div>
      <p class="dim" style="margin:12px 0 0;font-size:12.5px">The counts above are <b>data and
        prediction drift</b> &mdash; changes in the inputs and outputs. It is not concept drift,
        which is a change in the relationship between features and outcome and is only
        measurable once true labels come back${measured
          ? "; the status above says labels were available for this scan."
          : "."}</p>`, { flush:true });
  }, "drift");

  const dep = sect(r.current, d => d.deployment ? card("Deployment", `
    <div class="grid g4">
      ${kpi("Endpoint", `<span class="mono">${esc(d.deployment.endpoint_name)}</span>`)}
      ${kpi("Serving", `<span class="mono">v${int(d.deployment.current_version)}</span>`)}
      ${kpi("State", badge(d.deployment.state, d.deployment.state === "live" ? "ok" : "info"))}
      ${kpi("Strategy", esc(d.deployment.strategy || "-"))}
    </div>`, { flush:true })
    : card("Deployment", unavailable("No deployment on this endpoint."), { flush:true }), "deployment");

  return wizFrame(10, "Monitoring",
    `The workflow ends where operations begin. Everything below is live platform state, shared
     with the Command Center &mdash; this step is a starting point, not a separate copy.`,
    prjContext() + svc + dep + drift + card("Where to go next", `
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <a class="btn" href="#/monitoring">Monitoring</a>
        <a class="btn" href="#/drift">Drift Detection</a>
        <a class="btn" href="#/retraining">Retraining</a>
        <a class="btn" href="#/champion">Champion / Challenger</a>
        <a class="btn" href="#/audit">Audit Log</a>
        <a class="btn" href="#/overview">Command Center</a>
      </div>
      <p class="dim" style="margin:12px 0 0;font-size:12.5px">Retraining is driven by the
        platform's own triggers and by drift, not by this workflow.</p>`, { flush:true })
    + `<div class="wiznav">
        <button class="btn" data-goto="9">Back</button>
        <button class="btn" id="prjreset">Start another project</button></div>`);
}

/* ===================================================================== */
/* Page                                                                  */
/* ===================================================================== */
const STEP_RENDER = {
  1: stepDataset, 2: stepProfile, 3: stepTarget, 4: stepTraining, 5: stepEvaluation,
  6: stepRegistry, 7: stepApproval, 8: stepDeploy, 9: stepPredict, 10: stepMonitor,
};

PAGES.newproject = {
  title: "New ML Project",
  intro: "A guided path from a CSV to a deployed, monitored model. Each step drives the same "
       + "APIs the rest of the console uses; progress is kept in this browser tab.",
  async render(){
    const n = currentStep();
    return await STEP_RENDER[n]();
  },
  wire(){ wireProject(); },
};

/* ===================================================================== */
/* Wiring                                                                */
/* ===================================================================== */
let PRJ_POLL = null;

function wireProject(){
  if(PRJ_POLL){ clearTimeout(PRJ_POLL); PRJ_POLL = null; }

  document.querySelectorAll("[data-goto]").forEach(b => b.onclick = () => {
    const n = parseInt(b.dataset.goto, 10);
    if(stepState(n) === "locked"){ toast("That step is not available yet.", "bad"); return; }
    goStep(n);
  });

  document.querySelectorAll("#prjreset").forEach(b => b.onclick = async () => {
    const ok = await confirmAction({
      title: "Start over",
      body: "Clears this walkthrough's progress in your browser. Datasets, runs, model "
          + "versions and deployments already created stay exactly where they are.",
      confirm: "Start over",
    });
    if(!ok) return;
    prjReset(); api.bust(); goStep(1); render();
  });

  wireUpload();
  wireTarget();
  wireStart();
  wireDrawer();
  wireGate();
  wireDeploy();
  wirePredict();

  const poll = $("#prjpoll");
  if(poll) pollRunStep(poll.dataset.run, 0);
}

/* --- 01 upload ---------------------------------------------------------- */
function wireUpload(){
  const zone = $("#prjdrop"), input = $("#prjfile");
  if(!zone || !input) return;

  ["dragenter","dragover"].forEach(ev => zone.addEventListener(ev, e => {
    e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave","drop"].forEach(ev => zone.addEventListener(ev, e => {
    e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", e => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if(f) doUpload(f);
  });
  input.onchange = () => { const f = input.files && input.files[0]; if(f) doUpload(f); };
}

async function doUpload(file){
  const box = $("#prjupstatus");
  const fail = msg => { box.innerHTML =
    `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
      <b>Upload failed.</b><br><span style="font-size:12.5px">${esc(msg)}</span></div>`;
    toast(msg, "bad"); };

  if(!/\.csv$/i.test(file.name)) return fail(
    `${file.name} is not a .csv file. This endpoint accepts CSV only.`);
  if(file.size === 0) return fail("That file is empty.");

  let limit = null;
  try { limit = (await api.get("/api/v1/datasets/limits", 300000)).max_upload_bytes; } catch(e){}
  if(limit && file.size > limit) return fail(
    `${file.name} is ${(file.size/1048576).toFixed(1)} MB, over the `
    + `${Math.floor(limit/1048576)} MB limit the API accepts.`);

  const needsKey = await authRequired();
  let key = null;
  if(needsKey){
    const proceed = await confirmAction({
      title: "Upload dataset",
      body: `Register ${file.name} (${(file.size/1024).toFixed(0)} KB) as a new dataset version `
          + `and validate it.`,
      confirm: "Upload", needsKey: true,
    });
    if(!proceed) return;
    if(!proceed.key){ toast("An API key is required to upload.", "bad"); return; }
    key = proceed.key;
  }

  const desc = ($("#prjdesc") && $("#prjdesc").value) || "";
  box.innerHTML = `<div class="loadrow"><span class="spin" aria-hidden="true"></span>
    <span>Uploading ${esc(file.name)}&hellip;</span></div>
    <div class="bar" style="margin-top:8px"><i id="prjbar" style="width:0%"></i></div>`;

  /* XHR rather than fetch: only XHR reports upload progress, and a multi-MB
     CSV on a slow link is exactly when a progress bar earns its place. */
  const url = "/api/v1/datasets/upload?filename=" + encodeURIComponent(file.name)
    + "&description=" + encodeURIComponent(desc) + "&validate=true";
  const body = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "text/csv");
    xhr.setRequestHeader("Accept", "application/json");
    if(key) xhr.setRequestHeader("X-API-Key", key);
    xhr.upload.onprogress = e => {
      if(!e.lengthComputable) return;
      const bar = $("#prjbar");
      if(bar) bar.style.width = Math.round(e.loaded / e.total * 100) + "%";
    };
    xhr.onerror = () => reject(new Error(
      "The network dropped the request. Nothing was registered."));
    xhr.ontimeout = () => reject(new Error("The upload timed out."));
    xhr.onload = () => {
      let parsed = null;
      try { parsed = JSON.parse(xhr.responseText || "null"); } catch(e){ parsed = null; }
      if(xhr.status >= 200 && xhr.status < 300) return resolve(parsed);
      const msg = (parsed && parsed.error && parsed.error.message)
        || (parsed && parsed.detail) || `HTTP ${xhr.status}`;
      reject(new Error(msg));
    };
    xhr.send(file);
  }).catch(e => { fail(e.message); return null; });
  if(!body) return;

  PRJ = prjBlank();
  PRJ.version = body.version;
  PRJ.datasetName = file.name;
  PRJ.rows = body.rows;
  PRJ.columns = body.columns;
  PRJ.columnNames = body.column_names || [];
  PRJ.validation = body.validation || null;
  prjSave();
  api.bust();
  toast(`Registered as ${body.version}.`, "ok");
  render();
}

/* --- 03 target ---------------------------------------------------------- */
function wireTarget(){
  document.querySelectorAll("[data-usetarget]").forEach(b => b.onclick = () => {
    PRJ.target = b.dataset.usetarget; prjSave(); render();
  });
  const set = $("#prjsettarget"), sel = $("#prjtarget");
  if(set && sel) set.onclick = () => {
    if(!sel.value){ toast("Choose a column first.", "bad"); return; }
    PRJ.target = sel.value; prjSave(); render();
  };
  const clear = $("#prjcleartarget");
  if(clear) clear.onclick = () => {
    PRJ.target = null; PRJ.problemSupported = null; prjSave(); render();
  };
}

/* --- 04 start a run ----------------------------------------------------- */
function wireStart(){
  const boxes = [...document.querySelectorAll(".prjcand")];
  const count = $("#prjcandcount");
  const sync = () => {
    const n = boxes.filter(b => b.checked).length;
    if(count) count.textContent = n
      ? `AutoML will train ${n} candidate model(s), each a full training run.`
      : "Select at least one model.";
    const start = $("#prjstartautoml");
    if(start) start.disabled = !n;
  };
  boxes.forEach(b => b.onchange = sync);
  if(boxes.length) sync();

  const auto = $("#prjstartautoml");
  if(auto) auto.onclick = () => startRun("automl", {
    dataset_version: PRJ.version,
    target_column: PRJ.target,
    algorithms: boxes.filter(b => b.checked).map(b => b.value),
    primary_metric: PRJ.primaryMetric || "roc_auc",
    tune: false,
    target_stage: ($("#prjstage") && $("#prjstage").value) || "Staging",
    max_models: Math.max(1, boxes.filter(b => b.checked).length),
  });

  const man = $("#prjstartmanual");
  if(man) man.onclick = () => startRun("training", {
    dataset_version: PRJ.version,
    algorithm: ($("#prjalgo") && $("#prjalgo").value) || null,
    tune: !!($("#prjtune") && $("#prjtune").checked),
    promote: true,
    target_stage: ($("#prjstage") && $("#prjstage").value) || "Staging",
  });
}

async function startRun(kind, payload){
  const stage = payload.target_stage;
  const btn = kind === "automl" ? $("#prjstartautoml") : $("#prjstartmanual");
  const box = $("#prjstartresult");

  if(kind === "automl" && !payload.algorithms.length){
    toast("Select at least one candidate.", "bad"); return;
  }

  const needsKey = await authRequired();
  const proceed = await confirmAction({
    title: kind === "automl" ? "Start AutoML" : "Start training",
    body: kind === "automl"
      ? `Train ${payload.algorithms.length} candidate model(s) on ${PRJ.version}, target `
        + `${PRJ.target}. The best is registered and then judged by the approval gate — `
        + `promotion into ${stage} is not automatic.`
      : `Train ${payload.algorithm || "the configured default"} on ${PRJ.version}. The model is `
        + `registered and judged by the approval gate — promotion into ${stage} is not automatic.`,
    confirm: kind === "automl" ? "Start AutoML" : "Start training", needsKey,
  });
  if(!proceed) return;
  if(needsKey && !proceed.key){ toast("An API key is required.", "bad"); return; }

  const label = btn ? btn.textContent : "";
  if(btn){ btn.disabled = true; btn.textContent = "Starting…"; }
  try {
    const path = kind === "automl" ? "/api/v1/automl/runs" : "/api/v1/training/runs";
    const res = await api.post(path, payload, proceed.key);
    PRJ.mode = kind === "automl" ? "automl" : "manual";
    PRJ.runKind = kind;
    PRJ.runId = res.run_id;
    PRJ.runStatus = res.status || "queued";
    PRJ.modelVersion = null; PRJ.gateDecision = null; PRJ.deployedVersion = null;
    prjSave(); api.bust();
    toast("Run started.", "ok");
    goStep(5);
    render();
  } catch(e){
    const msg = (e.status === 401 || e.status === 403)
      ? "Rejected: the API key was missing or not accepted." : e.message;
    if(box) box.innerHTML =
      `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
        <b>Could not start the run.</b><br><span style="font-size:12.5px">${esc(msg)}</span></div>`;
    toast(msg, "bad");
  } finally {
    if(btn){ btn.disabled = false; btn.textContent = label; }
  }
}

/* --- 05 polling --------------------------------------------------------- */
/* Every three seconds while the run is live, and not one request after it
   reaches a terminal state. Also stops if the user navigates away. */
async function pollRunStep(runId, attempt){
  if(!runId || runId !== PRJ.runId) return;
  if(attempt > 400) return;
  if(route() !== "newproject") return;

  let run;
  try { run = await api.get(runPath(), 0); }
  catch(e){ PRJ_POLL = setTimeout(() => pollRunStep(runId, attempt + 1), 6000); return; }

  absorbRun(run);
  if(runDone(run.status)){
    api.bust();
    toast(`Run ${String(run.status).replace(/_/g," ")}.`,
      run.status === "failed" || run.status === "rejected" ? "bad" : "ok");
    render();
    return;
  }
  // Refresh the progress panel in place: re-rendering the page would discard
  // whatever the user has in flight elsewhere on it.
  const holder = document.querySelector("[data-progress]");
  if(holder && document.visibilityState === "visible"){
    holder.innerHTML = PRJ.runKind === "automl" ? automlBody(run, true) : trainingBody(run, true);
  }
  PRJ_POLL = setTimeout(() => pollRunStep(runId, attempt + 1), 3000);
}

/* --- 05 candidate drawer ------------------------------------------------ */
function wireDrawer(){
  document.querySelectorAll("[data-cand]").forEach(b => b.onclick = async () => {
    const algo = b.dataset.cand;
    let run;
    try { run = await api.get(runPath(), 0); }
    catch(e){ toast(e.message, "bad"); return; }
    const c = (run.candidates || []).find(x => x.algorithm === algo);
    if(!c){ toast("That candidate is no longer in the run record.", "bad"); return; }
    openDrawer(algo, candidateDetail(c, run));
  });
}

/* Metrics arrive in one bag, mixing scores with counts. Rendering everything
   to four decimals turns "1200 samples" into "1200.0000", so whole numbers are
   shown as whole numbers. */
function metricValue(v){
  if(typeof v !== "number" || !isFinite(v)) return NA;
  return Number.isInteger(v) ? int(v) : num(v, 4);
}

function candidateDetail(c, run){
  const m = c.metrics || {};
  const keys = Object.keys(m);
  return `<div class="grid g2" style="gap:10px">
      ${kpi("Algorithm", `<span class="mono">${esc(c.algorithm)}</span>`)}
      ${kpi("Status", statusBadge(c.status))}
      ${kpi("Rank", c.rank != null ? `#${int(c.rank)}` : NA)}
      ${kpi("Duration", c.duration_seconds != null
        ? `${Number(c.duration_seconds).toFixed(1)}s` : NA)}
    </div>
    <div style="margin-top:16px"><b style="font-size:13px">Metrics</b>
      ${keys.length ? `<div class="grid g2" style="gap:10px;margin-top:8px">
        ${keys.map(k => kpi(k.replace(/_/g," "), metricValue(m[k]))).join("")}</div>`
        : `<p class="dim" style="margin:8px 0 0;font-size:12.5px">
           No metrics were recorded for this candidate.</p>`}</div>
    <div style="margin-top:16px"><b style="font-size:13px">Provenance</b>
      <div class="kv" style="margin-top:8px">
        <div><span class="dim">Dataset</span> <span class="mono">${esc(run.dataset_version||"-")}</span></div>
        <div><span class="dim">Target</span> <span class="mono">${esc(run.target_column||"-")}</span></div>
        <div><span class="dim">Problem</span> ${esc(String(run.problem_type||"-").replace(/_/g," "))}</div>
        <div><span class="dim">Primary metric</span> <span class="mono">${esc(run.primary_metric||"-")}</span></div>
        <div><span class="dim">Tuning</span> ${run.tune ? "enabled" : "off"}</div>
        <div><span class="dim">Registered version</span> ${c.model_version != null
          ? `<span class="mono">v${int(c.model_version)}</span>` : "not registered"}</div>
      </div></div>
    ${c.error ? `<div style="margin-top:16px"><b style="font-size:13px">Error</b>
      <pre class="scroll" style="margin:8px 0 0">${esc(c.error)}</pre></div>` : ""}
    <p class="dim" style="margin:16px 0 0;font-size:12px">Hyperparameters are recorded with the
      MLflow run rather than on the candidate record; see the
      <a href="#/experiments">Experiments</a> page.</p>`;
}

function openDrawer(title, html){
  closeDrawer();
  const scrim = document.createElement("div");
  scrim.className = "scrim"; scrim.id = "prjscrim";
  const d = document.createElement("aside");
  d.className = "drawer"; d.id = "prjdrawer";
  d.setAttribute("role", "dialog"); d.setAttribute("aria-modal", "true");
  d.setAttribute("aria-label", title);
  d.innerHTML = `<div class="dhead"><b>${esc(title)}</b><span class="spacer"></span>
      <button class="btn icon" id="prjdclose" aria-label="Close">&times;</button></div>
    <div class="dbody">${html}</div>`;
  document.body.appendChild(scrim); document.body.appendChild(d);
  /* Force a reflow so the browser records the closed position before the open
     class lands, which is what makes it animate rather than jump. A
     requestAnimationFrame here would do the same when the tab is painting, but
     it does not fire in a backgrounded or non-rendering tab -- and a drawer
     that never opens is a worse failure than one that does not slide. */
  void d.offsetWidth;
  scrim.classList.add("on"); d.classList.add("open");
  $("#prjdclose").onclick = closeDrawer;
  scrim.onclick = closeDrawer;
  document.addEventListener("keydown", drawerEsc);
  $("#prjdclose").focus();
}
function drawerEsc(e){ if(e.key === "Escape") closeDrawer(); }
function closeDrawer(){
  document.removeEventListener("keydown", drawerEsc);
  ["#prjdrawer", "#prjscrim"].forEach(s => { const el = $(s); if(el) el.remove(); });
}

/* --- 07 gate ------------------------------------------------------------ */
function wireGate(){
  const btn = $("#prjrungate");
  if(!btn) return;
  btn.onclick = async () => {
    const key = ($("#prjgatekey") && $("#prjgatekey").value) || "";
    if(!key){ toast("Enter an API key.", "bad"); return; }
    btn.disabled = true;
    try {
      await api.post(`/api/v1/models/${encodeURIComponent(PRJ.modelName)}`
        + `/versions/${PRJ.modelVersion}/evaluate-gate`, null, key);
      api.bust(); render();
    } catch(e){
      toast(e.status === 401 || e.status === 403
        ? "The API key was not accepted." : e.message, "bad");
      btn.disabled = false;
    }
  };
}

/* --- 08 deploy ---------------------------------------------------------- */
function wireDeploy(){
  const btn = $("#prjdeploy");
  if(!btn) return;
  btn.onclick = async () => {
    const strategy = ($("#prjstrategy") && $("#prjstrategy").value) || "blue_green";
    const needsKey = await authRequired();
    const proceed = await confirmAction({
      title: "Deploy model version",
      body: `Roll ${PRJ.modelName} v${PRJ.modelVersion} onto the serving endpoint using the `
          + `${strategy.replace(/_/g," ")} strategy. This changes what live traffic is scored by.`,
      confirm: "Deploy", needsKey,
    });
    if(!proceed) return;
    if(needsKey && !proceed.key){ toast("An API key is required to deploy.", "bad"); return; }

    btn.disabled = true; btn.textContent = "Deploying…";
    try {
      const res = await api.post("/api/v1/deployments", {
        model_name: PRJ.modelName,
        model_version: PRJ.modelVersion,
        strategy,
        reason: "deployed from the guided build workflow",
      }, proceed.key);
      PRJ.deployedVersion = PRJ.modelVersion; prjSave(); api.bust();
      toast("Deployment created.", "ok");
      $("#prjdeployresult").innerHTML =
        `<div class="note" style="border-color:var(--ok-line);background:var(--ok-bg)">
          <b>Deployed.</b><br><span style="font-size:12.5px">
          ${esc(res.detail || res.message || "The deployment API accepted the rollout.")}</span></div>`;
      render();
    } catch(e){
      const msg = (e.status === 401 || e.status === 403)
        ? "Rejected: the API key was missing or not accepted." : e.message;
      $("#prjdeployresult").innerHTML =
        `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
          <b>Deployment refused.</b><br><span style="font-size:12.5px">${esc(msg)}</span></div>`;
      toast(msg, "bad");
      btn.disabled = false; btn.textContent = "Deploy this version";
    }
  };
}

/* --- 09 predict --------------------------------------------------------- */
function wirePredict(){
  const btn = $("#prjpredict");
  if(!btn) return;
  btn.onclick = async () => {
    const features = {};
    let bad = null;
    document.querySelectorAll(".prjfeat").forEach(i => {
      const raw = i.value.trim();
      if(raw === ""){ bad = bad || `${i.dataset.f} is empty.`; return; }
      if(i.dataset.num === "1"){
        const n = Number(raw);
        if(!isFinite(n)){ bad = bad || `${i.dataset.f} is not a number.`; return; }
        features[i.dataset.f] = n;
      } else features[i.dataset.f] = raw;
    });
    if(bad){ toast(bad, "bad"); return; }

    const needsKey = await authRequired();
    let key = null;
    if(needsKey){
      const proceed = await confirmAction({
        title: "Score a record", body: "Sends one request to the live prediction endpoint.",
        confirm: "Score", needsKey: true,
      });
      if(!proceed) return;
      if(!proceed.key){ toast("An API key is required.", "bad"); return; }
      key = proceed.key;
    }

    btn.disabled = true; btn.textContent = "Scoring…";
    try {
      const r = await api.post("/api/v1/predict", { features }, key);
      $("#prjpredresult").innerHTML = card("Response", `
        <div class="grid g4">
          ${kpi("Prediction", `<b>${esc(r.prediction_label)}</b>`, `class ${int(r.prediction)}`)}
          ${kpi("Probability", num(r.probability, 4), `threshold ${num(r.threshold, 2)}`)}
          ${kpi("Served by", `<span class="mono">v${int(r.model_version)}</span>`,
            `${esc(r.model_stage||"")} · ${esc(r.variant||"primary")}`)}
          ${kpi("Latency", ms(r.inference_latency_ms))}
        </div>
        <p class="dim" style="margin:12px 0 0;font-size:12.5px">The version and variant above are
          what actually served this request, not what the registry says is current.</p>`,
        { flush:true });
    } catch(e){
      const msg = (e.status === 401 || e.status === 403)
        ? "Rejected: the API key was missing or not accepted." : e.message;
      $("#prjpredresult").innerHTML =
        `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
          <b>The endpoint rejected this request.</b><br>
          <span style="font-size:12.5px">${esc(msg)}</span></div>`;
    } finally {
      btn.disabled = false; btn.textContent = "Score";
    }
  };
}
