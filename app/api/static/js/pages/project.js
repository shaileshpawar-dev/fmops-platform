

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


function profileFor(target){
  return api.get(`/api/v1/automl/profile/${encodeURIComponent(PRJ.version)}`
    + `?target=${encodeURIComponent(target)}`, 60000);
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


/* ===================================================================== */
/* Page                                                                  */
/* ===================================================================== */
/* Resolved when a step is rendered, not when this file loads: the renderers
   live in project-steps.js, which the shell loads after this module. Building
   the map eagerly would capture undefined for every entry. */
function stepRenderer(n){
  return {
    1: stepDataset, 2: stepProfile, 3: stepTarget, 4: stepTraining, 5: stepEvaluation,
    6: stepRegistry, 7: stepApproval, 8: stepDeploy, 9: stepPredict, 10: stepMonitor,
  }[n];
}


PAGES.newproject = {
  title: "New ML Project",
  intro: "A guided path from a CSV to a deployed, monitored model. Each step drives the same "
       + "APIs the rest of the console uses; progress is kept in this browser tab.",
  async render(){
    const n = currentStep();
    return await stepRenderer(n)();
  },
  wire(){ wireProject(); },
};

function metricValue(v){
  if(typeof v !== "number" || !isFinite(v)) return NA;
  return Number.isInteger(v) ? int(v) : num(v, 4);
}
