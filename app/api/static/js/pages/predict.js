/* Predict -- score a record with any model, against its own contract.
 *
 * The form is built from the recorded signature of the version that serves
 * this model: its features, their kinds, their training ranges and
 * categories. Nothing about the input shape is hard-coded here. The request
 * goes to POST /api/v1/models/{name}/predict, which checks it against the
 * version that actually scores it, and the answer comes back in the model's
 * own class labels.
 *
 * A prediction can then be given its true outcome. That label is what makes
 * live accuracy -- and retraining on new data -- possible at all.
 */

let LAST_PREDICTION = null;

function featureInput(f, value){
  const id = `pf-${f.name.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
  if(f.kind === "numeric"){
    const hint = f.minimum != null ? `training range ${num(f.minimum, 2)} … ${num(f.maximum, 2)}` : "";
    return `<div class="field"><label for="${id}">${esc(f.name)}</label>
      <input id="${id}" data-feature="${esc(f.name)}" data-kind="numeric" type="number" step="any"
        inputmode="decimal" value="${value != null ? esc(String(Number(Number(value).toFixed(4)))) : ""}" placeholder="${esc(
          f.median != null ? String(Number(f.median.toFixed(4))) : "")}">
      <span class="hint">${hint}</span></div>`;
  }
  const cats = f.categories || [];
  const complete = f.n_categories != null && f.n_categories <= cats.length;
  if(complete && cats.length <= 40){
    return `<div class="field"><label for="${id}">${esc(f.name)}</label>
      <select id="${id}" data-feature="${esc(f.name)}" data-kind="categorical">
        ${cats.map(c => `<option value="${esc(c)}" ${c === value ? "selected" : ""}>${esc(c)}</option>`).join("")}
      </select><span class="hint">${cats.length} categories seen in training</span></div>`;
  }
  return `<div class="field"><label for="${id}">${esc(f.name)}</label>
    <input id="${id}" data-feature="${esc(f.name)}" data-kind="categorical" list="${id}-list"
      value="${value != null ? esc(String(value)) : ""}" autocomplete="off">
    <datalist id="${id}-list">${cats.map(c => `<option value="${esc(c)}">`).join("")}</datalist>
    <span class="hint">${int(f.n_categories)} categories in training; the commonest are suggested</span></div>`;
}

function renderPrediction(p){
  const pos = p.positive_label || "positive";
  const isPos = p.prediction === 1;
  return card("Prediction", `
    <div class="predout ${isPos ? "pos" : "neg"}">
      <div class="lbl"><span class="eyebrow">Predicted</span><b>${esc(p.prediction_label)}</b></div>
      <div class="prob"><span class="eyebrow">P(${esc(pos)})</span><b class="mono">${num(p.probability, 4)}</b>
        <div class="bar" role="img" aria-label="probability ${esc(Number(p.probability).toFixed(4))} against threshold ${esc(Number(p.threshold).toFixed(3))}">
          <i style="width:${Math.max(0, Math.min(100, p.probability * 100)).toFixed(1)}%"></i>
          <u style="left:${Math.max(0, Math.min(100, p.threshold * 100)).toFixed(1)}%"></u></div>
        <span class="hint">decision threshold ${num(p.threshold, 3)} — the version's own operating point</span></div>
    </div>
    <div class="grid g4" style="margin-top:14px">
      ${kpi("Version", `<a class="mono" href="#/models/${encodeURIComponent(p.model_name)}/${p.model_version}">v${int(p.model_version)}</a>`, esc(p.model_stage || ""))}
      ${kpi("Routed as", badge(p.variant || "primary", p.variant === "canary" ? "warn" : "mute"))}
      ${kpi("Latency", ms(p.inference_latency_ms), "model inference")}
      ${kpi("Request id", copyable(p.request_id, String(p.request_id).slice(0, 12) + "…", { label:"request id" }))}
    </div>
    ${(p.warnings || []).length ? `<div class="note warnnote" style="margin-top:14px"><b>Accepted, with notes.</b>
      <ul class="plain">${p.warnings.map(w => `<li>${esc(w)}</li>`).join("")}</ul>
      <span class="dim" style="font-size:12px">Values the model never saw in training are scored, not
      refused — they are exactly what drift monitoring watches for.</span></div>` : ""}
    <div class="feedback">
      <span class="eyebrow">Record the true outcome</span>
      <p class="dim" style="margin:4px 0 9px;font-size:12.5px">When the real answer is known, send it.
        Labelled predictions are the only way live accuracy is measured and the only new data a
        retraining run can learn from.</p>
      <div class="actions">${(p.__labels || ["0","1"]).map(l =>
        `<button class="btn" data-label="${esc(l)}">It was “${esc(l)}”</button>`).join("")}</div>
    </div>`, { sub:`scored ${esc(ago(p.created_at))}` });
}

PAGES.predict = {
  title: "Predict",
  intro: "Score a record with any serving model, checked against the input contract that model "
       + "was trained on — then tell the platform what actually happened.",
  async render(){
    return withModel(async (name, models) => {
      const enc = encodeURIComponent(name);
      const r = await loadAll({ sig:`/api/v1/models/${enc}/signature`,
        recent:`/api/v1/predictions/recent?model_name=${enc}&limit=15` }, 5000);
      const picker = `<div class="pagebar">${modelPicker(models, name)}</div>`;
      if(!r.sig.ok) return picker + errorState(r.sig.error, "predict");
      const s = r.sig.data, sig = s.signature;
      const serving = models.find(m => m.name === name) || {};

      if(serving.serving_version == null) return picker + card(name, `<div class="state">
        <div class="big">Nothing is serving</div>${esc(name)} has no Production version. Only a
        Production version answers callers; approve one on its <a href="#/models/${enc}">model page</a>
        and deploy it.</div>`);
      if(!sig) return picker + card(name, `<div class="note"><b>No recorded input contract.</b><br>
        ${esc(s.detail || "")} Its typed request is documented at
        <a href="/docs#/predictions/predict_api_v1_predict_post" target="_blank" rel="noopener">POST /api/v1/predict</a>.</div>`);

      const labels = sig.display_labels || sig.class_labels;
      const form = card(`Score with ${name}`, `
        <div class="idl" style="margin:-2px 0 14px">
          <span>version <b>v${esc(String(s.version))} ${esc(s.stage)}</b></span>
          <span>predicts <b class="mono">${esc(sig.target)}</b></span>
          <span>classes <b>${esc(labels[0])}</b> / <b>${esc(labels[1])}</b> (positive)</span>
          <span>endpoint <b class="mono">${esc(s.endpoint)}</b></span>
        </div>
        <form id="predform" class="formgrid" novalidate>
          ${sig.features.map(f => featureInput(f, (s.example && s.example.features || {})[f.name])).join("")}
        </form>
        <div class="actions" style="margin-top:14px">
          <button class="btn pri" id="predgo">${icon("target",14)} Predict</button>
          <button class="btn" id="predfill">Fill with training medians</button>
          <span class="spacer"></span>
          <span class="dim" style="font-size:12px">Leave a field empty to send <span class="mono">null</span>;
            it is imputed exactly as missing training values were.</span>
        </div>
        <details class="curl" style="margin-top:14px"><summary>Same request from a terminal</summary>
          <pre class="jsonview" id="predcurl"></pre></details>`,
        { sub:`${sig.features.length} features from the recorded signature` });

      const recent = r.recent.ok ? r.recent.data.predictions : [];
      const hist = card("Recent predictions", table([
        { label:"When", render:x => when(x.created_at) },
        { label:"Version", render:x => `<span class="mono">v${esc(String(x.model_version ?? "?"))}</span>` },
        { label:"Variant", render:x => badge(x.variant || "primary", "mute") },
        { label:"Prediction", render:x => x.status === "ok"
            ? esc(x.prediction === 1 ? labels[1] : labels[0]) : runStatusBadge(x.status) },
        { label:"P(positive)", num:true, render:x => num(x.probability, 4) },
        { label:"Latency", num:true, render:x => ms(x.latency_ms) },
        { label:"Request", render:x => copyable(x.request_id, String(x.request_id).slice(0, 10) + "…", { label:"request id" }) },
      ], recent, { empty:"No predictions for this model yet." }), { flush:true, sub:"from the inference log" });

      window.__predctx = { name, labels, features: sig.features, example: (s.example || {}).features || {} };
      return picker + form + `<div id="predresult"></div>` + hist;
    });
  },
  wire(){ wirePredictForm(); },
};

/* Shared by the Predict page and the guided workflow's predict step: both
   render #predform from a signature and leave the context in __predctx. */
function wirePredictForm(){
    const ctx = window.__predctx;
    const form = $("#predform");
    if(!ctx || !form) return;
    const collect = () => {
      const out = {};
      form.querySelectorAll("[data-feature]").forEach(el => {
        const raw = (el.value || "").trim();
        if(raw === "") out[el.dataset.feature] = null;
        else out[el.dataset.feature] = el.dataset.kind === "numeric" ? Number(raw) : raw;
      });
      return out;
    };
    const curl = () => {
      const pre = $("#predcurl");
      if(pre) pre.textContent = `curl -X POST ${location.origin}/api/v1/models/${encodeURIComponent(ctx.name)}/predict \\\n`
        + `  -H "Content-Type: application/json" -H "X-API-Key: $FMOPS_API_KEY" \\\n`
        + `  -d '${JSON.stringify({ features: collect() })}'`;
    };
    form.addEventListener("input", curl);
    curl();
    $("#predfill").onclick = () => {
      form.querySelectorAll("[data-feature]").forEach(el => {
        const v = ctx.example[el.dataset.feature];
        el.value = v == null ? "" : (el.dataset.kind === "numeric" ? String(Number(Number(v).toFixed(4))) : v);
      });
      curl();
    };
    $("#predgo").onclick = async () => {
      const features = collect();
      const bad = Object.entries(features).filter(([, v]) => typeof v === "number" && !isFinite(v));
      if(bad.length){ toast(`Not a number: ${bad.map(b => b[0]).join(", ")}`, "bad"); return; }
      const res = await runAction({
        title: `Score with ${ctx.name}`,
        body: "Sends this record to the model's endpoint. The prediction is logged for monitoring.",
        confirm: "Predict", needsKey: true,
        path: `/api/v1/models/${encodeURIComponent(ctx.name)}/predict`, payload: { features },
        success: "Scored.", after: () => {},
      });
      if(!res) return;
      LAST_PREDICTION = { ...res, __labels: ctx.labels };
      $("#predresult").innerHTML = renderPrediction(LAST_PREDICTION);
      $("#predresult").scrollIntoView({ behavior:"smooth", block:"nearest" });
      document.querySelectorAll("#predresult [data-label]").forEach(b => b.onclick = () => runAction({
        title: `Record outcome “${b.dataset.label}”`,
        body: `Attaches the true outcome to request ${LAST_PREDICTION.request_id}. A second label for the `
            + "same request replaces the first.",
        confirm: "Record", needsKey: true,
        path: "/api/v1/feedback",
        payload: { request_id: LAST_PREDICTION.request_id, actual_label: b.dataset.label, source: "console" },
        success: "Outcome recorded.",
        after: fb => { toast(`${fb.labelled_total} labelled prediction(s) for ${fb.model_name}.`, "ok"); },
      }));
    };
}
