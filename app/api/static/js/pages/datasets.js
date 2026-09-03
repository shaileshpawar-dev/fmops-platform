PAGES.datasets = {
  title: "Datasets",
  intro: "Registered dataset versions. Versions are content-addressed: uploading identical bytes returns the existing version rather than creating a duplicate.",
  async render(){
    const r = await loadAll({ list:"/api/v1/datasets" }, 6000);
    const uploader = card("Upload a dataset", `
      <p class="dim" style="margin:0 0 12px">CSV only, up to 25 MB. The file is parsed with pandas,
        registered as a new version, and validated by the platform's own engine. It is never
        executed, and the filename never chooses a path.</p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <input type="file" id="dsfile" accept=".csv,text/csv" style="max-width:280px">
        <input type="text" id="dsdesc" placeholder="Description (optional)" style="max-width:250px">
        <button class="btn pri" id="dsupload">Upload and validate</button>
      </div>
      <div id="dsresult" style="margin-top:12px"></div>`);

    const tbl = sect(r.list, d => {
      const versions = d.versions || [];
      return card("Registered versions", table([
        { label:"Version", render:v => `<b class="mono">${esc(v.version)}</b>` },
        { label:"Dataset", render:v => `<span class="mono">${esc(v.dataset_name||"-")}</span>` },
        { label:"Rows", num:true, render:v => int(v.n_rows) },
        { label:"Columns", num:true, render:v => int(v.n_columns) },
        { label:"Hash", render:v => v.content_hash ?
            `<span class="mono dim">${esc(String(v.content_hash).slice(0,12))}</span>` : NA },
        { label:"DVC", render:v => has(v.dvc_tracked) ?
            (v.dvc_tracked ? badge("tracked","ok") : badge("not tracked","mute")) : NA },
        { label:"Created", render:v => when(v.created_at) },
        { label:"Actions", render:v =>
            `<button class="btn" data-act="ds-validate" data-v="${esc(v.version)}">Validate</button>
             <button class="btn" data-act="ds-preview" data-v="${esc(v.version)}">Preview</button>` },
      ], versions, { empty:"No datasets registered. Upload a CSV above to create the first version." }),
      { flush:true, sub:`${versions.length} version(s)` });
    }, "datasets");

    return uploader + tbl + `<div id="dsdetail"></div>`;
  }
};

function renderValidation(v){
  if(!v) return "";
  const head = `<div class="grid g4" style="margin-bottom:12px">
    ${kpi("Status", v.passed ? badge("PASSED","ok",true) : badge("FAILED","bad",true),
          `engine: ${esc(v.engine||"-")}`)}
    ${kpi("Expectations", int(v.expectations))}
    ${kpi("Passed", int(v.succeeded))}
    ${kpi("Failed", int(v.failed), has(v.warnings) ? `${v.warnings} warning(s)` : "")}
  </div>`;
  const fails = v.failures || [];
  const body = fails.length ? table([
    { label:"Expectation", render:f => `<span class="mono">${esc(f.expectation)}</span>` },
    { label:"Column", render:f => f.column ? `<span class="mono">${esc(f.column)}</span>` : NA },
    { label:"Severity", render:f => badge(f.severity, f.severity==="warning"?"warn":"bad") },
    { label:"Observed", render:f => f.observed ? `<span class="mono dim">${esc(f.observed)}</span>` : NA },
    { label:"Expected", render:f => f.expected ? `<span class="mono dim">${esc(f.expected)}</span>` : NA },
    { label:"Message", render:f => esc(f.message) },
  ], fails, { empty:"" }) : `<div class="state"><div class="big">All expectations passed</div>
      ${v.expectations} checks, no failures.</div>`;
  return head + body + (v.truncated ? `<p class="dim" style="margin:10px 0 0">
    Only the first 50 failures are listed.</p>` : "");
}

function renderPreview(p){
  const cols = p.column_profile || [];
  const sample = p.sample || [];
  const names = cols.map(c => c.name);
  const rows = sample.length ? `<div class="scroll"><table><thead><tr>${
    names.map(n => `<th>${esc(n)}</th>`).join("")}</tr></thead><tbody>${
    sample.map(row => `<tr>${names.map(n =>
      `<td class="mono" style="font-size:11.5px">${row[n] === null || row[n] === undefined
        ? `<span class="dim">null</span>` : esc(String(row[n]).slice(0,40))}</td>`).join("")}</tr>`).join("")
  }</tbody></table></div>` : emptyState("No sample rows.");

  return card(`Preview - ${esc(p.version)}`, `
    <div class="grid g3" style="margin-bottom:12px">
      ${kpi("Rows", int(p.rows))}${kpi("Columns", int(p.columns))}
      ${kpi("Showing", int(p.preview_rows), "sample capped server-side")}
    </div>
    <div class="dim" style="font-size:10.5px;font-weight:700;letter-spacing:.06em;
      text-transform:uppercase;margin:14px 0 7px">Column profile</div>` +
    table([
      { label:"Column", render:c => `<span class="mono">${esc(c.name)}</span>` },
      { label:"Type", render:c => `<span class="mono dim">${esc(c.dtype)}</span>` },
      { label:"Missing", num:true, render:c => `${int(c.missing)} <span class="dim">(${c.missing_pct}%)</span>` },
      { label:"Unique", num:true, render:c => int(c.unique) },
      { label:"Example", render:c => c.example ? `<span class="mono dim">${esc(c.example)}</span>` : NA },
    ], cols, { empty:"No columns." }) +
    `<div class="dim" style="font-size:10.5px;font-weight:700;letter-spacing:.06em;
      text-transform:uppercase;margin:14px 0 7px">Sample rows</div>` + rows);
}
