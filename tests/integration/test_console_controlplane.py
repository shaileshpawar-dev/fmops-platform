"""The control-plane console: structure, honesty, and self-containment.

These assert the properties the redesign is *for*, not its styling. A CSS
change should never fail them; dropping a capability, inventing a number, or
reaching for a CDN should.
"""

from __future__ import annotations

import re

CSS_URL = re.compile(r"url\(\s*([a-zA-Z0-9+.-]+):")


def _assets(api_client) -> dict[str, str]:
    """Every asset the console shell references, by path."""
    page = api_client.get("/dashboard")
    assert page.status_code == 200
    out = {"/dashboard": page.text}
    refs = re.findall(r'src="(/static/js/[^"]+)"', page.text)
    refs += re.findall(r'href="(/static/css/[^"]+)"', page.text)
    for path in refs:
        r = api_client.get(path)
        assert r.status_code == 200, f"console references {path} but it is not served"
        out[path] = r.text
    return out


def _joined(api_client) -> str:
    return "\n".join(_assets(api_client).values())


# --------------------------------------------------------------------------- #
# Self-containment
# --------------------------------------------------------------------------- #
def test_typefaces_are_embedded_not_fetched(api_client):
    """The console must render correctly with no egress.

    The container has no guaranteed outbound network. A webfont loaded from a
    CDN would fail open into an unstyled console in exactly the environment
    this is meant to run in, so the faces are embedded as data URIs.
    """
    assets = _assets(api_client)
    css = {p: t for p, t in assets.items() if p.endswith(".css")}
    assert css, "the console loads no stylesheet"

    faces = sum(t.count("@font-face") for t in css.values())
    assert faces >= 3, f"expected embedded font faces, found {faces}"

    for path, text in css.items():
        schemes = set(CSS_URL.findall(text))
        assert schemes <= {"data"}, f"{path} loads a font or image over {schemes - {'data'}}"


def test_console_ships_no_external_resource_in_any_asset(api_client):
    """Restates the no-egress guarantee over scripts *and* stylesheets."""
    body = _joined(api_client)
    for marker in ("https://", "http://", "//cdn", "integrity="):
        assert marker not in body, f"console references an external resource: {marker}"


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_signature_components_exist_and_are_used(api_client):
    """The six components the redesign is built from.

    Each renders from an API response; a component that exists but is never
    called would mean a page quietly went back to a generic table.
    """
    body = _joined(api_client)
    for fn in (
        "healthStrip",
        "lifecycleRail",
        "promotionRail",
        "versionRail",
        "evidenceTable",
        "attentionQueue",
    ):
        assert f"function {fn}(" in body, f"{fn} is not defined"
        # Defined once, called at least once somewhere else.
        assert body.count(f"{fn}(") >= 2, f"{fn} is defined but never used"


def test_model_version_is_addressable(api_client):
    """#/models/:version -- the version is the primary object.

    The router has to split a path segment for this to work at all; without it
    every model link silently falls back to the index.
    """
    body = _joined(api_client)
    assert "function routeParam(" in body, "the router cannot read a path parameter"
    assert "#/models/${" in body, "nothing links to a specific version"
    for tab in ("overview", "versions", "evaluation", "deployments", "observability", "audit"):
        assert f'"{tab}"' in body, f"the model page has no {tab} tab"


def test_rejected_versions_stay_visibly_rejected(api_client):
    """The registry must not look artificially green.

    A rejected candidate is the most informative thing a lineage strip can
    show, and the easiest thing for a redesign to quietly drop.
    """
    body = _joined(api_client)
    assert "rejected" in body
    assert ".vrail a.rejected" in body, "the version rail has no rejected styling"


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #
def test_runtime_page_does_not_claim_cloud_introspection(api_client):
    """ "Runtime", not "Infrastructure".

    The page can read the process; it cannot read AWS. The distinction is the
    whole reason the destination was renamed.
    """
    body = _joined(api_client)
    assert "PAGES.runtime =" in body
    assert '["runtime","Runtime"' in body
    assert "Documented architecture" in body, "topology is not labelled as documented"
    assert "Documented, not probed" in body
    assert "no AWS introspection" in body


def test_incidents_page_states_its_real_scope(api_client):
    """Alerts are not an incident-management system, and the page says so."""
    body = _joined(api_client)
    assert "PAGES.incidents =" in body
    assert "no assignment" in body, "the incidents page overstates its lifecycle"


def test_quality_gates_admits_it_has_no_decision_history(api_client):
    """There is no gate-collection endpoint; the page must not imply one."""
    body = _joined(api_client)
    assert "PAGES.gates =" in body
    assert "What this page cannot show" in body


def test_absent_baseline_is_not_rendered_as_a_zero_score(api_client):
    """``baseline_score: 0`` from evaluate-gate is a sentinel, not a measurement.

    When no production model exists the comparison reports a baseline of zero
    and says so in ``reason``. Rendering that as 0.0000 would invent a champion
    that scored nothing, and the improvement derived from it is meaningless.
    """
    body = _joined(api_client)
    assert "noIncumbent" in body, "the no-incumbent case is not handled"
    assert "no production model exists" in body, "the backend's own reason is not read"


def test_live_metrics_are_not_attributed_to_a_non_serving_version(api_client):
    """Service metrics are endpoint-wide, not per-version."""
    body = _joined(api_client)
    assert "IS NOT SERVING TRAFFIC" in body
    assert "not per model version" in body


# --------------------------------------------------------------------------- #
# Consistency and craft
# --------------------------------------------------------------------------- #
def test_run_status_has_exactly_one_colour_map(api_client):
    """The same state must not be a different colour on a different page.

    There were three helpers -- automlStatusBadge, runStatusBadge and
    statusBadge -- and a `rejected` run rendered grey on AutoML, amber on
    Training and red in the guided workflow.
    """
    body = _joined(api_client)
    assert "function runStatusBadge(" in body
    assert body.count("function runStatusBadge(") == 1, "more than one status map"
    for gone in ("function automlStatusBadge(", "function statusBadge("):
        assert gone not in body, f"{gone} is back; state colour will diverge again"
    assert "const RUN_STATE" in body


def test_icons_are_svg_not_glyphs(api_client):
    """One icon system, drawn rather than borrowed from a font.

    Glyph characters resolved differently per platform, sat on inconsistent
    baselines, and one of them was a heart labelling the Runtime page.
    """
    body = _joined(api_client)
    assert "function icon(" in body and "ICON_PATHS" in body
    assert 'stroke-width="1.75"' in body, "icons no longer share one stroke weight"
    # Nav entries name an icon, not a character.
    for pid, name in (("overview", "grid"), ("runtime", "server"), ("models", "layers")):
        assert f'["{pid}",' in body
        assert f'"{name}"' in body, f"{pid} has no {name} icon"


def test_tables_support_sorting_and_filtering(api_client):
    """Scanning is the point of a control plane; sorting is how you scan."""
    body = _joined(api_client)
    assert "function wireTables(" in body
    assert "aria-sort=" in body, "sorted columns are not announced"
    assert 'scope="col"' in body, "table headers are not scoped"
    assert "data-tfilter" in body and "data-tsort" in body


def test_command_palette_navigates_and_claims_no_search_backend(api_client):
    """A palette over existing routes, not a pretend search API.

    There is no search endpoint in this platform, and the empty state says so
    rather than implying results are missing.
    """
    body = _joined(api_client)
    assert "function openPalette(" in body
    assert "aria-keyshortcuts" in body
    assert "does not search" in body, "the palette overstates what it can find"
    # It must not invent an endpoint.
    assert "/api/v1/search" not in body


def test_copy_to_clipboard_has_a_plain_http_fallback(api_client):
    """The deployed console is served over HTTP, where navigator.clipboard is
    unavailable -- so the fallback is the path that actually runs."""
    body = _joined(api_client)
    assert "function copyable(" in body
    assert "navigator.clipboard" in body
    assert "execCommand" in body, "no fallback for a non-secure context"
    assert "Could not copy" in body, "a failed copy is silent"


def test_overlays_trap_focus(api_client):
    """Tab must not walk out of an open dialog into the page behind it."""
    body = _joined(api_client)
    assert "function trapFocus(" in body
    assert 'role="dialog"' in body and 'aria-modal="true"' in body


def test_workflow_is_split_and_every_module_is_served(api_client):
    """The guided workflow was one 1,700-line file."""
    page = api_client.get("/dashboard").text
    for mod in ("project.js", "project-steps.js", "project-wire.js"):
        assert f"/static/js/pages/{mod}" in page, f"{mod} is not loaded"
        assert api_client.get(f"/static/js/pages/{mod}").status_code == 200
    body = _joined(api_client)
    # The step map must resolve lazily: the renderers load after project.js.
    assert "function stepRenderer(" in body
    assert "const STEP_RENDER" not in body


def test_metrics_window_uses_the_parameter_the_api_accepts(api_client):
    """window_minutes is the only time parameter monitoring takes."""
    from app.api.routes import monitoring  # noqa: F401

    spec = api_client.get("/openapi.json").json()
    params = spec["paths"]["/api/v1/monitoring/summary"]["get"].get("parameters", [])
    assert any(p["name"] == "window_minutes" for p in params)

    body = _joined(api_client)
    assert "OBS_WINDOWS" in body
    assert "window_minutes=${" in body, "the window selector does not reach the API"


def test_audit_table_reads_fields_the_api_returns(api_client):
    """Resource and Status read `e.resource` and `e.status`, which this API has
    never returned -- both columns rendered empty on every row."""
    entries = api_client.get("/api/v1/audit?limit=1").json()["entries"]
    if entries:
        row = entries[0]
        assert "resource_type" in row and "outcome" in row
        assert "resource" not in row and "status" not in row

    body = _joined(api_client).replace(" ", "")
    assert "e.resource_type" in body and "e.outcome" in body
    # `e.resource` and `e.status` are not fields this API returns. Guard the
    # exact phantom accessors rather than any string containing them.
    for phantom in ("e.resource||", "e.resource)", "e.status?", "e.status||"):
        assert phantom not in body, f"phantom audit field is back: {phantom}"
