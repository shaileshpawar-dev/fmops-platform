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
