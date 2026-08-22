from pathlib import Path

import yaml

from scripts.target_analysis_needs_schema import audit

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "target-analysis-pipeline.yml"


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def test_needs_graph_is_sound():
    # No job may read (Content) or require-the-existence-of (Presence) an
    # artifact from a job it does not declare needs: on -- a violation here
    # is a real race, not just wasted time.
    soundness, _ = audit(_jobs())
    assert soundness == [], (
        f"jobs read from other jobs' artifacts without declaring needs: on them "
        f"(a real race): {soundness}"
    )


def test_needs_graph_is_complete():
    # Every declared needs: entry must be grounded by a real Content or
    # Presence edge -- an ungrounded entry costs only wall-clock time, but
    # costs it forever, silently, since nothing else checks this.
    _, completeness = audit(_jobs())
    assert completeness == [], (
        f"jobs declare needs: on other jobs they never read from or check "
        f"the existence of -- pure scheduling waste: {completeness}"
    )
