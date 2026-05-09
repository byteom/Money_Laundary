from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment.app import app


def test_frontend_shell_contains_live_demo_and_explainability_panels():
    with app.test_client() as client:
        response = client.get('/')

    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert 'data-tab="singleTab"' in html
    assert 'data-tab="batchTab"' in html
    assert 'data-tab="contextTab"' in html
    assert 'id="singleResults"' in html
    assert 'id="batchTableBody"' in html
    assert 'id="contextResults"' in html
    assert 'id="recentTransactionsJson"' in html
    assert 'id="singleTopFactors"' in html
    assert 'id="contextDecisionSteps"' in html
    assert '/static/js/app.js' in html
    assert '/static/css/styles.css' in html
