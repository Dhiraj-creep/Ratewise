from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_every_empty_state_screen_loads():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    for page in ["Collect data", "Explore dataset", "Train & evaluate", "Predict ratings", "Project guide"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"Failed page: {page}"
