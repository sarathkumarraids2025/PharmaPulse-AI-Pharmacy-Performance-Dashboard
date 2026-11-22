"""End-to-end smoke test using Streamlit's AppTest. Skipped when streamlit/plotly are missing.
Run: python -m unittest discover -s tests -v
"""
import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
    import plotly  # noqa: F401
    HAS_DEPS = True
except ImportError:  # pragma: no cover
    HAS_DEPS = False

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "📊 Dashboard Overview", "👥 Customer Analytics", "⚠️ Churn Analysis", "💰 Revenue Analysis",
    "🧩 Customer Segmentation", "🗂️ Data Explorer", "💡 Business Insights", "ℹ️ About Project",
]


@unittest.skipUnless(HAS_DEPS, "streamlit/plotly not installed")
class AppSmokeTests(unittest.TestCase):
    def test_every_page_loads_without_exception(self):
        at = AppTest.from_file(APP, default_timeout=90).run()
        self.assertEqual(len(at.exception), 0)
        for page in PAGES:
            at.session_state["nav_page"] = page
            at.run()
            self.assertEqual(len(at.exception), 0, f"exception on page {page}")

    def test_empty_filter_does_not_crash(self):
        at = AppTest.from_file(APP, default_timeout=90).run()
        at.session_state["f_contract"] = []
        at.run()
        self.assertEqual(len(at.exception), 0)


if __name__ == "__main__":
    unittest.main()
