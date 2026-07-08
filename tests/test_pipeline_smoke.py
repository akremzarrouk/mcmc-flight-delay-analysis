"""End-to-end smoke test: runs the full pipeline against the small synthetic
dataset in data/sample/ (see scripts/generate_sample_data.py) so CI can catch
breakage without the ~3.3 GB real download.
"""
import os

from src.flight_pipeline import run_analysis

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")


def test_run_analysis_end_to_end_on_sample_data():
    results = run_analysis(
        flight_files=[os.path.join(SAMPLE_DIR, "flights.csv")],
        plane_file=os.path.join(SAMPLE_DIR, "plane-data.csv"),
        airport_file=os.path.join(SAMPLE_DIR, "airports.csv"),
        carrier_file=os.path.join(SAMPLE_DIR, "carriers.csv"),
    )

    assert set(results.keys()) == {"delay_patterns", "aircraft_age", "diversion_models"}
    assert 2007 in results["delay_patterns"]
    assert 2007 in results["aircraft_age"]
    assert len(results["diversion_models"]) == 1

    delay_result = results["delay_patterns"][2007]
    assert "best_day" in delay_result
    assert "best_combination" in delay_result

    age_result = results["aircraft_age"][2007]
    assert 0.0 <= age_result["roc_auc"] <= 1.0

    diversion_result = results["diversion_models"][0]
    assert 0.0 <= diversion_result["roc_auc"] <= 1.0
