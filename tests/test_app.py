import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.agent import grounded, parse_query, run_agent
from backend.app import app
from backend.pipeline import Pipeline, load_data, metrics

client = TestClient(app)


@pytest.fixture(scope="module")
def pipe():
    return Pipeline()


def test_load_data_assigns_two_assets_per_date():
    df = load_data()
    assert len(df) == 60 and df["date"].nunique() == 30
    assert set(df["asset"]) == {"A", "B"}
    assert df.groupby("date")["asset"].nunique().eq(2).all()


def test_load_data_rejects_bad_outage_flag(tmp_path):
    df = pd.read_csv("data/generation_forecast.csv")
    df.loc[0, "outage_flag"] = 2
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_data(p)


def test_metrics_known_values():
    m = metrics([10, 20], [12, 18])
    assert m["mae"] == 2 and m["bias"] == 0 and m["wape_pct"] == pytest.approx(13.33, abs=0.01)


def test_walk_forward_has_no_leakage(pipe):
    """OOF predictions exist only for dates after the first 15, and never for training-only dates."""
    dates = np.sort(pipe.df["date"].unique())
    first_scored = pipe.df.loc[pipe.cv["scored_index"], "date"].min()
    assert first_scored == dates[15]
    assert pipe.df.loc[pipe.df["date"] < first_scored, "model_forecast_oof"].isna().all()


def test_outage_is_reliable_driver_and_attribution_sums(pipe):
    out = next(d for d in pipe.summary()["drivers"] if d["driver"] == "outage")
    assert out["significant"] and out["coef_mwh_per_unit"] < -5
    row = pipe.df.loc[pipe.df["outage_flag"] == 1].iloc[0]
    c = pipe.driver_model.contributions(row)
    assert sum(c.values()) == pytest.approx(row["err"], abs=1e-6)


def test_predict_outage_lowers_forecast(pipe):
    base = pipe.predict(700, 8, 32, 0)["forecast_mwh"]
    out = pipe.predict(700, 8, 32, 1)["forecast_mwh"]
    assert out < base


def test_parse_query():
    assert parse_query("Why was 2026-08-15 unit B off?", 2026)["asset"] == "B"
    assert parse_query("explain Aug 8", 2026)["date"] == "2026-08-08"
    assert parse_query("15th august", 2026)["date"] == "2026-08-15"
    assert parse_query("list outage days", 2026)["intent"] == "outages"
    assert parse_query("hello", 2026)["intent"] == "fleet"


def test_agent_explains_outage_day(pipe):
    r = run_agent(pipe, "why was 2026-08-15 unit A off", use_llm=False)
    e = r["explanations"][0]
    assert e["primary_driver"] == "outage" and e["confidence"] == "high"
    assert r["llm_used"] is False and any(t["tool"] == "explain_row" for t in r["trace"])


def test_agent_unknown_date(pipe):
    assert "No data" in run_agent(pipe, "2030-01-01", use_llm=False)["answer"]


def test_grounding_guard():
    ev = '{"miss_mwh": -17.17, "driver": 12.4}'
    assert grounded("The miss was 17.2 MWh, mostly outage (12.4).", ev)
    assert not grounded("The miss was 99.9 MWh.", ev)


def test_api_endpoints():
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/summary").status_code == 200
    assert len(client.get("/api/series?asset=A").json()) == 30
    assert client.get("/api/variance?date=2026-08-15&asset=A").json()["primary_driver"] == "outage"
    assert client.get("/api/variance?date=2030-01-01&asset=A").status_code == 404
    assert client.get("/api/variance?date=bad&asset=A").status_code == 422
    r = client.post("/api/forecast", json={"irradiance_wm2": 600, "wind_speed_ms": 7, "temperature_c": 30})
    assert r.status_code == 200 and r.json()["forecast_mwh"] > 0
    assert client.post("/api/forecast", json={"irradiance_wm2": -5, "wind_speed_ms": 7, "temperature_c": 30}).status_code == 422
    assert client.post("/api/run", json={"text": "summary", "use_llm": False}).json()["intent"] == "fleet"
    assert "<html" in client.get("/").text
