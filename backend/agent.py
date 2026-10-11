"""Lightweight variance-explanation agent.

Design
------
The agent is a small *tool-using planner*: it routes a natural-language question to an intent,
calls deterministic analysis tools (each call is recorded in a trace shown in the UI) and then
writes the answer.

* All numbers come from the tools (pandas / numpy), never from an LLM.
* Narration is deterministic by default, so the app works offline and in the Railway free tier.
* If ``GEMINI_API_KEY`` is set, Gemini rewrites the evidence into a short analyst note.
  The reply is rejected (falls back to the template) if it contains numbers that are not present in
  the evidence, so the LLM cannot invent figures. Configure with ``LLM_MODEL`` (default below).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

from .pipeline import DRIVERS, RAW, UNITS, Pipeline

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DEFAULT_LLM_MODEL = "gemini-2.5-flash"


# --------------------------------------------------------------------------- tools
def tool_similar_conditions(p: Pipeline, row: pd.Series, k: int = 6) -> dict:
    """Misses on the k most similar-weather, non-outage rows (is this miss typical for such weather?)."""
    d = p.df[(p.df["outage_flag"] == 0) & (p.df["row_id"] != row["row_id"])]
    cols = RAW[:3]
    sd = p.df[cols].std().replace(0, 1)
    dist = (((d[cols] - row[cols].astype(float)) / sd) ** 2).sum(axis=1) ** 0.5
    nn = d.loc[dist.nsmallest(k).index]
    return {
        "k": int(len(nn)),
        "mean_miss_mwh": round(float(nn["err"].mean()), 2),
        "std_miss_mwh": round(float(nn["err"].std(ddof=0)), 2),
        "rows": nn["row_id"].tolist(),
    }


def tool_explain_row(p: Pipeline, row: pd.Series) -> dict:
    err = float(row["err"])
    contrib = p.driver_model.contributions(row)
    drivers_only = {k: v for k, v in contrib.items() if k in DRIVERS.values()}
    ranked = sorted(drivers_only.items(), key=lambda kv: abs(kv[1]), reverse=True)
    sim = tool_similar_conditions(p, row)
    outage = bool(row["outage_flag"])
    sev = row["severity"]
    unexplained = contrib["unexplained"]
    explained_share = 0.0 if err == 0 else max(0.0, min(1.0, 1 - abs(unexplained) / max(abs(err), 1e-9)))
    direction = "under-forecast (actual above forecast)" if err > 0 else "over-forecast (actual below forecast)"

    caveats = []
    if outage:
        o_coef = p.driver_model.coef[4]
        lo, hi = p.driver_model.ci["outage_flag"]
        primary = "outage"
        confidence = "high" if (hi < 0 or lo > 0) else "medium"
        headline = (
            f"Outage-flagged day: actual was {abs(err):.1f} MWh {'below' if err < 0 else 'above'} forecast. "
            f"The forecast already derates for outages, but on average by ~{abs(o_coef):.0f} MWh too little (90% CI {lo:.0f} to {hi:.0f})."
        )
        caveats.append(f"Outage effect is estimated from only {int(p.df['outage_flag'].sum())} flagged rows; treat the size as indicative.")
    elif sev == "minor":
        primary = "noise"
        confidence = "medium"
        headline = f"Within normal forecast noise: miss of {err:+.1f} MWh is below the typical daily miss of {p.mae_noise_threshold:.1f} MWh."
    else:
        top, top_v = ranked[0]
        if abs(unexplained) > 0.7 * abs(err):
            primary = "unexplained"
            confidence = "low"
            headline = (
                f"{sev.capitalize()} miss of {err:+.1f} MWh that the available drivers explain little of "
                f"(largest: {top} at {top_v:+.1f} MWh). Likely driven by something not in the data."
            )
            caveats.append(
                "Unmodelled factors worth checking: intra-day cloud/wind variability, curtailment, soiling, "
                "inverter/turbine derating, forecast-model error. These are hypotheses, not findings."
            )
        else:
            primary = top
            raw_col = RAW[list(DRIVERS.values()).index(top)]
            lo_, hi_ = p.driver_model.ci[raw_col]
            confidence = "medium" if lo_ * hi_ > 0 else "low"  # CI excludes 0 -> medium
            headline = f"{sev.capitalize()} miss of {err:+.1f} MWh, mostly attributed to {top} ({top_v:+.1f} MWh)."
    if primary != "outage" and not any(d["significant"] for d in p.driver_model.table(p.df) if d["driver"] != "outage"):
        caveats.append("No weather driver is statistically significant at fleet level (90% bootstrap CI includes 0), so weather attributions are weak.")
    if sim["k"] and abs(sim["mean_miss_mwh"]) < 0.5 * p.mae_noise_threshold and not outage:
        caveats.append(f"Similar-weather days average a miss of {sim['mean_miss_mwh']:+.1f} MWh (σ {sim['std_miss_mwh']:.1f}), i.e. no systematic weather bias.")

    row_vals = {DRIVERS[c]: f"{row[c]:g} {UNITS[c]}" if c != "outage_flag" else ("yes" if outage else "no") for c in RAW}
    return {
        "row_id": row["row_id"],
        "date": row["date"].strftime("%Y-%m-%d"),
        "asset": row["asset"],
        "inputs": row_vals,
        "actual_mwh": float(row["actual_mwh"]),
        "forecast_mwh": float(row["forecast_mwh"]),
        "model_forecast_mwh": None if pd.isna(row.get("model_forecast_oof", np.nan)) else round(float(row["model_forecast_oof"]), 2),
        "miss_mwh": round(err, 2),
        "miss_pct_of_forecast": round(err / row["forecast_mwh"] * 100, 1) if row["forecast_mwh"] else None,
        "direction": direction,
        "severity": sev,
        "robust_z": round(float(row["z"]), 2),
        "contributions_mwh": [{"driver": k, "mwh": round(v, 2)} for k, v in ranked]
        + [
            {"driver": "systematic bias", "mwh": round(contrib["systematic bias"], 2)},
            {"driver": "unexplained", "mwh": round(unexplained, 2)},
        ],
        "primary_driver": primary,
        "explained_share": round(explained_share, 2),
        "confidence": confidence,
        "headline": headline,
        "similar_conditions": sim,
        "caveats": caveats,
    }


def tool_rank_misses(p: Pipeline, n: int = 5, outage_only: bool = False) -> pd.DataFrame:
    d = p.df[p.df["outage_flag"] == 1] if outage_only else p.df
    return d.reindex(d["err"].abs().sort_values(ascending=False).index).head(n)


# ----------------------------------------------------------------- query parsing
def parse_query(text: str, default_year: int) -> dict:
    t = text.lower()
    out = {"date": None, "asset": None}
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        out["date"] = m.group(0)
    else:
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b", t) or None
        if m:
            out["date"] = f"{default_year}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"
        else:
            m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", t)
            if m:
                out["date"] = f"{default_year}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    a = re.search(r"\b(?:unit|asset|site|plant)\s*([a-z])\b", t)
    if a:
        out["asset"] = a.group(1).upper()
    n = re.search(r"\b(?:top|worst|biggest|largest)\s*(\d{1,2})\b", t)
    out["n"] = int(n.group(1)) if n else 5
    if out["date"]:
        out["intent"] = "day"
    elif "outage" in t:
        out["intent"] = "outages"
    elif re.search(r"worst|biggest|largest|top|miss(es)?\b|rank", t):
        out["intent"] = "top_misses"
    else:
        out["intent"] = "fleet"
    return out


# ------------------------------------------------------------------ narration
def narrate_day(e: dict) -> str:
    lines = [
        f"{e['date']} (unit {e['asset']}): actual {e['actual_mwh']:.1f} MWh vs forecast {e['forecast_mwh']:.1f} MWh "
        f"→ {e['miss_mwh']:+.1f} MWh ({e['direction']}).",
        e["headline"],
    ]
    top = [c for c in e["contributions_mwh"] if c["driver"] not in ("unexplained", "systematic bias")][:2]
    if e["primary_driver"] not in ("noise",):
        parts = ", ".join(f"{c['driver']} {c['mwh']:+.1f}" for c in top)
        unexpl = next(c["mwh"] for c in e["contributions_mwh"] if c["driver"] == "unexplained")
        lines.append(f"Attribution (MWh): {parts}; unexplained {unexpl:+.1f}.")
    lines.append(f"Confidence: {e['confidence']}.")
    lines += [f"Caveat: {c}" for c in e["caveats"]]
    return " ".join(lines)


def narrate_fleet(s: dict) -> str:
    wf = s["walk_forward"]
    op, mdl = wf["operational_forecast"], wf[s["selected_model"]]
    sig = [d for d in s["drivers"] if d["significant"]]
    insig = [d["driver"] for d in s["drivers"] if not d["significant"]]
    txt = (
        f"Across {s['data']['rows']} unit-days the operational forecast misses by {s['operational_forecast_full_sample']['mae']:.1f} MWh on average "
        f"({s['operational_forecast_full_sample']['wape_pct']:.1f}% WAPE) with little overall bias. "
    )
    if sig:
        d = sig[0]
        txt += (
            f"The one statistically reliable miss driver is {d['driver']}: when it is flagged, actual generation is ~{abs(d['coef_mwh_per_unit']):.0f} MWh "
            f"{'lower' if d['coef_mwh_per_unit'] < 0 else 'higher'} than the forecast already says (90% CI {d['ci90'][0]:.0f} to {d['ci90'][1]:.0f}). "
        )
    if insig:
        txt += f"No reliable signal for {', '.join(insig)}; the remaining miss looks like noise of roughly ±{s['noise_floor_mwh']:.0f} MWh. "
    txt += (
        f"Out-of-sample, our weather-only baseline ({s['selected_model']}) has MAE {mdl['mae']:.1f} vs {op['mae']:.1f} for the operational forecast, "
        "so it does not beat it; it is an independent sanity check, not a replacement."
    )
    return txt


# --------------------------------------------------------------------- LLM (optional)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    """Extract numeric claims while ignoring date components."""
    # Ignore ISO dates, including single-digit month/day formats.
    text = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", text)
    months = (
        r"January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|"
        r"Oct|Nov|Dec"
    )
    # Ignore dates such as "August 16, 2026", "Aug 16", or "16 Aug".
    text = re.sub(
        rf"\b(?:{months})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{months})\.?\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return [float(x) for x in _NUM.findall(text)]


def grounded(llm_text: str, evidence_json: str, tol: float = 0.051) -> bool:
    """Reject unsupported numeric claims while ignoring dates in the narration."""
    ev = [abs(x) for x in _numbers(evidence_json)]
    for n in _numbers(llm_text):
        n = abs(n)
        if n <= 10 and float(n).is_integer():  # counts / "top 3" etc.
            continue
        if not any(abs(n - e) <= tol or abs(n - round(e)) < 1e-9 for e in ev):
            print(f"Grounding check rejected number: {n}")
            return False
    return True


def llm_narrate(question: str, evidence: dict) -> str | None:
    """Use Gemini to narrate grounded evidence; fall back to the template."""
    try:
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()

        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key or key == "PASTE_YOUR_NEW_API_KEY_HERE":
            return None

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        ev = json.dumps(evidence, default=str)
        prompt = (
            "You are a renewable-energy operations analyst. Write a concise "
            "plain-English explanation in at most 120 words using only the "
            "JSON evidence below. Do not invent numerical claims. State "
            "confidence and the main caveat. Label possible causes as "
            "hypotheses, not facts.\\n\\n"
            f"Question: {question}\\n\\nEvidence JSON:\\n{ev}"
        )

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        answer = (response.text or "").strip()
        if not answer:
            print("Gemini returned empty text")
            return None
        if not grounded(answer, ev):
            print("Gemini response rejected by grounded() validation")
            print("Gemini response:", answer)
            return None
        print("Gemini response accepted by grounded() validation")
        return answer
    except Exception as exc:
        print(f"Gemini request failed: {type(exc).__name__}: {exc}")
        return None


# ------------------------------------------------------------------------ agent
def run_agent(p: Pipeline, question: str, use_llm: bool = True) -> dict:
    trace: list[dict] = []
    year = int(p.df["date"].min().year)
    plan = parse_query(question or "", year)
    trace.append({"step": "route", "tool": "parse_query", "result": {k: v for k, v in plan.items()}})

    explanations: list[dict] = []
    fleet = None
    notes = ""

    if plan["intent"] == "day":
        rows = p.df[p.df["date"] == pd.Timestamp(plan["date"])]
        if plan["asset"]:
            rows = rows[rows["asset"] == plan["asset"]]
        trace.append({"step": "lookup", "tool": "get_records", "result": f"{len(rows)} row(s) for {plan['date']}" + (f" unit {plan['asset']}" if plan["asset"] else "")})
        if rows.empty:
            lo, hi = p.df["date"].min().date(), p.df["date"].max().date()
            return {"intent": "day", "trace": trace, "answer": f"No data for {plan['date']}. The dataset covers {lo} to {hi}.", "explanations": [], "llm_used": False}
        for _, r in rows.iterrows():
            explanations.append(tool_explain_row(p, r))
    elif plan["intent"] in ("top_misses", "outages"):
        sel = tool_rank_misses(p, plan["n"], outage_only=plan["intent"] == "outages")
        trace.append({"step": "rank", "tool": "rank_misses", "result": f"{len(sel)} rows by |miss|" + (", outage-flagged only" if plan["intent"] == "outages" else "")})
        for _, r in sel.iterrows():
            explanations.append(tool_explain_row(p, r))
        if plan["intent"] == "outages" and not len(sel):
            notes = "No outage-flagged rows in the data."
    else:
        fleet = p.summary()
        trace.append({"step": "aggregate", "tool": "fleet_summary", "result": "walk-forward metrics + driver regression"})

    for e in explanations:
        trace.append({"step": "attribute", "tool": "explain_row", "result": f"{e['row_id']}: miss {e['miss_mwh']:+.1f} MWh → {e['primary_driver']} ({e['confidence']} confidence)"})

    if fleet is not None:
        base = narrate_fleet(fleet)
        evidence = {k: fleet[k] for k in ("walk_forward", "drivers", "noise_floor_mwh", "selected_model", "operational_forecast_full_sample")}
    else:
        base = notes or "\n".join(narrate_day(e) for e in explanations)
        evidence = [{k: v for k, v in e.items() if k not in ("inputs",)} for e in explanations]

    llm_text = llm_narrate(question, evidence) if use_llm and (fleet is not None or explanations) else None
    trace.append({"step": "narrate", "tool": "llm" if llm_text else "template", "result": "LLM note passed numeric grounding check" if llm_text else "deterministic template"})
    return {
        "intent": plan["intent"],
        "trace": trace,
        "answer": llm_text or base,
        "template_answer": base if llm_text else None,
        "explanations": explanations,
        "fleet": fleet,
        "llm_used": bool(llm_text),
    }
