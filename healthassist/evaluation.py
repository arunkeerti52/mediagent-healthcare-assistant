"""LLMOps evaluation: planner accuracy, booking success, QAEvalChain on summaries & search, tool success rates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from langchain_classic.evaluation.qa import QAEvalChain
except ImportError:
    from langchain.evaluation.qa import QAEvalChain

from .agent import HealthcareAgent
from .db import ROOT, HospitalDB
from .memory import SessionMemory

EVAL_DB = ROOT / "data" / "eval_hospital.db"
OUT = ROOT / "data"

# Expected plans follow the planner's rules: history is loaded before booking, summarising or searching.
PLANNER_CASES = [
    ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
     ["identify_patient", "get_medical_history", "summarize_history", "check_doctor_availability", "book_appointment", "medical_search"]),
    ("Book a cardiologist for my mother next Monday morning",
     ["identify_patient", "get_medical_history", "check_doctor_availability", "book_appointment"]),
    ("Summarize my father's medical history", ["identify_patient", "get_medical_history", "summarize_history"]),
    ("What are the latest treatment options for type 2 diabetes?", ["identify_patient", "get_medical_history", "medical_search"]),
    ("Add a note that my father was prescribed iron tablets 100 mg daily", ["identify_patient", "add_medical_record"]),
    ("Show my upcoming appointments", ["identify_patient", "list_appointments"]),
    ("My mother has knee pain, book an orthopedic doctor and explain osteoarthritis treatment",
     ["identify_patient", "get_medical_history", "check_doctor_availability", "book_appointment", "medical_search"]),
    ("Record that my father's latest eGFR test result is 32", ["identify_patient", "add_medical_record"]),
    ("I need an appointment with a pulmonologist tomorrow",
     ["identify_patient", "get_medical_history", "check_doctor_availability", "book_appointment"]),
    ("What are the symptoms of heart failure?", ["identify_patient", "get_medical_history", "medical_search"]),
]

BOOKING_CASES = [
    ("Arun Kumar", "Book a nephrologist for my father", "Nephrology", "P001"),
    ("Arun Kumar", "Book a cardiologist for my mother next Monday morning", "Cardiology", "P003"),
    ("Arun Kumar", "Schedule a general physician check-up for me tomorrow", "General Medicine", "P002"),
    ("Sara Thomas", "I need an appointment with a pulmonologist for my asthma", "Pulmonology", "P004"),
    ("Imran Qureshi", "Book a cardiologist for me in the afternoon", "Cardiology", "P005"),
    ("Arun Kumar", "My mother has knee pain, book an orthopedic doctor", "Orthopedics", "P003"),
    ("Arun Kumar", "Book an endocrinologist for my father's diabetes", "Endocrinology", "P001"),
    ("Arun Kumar", "Book an appointment for my father", "Nephrology", "P001"),  # specialty inferred from history
]

SUMMARY_REFS = {
    "P001": "Ramesh Kumar has chronic kidney disease stage 3b, type 2 diabetes and hypertension; takes telmisartan, metformin, dapagliflozin and atorvastatin; alert to avoid NSAIDs and sulfa allergy.",
    "P003": "Lakshmi Kumar has osteoarthritis of both knees and hypothyroidism treated with levothyroxine.",
    "P005": "Imran Qureshi has coronary artery disease with a stent and takes aspirin, atorvastatin and metoprolol.",
}
SEARCH_CASES = [
    ("Arun Kumar", "What are the latest treatment methods for chronic kidney disease?",
     "Chronic kidney disease treatment slows progression; kidney failure is treated with dialysis or kidney transplant."),
    ("Sara Thomas", "What is the treatment for asthma?", "Asthma is treated with inhaled corticosteroids and bronchodilator inhalers."),
    ("Imran Qureshi", "How is coronary artery disease treated?", "Coronary artery disease is treated with statins, aspirin, angioplasty with stents or bypass surgery."),
    ("Arun Kumar", "What are the treatment options for type 2 diabetes?", "Type 2 diabetes is treated with diet, physical activity, metformin and insulin when needed."),
]


def _grade(llm, examples, preds) -> list[str]:
    graded = QAEvalChain.from_llm(llm).evaluate(examples, preds, question_key="question", answer_key="answer", prediction_key="result")
    return ["INCORRECT" if "INCORRECT" in (g.get("results") or g.get("text", "")).upper() else "CORRECT" for g in graded]


def run_evaluation() -> dict:
    db = HospitalDB(EVAL_DB)
    db.reset()
    agent = HealthcareAgent(db)

    # 1. Planner / goal decomposition
    prow = []
    for q, expected in PLANNER_CASES:
        plan = agent.planner.plan(q, "")
        got = [s["tool"] for s in plan]
        recall = len(set(expected) & set(got)) / len(set(expected))
        precision = len(set(expected) & set(got)) / len(set(got))
        prow.append({"query": q, "expected_tools": " → ".join(expected), "planned_tools": " → ".join(got),
                     "tool_recall": round(recall, 2), "tool_precision": round(precision, 2), "exact_match": got == expected})
    planner = pd.DataFrame(prow)

    # 2. Booking success
    brow = []
    for user, q, spec, pid in BOOKING_CASES:
        r = agent.handle(SessionMemory(user), q)
        appt = db.q("SELECT * FROM appointments ORDER BY appointment_id DESC LIMIT 1")
        booked = r["plan"] and any(s["tool"] == "book_appointment" and s["success"] for s in r["plan"])
        ok_spec = ok_patient = False
        if booked and len(appt):
            a = appt.iloc[0]
            ok_spec = agent.schedule.doctor(a.doctor_id)["specialty"] == spec
            ok_patient = a.patient_id == pid
        brow.append({"request": q, "user": user, "expected_specialty": spec, "booked": bool(booked), "right_specialty": ok_spec,
                     "right_patient": ok_patient, "success": bool(booked and ok_spec and ok_patient), "latency_s": r["latency_s"]})
    booking = pd.DataFrame(brow)
    dup = db.q("SELECT doctor_id, slot_start, COUNT(*) n FROM appointments GROUP BY 1,2 HAVING n>1")

    # 3. QAEvalChain - history summaries
    sess = SessionMemory("eval")
    examples, preds, srow = [], [], []
    for pid, ref in SUMMARY_REFS.items():
        ctx = {"query": "summarise history", "patient": agent.ehr.patient(pid)}
        summary = agent._summarize_history(sess, {}, ctx)
        examples.append({"question": f"Summarise the medical history of {pid}", "answer": ref})
        preds.append({"result": summary})
        srow.append({"case": f"History summary {pid}", "reference": ref, "prediction": summary})
    # 4. QAEvalChain - medical search (RAG)
    for user, q, ref in SEARCH_CASES:
        r = agent.handle(SessionMemory(user), q)
        out = next((s["output"] for s in r["plan"] if s["tool"] == "medical_search"), "")
        examples.append({"question": q, "answer": ref})
        preds.append({"result": out})
        srow.append({"case": f"Search: {q}", "reference": ref, "prediction": out})
    grades = _grade(agent.llm, examples, preds)
    # Negative control: deliberately wrong answers must be graded INCORRECT
    ctrl = _grade(agent.llm, [examples[0], examples[len(SUMMARY_REFS)]],
                  [{"result": "The patient is healthy with no chronic conditions and takes no medication."},
                   {"result": "Chronic kidney disease is cured with antibiotics and does not need dialysis or transplant."}])
    caught = sum(g == "INCORRECT" for g in ctrl)
    qa = pd.DataFrame(srow)
    qa["grade"] = grades
    qa["type"] = ["summary"] * len(SUMMARY_REFS) + ["search"] * len(SEARCH_CASES)

    # 5. Tool success from logs
    logs = db.q("SELECT tool, success, latency_ms FROM tool_logs")
    tools = logs.groupby("tool").agg(calls=("success", "size"), success_rate=("success", "mean"), avg_latency_ms=("latency_ms", "mean")).reset_index()
    tools["success_rate"] = (tools["success_rate"] * 100).round(1)
    tools["avg_latency_ms"] = tools["avg_latency_ms"].round(1)

    summary = {
        "planner_tool_recall_%": float(round(100 * planner["tool_recall"].mean(), 1)),
        "planner_tool_precision_%": float(round(100 * planner["tool_precision"].mean(), 1)),
        "planner_exact_match_%": float(round(100 * planner["exact_match"].mean(), 1)),
        "booking_success_%": float(round(100 * booking["success"].mean(), 1)),
        "double_bookings": int(len(dup)),
        "summary_qa_accuracy_%": float(round(100 * (qa[qa.type == "summary"].grade == "CORRECT").mean(), 1)),
        "search_qa_accuracy_%": float(round(100 * (qa[qa.type == "search"].grade == "CORRECT").mean(), 1)),
        "tool_success_%": float(round(100 * logs["success"].mean(), 1)),
        "grader_negative_controls_caught": f"{caught}/{len(ctrl)}",
    }
    planner.to_csv(OUT / "eval_planner.csv", index=False)
    booking.to_csv(OUT / "eval_booking.csv", index=False)
    qa.to_csv(OUT / "eval_qa.csv", index=False)
    tools.to_csv(OUT / "eval_tools.csv", index=False)
    (OUT / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "planner": planner, "booking": booking, "qa": qa, "tools": tools}


def load_saved() -> dict | None:
    """Results of the last evaluation run (so the dashboard can show them without spending LLM quota)."""
    files = {k: OUT / f"eval_{k}.csv" for k in ("planner", "booking", "qa", "tools")}
    if not (OUT / "eval_summary.json").exists() or not all(f.exists() for f in files.values()):
        return None
    return {"summary": json.loads((OUT / "eval_summary.json").read_text()), **{k: pd.read_csv(f) for k, f in files.items()}}


if __name__ == "__main__":
    res = run_evaluation()
    print(res["planner"][["planned_tools", "tool_recall", "tool_precision", "exact_match"]].to_string())
    print(res["booking"].to_string())
    print(res["qa"][["case", "grade"]].to_string())
    print(json.dumps(res["summary"], indent=2))
