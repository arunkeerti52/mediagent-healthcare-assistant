"""MediAgent - Streamlit dashboard for the Agentic Healthcare Assistant.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from healthassist.agent import HealthcareAgent
from healthassist.db import HospitalDB
from healthassist.evaluation import load_saved, run_evaluation
from healthassist.llm import provider_name
from healthassist.memory import SessionMemory
from healthassist.tools import DOCTORS

st.set_page_config(page_title="MediAgent Healthcare Assistant", page_icon="🩺", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem}
div[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px}
.step{border-left:4px solid #0ea5e9;background:#f0f9ff;padding:6px 12px;margin:4px 0;border-radius:4px;font-size:0.9rem}
.step.fail{border-color:#ef4444;background:#fef2f2}
</style>""", unsafe_allow_html=True)

SCENARIOS = [
    "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
    "Summarize my father's medical history",
    "Add a note that he was prescribed iron tablets 100 mg daily for anaemia",
    "Book a cardiologist for my mother next Monday morning",
    "My mother has knee pain, book an orthopedic doctor and explain osteoarthritis treatment",
    "Show my father's upcoming appointments",
]


@st.cache_resource
def get_agent():
    return HealthcareAgent(HospitalDB())


agent = get_agent()
db = agent.db

with st.sidebar:
    st.title("🩺 MediAgent")
    st.caption("Agentic Healthcare Assistant")
    st.info(f"**LLM:** {provider_name()}")
    user = st.selectbox("Signed-in user (patient / caregiver)", ["Arun Kumar", "Sara Thomas", "Imran Qureshi"])
    if st.session_state.get("user") != user:
        st.session_state["user"] = user
        st.session_state["session"] = SessionMemory(user)
        st.session_state["chat"] = []
    st.markdown("**Test scenarios**")
    for i, sc in enumerate(SCENARIOS):
        if st.button(sc, key=f"sc{i}", use_container_width=True):
            st.session_state["pending"] = sc
    if st.button("Reset demo database", use_container_width=True):
        db.reset()
        agent.memory.rebuild()
        st.session_state["session"] = SessionMemory(user)
        st.session_state["chat"] = []
        st.rerun()

session: SessionMemory = st.session_state["session"]
st.header("Agentic Healthcare Assistant for Medical Task Automation")

appts = agent.schedule.appointments()
logs = db.q("SELECT * FROM tool_logs ORDER BY id DESC")
m = st.columns(5)
m[0].metric("Patients", int(db.q("SELECT COUNT(*) n FROM patients").n[0]))
m[1].metric("Upcoming appointments", int((appts.status == "Booked").sum()))
m[2].metric("EHR records", int(db.q("SELECT COUNT(*) n FROM medical_records").n[0]))
m[3].metric("Tool calls", len(logs))
m[4].metric("Tool success rate", f"{100 * logs.success.mean():.1f}%" if len(logs) else "–")

tabs = st.tabs(["🤖 Assistant", "🧑 Patient View", "👩‍⚕️ Doctor View", "📅 Appointments", "🔬 Medical Info", "📊 Evaluation", "🧠 Memory & Logs"])

# ----------------------------------------------------------------- assistant --
with tabs[0]:
    for turn in st.session_state["chat"]:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant", avatar="🩺"):
            with st.expander(f"🧭 LLM plan: {len(turn['plan'])} sub-goals · {turn['latency_s']}s", expanded=True):
                for s in turn["plan"]:
                    args = {k: v for k, v in (s.get("args") or {}).items() if k not in ("reference", "question", "reason") and v}
                    st.markdown(f"<div class='step{'' if s['success'] else ' fail'}'><b>Step {s['step']}</b> · {s['goal']} → "
                                f"<code>{s['tool']}</code> {json.dumps(args) if args else ''} · {'✅' if s['success'] else '❌'} {s['latency_ms']} ms</div>",
                                unsafe_allow_html=True)
            st.markdown(turn["response"])
    q = st.chat_input("Describe what you need, e.g. book a doctor, update records, summarise history, disease info...") or st.session_state.pop("pending", None)
    if q:
        try:
            with st.spinner("Planning and executing..."):
                r = agent.handle(session, q)
            st.session_state["chat"].append({"query": q, **r})
            st.rerun()
        except Exception as e:  # e.g. Cloudflare quota / network errors
            st.error(f"LLM request failed: {type(e).__name__}: {e}")

# -------------------------------------------------------------- patient view --
with tabs[1]:
    pats = db.q("SELECT * FROM patients")
    pid = st.selectbox("Patient", pats.patient_id, format_func=lambda x: f"{x} - {pats.set_index('patient_id').name[x]}")
    p = agent.ehr.patient(pid)
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"### {p['name']} · {p['age']}y {p['gender']} · {p['blood_group']}")
        if st.button("✨ Generate AI history summary"):
            st.session_state[f"sum_{pid}"] = agent._summarize_history(session, {}, {"query": "summary", "patient": p})
        if f"sum_{pid}" in st.session_state:
            st.markdown(st.session_state[f"sum_{pid}"])
        st.markdown("**Medical records (EHR)**")
        st.dataframe(agent.ehr.records(pid), use_container_width=True, hide_index=True, height=300)
    with b:
        st.markdown("**Add / update history** (attendant)")
        with st.form("add_rec", clear_on_submit=True):
            kind = st.selectbox("Record type", ["note", "diagnosis", "medication", "lab", "alert"])
            desc = st.text_area("Description (structured or free text)")
            if st.form_submit_button("Save record") and desc.strip():
                agent.ehr.add_record(pid, kind, desc.strip(), added_by="attendant")
                agent.memory.add(pid, f"{date.today()} {kind}: {desc.strip()}", kind)
                st.success("Saved to EHR and long-term memory")
                st.rerun()
        st.markdown("**Upcoming appointments**")
        st.dataframe(appts[(appts.patient_id == pid)][["slot_start", "doctor", "specialty", "status"]], hide_index=True, use_container_width=True)
        with st.expander("Register new patient"):
            with st.form("new_pat", clear_on_submit=True):
                n = st.text_input("Name"); ag = st.number_input("Age", 0, 120, 40); g = st.selectbox("Gender", ["Male", "Female", "Other"])
                rel = st.selectbox("Relation to signed-in user", ["self", "father", "mother", "wife", "husband", "son", "daughter"])
                if st.form_submit_button("Register") and n:
                    new = agent.ehr.add_patient(n, int(ag), g, caregiver=user, relation=rel)
                    agent.memory.rebuild()
                    st.success(f"Registered {new}")

# --------------------------------------------------------------- doctor view --
with tabs[2]:
    doc = st.selectbox("Doctor", [d["doctor_id"] for d in DOCTORS],
                       format_func=lambda x: next(f"{d['name']} - {d['specialty']}" for d in DOCTORS if d["doctor_id"] == x))
    d = agent.schedule.doctor(doc)
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"### {d['name']} · {d['specialty']}\n{d['location']} · {', '.join(d['days'])} {d['start']}-{d['end']}")
        mine = appts[appts.doctor_id == doc]
        st.markdown("**Patient appointments**")
        st.dataframe(mine[["appointment_id", "slot_start", "patient", "reason", "status"]], hide_index=True, use_container_width=True)
        if len(mine):
            c1, c2, c3 = st.columns(3)
            aid = c1.selectbox("Appointment", mine.appointment_id)
            stt = c2.selectbox("Set status", ["Booked", "Completed", "Cancelled", "No-show"])
            if c3.button("Update", use_container_width=True):
                agent.schedule.update_status(int(aid), stt)
                st.rerun()
    with b:
        st.markdown("**Next free slots**")
        free = agent.schedule.available_slots(doc, limit=12)
        st.dataframe(pd.DataFrame({"slot": [datetime.fromisoformat(s).strftime("%a %d %b, %I:%M %p") for s in free]}), hide_index=True, use_container_width=True)

# -------------------------------------------------------------- appointments --
with tabs[3]:
    st.markdown("**Real-time appointment tracking**")
    if appts.empty:
        st.caption("No appointments yet - try a booking scenario in the Assistant tab.")
    else:
        a, b = st.columns([3, 2])
        a.dataframe(appts[["appointment_id", "slot_start", "patient", "doctor", "specialty", "status", "reason"]], hide_index=True, use_container_width=True, height=320)
        b.plotly_chart(px.bar(appts.groupby(["specialty", "status"]).size().reset_index(name="count"), x="specialty", y="count", color="status",
                              title="Appointments by specialty & status").update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0)), use_container_width=True)
        ap = appts.copy()
        ap["start"] = pd.to_datetime(ap.slot_start)
        ap["end"] = ap.start + pd.Timedelta(minutes=30)
        st.plotly_chart(px.timeline(ap, x_start="start", x_end="end", y="doctor", color="patient", hover_data=["status", "specialty"],
                                    title="Schedule timeline").update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0)), use_container_width=True)

# --------------------------------------------------------------- medical info --
with tabs[4]:
    c1, c2 = st.columns([4, 1])
    topic = c1.text_input("Search a disease / treatment (live MedlinePlus + WHO)", "chronic kidney disease treatment")
    if c2.button("Search", use_container_width=True, type="primary"):
        with st.spinner("Searching trusted sources and summarising..."):
            out = agent._medical_search(session, {"topic": topic, "question": topic}, {"query": topic})
    ls = agent.last_search
    if ls:
        st.markdown(f"**Latest retrieved information** · topic: *{ls['topic']}* · {ls['ts']} · mode: {ls['mode']} · {ls['n_chunks']} chunks indexed")
        st.markdown(ls["answer"])
        st.markdown("**Sources**\n" + "\n".join(f"{i + 1}. [{t} - {src}]({u})" for i, (t, src, u) in enumerate(ls["sources"])))
    else:
        st.caption("Run a search here or ask the assistant a disease question.")

# ---------------------------------------------------------------- evaluation --
with tabs[5]:
    st.markdown("Planner decomposition accuracy, booking success, **QAEvalChain** grading of history summaries and RAG search answers, "
                "and per-tool success rates (run on an isolated evaluation DB).")
    if st.button("▶️ Run evaluation", type="primary"):
        with st.spinner("Running evaluation suite..."):
            st.session_state["eval"] = run_evaluation()
    elif "eval" not in st.session_state and load_saved():
        st.session_state["eval"] = load_saved()
        st.caption("Showing the results of the last evaluation run (data/eval_*.csv). Click the button to re-run.")
    if "eval" in st.session_state:
        e = st.session_state["eval"]; s = e["summary"]
        r = st.columns(6)
        r[0].metric("Planner tool recall", f"{s['planner_tool_recall_%']}%")
        r[1].metric("Planner exact match", f"{s['planner_exact_match_%']}%")
        r[2].metric("Booking success", f"{s['booking_success_%']}%")
        r[3].metric("Summary QA (QAEvalChain)", f"{s['summary_qa_accuracy_%']}%")
        r[4].metric("Search QA (QAEvalChain)", f"{s['search_qa_accuracy_%']}%")
        r[5].metric("Tool success", f"{s['tool_success_%']}%")
        st.caption(f"QAEvalChain negative control: {s['grader_negative_controls_caught']} deliberately wrong answers graded INCORRECT.")
        a, b = st.columns(2)
        a.markdown("**Tool performance**")
        a.dataframe(e["tools"], hide_index=True, use_container_width=True)
        b.plotly_chart(px.bar(e["tools"], x="tool", y="success_rate", text="calls", title="Tool success rate % (label = calls)", range_y=[0, 105])
                       .update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0)), use_container_width=True)
        st.markdown("**QAEvalChain results**")
        st.dataframe(e["qa"][["type", "case", "reference", "grade"]], hide_index=True, use_container_width=True)
        st.markdown("**Booking scenarios**")
        st.dataframe(e["booking"], hide_index=True, use_container_width=True)
        st.markdown("**Planner goal decomposition**")
        st.dataframe(e["planner"], hide_index=True, use_container_width=True)

# ------------------------------------------------------------ memory & logs --
with tabs[6]:
    a, b = st.columns(2)
    with a:
        st.markdown("**Session memory (short-term facts)**")
        st.json(session.facts or {})
        st.markdown("**Memory trace (reads / writes / recalls)**")
        st.dataframe(pd.DataFrame(session.trace[-25:][::-1]), hide_index=True, use_container_width=True, height=260)
    with b:
        st.markdown("**Long-term patient memory (FAISS) — semantic recall**")
        pats = db.q("SELECT * FROM patients")
        mp = st.selectbox("Patient memory", pats.patient_id, key="mem_p", format_func=lambda x: f"{x} - {pats.set_index('patient_id').name[x]}")
        mq = st.text_input("Recall query", "kidney function and medications")
        for d in agent.memory.recall(mp, mq, k=5):
            st.markdown(f"- `{d.metadata['kind']}` {d.page_content}")
    st.markdown("**Planning breakdowns (history)**")
    plans = db.q("SELECT * FROM plans ORDER BY ts DESC LIMIT 20")
    for pl in plans.itertuples():
        with st.expander(f"{pl.ts} · {pl.user_name}: {pl.query[:90]}"):
            st.dataframe(pd.DataFrame(json.loads(pl.plan))[["step", "goal", "tool", "success", "latency_ms"]], hide_index=True, use_container_width=True)
    st.markdown("**Tool usage log**")
    if len(logs):
        c1, c2 = st.columns([2, 3])
        agg = logs.groupby("tool").agg(calls=("id", "count"), failures=("success", lambda x: int((x == 0).sum()))).reset_index()
        c1.dataframe(agg, hide_index=True, use_container_width=True)
        c2.dataframe(logs[["ts", "request_id", "step", "tool", "success", "latency_ms", "args"]], hide_index=True, use_container_width=True, height=260)
