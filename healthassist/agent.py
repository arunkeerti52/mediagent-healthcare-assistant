"""Executor agent: runs the planner's sub-goals with tools, memory and LLM prompt chains."""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import date, datetime

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .db import HospitalDB
from .llm import get_embeddings, get_llm
from .memory import LongTermPatientMemory, SessionMemory
from .planner import SPECIALTIES, Planner
from .tools import DoctorScheduleAPI, EHRAPI, MedicalSearchAPI

DISCLAIMER = "_This assistant supports care coordination and does not replace professional medical advice._"

PATIENT_PROMPT = PromptTemplate.from_template("""Identify which patient this request is about.
The signed-in user is {user}. Patients the user may act for: {linked}.
Session context: {context}
Request: {query}
Return JSON only: {{"relation": "self" | "father" | "mother" | "wife" | "husband" | "son" | "daughter" | "previous" | null,
"name": "<patient name if explicitly named, else null>"}}
Rules: first-person words ("I", "me", "my appointments", "for me") mean "self". Use "previous" ONLY for third-person
pronouns (he/she/him/her/his) that refer to the patient discussed earlier in the session.""")

ROUTING_PROMPT = PromptTemplate.from_template("""You route patients to the right clinic.
Allowed specialties: {specialties}
Patient history:
{history}
Request: {query}
Return JSON only: {{"specialty": "<one allowed specialty most relevant to the request, or to the most recent/serious condition if the
request does not say>", "condition": "<the main disease relevant to the request, as a short search term>"}}""")

HISTORY_PROMPT = PromptTemplate.from_template("""You are a clinical documentation assistant. Summarise the patient's history for a
treating doctor. Use ONLY the records below. Sections: **Active conditions**, **Current medications**, **Recent results**,
**Alerts & allergies**, **Recent notes**. Be concise (bullets), keep exact values and dates, and do not invent information.

PATIENT RECORDS:
{records}

LONG-TERM MEMORY RECALL:
{memory}

SUMMARY:""")

MED_SUMMARY_PROMPT = PromptTemplate.from_template("""You are a medical information assistant. Answer the QUESTION using ONLY the
retrieved passages from trusted sources (MedlinePlus / WHO). Cite passages as [n] and summarise the latest treatment or
management approaches they describe. Then add a short "**For this patient:**" section that relates the information to the
PATIENT CONTEXT (age, conditions, current medicines, alerts) without prescribing. Keep it under 250 words.

PATIENT CONTEXT: {patient}
QUESTION: {question}
PASSAGES:
{passages}

ANSWER:""")


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


class HealthcareAgent:
    def __init__(self, db: HospitalDB | None = None):
        self.db = db or HospitalDB()
        self.llm = get_llm()
        self.schedule = DoctorScheduleAPI(self.db)
        self.ehr = EHRAPI(self.db)
        self.search_api = MedicalSearchAPI()
        self.memory = LongTermPatientMemory(self.db)
        self.planner = Planner(self.llm)
        parse = StrOutputParser()
        self.patient_chain = PATIENT_PROMPT | self.llm | parse
        self.routing_chain = ROUTING_PROMPT | self.llm | parse
        self.history_chain = HISTORY_PROMPT | self.llm | parse
        self.med_chain = MED_SUMMARY_PROMPT | self.llm | parse
        self.last_search: dict | None = None

    # ------------------------------------------------------------------ tools --
    def _identify_patient(self, s: SessionMemory, args, ctx):
        linked = self.ehr.linked_patients(s.user_name)
        out = _json(self.patient_chain.invoke({
            "user": s.user_name, "query": ctx["query"], "context": s.context_text(),
            "linked": "; ".join(f"{r.name} ({r.relation}, {r.age}y)" for r in linked.itertuples())}))
        relation, name = (out.get("relation") or "").lower(), out.get("name")
        p = None
        if relation == "previous" and s.facts.get("current_patient_id"):
            p = {**self.ehr.patient(s.facts["current_patient_id"]), "relation": "previous"}
        elif relation and relation not in ("self", "previous"):
            p = self.ehr.by_relation(s.user_name, relation)
        if p is None and name:
            p = self.ehr.find_by_name(name)
        if p is None:
            p = self.ehr.by_relation(s.user_name, "self")
        if p is None:
            raise LookupError("Could not identify the patient - please mention the name or relation.")
        ctx["patient"] = p
        s.remember("current_patient_id", p["patient_id"])
        rel = p.get("relation", "")
        suffix = {"self": "", "named": "", "": "", "previous": " (same patient as earlier in this conversation)"}.get(rel, f" - your {rel}")
        return f"{p['name']} ({p['patient_id']}), {p['age']}y {p['gender']}{suffix}"

    def _get_medical_history(self, s, args, ctx):
        pid = ctx["patient"]["patient_id"]
        ctx["history"] = self.ehr.history_text(pid)
        recalled = self.memory.recall(pid, ctx["query"])
        ctx["memory_recall"] = [d.page_content for d in recalled]
        s.trace.append({"op": "RECALL", "key": f"FAISS patient memory ({pid})", "value": " | ".join(ctx["memory_recall"][:3])})
        return f"{len(self.ehr.records(pid))} EHR records loaded; {len(recalled)} memory items recalled."

    def _summarize_history(self, s, args, ctx):
        if "history" not in ctx:
            self._get_medical_history(s, {}, ctx)
        summary = self.history_chain.invoke({"records": ctx["history"], "memory": "\n".join(ctx.get("memory_recall", []))})
        ctx["history_summary"] = summary
        return summary

    def _route(self, ctx) -> dict:
        """LLM picks the specialty / condition from the patient's history when the request does not state it."""
        if "routing" not in ctx:
            history = ctx.get("history") or (self.ehr.history_text(ctx["patient"]["patient_id"]) if "patient" in ctx else "unknown")
            ctx["routing"] = _json(self.routing_chain.invoke({"specialties": ", ".join(SPECIALTIES), "history": history,
                                                              "query": ctx["query"]}))
        return ctx["routing"]

    def _check_doctor_availability(self, s, args, ctx):
        spec = args.get("specialty")
        if spec not in SPECIALTIES:
            spec = self._route(ctx).get("specialty")
        if spec not in SPECIALTIES:
            raise LookupError(f"Could not determine a specialty (got {spec!r}).")
        start = date.fromisoformat(args["preferred_date"]) if args.get("preferred_date") else None
        found = self.schedule.earliest_slot(spec, start, args.get("preferred_time"))
        if not found:
            raise LookupError(f"No {spec} slots available in the next 2 weeks.")
        ctx["slot"] = found
        s.remember("last_specialty", spec)
        d = found["doctor"]
        return f"Earliest {spec} slot: {d['name']} on {_fmt(found['slot'])} at {d['location']}."

    def _book_appointment(self, s, args, ctx):
        if "slot" not in ctx:
            self._check_doctor_availability(s, {}, ctx)
        f = ctx["slot"]
        appt = self.schedule.book(ctx["patient"]["patient_id"], f["doctor"]["doctor_id"], f["slot"],
                                  (args.get("reason") or ctx["query"])[:200])
        ctx["appointment"] = {"id": appt, **f}
        s.remember("last_appointment", f"#{appt} {f['doctor']['name']} {_fmt(f['slot'])}")
        self.memory.add(ctx["patient"]["patient_id"],
                        f"Appointment #{appt} booked with {f['doctor']['name']} ({f['doctor']['specialty']}) on {f['slot']}", "appointment")
        return (f"✅ Appointment #{appt} confirmed with **{f['doctor']['name']}** ({f['doctor']['specialty']}) on "
                f"**{_fmt(f['slot'])}**, {f['doctor']['location']}.")

    def _medical_search(self, s, args, ctx):
        topic = args.get("topic") or self._route(ctx).get("condition") or ctx["query"]
        question = args.get("question") or ctx["query"]
        results, mode = self.search_api.search(topic)
        if not results:
            raise LookupError(f"No MedlinePlus/WHO results for '{topic}' ({mode}).")
        # RAG: chunk the retrieved pages -> FAISS -> keep the passages most relevant to the question
        chunks = []
        for r in results:
            for piece in re.split(r"(?<=\.)\s+(?=[A-Z])", r["text"]):
                if len(piece) > 40:
                    chunks.append(Document(page_content=piece, metadata={"title": r["title"], "url": r["url"], "source": r["source"]}))
        index = FAISS.from_documents(chunks, get_embeddings())
        hits = index.similarity_search(f"{topic}: {question}", k=12)
        src = list(dict.fromkeys((h.metadata["title"], h.metadata["source"], h.metadata["url"]) for h in hits))
        num = {u: i + 1 for i, (_, _, u) in enumerate(src)}
        passages = "\n".join(f"[{num[h.metadata['url']]}] {h.page_content}" for h in hits)
        patient_ctx = "not specified"
        if "patient" in ctx:
            if "history" not in ctx:
                self._get_medical_history(s, {}, ctx)
            patient_ctx = ctx["history"].replace("\n", " ")
        answer = self.med_chain.invoke({"patient": patient_ctx, "question": question, "passages": passages})
        refs = "\n".join(f"[{i + 1}] [{t} - {src_}]({u})" for i, (t, src_, u) in enumerate(src))
        ctx["search"] = {"topic": topic, "mode": mode, "answer": answer, "sources": src, "n_chunks": len(chunks)}
        self.last_search = {**ctx["search"], "question": question, "ts": datetime.now().isoformat(timespec="seconds")}
        s.remember("last_topic", topic)
        return f"{answer}\n\n**Sources** ({mode}):\n{refs}"

    def _add_medical_record(self, s, args, ctx):
        pid = ctx["patient"]["patient_id"]
        kind = args.get("record_type") if args.get("record_type") in ("diagnosis", "medication", "lab", "alert", "note") else "note"
        desc = args.get("description") or ctx["query"]
        rid = self.ehr.add_record(pid, kind, desc, added_by=s.user_name)
        self.memory.add(pid, f"{date.today()} {kind}: {desc}", kind)
        return f"📝 Added {kind} record #{rid} to {ctx['patient']['name']}'s chart: “{desc}”"

    def _list_appointments(self, s, args, ctx):
        df = self.schedule.appointments(patient_id=ctx["patient"]["patient_id"])
        df = df[df.status == "Booked"]
        if df.empty:
            return "No upcoming appointments."
        return "\n".join(f"- #{r.appointment_id} {r.doctor} ({r.specialty}) on {_fmt(r.slot_start)}" for r in df.itertuples())

    # ------------------------------------------------------------------- main --
    def handle(self, session: SessionMemory, query: str) -> dict:
        rid = uuid.uuid4().hex[:8]
        t0 = time.time()
        plan = self.planner.plan(query, session.context_text())
        ctx = {"query": query}
        steps = []
        for step in plan:
            fn = getattr(self, f"_{step['tool']}")
            ts = time.time()
            try:
                out, ok = fn(session, step["args"], ctx), True
            except Exception as e:
                out, ok = f"⚠️ {type(e).__name__}: {e}", False
            ms = round((time.time() - ts) * 1000, 1)
            self.db.log_tool(rid, step["step"], step["tool"], step["args"], ok, ms, out)
            steps.append({**step, "output": out, "success": ok, "latency_ms": ms})
            if not ok and step["tool"] == "identify_patient":
                break
        response = self._compose(steps)
        session.turns.append({"query": query, "response": response, "request_id": rid})
        self.db.save_plan(rid, session.user_name, query, steps, response)
        return {"request_id": rid, "plan": steps, "response": response, "latency_s": round(time.time() - t0, 2),
                "patient": ctx.get("patient")}

    @staticmethod
    def _compose(steps: list[dict]) -> str:
        titles = {"identify_patient": "👤 Patient", "summarize_history": "📋 Medical history summary", "book_appointment": "📅 Appointment",
                  "medical_search": "🔬 Latest medical information", "add_medical_record": "📝 Record update",
                  "list_appointments": "📅 Upcoming appointments"}
        parts = [f"#### {titles.get(s['tool'], s['goal'])}\n{s['output']}" for s in steps if s["tool"] in titles or not s["success"]]
        return "\n\n".join(parts) + f"\n\n{DISCLAIMER}"


def _fmt(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%a %d %b %Y, %I:%M %p")
