"""Planner agent: an LLM decomposes a multi-step patient request into ordered sub-goals mapped to tools."""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

TOOLS = {
    "identify_patient": "Resolve who the request is about (self, a relative such as 'my father', or a named patient). args: {}",
    "get_medical_history": "Fetch the patient's EHR records and long-term memory. args: {}",
    "summarize_history": "Summarise past diagnoses, treatments, results and alerts. args: {}",
    "check_doctor_availability": "Find the earliest free slot with a specialist. args: {\"specialty\": one of SPECIALTIES or "
                                 "null if the user did not say (it is then inferred from the history), \"preferred_date\": "
                                 "\"YYYY-MM-DD\" or null, \"preferred_time\": \"morning\"|\"afternoon\"|null}",
    "book_appointment": "Book the slot found by check_doctor_availability. args: {\"reason\": short reason}",
    "medical_search": "Search MedlinePlus and WHO and summarise with RAG. args: {\"topic\": disease name, or null to use the "
                      "patient's main condition, \"question\": what to answer}",
    "add_medical_record": "Add structured/unstructured history to the EHR. args: {\"record_type\": "
                          "\"diagnosis\"|\"medication\"|\"lab\"|\"alert\"|\"note\", \"description\": text to store}",
    "list_appointments": "List the patient's upcoming appointments. args: {}",
}
SPECIALTIES = ["Nephrology", "Cardiology", "General Medicine", "Endocrinology", "Neurology", "Orthopedics",
               "Pulmonology", "Geriatrics"]

PLAN_PROMPT = PromptTemplate.from_template("""You are the planning module of an agentic healthcare assistant used by patients,
caregivers and hospital attendants. Break the user's request into the minimal ordered list of sub-goals.
Each sub-goal uses exactly one tool:
{tools}
SPECIALTIES: {specialties}

Rules:
- Always start with identify_patient.
- Include get_medical_history before check_doctor_availability, summarize_history or medical_search so patient context is used.
- Do NOT include get_medical_history for add_medical_record or list_appointments - they do not need it.
- Include summarize_history only if the user asks for a summary/history, OR the request describes the patient's existing
  chronic condition (e.g. "my father has chronic kidney disease") while booking. A plain booking request does not need it.
- check_doctor_availability must come directly before book_appointment.
- Only include tools the request actually needs. Do not invent extra steps.
- Today's date is {today} ({weekday}). Resolve relative dates ("tomorrow", "Monday", "next Monday") to YYYY-MM-DD using
  this calendar of the coming days, and pick the FIRST matching weekday after today: {calendar}

SESSION CONTEXT (earlier turns of this conversation):
{context}

USER REQUEST: {query}

Return ONLY a JSON array, no prose:
[{{"step": 1, "goal": "<short goal>", "tool": "<tool name>", "args": {{...}}}}, ...]""")


class PlanError(RuntimeError):
    pass


class Planner:
    def __init__(self, llm):
        self.chain = PLAN_PROMPT | llm | StrOutputParser()

    def plan(self, query: str, context: str, attempts: int = 2) -> list[dict]:
        today = date.today()
        raw = ""
        for _ in range(attempts):  # retry once if the model returns an invalid plan
            raw = self.chain.invoke({"query": query, "context": context, "today": today.isoformat(),
                                     "weekday": today.strftime("%A"), "specialties": ", ".join(SPECIALTIES),
                                     "calendar": ", ".join(f"{d:%a} {d:%Y-%m-%d}" for d in (today + timedelta(days=i) for i in range(1, 15))),
                                     "tools": "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())})
            try:
                plan = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
                if plan and all(isinstance(s, dict) and s.get("tool") in TOOLS for s in plan):
                    for i, s in enumerate(plan, 1):
                        s["step"], s["args"] = i, s.get("args") or {}
                    return plan
            except (AttributeError, json.JSONDecodeError):
                pass
        raise PlanError(f"Planner did not return a valid plan: {raw[:300]}")
