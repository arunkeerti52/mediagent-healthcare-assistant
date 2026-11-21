"""Agent tools: Doctor Schedule API, EHR (patient DB) API and Medical Search API (MedlinePlus + WHO corpus)."""
from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .db import ROOT, HospitalDB
from .llm import get_embeddings

DOCTORS = json.loads((ROOT / "data" / "doctors.json").read_text())
DAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --------------------------------------------------------------------------- #
# Doctor Schedule API
# --------------------------------------------------------------------------- #
class DoctorScheduleAPI:
    def __init__(self, db: HospitalDB):
        self.db = db

    def find_doctors(self, specialty: str) -> list[dict]:
        return [d for d in DOCTORS if d["specialty"].lower() == specialty.lower()]

    def doctor(self, doctor_id: str) -> dict:
        return next(d for d in DOCTORS if d["doctor_id"] == doctor_id)

    def available_slots(self, doctor_id: str, start: date | None = None, days: int = 14, limit: int = 20) -> list[str]:
        d = self.doctor(doctor_id)
        booked = set(self.db.q("SELECT slot_start FROM appointments WHERE doctor_id=? AND status!='Cancelled'", (doctor_id,))["slot_start"])
        start = start or date.today() + timedelta(days=1)
        slots = []
        for i in range(days):
            day = start + timedelta(days=i)
            if DAY[day.weekday()] not in d["days"]:
                continue
            t = datetime.combine(day, datetime.strptime(d["start"], "%H:%M").time())
            end = datetime.combine(day, datetime.strptime(d["end"], "%H:%M").time())
            while t < end:
                iso = t.isoformat(timespec="minutes")
                if iso not in booked and t > datetime.now():
                    slots.append(iso)
                    if len(slots) >= limit:
                        return slots
                t += timedelta(minutes=d["slot_minutes"])
        return slots

    def earliest_slot(self, specialty: str, start: date | None = None, preferred_time: str | None = None) -> dict | None:
        best = None
        for d in self.find_doctors(specialty):
            for s in self.available_slots(d["doctor_id"], start, limit=40):
                hour = int(s[11:13])
                if preferred_time == "morning" and hour >= 12 or preferred_time == "afternoon" and hour < 12:
                    continue
                if best is None or s < best["slot"]:
                    best = {"doctor": d, "slot": s}
                break  # earliest matching slot for this doctor found
        return best

    def book(self, patient_id: str, doctor_id: str, slot: str, reason: str) -> int:
        if slot not in self.available_slots(self.doctor(doctor_id)["doctor_id"], date.fromisoformat(slot[:10]), days=1, limit=100):
            raise ValueError(f"Slot {slot} is no longer available for {doctor_id}")
        return self.db.execute("INSERT INTO appointments (patient_id, doctor_id, slot_start, status, reason, created_at) VALUES (?,?,?,?,?,?)",
                               (patient_id, doctor_id, slot, "Booked", reason, datetime.now().isoformat(timespec="seconds")))

    def update_status(self, appointment_id: int, status: str):
        self.db.execute("UPDATE appointments SET status=? WHERE appointment_id=?", (status, appointment_id))

    def appointments(self, patient_id: str | None = None, doctor_id: str | None = None):
        df = self.db.q("""SELECT a.appointment_id, a.slot_start, a.status, a.reason, a.patient_id, p.name AS patient, a.doctor_id
                          FROM appointments a JOIN patients p USING(patient_id) ORDER BY a.slot_start""")
        df["doctor"] = df["doctor_id"].map({d["doctor_id"]: d["name"] for d in DOCTORS})
        df["specialty"] = df["doctor_id"].map({d["doctor_id"]: d["specialty"] for d in DOCTORS})
        if patient_id:
            df = df[df.patient_id == patient_id]
        if doctor_id:
            df = df[df.doctor_id == doctor_id]
        return df


# --------------------------------------------------------------------------- #
# EHR / Patient DB API
# --------------------------------------------------------------------------- #
class EHRAPI:
    def __init__(self, db: HospitalDB):
        self.db = db

    def linked_patients(self, user_name: str):
        return self.db.q("""SELECT p.*, c.relation FROM caregivers c JOIN patients p USING(patient_id)
                            WHERE lower(c.user_name)=lower(?)""", (user_name,))

    def by_relation(self, user_name: str, relation: str) -> dict | None:
        df = self.linked_patients(user_name)
        df = df[df.relation == relation]
        return df.iloc[0].to_dict() if len(df) else None

    def find_by_name(self, name: str) -> dict | None:
        df = self.db.q("SELECT * FROM patients WHERE lower(name) LIKE lower(?)", (f"%{name.strip()}%",))
        return {**df.iloc[0].to_dict(), "relation": "named"} if len(df) else None

    def patient(self, patient_id: str) -> dict:
        return self.db.q("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).iloc[0].to_dict()

    def records(self, patient_id: str):
        return self.db.q("SELECT record_date, record_type, description, added_by FROM medical_records WHERE patient_id=? ORDER BY record_date",
                         (patient_id,))

    def add_record(self, patient_id: str, record_type: str, description: str, added_by: str = "attendant") -> int:
        return self.db.execute("INSERT INTO medical_records (patient_id, record_date, record_type, description, added_by) VALUES (?,?,?,?,?)",
                               (patient_id, date.today().isoformat(), record_type, description, added_by))

    def add_patient(self, name: str, age: int, gender: str, blood_group: str = "", phone: str = "", caregiver: str | None = None,
                    relation: str = "self") -> str:
        n = int(self.db.q("SELECT COUNT(*) n FROM patients").n[0]) + 1
        pid = f"P{n:03d}"
        self.db.execute("INSERT INTO patients VALUES (?,?,?,?,?,?)", (pid, name, age, gender, blood_group, phone))
        if caregiver:
            self.db.execute("INSERT INTO caregivers VALUES (?,?,?)", (caregiver, pid, relation))
        return pid

    def history_text(self, patient_id: str) -> str:
        p = self.patient(patient_id)
        recs = self.records(patient_id)
        lines = [f"Patient {p['name']} ({p['patient_id']}), {p['age']}-year-old {p['gender'].lower()}, blood group {p['blood_group']}."]
        lines += [f"[{r.record_date}] {r.record_type.upper()}: {r.description}" for r in recs.itertuples()]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Medical Search API (live MedlinePlus web service + live WHO fact sheets)
# --------------------------------------------------------------------------- #
def _strip(s: str) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s))).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


class MedicalSearchAPI:
    MEDLINE_URL = "https://wsearch.nlm.nih.gov/ws/query"
    WHO_INDEX = "https://www.who.int/news-room/fact-sheets"
    HEADERS = {"User-Agent": "MediAgent/1.0 (capstone project)"}
    _who_index: FAISS | None = None

    def medlineplus(self, term: str, n: int = 3) -> list[dict]:
        r = requests.get(self.MEDLINE_URL, params={"db": "healthTopics", "term": term, "retmax": n}, timeout=10)
        r.raise_for_status()
        out = []
        for doc in ET.fromstring(r.content).iter("document"):
            fields = {c.get("name"): _strip(c.text or "") for c in doc.findall("content")}
            if fields.get("FullSummary"):
                out.append({"title": fields.get("title", ""), "url": doc.get("url"), "source": "MedlinePlus", "text": fields["FullSummary"]})
        return out

    def _who_titles(self) -> FAISS:
        """Index the titles of all WHO fact sheets once, so a topic can be matched semantically."""
        if MedicalSearchAPI._who_index is None:
            page = requests.get(self.WHO_INDEX, headers=self.HEADERS, timeout=15).text
            links = dict(re.findall(r'<a[^>]+href="(/news-room/fact-sheets/detail/[^"]+)"[^>]*>(.*?)</a>', page, re.S))
            docs = [Document(page_content=_strip(t), metadata={"url": "https://www.who.int" + u}) for u, t in links.items() if _strip(t)]
            MedicalSearchAPI._who_index = FAISS.from_documents(docs, get_embeddings())
        return MedicalSearchAPI._who_index

    def who(self, term: str, n: int = 1) -> list[dict]:
        out = []
        for hit in self._who_titles().similarity_search(term, k=n):
            page = requests.get(hit.metadata["url"], headers=self.HEADERS, timeout=15).text
            for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', page, re.S):
                try:  # WHO pages sometimes append extra data after the JSON-LD object
                    data, _ = json.JSONDecoder(strict=False).raw_decode(block.strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("description"):
                    out.append({"title": data.get("headline", hit.page_content), "url": hit.metadata["url"],
                                "source": "WHO fact sheet", "text": _strip(data["description"])})
                    break
        return out

    def search(self, term: str) -> tuple[list[dict], str]:
        results, used = [], []
        for name, fn in (("MedlinePlus", self.medlineplus), ("WHO", self.who)):
            try:
                found = fn(term)
                results += found
                used.append(f"{name} ({len(found)})")
            except Exception as e:  # one source down should not break the other
                used.append(f"{name} failed: {type(e).__name__}")
        return results, " + ".join(used)
