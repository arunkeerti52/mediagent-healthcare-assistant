"""Memory modules: long-term patient memory (FAISS vector store) + short-term session memory."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .db import HospitalDB
from .llm import get_embeddings


class LongTermPatientMemory:
    """Stores patient summaries and individual EHR entries as vectors; recalled per patient by similarity."""

    def __init__(self, db: HospitalDB):
        self.db = db
        self.store: FAISS | None = None
        self.rebuild()

    def _docs(self) -> list[Document]:
        docs = []
        pats = self.db.q("SELECT * FROM patients")
        recs = self.db.q("SELECT * FROM medical_records")
        for p in pats.itertuples():
            pr = recs[recs.patient_id == p.patient_id]
            dx = "; ".join(pr[pr.record_type == "diagnosis"].description) or "no recorded diagnoses"
            meds = "; ".join(pr[pr.record_type == "medication"].description) or "no medications"
            docs.append(Document(page_content=f"Patient summary - {p.name}, {p.age}y {p.gender}. Conditions: {dx}. Medications: {meds}.",
                                 metadata={"patient_id": p.patient_id, "kind": "summary"}))
            for r in pr.itertuples():
                docs.append(Document(page_content=f"{r.record_date} {r.record_type}: {r.description}",
                                     metadata={"patient_id": p.patient_id, "kind": r.record_type}))
        return docs

    def rebuild(self):
        self.store = FAISS.from_documents(self._docs(), get_embeddings())

    def add(self, patient_id: str, text: str, kind: str = "note"):
        self.store.add_documents([Document(page_content=text, metadata={"patient_id": patient_id, "kind": kind})])

    def recall(self, patient_id: str, query: str, k: int = 6) -> list[Document]:
        return self.store.similarity_search(query, k=k, filter={"patient_id": patient_id}, fetch_k=200)


@dataclass
class SessionMemory:
    """Short-term working memory for one conversation (context carried across turns)."""

    user_name: str
    turns: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)  # current_patient_id, last_specialty, last_topic, last_appointment
    trace: list = field(default_factory=list)  # memory reads/writes for the UI

    def remember(self, key: str, value):
        self.facts[key] = value
        self.trace.append({"op": "WRITE", "key": key, "value": str(value)})

    def get(self, key: str, default=None):
        val = self.facts.get(key, default)
        self.trace.append({"op": "READ", "key": key, "value": str(val)})
        return val

    def context_text(self) -> str:
        recent = "\n".join(f"User: {t['query']}\nAssistant: {t['response'][:300]}" for t in self.turns[-3:])
        facts = ", ".join(f"{k}={v}" for k, v in self.facts.items()) or "none"
        return f"Known facts: {facts}\nRecent conversation:\n{recent or '(none)'}"
