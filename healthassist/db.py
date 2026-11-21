"""SQLite EHR + scheduling database (patients, records, appointments, caregivers, tool logs)."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "hospital.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (patient_id TEXT PRIMARY KEY, name TEXT, age INTEGER, gender TEXT, blood_group TEXT, phone TEXT);
CREATE TABLE IF NOT EXISTS caregivers (user_name TEXT, patient_id TEXT, relation TEXT);
CREATE TABLE IF NOT EXISTS medical_records (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, record_date TEXT,
    record_type TEXT, description TEXT, added_by TEXT);
CREATE TABLE IF NOT EXISTS appointments (appointment_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, doctor_id TEXT,
    slot_start TEXT, status TEXT, reason TEXT, created_at TEXT, UNIQUE(doctor_id, slot_start));
CREATE TABLE IF NOT EXISTS tool_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, request_id TEXT, step INTEGER, tool TEXT,
    args TEXT, success INTEGER, latency_ms REAL, output TEXT);
CREATE TABLE IF NOT EXISTS plans (request_id TEXT PRIMARY KEY, ts TEXT, user_name TEXT, query TEXT, plan TEXT, response TEXT);
"""

PATIENTS = [
    ("P001", "Ramesh Kumar", 70, "Male", "B+", "+91-98450-11223"),
    ("P002", "Arun Kumar", 41, "Male", "O+", "+91-98450-99887"),
    ("P003", "Lakshmi Kumar", 66, "Female", "A+", "+91-98450-44556"),
    ("P004", "Sara Thomas", 34, "Female", "AB+", "+91-99000-12345"),
    ("P005", "Imran Qureshi", 58, "Male", "O-", "+91-97400-67890"),
]
CAREGIVERS = [("Arun Kumar", "P001", "father"), ("Arun Kumar", "P003", "mother"), ("Arun Kumar", "P002", "self"),
              ("Sara Thomas", "P004", "self"), ("Imran Qureshi", "P005", "self")]
RECORDS = [
    ("P001", -1460, "diagnosis", "Type 2 diabetes mellitus (diagnosed 2022)"),
    ("P001", -1100, "diagnosis", "Essential hypertension"),
    ("P001", -400, "diagnosis", "Chronic kidney disease stage 3b (eGFR 38 mL/min/1.73m2), likely diabetic nephropathy"),
    ("P001", -400, "medication", "Telmisartan 40 mg once daily"),
    ("P001", -400, "medication", "Metformin dose reduced to 500 mg twice daily due to eGFR"),
    ("P001", -200, "medication", "Dapagliflozin 10 mg once daily started for CKD"),
    ("P001", -200, "medication", "Atorvastatin 20 mg at night"),
    ("P001", -60, "lab", "eGFR 34 mL/min/1.73m2; urine albumin-creatinine ratio 320 mg/g; HbA1c 7.4%; potassium 5.1 mmol/L"),
    ("P001", -60, "lab", "Haemoglobin 10.8 g/dL (mild anaemia of CKD)"),
    ("P001", -60, "alert", "Avoid NSAIDs (ibuprofen, diclofenac) - nephrotoxic; adjust drug doses to eGFR"),
    ("P001", -60, "alert", "Allergy: sulfa drugs (rash)"),
    ("P001", -30, "note", "Reports ankle swelling and fatigue; BP 148/88. Advised low-salt diet and nephrology review."),
    ("P002", -300, "diagnosis", "Seasonal allergic rhinitis"),
    ("P002", -90, "lab", "Fasting lipid profile: LDL 142 mg/dL"),
    ("P003", -900, "diagnosis", "Osteoarthritis of both knees"),
    ("P003", -500, "diagnosis", "Hypothyroidism"),
    ("P003", -500, "medication", "Levothyroxine 50 mcg daily"),
    ("P003", -45, "note", "Knee pain worse on stairs; physiotherapy recommended"),
    ("P004", -120, "diagnosis", "Mild persistent asthma"),
    ("P004", -120, "medication", "Budesonide-formoterol inhaler as maintenance and reliever"),
    ("P005", -700, "diagnosis", "Coronary artery disease - PCI with stent to LAD (2024)"),
    ("P005", -700, "medication", "Aspirin 75 mg, Atorvastatin 40 mg, Metoprolol 25 mg"),
    ("P005", -20, "lab", "LDL 68 mg/dL; ECG normal sinus rhythm"),
]


class HospitalDB:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)
            if c.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0:
                c.executemany("INSERT INTO patients VALUES (?,?,?,?,?,?)", PATIENTS)
                c.executemany("INSERT INTO caregivers VALUES (?,?,?)", CAREGIVERS)
                today = date.today()
                c.executemany("INSERT INTO medical_records (patient_id, record_date, record_type, description, added_by) VALUES (?,?,?,?,?)",
                              [(p, (today + timedelta(days=d)).isoformat(), t, desc, "seed") for p, d, t, desc in RECORDS])

    def conn(self):
        return sqlite3.connect(self.path)

    def q(self, sql: str, params=()) -> pd.DataFrame:
        with self.conn() as c:
            return pd.read_sql(sql, c, params=params)

    def execute(self, sql: str, params=()) -> int:
        with self.conn() as c:
            return c.execute(sql, params).lastrowid

    def reset(self):
        self.path.unlink(missing_ok=True)
        self.__init__(self.path)

    def log_tool(self, request_id: str, step: int, tool: str, args: dict, success: bool, latency_ms: float, output: str):
        self.execute("INSERT INTO tool_logs (ts, request_id, step, tool, args, success, latency_ms, output) VALUES (?,?,?,?,?,?,?,?)",
                     (datetime.now().isoformat(timespec="seconds"), request_id, step, tool, json.dumps(args, default=str),
                      int(success), latency_ms, output[:2000]))

    def save_plan(self, request_id: str, user: str, query: str, plan: list, response: str):
        self.execute("INSERT OR REPLACE INTO plans VALUES (?,?,?,?,?,?)",
                     (request_id, datetime.now().isoformat(timespec="seconds"), user, query, json.dumps(plan), response))
