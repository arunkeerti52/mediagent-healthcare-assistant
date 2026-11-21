# MediAgent: Agentic Healthcare Assistant

A planner–executor agent that books appointments, manages EHR records, summarises medical history and searches MedlinePlus and WHO live using RAG. LLM: Llama 3.3 70B on Cloudflare Workers AI. It uses FAISS long-term patient memory and includes a Streamlit dashboard and LLMOps evaluation.

## Quick start
```bash
pip install -r requirements.txt
cp .env.example .env        # set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN
streamlit run app.py
python -m healthassist.evaluation
```

## Project layout
| Path | Purpose |
|---|---|
| `healthassist/planner.py` | LLM goal decomposition into tool sub-goals |
| `healthassist/agent.py` | Executor, tools, prompt chains (patient identification, specialty routing, history summary, medical RAG) |
| `healthassist/tools.py` | Doctor Schedule API, EHR API, Medical Search API (live MedlinePlus + WHO fact sheets) |
| `healthassist/memory.py` | FAISS long-term patient memory and session memory |
| `healthassist/db.py` | SQLite schema, seed data, tool and plan logs |
| `healthassist/evaluation.py` | Planner, booking, QAEvalChain and tool success evaluation |
| `data/doctors.json` | Doctor calendar |
| `app.py` | Streamlit dashboard |

All patient data is synthetic. This is not a medical device and does not give medical advice.
