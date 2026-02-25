# Ragging Face

This project is a **Streamlit-based smart industrial quality assistant** demonstrating:

- Industrial computer vision (defect/anomaly detection)
- Production traceability analytics
- Retrieval-Augmented Generation with local embeddings
- Local LLM integration for report generation
- Clean modular Python architecture
- Industry 4.0 style dashboard

## Setup

1. Create a virtual environment (venv/conda) and activate it.
2. Install dependencies (CPU-only PyTorch wheel ensures a lean container):

```bash
pip install -r requirements.txt

# requirements.txt already contains a --find-links pointing to the
# PyTorch CPU wheels and specifies torch==2.1.2+cpu, torchvision==0.16.2+cpu.

# the project no longer depends on OpenCV, but headless package
# is kept for compatibility with earlier versions.
```

3. Run the app:

```bash
streamlit run ragging_face/app.py
```

No paid APIs or authentication are required.

**System dependencies:**
The vision module now uses `torchvision` and does not require OpenCV; there is no `libGL` dependency.  If a full `opencv-python` package is installed by accident, the app silently ignores it.

## Structure

- `ragging_face/app.py` – main Streamlit application
- `ragging_face/modules/` – feature modules (vision, production, RAG, LLM)
- `ragging_face/utils/` – helper utilities
- `ragging_face/datasets/` – sample data generators and storage
  - `testing.py` generates fake production logs and saves results under `datasets/results/{production,rag,vision}`
- `ragging_face/cleanup.py` – maintenance script to remove uploaded files older than a given number of days. Intended to be run daily (e.g. via cron) to avoid data buildup.

- `requirements.txt` – Python dependencies

## Modules

See the sidebar in the running application for:

- **Vision Module**: upload images and detect objects/defects using a local torchvision object detector (pre‑trained Faster R‑CNN)
- **Production Module**: analyze CSV production logs, compute KPIs and anomalies
- **RAG & LLM Module**: ingest documents, run retrieval and generate AI-driven answers/reports using a local model
- **Dashboard**: overview of KPIs and trends

The project is designed for easy deployment and showcases professional, maintainable code suitable for an Industry 4.0 portfolio.
