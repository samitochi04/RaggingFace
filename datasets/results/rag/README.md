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
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run ragging_face/app.py
```

No paid APIs or authentication are required.

## Structure

- `ragging_face/app.py` – main Streamlit application
- `ragging_face/modules/` – feature modules (vision, production, RAG, LLM)
- `ragging_face/utils/` – helper utilities
- `ragging_face/datasets/` – sample data generators and storage
  - `testing.py` generates fake production logs and saves results under `datasets/results/{production,rag,vision}`
- `requirements.txt` – Python dependencies

## Modules

See the sidebar in the running application for:

- **Vision Module**: upload images and detect objects/defects via YOLOv8
- **Production Module**: analyze CSV production logs, compute KPIs and anomalies
- **RAG & LLM Module**: ingest documents, run retrieval and generate AI-driven answers/reports using a local model
- **Dashboard**: overview of KPIs and trends

The project is designed for easy deployment and showcases professional, maintainable code suitable for an Industry 4.0 portfolio.
