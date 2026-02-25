import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from modules import vision, production, llm_module
from modules.rag import RAGStore, ingest_files

# configure page
st.set_page_config(page_title="Ragging Face", layout="wide", initial_sidebar_state="auto")

# ensure dataset result folders exist
BASE_DATASET = os.path.join(os.getcwd(), "ragging_face", "datasets", "results")
os.makedirs(os.path.join(BASE_DATASET, "vision"), exist_ok=True)
os.makedirs(os.path.join(BASE_DATASET, "production"), exist_ok=True)
os.makedirs(os.path.join(BASE_DATASET, "rag"), exist_ok=True)

# dark theme switch via config - user should set in Streamlit config
# simple header
st.title("Ragging Face – Smart Quality Assistant")

if 'rag_store' not in st.session_state:
    st.session_state['rag_store'] = RAGStore(persist_path="./rag_index")

if 'uploaded_docs' not in st.session_state:
    st.session_state['uploaded_docs'] = []


def vision_tab():
    st.header("Vision Module")
    if vision.cv2 is None:
        st.error("Vision functionality is unavailable (OpenCV import failed). Ensure libGL is installed on the host.")
        return
    uploaded = st.file_uploader("Upload image for defect detection", type=['png', 'jpg', 'jpeg'])
    if uploaded:
        img = Image.open(uploaded)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        outdir = os.path.join(BASE_DATASET, "vision")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, uploaded.name)
        with open(path, 'wb') as f:
            f.write(buf.getvalue())
        st.success(f"Image saved to {path}")
        try:
            results = vision.detect_defects(path)
            st.image(results['image'], caption="uploaded image", use_column_width=True)
            if results['boxes']:
                st.write("Detected objects / defects:")
                for b in results['boxes']:
                    st.write(b)
            else:
                st.write("No objects detected.")
        except Exception as e:
            st.error(f"Detection failed: {e}")
        # no cleanup; keep file for records


def production_tab():
    st.header("Production / Traceability Module")
    uploaded = st.file_uploader("Upload production CSV log", type=['csv'])
    if uploaded:
        outdir = os.path.join(BASE_DATASET, "production")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, uploaded.name)
        with open(path, 'wb') as f:
            f.write(uploaded.getvalue())
        st.success(f"CSV saved to {path}")
        analysis = production.analyze_csv(path)
        st.session_state['last_production'] = analysis
        kpis = analysis['kpis']
        st.subheader("Key Performance Indicators")
        st.write(kpis)
        df = analysis['df']
        st.subheader("Sample data")
        st.dataframe(df.head())
        if kpis.get('defect_rate') is not None:
            fig = px.histogram(df, x='defect', title='Defect distribution')
            st.plotly_chart(fig, use_container_width=True)
        # show correlations
        if analysis['correlation']:
            corr_df = pd.DataFrame.from_dict(analysis['correlation'], orient='index', columns=['corr_with_defect'])
            st.subheader('Correlation with defect')
            st.table(corr_df)
            # anomaly detection on numeric columns
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            col = st.selectbox('Select column for anomaly detection', num_cols)
            if col:
                idxs = production.detect_anomalies(df, col)
                st.write(f"Anomalous rows for {col}: {idxs}")


def rag_tab():
    st.header("RAG Module")
    st.write("Upload maintenance reports or docs (pdf/text)")
    uploaded = st.file_uploader("Add documents", accept_multiple_files=True, type=['pdf', 'txt', 'md'])
    if uploaded:
        paths = []
        outdir = os.path.join(BASE_DATASET, "rag")
        os.makedirs(outdir, exist_ok=True)
        for f in uploaded:
            p = os.path.join(outdir, f.name)
            with open(p, 'wb') as out:
                out.write(f.read())
            paths.append(p)
            st.session_state.uploaded_docs.append(p)
        ingest_files(paths, st.session_state.rag_store)
        st.success(f"Ingested {len(paths)} documents")
    query = st.text_input("Ask a question about the documents")
    if query and st.session_state.rag_store.index:
        results = st.session_state.rag_store.query(query)
        context = "\n\n".join([r.page_content for r in results])
        if len(context) > 800:
            st.warning("Retrieved context is long and will be truncated for the LLM.")
        st.subheader("Retrieved context (preview)")
        for r in results:
            src = r.metadata.get('source', '')
            if src:
                st.write(f"**Source:** {src}")
            st.write(r.page_content[:500])
            st.write("----")
        try:
            answer = llm_module.generate_answer(query, context)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            answer = None
        try:
            report = llm_module.generate_report(query, context)
            st.subheader("AI-generated Report")
            st.write(report)
        except Exception as e:
            st.error(f"Error generating report: {e}")
            report = None
        if report and st.button("Download report"):
            b = io.BytesIO()
            b.write(report.encode('utf-8'))
            st.download_button("Download", data=b, file_name="quality_report.txt")


def dashboard_tab():
    st.header("Smart Dashboard")
    prod_dir = os.path.join(BASE_DATASET,"production")
    prod_files = os.listdir(prod_dir) if os.path.isdir(prod_dir) else []
    if prod_files:
        st.write(f"Stored production files: {len(prod_files)}")
    vis_dir = os.path.join(BASE_DATASET,"vision")
    vis_files = os.listdir(vis_dir) if os.path.isdir(vis_dir) else []
    if vis_files:
        st.write(f"Stored vision images: {len(vis_files)}")
    if 'last_production' in st.session_state:
        analysis = st.session_state['last_production']
        kpis = analysis['kpis']
        st.subheader("Latest Production KPIs")
        st.write(kpis)
        df = analysis['df']
        if kpis.get('defect_rate') is not None:
            fig = px.histogram(df, x='defect', title='Defect distribution')
            st.plotly_chart(fig, use_container_width=True)
    else:
        if not prod_files:
            st.write("No production data uploaded yet.")
    st.markdown("---")
    rag_dir = os.path.join(BASE_DATASET,"rag")
    rag_files = os.listdir(rag_dir) if os.path.isdir(rag_dir) else []
    if 'rag_store' in st.session_state and st.session_state.rag_store.index is not None:
        st.subheader("Documents indexed for RAG")
        st.write(f"{len(st.session_state.uploaded_docs)} files")
    else:
        st.write("No documents ingested.")
    if rag_files:
        st.write(f"Stored documents: {len(rag_files)}")
    st.markdown("---")
    st.write("Use the other tabs to upload data and interact with modules.")


# main navigation
tabs = ["Vision", "Production", "RAG+LLM", "Dashboard"]
selected = st.sidebar.selectbox("Module", tabs)
if selected == "Vision":
    vision_tab()
elif selected == "Production":
    production_tab()
elif selected == "RAG+LLM":
    rag_tab()
elif selected == "Dashboard":
    dashboard_tab()
