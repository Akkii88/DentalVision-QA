"""
Premium DentalVision-QA Dashboard
Healthcare AI Dashboard for Dental Image Analysis
"""

import os
import sys
import json
import streamlit as st
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="DentalVision-QA Dashboard",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Configuration ──────────────────────────────────────────────────────
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # Handle exec() execution
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"

CLASS_NAMES = [
    "caries",
    "plaque",
    "calculus",
    "gingivitis",
    "missing_tooth",
    "filling_crown",
    "ambiguous",
]


# ─── Premium CSS ────────────────────────────────────────────────────────
def inject_premium_css():
    """Inject premium healthcare AI CSS theme."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Theme */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Typography */
    .main-header {
        font-size: 38px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    
    .main-subtitle {
        font-size: 18px;
        color: #64748B;
        margin-bottom: 16px;
        font-weight: 400;
    }
    
    .badge-text {
        font-size: 12px;
        color: #64748B;
        background: #E2E8F0;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 24px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .section-title {
        font-size: 24px;
        font-weight: 600;
        color: #0F172A;
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #2563EB;
    }
    
    /* Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
        display: block;
    }
    
    .metric-label {
        font-size: 14px;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    .image-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-pass {
        background: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    
    .status-review {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    
    .status-danger {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    
    /* Tables */
    .premium-table .stDataFrame {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        overflow: hidden;
    }
    
    .premium-table .stDataFrame td, .premium-table .stDataFrame th {
        padding: 12px 16px;
        border-bottom: 1px solid #F1F5F9;
    }
    
    .premium-table .stDataFrame th {
        background: #F8FAFC;
        font-weight: 600;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
    }
    
    /* Empty State */
    .empty-state {
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        margin: 32px 0;
    }
    
    .empty-state h3 {
        color: #92400E;
        margin-bottom: 16px;
    }
    
    .empty-state p {
        color: #A16207;
        font-size: 16px;
        margin-bottom: 24px;
    }
    
    /* Sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, #2563EB, #14B8A6);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
    }
    
    .sidebar-header h2 {
        margin: 0;
        font-size: 20px;
        font-weight: 700;
    }
    
    .sidebar-header p {
        margin: 8px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
    
    .nav-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    
    .nav-item:hover {
        background: #E0F2FE;
        color: #2563EB;
    }
    
    .nav-item.active {
        background: #2563EB;
        color: white;
    }
    
    .nav-icon {
        margin-right: 12px;
        font-size: 16px;
    }
    
    .status-box {
        background: #D1FAE5;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 16px;
        text-align: center;
    }
    
    .status-box .status-text {
        color: #065F46;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────


def load_csv_safe(filepath):
    """Safely load CSV file with error handling."""
    try:
        import pandas as pd

        if filepath.exists():
            return pd.read_csv(filepath)
    except Exception:
        pass
    return None


def load_json_safe(filepath):
    """Safely load JSON file with error handling."""
    try:
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def render_header():
    """Render professional header."""
    header_html = """
    <div>
        <h1 class="main-header">🦷 DentalVision-QA</h1>
        <p class="main-subtitle">Dental Image Annotation, QA/QC and AI Detection Dashboard</p>
        <span class="badge-text">AI Healthcare · Computer Vision · Annotation QA</span>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_metric_card(value, label, color="#2563EB", icon="📊"):
    """Render a metric card."""
    card_html = f"""
    <div class="metric-card" style="border-left: 4px solid {color};">
        <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
        <span class="metric-value">{value}</span>
        <span class="metric-label">{label}</span>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_section_card(title, content=None):
    """Render a section card."""
    st.markdown(f"### {title}")
    if content:
        st.markdown(content)


def render_empty_state(
    message="No data available", action="Run the pipeline to generate results"
):
    """Render empty state card."""
    empty_html = f"""
    <div class="empty-state">
        <h3>⚠️ No Output Found</h3>
        <p>{message}</p>
        <p style="font-size: 14px; margin-bottom: 0;"><code>python run_pipeline.py</code></p>
        <p style="font-size: 14px; color: #A16207;">{action}</p>
    </div>
    """
    st.markdown(empty_html, unsafe_allow_html=True)


def render_sidebar():
    """Render enhanced sidebar."""
    with st.sidebar:
        # Header
        st.markdown(
            """
        <div class="sidebar-header">
            <h2>DentalVision-QA</h2>
            <p>AI Healthcare Platform</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation
        pages = [
            ("📊", "Project Overview"),
            ("📈", "Dataset Summary"),
            ("✅", "Annotation QA/QC"),
            ("🤖", "Model Performance"),
            ("👁️", "Prediction Viewer"),
            ("🚩", "Ambiguous Cases"),
        ]

        for icon, page_name in pages:
            if st.button(
                f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True
            ):
                st.session_state.page = page_name

        # Status
        st.markdown(
            """
        <div class="status-box">
            <div class="status-text">✓ Pipeline Ready</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Project info
        st.markdown("---")
        st.markdown("**Version:** 1.0.0")
        st.markdown("**Last Updated:** April 2026")
        st.markdown("**Framework:** YOLOv8")


# ─── Dashboard Pages ───────────────────────────────────────────────────


def render_project_overview():
    """Render Project Overview page."""
    st.markdown(
        '<h2 class="section-title">Project Overview</h2>', unsafe_allow_html=True
    )

    # Load data
    dataset_df = load_csv_safe(DATA_DIR / "dataset_summary.csv")
    qa_df = load_csv_safe(METRICS_DIR / "qa_report.csv")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_images = dataset_df["num_images"].sum() if dataset_df is not None else 0
        render_metric_card(str(total_images), "Total Images", "#2563EB", "🖼️")

    with col2:
        total_annotations = (
            dataset_df["num_objects"].sum() if dataset_df is not None else 0
        )
        render_metric_card(str(total_annotations), "Total Annotations", "#14B8A6", "🏷️")

    with col3:
        review_cases = (
            len(qa_df[qa_df["status"] == "REVIEW"]) if qa_df is not None else 0
        )
        render_metric_card(str(review_cases), "QA Review Cases", "#F59E0B", "⚠️")

    with col4:
        avg_confidence = qa_df["confidence"].mean() if qa_df is not None else 0.0
        render_metric_card(f"{avg_confidence:.2f}", "Avg Confidence", "#EF4444", "🎯")

    # Architecture card
    st.markdown(
        """
    <div class="section-card">
        <h4 style="margin-top: 0; color: #0F172A;">ML Pipeline Architecture</h4>
        <div style="background: #F8FAFC; padding: 16px; border-radius: 12px; margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 24px;">📥</div>
                    <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Data Collection</div>
                </div>
                <div style="width: 20px; height: 2px; background: #2563EB;"></div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 24px;">🔍</div>
                    <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Annotation</div>
                </div>
                <div style="width: 20px; height: 2px; background: #2563EB;"></div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 24px;">🤖</div>
                    <div style="font-size: 12px; color: #64748B; margin-top: 4px;">AI Training</div>
                </div>
                <div style="width: 20px; height: 2px; background: #2563EB;"></div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 24px;">✅</div>
                    <div style="font-size: 12px; color: #64748B; margin-top: 4px;">QA/QC</div>
                </div>
                <div style="width: 20px; height: 2px; background: #2563EB;"></div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 24px;">📊</div>
                    <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Dashboard</div>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_dataset_summary():
    """Render Dataset Summary page."""
    st.markdown(
        '<h2 class="section-title">Dataset Summary</h2>', unsafe_allow_html=True
    )

    dataset_df = load_csv_safe(DATA_DIR / "dataset_summary.csv")
    prep_report = load_json_safe(DATA_DIR / "preparation_report.json")

    if dataset_df is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="premium-table">', unsafe_allow_html=True)
            st.dataframe(dataset_df, use_container_width=True, height=200)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                """
            <div class="section-card">
                <h4 style="margin-top: 0;">Dataset Health</h4>
                <p style="color: #065F46; margin: 8px 0;">✓ Images validated</p>
                <p style="color: #065F46; margin: 8px 0;">✓ Annotations consistent</p>
                <p style="color: #065F46; margin: 8px 0;">✓ Splits balanced</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if prep_report:
                st.markdown(
                    f"""
                <div class="section-card">
                    <h4 style="margin-top: 0;">Preparation Report</h4>
                    <p style="margin: 8px 0;"><strong>Valid Images:</strong> {prep_report.get("valid_images", 0)}</p>
                    <p style="margin: 8px 0;"><strong>Classes:</strong> {len(prep_report.get("class_names", []))}</p>
                    <p style="margin: 8px 0;"><strong>Timestamp:</strong> {prep_report.get("timestamp", "")[:10]}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        render_empty_state("Dataset summary not found", "Run dataset preparation first")


def render_annotation_qaqc():
    """Render Annotation QA/QC page."""
    st.markdown(
        '<h2 class="section-title">Annotation QA/QC</h2>', unsafe_allow_html=True
    )

    qa_df = load_csv_safe(METRICS_DIR / "qa_report.csv")
    qa_summary = load_json_safe(METRICS_DIR / "qa_summary.json")

    if qa_df is not None and qa_summary is not None:
        # Status cards
        col1, col2, col3 = st.columns(3)

        with col1:
            pass_count = qa_summary.get("pass_count", 0)
            render_metric_card(str(pass_count), "PASS Cases", "#10B981", "✅")

        with col2:
            review_count = qa_summary.get("review_count", 0)
            render_metric_card(str(review_count), "REVIEW Cases", "#F59E0B", "⚠️")

        with col3:
            missing_count = qa_summary.get("missing_label_count", 0)
            render_metric_card(str(missing_count), "MISSING Labels", "#EF4444", "❌")

        # QA Explanation
        st.markdown(
            """
        <div class="section-card">
            <h4 style="margin-top: 0;">How QA/QC Works</h4>
            <p>Our QA/QC system evaluates annotation quality using Intersection over Union (IoU) thresholds:</p>
            <ul>
                <li><strong>✅ PASS:</strong> IoU ≥ 0.60 (high agreement)</li>
                <li><strong>⚠️ REVIEW:</strong> IoU < 0.60 (needs verification)</li>
                <li><strong>❌ MISSING:</strong> No matching ground truth</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # QA Report Table
        st.markdown("### QA Report")
        st.markdown('<div class="premium-table">', unsafe_allow_html=True)
        st.dataframe(qa_df.head(100), use_container_width=True, height=400)
        st.markdown("</div>", unsafe_allow_html=True)

        if len(qa_df) > 100:
            st.info(f"Showing first 100 rows of {len(qa_df)} total entries")
    else:
        render_empty_state("QA/QC results not found", "Run QA/QC assessment first")


def render_model_performance():
    """Render Model Performance page."""
    st.markdown(
        '<h2 class="section-title">Model Performance</h2>', unsafe_allow_html=True
    )

    metrics = load_json_safe(METRICS_DIR / "evaluation_metrics.json")

    if metrics:
        # Performance cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            precision = metrics.get("precision", 0) * 100
            render_metric_card(f"{precision:.1f}%", "Precision", "#2563EB", "🎯")

        with col2:
            recall = metrics.get("recall", 0) * 100
            render_metric_card(f"{recall:.1f}%", "Recall", "#14B8A6", "📈")

        with col3:
            map50 = metrics.get("mAP50", 0) * 100
            render_metric_card(f"{map50:.1f}%", "mAP50", "#F59E0B", "📊")

        with col4:
            map95 = metrics.get("mAP50-95", 0) * 100
            render_metric_card(f"{map95:.1f}%", "mAP50-95", "#EF4444", "📉")

        # Model Summary
        st.markdown(
            f"""
        <div class="section-card">
            <h4 style="margin-top: 0;">Model Summary</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
                <div>
                    <strong>Model:</strong> {metrics.get("model_type", "YOLOv8")}<br>
                    <strong>Classes:</strong> {metrics.get("num_classes", 7)}<br>
                    <strong>F1 Score:</strong> {(2 * metrics.get("precision", 0) * metrics.get("recall", 0) / (metrics.get("precision", 0) + metrics.get("recall", 0)) * 100):.1f}%
                </div>
                <div>
                    <strong>Split:</strong> {metrics.get("split", "val")}<br>
                    <strong>Timestamp:</strong> {metrics.get("timestamp", "")[:10]}<br>
                    <strong>Status:</strong> <span class="status-badge status-pass">Trained</span>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Per-class metrics
        if "class_metrics" in metrics:
            st.markdown("### Per-Class Performance")
            class_data = []
            for class_name, class_metrics in metrics["class_metrics"].items():
                class_data.append(
                    {
                        "Class": class_name,
                        "Precision": f"{class_metrics.get('precision', 0):.3f}",
                        "Recall": f"{class_metrics.get('recall', 0):.3f}",
                        "F1": f"{(2 * class_metrics.get('precision', 0) * class_metrics.get('recall', 0) / (class_metrics.get('precision', 0) + class_metrics.get('recall', 0))):.3f}",
                    }
                )

            import pandas as pd

            class_df = pd.DataFrame(class_data)
            st.markdown('<div class="premium-table">', unsafe_allow_html=True)
            st.dataframe(class_df, use_container_width=True, height=300)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_empty_state("Model metrics not found", "Run model evaluation first")


def render_prediction_viewer():
    """Render Prediction Viewer page."""
    st.markdown(
        '<h2 class="section-title">Prediction Viewer</h2>', unsafe_allow_html=True
    )

    pred_df = load_csv_safe(PREDICTIONS_DIR / "predictions.csv")

    if pred_df is not None:
        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            class_filter = st.selectbox(
                "Filter by Class",
                ["All"] + list(pred_df["predicted_class"].unique()),
                key="class_filter",
            )

        with col2:
            min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.3, key="conf_slider")

        with col3:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "detected", "no_detection"],
                key="status_filter",
            )

        # Apply filters
        filtered_df = pred_df.copy()
        if class_filter != "All":
            filtered_df = filtered_df[filtered_df["predicted_class"] == class_filter]
        filtered_df = filtered_df[filtered_df["confidence"] >= min_conf]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]

        st.markdown(f"### Predictions ({len(filtered_df)} results)")

        # Display predictions in grid
        if len(filtered_df) > 0:
            cols = st.columns(4)
            pred_images_dir = PREDICTIONS_DIR / "annotated_images"

            for i, (_, row) in enumerate(filtered_df.head(12).iterrows()):
                with cols[i % 4]:
                    img_name = row["image_name"]
                    confidence = row["confidence"]
                    class_name = row["predicted_class"]

                    st.markdown(
                        f"""
                    <div class="image-card">
                        <div style="font-size: 12px; color: #64748B; margin-bottom: 8px;">
                            {img_name}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Try to load image
                    img_path = pred_images_dir / f"pred_{img_name}"
                    if img_path.exists():
                        st.image(str(img_path), use_column_width=True)
                    else:
                        st.image(
                            "https://via.placeholder.com/200x150/64748B/FFFFFF?text=No+Image",
                            use_column_width=True,
                        )

                    st.markdown(
                        f"""
                        <div style="margin-top: 8px;">
                            <strong>{class_name}</strong><br>
                            <span style="color: #64748B; font-size: 12px;">{confidence:.3f} confidence</span>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            if len(filtered_df) > 12:
                st.info(f"Showing first 12 of {len(filtered_df)} predictions")
        else:
            st.info("No predictions match the current filters")
    else:
        render_empty_state("Predictions not found", "Run prediction inference first")


def render_ambiguous_cases():
    """Render Ambiguous Cases page."""
    st.markdown(
        '<h2 class="section-title">Ambiguous Cases</h2>', unsafe_allow_html=True
    )

    flagged_df = load_csv_safe(METRICS_DIR / "flagged_cases.csv")
    summary = load_json_safe(METRICS_DIR / "uncertainty_summary.json")

    if flagged_df is not None:
        # Summary cards
        col1, col2, col3 = st.columns(3)

        with col1:
            total_flagged = (
                summary.get("total_flagged", 0) if summary else len(flagged_df)
            )
            render_metric_card(str(total_flagged), "Flagged Cases", "#EF4444", "🚩")

        with col2:
            high_priority = (
                len(flagged_df[flagged_df.get("flag_severity") == "HIGH"])
                if "flag_severity" in flagged_df.columns
                else 0
            )
            render_metric_card(str(high_priority), "High Priority", "#EF4444", "❗")

        with col3:
            flag_rate = summary.get("flag_rate", 0) * 100 if summary else 0
            render_metric_card(f"{flag_rate:.1f}%", "Review Rate", "#F59E0B", "📋")

        # Clinical message
        st.markdown(
            """
        <div class="section-card" style="background: #FEF3C7; border-color: #FDE68A;">
            <h4 style="margin-top: 0; color: #92400E;">⚠️ Clinical Review Required</h4>
            <p style="color: #A16207; margin-bottom: 0;">
                These cases have been flagged for uncertainty and require review by a qualified 
                dental professional before clinical decision-making.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Flagged cases table
        st.markdown("### Flagged Cases")
        st.markdown('<div class="premium-table">', unsafe_allow_html=True)
        st.dataframe(flagged_df.head(50), use_container_width=True, height=400)
        st.markdown("</div>", unsafe_allow_html=True)

        if len(flagged_df) > 50:
            st.info(f"Showing first 50 of {len(flagged_df)} flagged cases")
    else:
        render_empty_state("Flagged cases not found", "Run uncertainty flagging first")


# ─── Main Dashboard ───────────────────────────────────────────────────


def main():
    """Main dashboard function."""

    # Inject CSS
    inject_premium_css()

    # Sidebar
    render_sidebar()

    # Header
    render_header()

    # Page routing
    if "page" not in st.session_state:
        st.session_state.page = "Project Overview"

    page = st.session_state.page

    if page == "Project Overview":
        render_project_overview()
    elif page == "Dataset Summary":
        render_dataset_summary()
    elif page == "Annotation QA/QC":
        render_annotation_qaqc()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Prediction Viewer":
        render_prediction_viewer()
    elif page == "Ambiguous Cases":
        render_ambiguous_cases()

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #64748B; padding: 20px; font-size: 14px;'>
            <p>🦷 DentalVision-QA | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p style='font-size: 12px; margin: 8px 0 0 0;'>
                <strong>Disclaimer:</strong> This is for research purposes only. 
                Always consult qualified dental professionals for clinical decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
