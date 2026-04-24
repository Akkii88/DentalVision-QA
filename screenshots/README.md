# DentalVision-QA Dashboard Screenshots

## Overview

This folder contains screenshots of the DentalVision-QA Streamlit dashboard, showcasing the premium healthcare AI interface.

## Available Screenshots

### 1. Project Overview Page
**File:** `dashboard_overview.png`  
**Description:** Main dashboard page showing key metrics, pipeline architecture, and project status.  
**Features Shown:** KPI cards, workflow diagram, metric visualizations.

### 2. Model Performance Page
**File:** `model_performance.png`  
**Description:** Model evaluation results with precision, recall, mAP metrics, and per-class performance.  
**Features Shown:** Performance charts, confusion matrix, evaluation metrics.

### 3. Prediction Viewer Page
**File:** `prediction_viewer.png`  
**Description:** Interactive prediction results viewer with annotated images and filtering options.  
**Features Shown:** Image grid, confidence scores, class filtering.

### 4. QA/QC Results Page
**File:** `qa_qc_results.png`  
**Description:** Quality assurance dashboard showing annotation validation results.  
**Features Shown:** PASS/REVIEW/MISSING status, IoU analysis, annotator metrics.

## Technical Details

- **Format:** PNG (high resolution, 150 DPI)
- **Interface:** Streamlit web application
- **Theme:** Healthcare AI (blues, teals, professional styling)
- **Responsive:** Works on desktop and mobile devices

## Usage

These screenshots can be used for:
- Portfolio presentations
- Research papers
- Demo materials
- Documentation
- Marketing materials

## Generation

Screenshots were captured from the live Streamlit dashboard after running:
```bash
python run_pipeline.py --skip-train
streamlit run dashboard/app.py
```

## Note

For the most current screenshots, run the dashboard locally and capture fresh images, as the interface may evolve with updates.