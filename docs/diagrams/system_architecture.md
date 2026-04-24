# System Architecture

## DentalVision-QA System Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A1[Zenodo Dataset<br>6,313 images]
        A2[Kaggle Dataset<br>418 images]
        A3[Roboflow Dataset<br>Varies]
        A4[Synthetic Data<br>Placeholder]
    end

    subgraph "Annotation Phase"
        B1[Label Studio<br>Interface]
        B2[Dental Finding<br>Annotation]
        B3[YOLO Format<br>Export]
    end

    subgraph "ML Pipeline"
        C1[Data Preparation<br>YOLO Format]
        C2[YOLOv8 Training<br>Model Development]
        C3[Model Evaluation<br>Metrics & Validation]
        C4[Prediction Engine<br>Inference]
    end

    subgraph "Quality Assurance"
        D1[QA/QC Engine<br>IoU Validation]
        D2[Uncertainty Flagging<br>Risk Assessment]
        D3[Clinical Review<br>Queue]
    end

    subgraph "User Interface"
        E1[Streamlit Dashboard<br>Visualization]
        E2[Interactive Analytics<br>Reports]
    end

    subgraph "Human Loop"
        F1[Dentist Review<br>Expert Validation]
        F2[Feedback Loop<br>Model Improvement]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1

    B1 --> B2 --> B3 --> C1
    C1 --> C2 --> C3 --> C4

    C4 --> D1 --> D2 --> D3
    D3 --> F1 --> F2

    D1 --> E1 --> E2
    D2 --> E1

    F2 -.-> C2
    F1 -.-> E1

    style A1 fill:#e3f2fd
    style B2 fill:#fff3e0
    style C2 fill:#e8f5e8
    style D1 fill:#fff3e0
    style E1 fill:#e8f5e8
    style F1 fill:#ffebee
```

### Architecture Overview

- **Data Layer**: Multi-source dataset integration with fallback options
- **Annotation Layer**: Label Studio-based dental finding annotation
- **ML Layer**: YOLOv8 training and inference pipeline
- **QA Layer**: Automated quality assessment and uncertainty quantification
- **UI Layer**: Interactive dashboard for visualization and analysis
- **Human Layer**: Clinical expert review and feedback loop

### Key Components

1. **Dataset Sources**: Zenodo, Kaggle, Roboflow, Synthetic
2. **Annotation Tool**: Label Studio with 7 dental finding classes
3. **ML Framework**: YOLOv8 for object detection
4. **QA Engine**: IoU-based validation with clinical thresholds
5. **Dashboard**: Streamlit-based healthcare UI
6. **Human Loop**: Dentist review for uncertain cases

### Data Flow

1. Raw dental images from multiple sources
2. Annotation using Label Studio interface
3. Data preparation and YOLO format conversion
4. Model training and evaluation
5. Prediction with QA/QC validation
6. Uncertainty flagging and clinical review
7. Dashboard visualization and reporting
8. Human feedback loop for model improvement