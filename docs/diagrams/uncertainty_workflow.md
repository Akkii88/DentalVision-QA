# Uncertainty Workflow

## Uncertainty Quantification and Flagging Workflow

```mermaid
flowchart TD
    A[Model Prediction<br>Bounding Box + Class + Confidence] --> B{Confidence ≥ 0.70?}

    B -->|Yes| C[High Confidence<br>Auto-Accept]
    B -->|No| D[Low Confidence<br>Flag for Review]

    A --> E[Check Overlap<br>IoU Analysis]

    E --> F{Multiple Boxes<br>IoU > 0.50?}

    F -->|Yes| G[Overlapping Predictions<br>Flag for Review]
    F -->|No| H[No Overlap<br>Proceed]

    A --> I{Class = 'ambiguous'?}

    I -->|Yes| J[Ambiguous Class<br>Flag for Review]
    I -->|No| K[Clear Class<br>Proceed]

    A --> L[Size Check<br>Box Area Analysis]

    L --> M{Box Area > 40%<br>of Image?}

    M -->|Yes| N[Large Bounding Box<br>Flag for Review]
    M -->|No| O[Normal Size<br>Proceed]

    A --> P[Edge Check<br>Boundary Analysis]

    P --> Q{Box within<br>5% of Edge?}

    Q -->|Yes| R[Edge Detection<br>Flag for Review]
    Q -->|No| S[Central Location<br>Proceed]

    C --> T[Final Decision<br>Auto-Accept]
    H --> T
    K --> T
    O --> T
    S --> T

    D --> U[Clinical Review<br>Required]
    G --> U
    J --> U
    N --> U
    R --> U

    T --> V[Dashboard Display<br>Green Status]
    U --> W[Clinical Queue<br>Amber/Red Status]

    W --> X[Dentist Review<br>Expert Decision]
    X --> Y{Acceptable?}

    Y -->|Yes| Z[Confirm Prediction<br>Update Records]
    Y -->|No| AA[Reject Prediction<br>Request Re-annotation]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#d1fae5
    style D fill:#fef3c7
    style G fill:#fef3c7
    style J fill:#fef3c7
    style N fill:#fef3c7
    style R fill:#fef3c7
    style U fill:#ffebee
    style X fill:#ffebee
```

### Uncertainty Quantification Process

The uncertainty flagging system identifies predictions that require human clinical review based on multiple risk factors and quality indicators.

### Flagging Criteria

#### 1. Confidence Threshold
- **High Confidence** (≥0.70): Auto-accept with minimal risk
- **Low Confidence** (<0.70): Flag for clinical review
- **Rationale**: Empirical threshold based on clinical acceptability

#### 2. Prediction Overlap
- **IoU Analysis**: Check for overlapping bounding boxes
- **Threshold**: IoU > 0.50 between predictions
- **Risk**: Conflicting detections may indicate uncertainty

#### 3. Class Ambiguity
- **Ambiguous Class**: Explicit "ambiguous" predictions
- **Risk**: Model acknowledges uncertainty in classification
- **Action**: Always flag for expert review

#### 4. Bounding Box Size
- **Large Boxes**: Area > 40% of image
- **Risk**: Overly aggressive detection may be artifact
- **Threshold**: Clinically reasonable size limits

#### 5. Edge Proximity
- **Edge Detection**: Boxes within 5% of image boundaries
- **Risk**: Boundary artifacts or incomplete anatomy
- **Safety**: Conservative edge margins

### Risk Levels

| Combination | Severity | Action | Timeline |
|-------------|----------|--------|----------|
| Single Flag | Low | Review | Within 24h |
| Multiple Flags | Medium | Priority Review | Within 8h |
| Critical + Low Conf | High | Urgent Review | Within 2h |

### Clinical Review Process

#### Review Queue Prioritization
1. **High Priority**: Low confidence + multiple flags
2. **Medium Priority**: Single flag conditions
3. **Low Priority**: Edge cases and minor issues

#### Expert Decision Options
- **Accept**: Confirm prediction as clinically valid
- **Reject**: Mark as false positive, remove from records
- **Re-annotate**: Request new ground truth annotation
- **Escalate**: Forward to senior clinician or committee

### Quality Metrics

#### Flagging Statistics
- **Flag Rate**: Percentage of predictions requiring review
- **False Positive Rate**: Incorrectly flagged predictions
- **Review Turnaround**: Average time for clinical decisions
- **Expert Agreement**: Consistency between reviewers

#### Clinical Outcomes
- **Safety Score**: Reduction in potential clinical errors
- **Efficiency Gain**: Automated handling of high-confidence cases
- **Quality Improvement**: Data-driven model and process enhancement

### Integration with QA/QC

- **Complementary Systems**: Uncertainty flagging works alongside IoU-based QA
- **Multi-layer Safety**: Different validation approaches provide redundancy
- **Feedback Loop**: Review outcomes inform future flagging thresholds

### Benefits

- **Clinical Safety**: Identifies cases needing expert attention
- **Efficiency**: Automates review of high-confidence predictions
- **Quality Assurance**: Multiple validation layers ensure accuracy
- **Continuous Learning**: Review data improves model performance
- **Risk Mitigation**: Systematic approach to uncertainty management

### Dashboard Integration

- **Real-time Monitoring**: Live flagging statistics and trends
- **Review Queue**: Prioritized case management interface
- **Performance Analytics**: Review efficiency and quality metrics
- **Audit Trail**: Complete record of decisions and rationales