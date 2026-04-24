# Dental Image Annotation Guidelines

## Overview

This document provides comprehensive guidelines for annotating dental intraoral images in the DentalVision-QA project. Proper annotation is critical for training accurate computer vision models and ensuring clinical relevance of AI predictions.

### Purpose

- Standardize annotation practices across annotators
- Minimize inter-annotator variability
- Ensure clinical validity of annotations
- Support robust model training and evaluation
- Enable consistent quality assurance

### Scope

This guide covers annotation of:
- Intraoral photographic images
- Visible dental findings and conditions
- Restorative materials and dental work
- Basic periodontal conditions

### Target Users

- Dental professionals (dentists, hygienists)
- Trained medical annotators
- Quality assurance reviewers
- Research coordinators

---

## System Setup

### Hardware Requirements

- **Monitor:** 1920×1080 minimum resolution, color-calibrated
- **Workspace:** Consistent lighting, ergonomic setup
- **Input Device:** Precision mouse or graphics tablet
- **Internet:** Stable connection for Label Studio

### Software Requirements

- **Label Studio:** Latest version (recommended)
- **Web Browser:** Chrome 90+ or Firefox 88+
- **Image Viewer:** For pre-annotation quality assessment

### Account Setup

1. Request access from project administrator
2. Complete HIPAA training certification
3. Sign data use agreement
4. Enable two-factor authentication
5. Complete annotation training module

---

## Annotation Platform

### Label Studio Interface

The project uses Label Studio for annotation with the following configuration:

```xml
<View>
  <Header value="Dental Image Annotation - DentalVision-QA"/>
  <Image name="image" value="$image"/>
  <RectangleLabels name="dental_findings" toName="image">
    <Label value="caries" background="#FF6B6B"/>
    <Label value="plaque" background="#4ECDC4"/>
    <Label value="calculus" background="#45B7D1"/>
    <Label value="gingivitis" background="#96CEB4"/>
    <Label value="missing_tooth" background="#FFEAA7"/>
    <Label value="filling_crown" background="#DDA0DD"/>
    <Label value="ambiguous" background="#A0A0A0"/>
  </RectangleLabels>
</View>
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `1-7` | Select class (1=caries, 2=plaque, etc.) |
| `R` | Rectangle tool |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+S` | Save |
| `Space` | Next image |
| `Shift+Space` | Previous image |

---

## Label Definitions

### 1. Caries (Dental Decay) 🔴
**Color:** #FF6B6B (Red)

**Definition:** Demineralization of tooth structure resulting in cavitation or surface destruction detectable visually.

**Clinical Criteria:**
- White spot lesions (early demineralization)
- Enamel caries (surface cavitation)
- Dentin caries (visible dentin involvement)
- Root caries (cementum/dentin)
- Recurrent caries (around restorations)

**Annotation Rules:**
1. Outline the entire carious lesion
2. Include surrounding demineralized enamel
3. Exclude unaffected enamel margins
4. For interproximal caries: annotate visible extent only
5. Use separate boxes for multiple distinct lesions

**Examples:**
- Small occlusal pit caries
- Smooth surface white spot lesions
- Cervical root caries
- Recurrent decay at restoration margins

### 2. Plaque (Bacterial Biofilm) 🟢
**Color:** #4ECDC4 (Teal)

**Definition:** Soft, adherent bacterial biofilm accumulation on tooth surfaces.

**Clinical Criteria:**
- Visible plaque on facial/labial surfaces
- Interproximal plaque (visible portions)
- Subgingival plaque (if visible)
- Heavy plaque accumulation
- Plaque with color changes

**Annotation Rules:**
1. Annotate extent of visible plaque
2. Follow anatomical contours
3. Use conservative bounding boxes
4. Distinguish from calculus (soft vs. hard deposits)

**Examples:**
- Thick plaque on cervical areas
- Interproximal plaque triangles
- Plaque on restoration surfaces

### 3. Calculus (Tartar) 🔵
**Color:** #45B7D1 (Blue)

**Definition:** Mineralized dental deposits (tartar) firmly attached to tooth surfaces.

**Clinical Criteria:**
- Supragingival calculus (above gumline)
- Subgingival calculus (below gumline, if visible)
- Heavy calculus deposits
- Calculus on root surfaces

**Annotation Rules:**
1. Outline all visible calculus deposits
2. Combine adjacent deposits in single box
3. Include slight margin around deposit
4. Distinguish from extrinsic stains

**Examples:**
- Large supragingival calculus ledges
- Subgingival calculus extensions
- Interproximal calculus triangles

### 4. Gingivitis (Gum Inflammation) 🟡
**Color:** #96CEB4 (Greenish)

**Definition:** Inflammation of gingival tissues characterized by erythema and edema.

**Clinical Criteria:**
- Redness/inflammation of gingiva
- Swelling/edema
- Loss of stippling
- Rolled gingival margins
- Bleeding on probing (if applicable)

**Annotation Rules:**
1. Outline affected gingival area
2. Include adjacent affected tissue
3. Exclude normal attached gingiva
4. Note generalized vs. localized inflammation

**Examples:**
- Marginal gingivitis
- Interproximal papilla inflammation
- Generalized gingival redness

### 5. Missing Tooth (Edentulous Space) 🟠
**Color:** #FFEAA7 (Orange)

**Definition:** Complete absence of tooth crown in dental arch.

**Clinical Criteria:**
- Edentulous space in arch
- Extracted teeth
- Congenitally missing teeth
- Fractured at gingival level

**Annotation Rules:**
1. Outline the entire edentulous space
2. Include adjacent teeth for context
3. Use FDI tooth numbering for reference
4. Exclude orthodontic spaces

**Examples:**
- Recent extraction sites
- Congenitally missing lateral incisors
- Trauma-related tooth loss

### 6. Filling/Crown (Restorative Materials) 🟣
**Color:** #DDA0DD (Purple)

**Definition:** Restorative materials or prosthetic crowns present on teeth.

**Clinical Criteria:**
- Amalgam restorations
- Composite restorations
- Glass ionomer restorations
- Full/partial crowns
- Inlays/onlays
- Veneers
- Core buildups

**Annotation Rules:**
1. Outline entire restoration outline
2. Include margins completely
3. Combine adjacent restorations
4. Note material type if distinguishable

**Examples:**
- Multi-surface amalgam fillings
- Ceramic crowns
- Composite veneers
- Gold inlays

### 7. Ambiguous (Uncertain Findings) ⚪
**Color:** #A0A0A0 (Gray)

**Definition:** Dental findings that are unclear, questionable, or require additional information.

**Clinical Criteria:**
- Unclear lesion nature
- Questionable caries diagnosis
- Insufficient image quality
- Borderline cases
- Requires additional views/imaging

**Annotation Rules:**
1. Mark entire questionable area
2. Add detailed notes in comments
3. Flag for expert review
4. Recommend additional imaging

**Examples:**
- Possible early demineralization
- Unclear stain vs. caries
- Poor image quality areas
- Overlapping anatomical structures

---

## Annotation Workflow

### Phase 1: Image Assessment

1. **Quality Check:**
   - Image focus and clarity
   - Exposure and lighting
   - Field of view adequacy
   - Anatomical landmarks visibility

2. **Preliminary Scan:**
   - Systematically examine all teeth
   - Note obvious findings
   - Identify challenging areas

### Phase 2: Detailed Annotation

1. **Quadrant-by-Quadrant Approach:**
   - Upper right quadrant
   - Upper left quadrant
   - Lower right quadrant
   - Lower left quadrant

2. **Tooth-by-Tooth Examination:**
   - Facial/labial surfaces
   - Lingual/palatal surfaces
   - Occlusal/incisal surfaces
   - Interproximal areas
   - Cervical areas

### Phase 3: Quality Assurance

1. **Self-Review:**
   - Verify all annotations
   - Check bounding box accuracy
   - Validate class assignments
   - Review edge cases

2. **Documentation:**
   - Add comments for ambiguous cases
   - Note unusual findings
   - Document annotation confidence

---

## Bounding Box Guidelines

### General Principles

- **Accuracy:** ±2-3 pixels precision
- **Completeness:** Include entire finding
- **Minimalism:** Avoid unnecessary background
- **Consistency:** Maintain standards across images

### Specific Guidelines

#### Caries Lesions
- Include surrounding demineralized enamel
- Extend to sound tooth structure boundaries
- Separate boxes for distinct lesions

#### Soft Tissue (Gingivitis)
- Follow gingival contours
- Include affected papillae
- Exclude healthy attached gingiva

#### Restorations
- Outline entire restoration
- Include preparation margins
- Combine contiguous restorations

#### Missing Teeth
- Include entire edentulous space
- Extend to adjacent teeth contacts
- Account for healing or granulation tissue

### Edge Cases

#### Interproximal Lesions
- Annotate visible portions only
- Don't extrapolate behind contact points
- Note if likely extends interproximally

#### Overlapping Findings
- Prioritize primary pathology
- Use separate boxes when distinct
- Document relationships in comments

#### Poor Image Quality
- Annotate confidently visible findings
- Mark uncertain areas as "ambiguous"
- Skip completely obscured areas

---

## Quality Standards

### Annotation Accuracy

| Metric | Target | Tolerance |
|--------|--------|-----------|
| Bounding Box Precision | ±2 pixels | ±5 pixels max |
| Class Assignment Accuracy | 95% | 90% minimum |
| Completeness (Recall) | 90% | 80% minimum |
| Inter-annotator Agreement | 0.85 IoU | 0.75 minimum |

### Review Process

#### Level 1: Annotator Self-Check
- **Rate:** 100% of annotations
- **Time:** 30-60 seconds per image
- **Focus:** Completeness, obvious errors
- **Tools:** Label Studio review mode

#### Level 2: Peer Review
- **Rate:** 10-20% of annotations
- **Time:** 60-90 seconds per image
- **Focus:** Accuracy, consistency
- **Reviewer:** Senior annotator

#### Level 3: Expert Review
- **Rate:** 5-10% of annotations
- **Time:** 2-5 minutes per image
- **Focus:** Clinical validity
- **Reviewer:** Licensed dentist

### Quality Metrics Tracking

**Weekly Metrics:**
- Annotations per hour
- Self-review correction rate
- Peer review feedback rate
- Expert override rate

**Monthly Metrics:**
- Inter-annotator agreement scores
- Class-wise accuracy rates
- Completeness vs. accuracy trade-offs
- Training needs identification

---

## Common Pitfalls

### Over-Annotation Errors

❌ **Including Normal Anatomy:**
- Annotating normal enamel texture as plaque
- Marking healthy gingiva as gingivitis
- Including anatomical landmarks as findings

✅ **Solution:** Focus on pathological changes only

### Under-Annotation Errors

❌ **Missing Subtle Findings:**
- Ignoring early white spot lesions
- Overlooking interproximal caries
- Missing mild gingivitis

✅ **Solution:** Systematic scanning protocols

### Class Confusion Errors

❌ **Common Misclassifications:**
- Plaque vs. calculus (soft vs. hard)
- Stain vs. caries (extrinsic vs. demineralization)
- Healthy vs. inflamed gingiva

✅ **Solution:** Reference clinical criteria and examples

### Technical Errors

❌ **Bounding Box Issues:**
- Boxes too tight (cropping findings)
- Boxes too loose (including noise)
- Inconsistent sizing standards

✅ **Solution:** Standardize based on anatomical boundaries

---

## Training and Certification

### Initial Training

1. **Platform Familiarization** (2 hours)
   - Label Studio interface
   - Keyboard shortcuts
   - Project-specific configuration

2. **Clinical Knowledge** (4 hours)
   - Dental anatomy and terminology
   - Common dental conditions
   - Diagnostic criteria

3. **Annotation Practice** (4 hours)
   - Guided annotation exercises
   - Feedback and correction
   - Speed vs. accuracy balance

### Certification Process

1. **Written Assessment** (1 hour)
   - Clinical knowledge quiz
   - Guideline comprehension test

2. **Practical Assessment** (2 hours)
   - Annotate sample images
   - Achieve 85% accuracy threshold
   - Demonstrate systematic approach

3. **Calibration Session** (1 hour)
   - Annotate with expert reviewer
   - Discuss discrepancies
   - Refine annotation approach

### Continuing Education

**Monthly Requirements:**
- Review updated guidelines
- Complete refresher quizzes
- Participate in calibration sessions

**Annual Requirements:**
- Recertification assessment
- Clinical knowledge updates
- Technology training

---

## Troubleshooting

### Common Issues

#### Image Loading Problems
- **Issue:** Images fail to load in Label Studio
- **Solution:** Check image format (JPEG/PNG), file size (<10MB), network connectivity

#### Annotation Saving Errors
- **Issue:** Annotations not saving properly
- **Solution:** Check internet connection, browser compatibility, local storage

#### Class Selection Issues
- **Issue:** Labels not appearing or selecting incorrectly
- **Solution:** Refresh browser, clear cache, check project configuration

#### Performance Issues
- **Issue:** Slow loading or lagging interface
- **Solution:** Close other browser tabs, check system resources, use recommended browser

### Escalation Procedures

1. **Level 1:** Check troubleshooting guide and FAQs
2. **Level 2:** Contact project coordinator
3. **Level 3:** Escalate to technical support team
4. **Level 4:** Report as system bug with detailed reproduction steps

---

## Resources and References

### Clinical References

- FDI World Dental Federation tooth numbering system
- American Dental Association diagnostic criteria
- WHO oral health assessment guidelines
- Periodontology clinical practice guidelines

### Technical References

- Label Studio documentation
- Computer vision annotation best practices
- Medical image annotation standards
- Quality assurance methodologies

### Training Materials

- Interactive training modules
- Sample annotation exercises
- Video tutorials
- Case study examples

---

## Document Control

| Field | Value |
|-------|-------|
| Document Title | Dental Image Annotation Guidelines |
| Version | 1.0.0 |
| Effective Date | April 2026 |
| Review Date | April 2027 |
| Author | DentalVision-QA Clinical Team |
| Approver | Project Medical Director |
| Distribution | Annotators, Reviewers, QA Team |

### Change History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | April 2026 | Initial release | Clinical Team |

### Document Approval

- [x] Clinical Team Review
- [x] QA Team Review
- [x] Technical Team Review
- [x] Medical Director Approval

---

## Contact Information

### Support Channels

**Technical Support:**
- Email: support@dentalvision-qa.org
- Hours: Monday-Friday, 9 AM - 5 PM EST

**Clinical Questions:**
- Email: clinical@dentalvision-qa.org
- For annotation interpretation guidance

**Training Support:**
- Email: training@dentalvision-qa.org
- For certification and continuing education

### Emergency Contacts

**System Downtime:** +1-555-0123 (24/7)
**Data Security Incident:** +1-555-0124 (24/7)
**Medical Emergency:** Local emergency services

---

*This document is confidential and intended for authorized DentalVision-QA project personnel only. All annotations must comply with HIPAA and applicable data privacy regulations.*

**© 2026 DentalVision-QA. All rights reserved.**