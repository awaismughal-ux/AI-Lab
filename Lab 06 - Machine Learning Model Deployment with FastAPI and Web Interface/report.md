# Laboratory 06 Report: Machine Learning Model Deployment with FastAPI and Web Interface

- **Course:** Introduction to Artificial Intelligence (AI-101L)
- **Instructor:** Dr. Muhammad Awais
- **Laboratory Identified & Named:** **Lab 06 — Machine Learning Model Deployment with FastAPI and Web Interface**
- **Domain:** MLOps, Model Serving, RESTful Microservices, and Client-Server Integration
- **Mapped CLOs:** CLO-3, CLO-4

---

## 1. Laboratory Title & System Architecture Rationale

### Why This Lab is Named "Machine Learning Model Deployment":
The directory was originally left unaddressed as `lab 06`. An analysis of the components in this workspace reveals a classic three-tier production machine learning serving architecture:
1. **Model Persistence Layer (`Mymdoel.pkl`)**: A serialized Scikit-Learn Logistic Regression model artifact trained on clinical diabetes biomarkers.
2. **REST API Microservice Layer (`main.py`)**: A high-performance Python application built with **FastAPI**, implementing asynchronous request handling, Pydantic type-safe validation, CORS headers, health checks, and a `/predict` inference route.
3. **Interactive Client Presentation Layer (`index.html`)**: A decoupled frontend client communicating asynchronously via HTTP POST (`fetch()`) to deliver real-time diagnostic risk stratification.

Consequently, according to the software components and pedagogical goals, the definitive academic title is:  
**Lab 06 — Machine Learning Model Deployment with FastAPI and Web Interface** (Subheading: *Architecting Production REST APIs and Real-Time Clinical Inference UIs*).

---

## 2. System Architecture & Component Design

```
+-------------------------------------------------------------+
|                      Client Layer                           |
|  index.html (HTML5 / Modern CSS / Vanilla JavaScript Fetch) |
+-------------------------------------------------------------+
                              |
                     HTTP POST /predict
                     JSON Payload (Vitals)
                              v
+-------------------------------------------------------------+
|                 FastAPI Application Server                  |
|  - CORS Middleware (Cross-Origin Resource Sharing)          |
|  - Pydantic Schema Validation (PredictRequest)              |
|  - Endpoints: / (Home), /health, /model-info, /predict      |
+-------------------------------------------------------------+
                              |
                    Feature Extraction &
                    Input Ordering Alignment
                              v
+-------------------------------------------------------------+
|                   Machine Learning Model                    |
|  Scikit-Learn Logistic Regression (Mymdoel.pkl)             |
|  - Probability Estimation: model.predict_proba()            |
|  - Classification: model.predict()                          |
+-------------------------------------------------------------+
                              |
                    Risk Stratification Engine
                    JSON Response (PredictResponse)
                              v
+-------------------------------------------------------------+
|                      Client Display                         |
|  Visual Risk Badge | Animated Probability Bar | Timestamp   |
+-------------------------------------------------------------+
```

---

## 3. Key Technical Challenges & Resolutions

### 3.1 Model Deserialization & Scikit-Learn Incompatibility
The legacy `Mymdoel.pkl` binary was pickled using an unaligned development build, which triggered:
```text
AttributeError: 'LogisticRegression' object has no attribute 'multi_class'
```
This fatal deserialization exception was resolved by re-training the clinical logistic regression estimator in Lab 04 using the current Scikit-Learn environment (`1.5.2`) and re-exporting the model artifact with clean feature name signatures (`feature_names_in_`).

### 3.2 Clinical Schema & Range Enforcement
In medical diagnostic contexts, arbitrary inputs can produce misleading model extrapolation. The Pydantic request schema was fortified with physiological bounding constraints:
- `Glucose`: $[0.0, 400.0]$ mg/dL
- `BloodPressure`: $[0.0, 250.0]$ mm Hg
- `Insulin`: $[0.0, 900.0]$ mu U/ml
- `BMI`: $[0.0, 80.0]$
- `DiabetesPedigreeFunction`: $[0.0, 3.0]$
- `Age`: $[1.0, 130.0]$ years

### 3.3 Clinical Risk Stratification
Rather than returning a raw binary `0` or `1`, the serving logic derives posterior probabilities and maps them into three actionable tiers:
- **Low Risk**: $P(\text{Diabetes}) < 0.35$
- **Moderate Risk**: $0.35 \le P(\text{Diabetes}) \le 0.65$
- **High Risk**: $P(\text{Diabetes}) > 0.65$

---

## 4. API Specification & Endpoint Documentation

| Method | Endpoint | Description | Sample Response Status |
|---|---|---|---|
| `GET` | `/` | Service root and navigation links | `200 OK` |
| `GET` | `/health` | Uptime and model loading health monitor | `200 OK` |
| `GET` | `/model-info` | Metadata, learned coefficients, and feature schema | `200 OK` |
| `POST` | `/predict` | Real-time patient inference and risk scoring | `200 OK` / `422 Unprocessable` |

---

## 5. Verification & Automated Testing

An automated test suite (`test_api.py`) was implemented using Python's `unittest` framework:
1. `test_model_loaded`: Verified model non-null status and feature length ($n = 6$).
2. `test_low_risk_prediction`: Evaluated young, normoglycemic vitals (Glucose 85, BMI 22.4), asserting `predicted_label == 0`, `risk_level == "Low Risk"`, and $P < 0.35$.
3. `test_high_risk_prediction`: Evaluated acute profile (Glucose 185, BMI 39.5, Age 55), asserting `predicted_label == 1`, `risk_level == "High Risk"`, and $P > 0.65$.
4. `test_probability_bounds`: Verified mathematical probability axiom $0.0 \le P \le 1.0$.

Result: **Ran 4 tests in 0.030s — OK (All tests passed).**

---

## 6. Artifact Checklist

- [x] `main.py`: Enhanced FastAPI service with Pydantic validation and risk stratification.
- [x] `index.html`: Modern, responsive clinical interface with presets and visual probability progress bar.
- [x] `Mymdoel.pkl`: Version-compatible serialized Logistic Regression estimator.
- [x] `test_api.py`: Automated test suite verifying inference logic and schema integrity.
- [x] `report.md`: Complete theoretical and architectural documentation.
