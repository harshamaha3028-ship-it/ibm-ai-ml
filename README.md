<<<<<<< HEAD
# Full Stack Web Development

This repository contains coursework, observations, e-records, assignments, and projects for the Full Stack Web Development course.

## Repository Structure

- **Observation/** - Observation notes and PDF documents
- **E-Record/** - Electronic records and documentation
- **Assignments/**
  - **Assignment-1/** - Assignment 1 submissions
  - **Assignment-2-Individual-Project/** - Individual web project
- **Group-Project/**
  - **client/** - Frontend application (React)
  - **server/** - Backend application (Node.js + Express + MongoDB)
=======
# Automobile Efficiency Regressor & Vehicle Class Profiler

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-MySQL%20%7C%20SQLite-4479A1.svg)](https://www.mysql.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-style AI/ML platform that analyzes automobile engineering specifications, estimates fuel economy in **Miles Per Gallon (MPG)** using regularized regression, automatically discovers vehicle archetypes via unsupervised **K-Means clustering** and **2D PCA projection**, and delivers explainable predictions through a responsive web dashboard backed by a **FastAPI** REST API and **MySQL** database persistence.

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Key Objectives](#2-key-objectives)
3. [Dataset Description](#3-dataset-description)
4. [System Architecture](#4-system-architecture)
5. [Machine Learning Architecture](#5-machine-learning-architecture)
   - [Supervised Regression Pipeline](#supervised-regression-pipeline)
   - [Unsupervised Vehicle Profiling / Clustering](#unsupervised-vehicle-profiling--clustering)
6. [Feature Engineering & Preprocessing](#6-feature-engineering--preprocessing)
7. [Explainability & Model Diagnostics](#7-explainability--model-diagnostics)
8. [Database Architecture](#8-database-architecture)
9. [REST API Documentation](#9-rest-api-documentation)
10. [Web Dashboard & User Interface](#10-web-dashboard--user-interface)
11. [Installation & Setup Guide](#11-installation--setup-guide)
12. [Running Model Training](#12-running-model-training)
13. [Running the Application](#13-running-the-application)
14. [Testing Suite](#14-testing-suite)
15. [Limitations & Ethical Considerations](#15-limitations--ethical-considerations)

---

## 1. Project Overview & Problem Statement

Vehicle fuel economy is driven by complex interactions between mechanical configurations (displacement, cylinder count), vehicle mass, power output, aerodynamic age factors, and manufacturing origin.

This project solves two interconnected problems:
1. **Supervised Regression**: Accurately predicting continuous fuel efficiency (**MPG**) given vehicle specifications while avoiding overfitting and collinearity traps through regularized models (**Ridge**, **Lasso**, **ElasticNet**).
2. **Unsupervised Vehicle Archetyping**: Grouping automobiles into physical and efficiency clusters (**K-Means**, $k=3$) to identify whether a vehicle resembles a *Lightweight Efficient Compact*, *Balanced Midsize Cruiser*, or *Heavy High-Power V8*.

---

## 2. Key Objectives

- **Predict MPG**: Provide instantaneous, reliable estimates of expected fuel economy.
- **Profile Archetypes**: Automatically categorize vehicles based on physical/performance footprints.
- **Feature Attribution**: Explain *why* a vehicle achieved its score (e.g. weight inertia drag vs. modern era aerodynamics).
- **Prevent Data Leakage**: Enforce rigorous ML hygiene by fitting transformers and imputers strictly on training splits.
- **Production Architecture**: Decouple model inference, API routing, database persistence, and presentation layers.

---

## 3. Dataset Description

The system uses the classical **UCI Auto MPG Dataset** ($N=398$ vehicles):

| Feature | Type | Unit / Description |
| :--- | :--- | :--- |
| `mpg` | Continuous (Target) | Miles Per Gallon fuel economy |
| `cylinders` | Discrete | Number of combustion cylinders (3 to 8) |
| `displacement` | Continuous | Engine displacement volume (cubic inches) |
| `horsepower` | Continuous | Brake horsepower (contains 6 missing `?` instances imputed via training median) |
| `weight` | Continuous | Vehicle curb weight in pounds (lbs) |
| `acceleration` | Continuous | Time to accelerate from 0 to 60 mph (seconds) |
| `model_year` | Discrete | Production model year (1970 to 1982) |
| `origin` | Categorical | 1: USA, 2: Europe, 3: Japan |
| `car_name` | String | Vehicle model label (metadata only) |

---

## 4. System Architecture

```text
                         ┌─────────────────────────┐
                         │     User / Researcher   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Interactive Web UI    │
                         │ HTML5 + CSS3 + Chart.js │
                         └────────────┬────────────┘
                                      │
                         REST API (HTTP / JSON)
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     FastAPI Backend     │
                         └────────────┬────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
      ┌───────────────┐      ┌────────────────┐      ┌─────────────────┐
      │  Prediction   │      │    Vehicle     │      │   Analytics &   │
      │    Service    │      │   Profiling    │      │    Auditing     │
      └───────┬───────┘      └───────┬────────┘      └────────┬────────┘
              │                       │                        │
              ▼                       ▼                        ▼
      ┌───────────────┐      ┌────────────────┐      ┌─────────────────┐
      │  Regularized  │      │  KMeans (k=3)  │      │  Parameterized  │
      │  Regression   │      │   + 2D PCA     │      │  MySQL Queries  │
      └───────┬───────┘      └───────┬────────┘      └────────┬────────┘
              │                       │                        │
              └───────────────────────┼────────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ MySQL DB (automobile_ml)│
                         │   (SQLite Fallback)     │
                         └─────────────────────────┘
```

---

## 5. Machine Learning Architecture

### Supervised Regression Pipeline

1. **Train/Test Holdout Split**: 80% Training ($N=318$), 20% Testing Holdout ($N=80$) with fixed `random_state=42`.
2. **Median Imputation**: Fitted solely on the training partition.
3. **Domain Feature Transformation**: Deriving power-to-weight, displacement-per-cylinder, weight-per-cylinder, vehicle age, and one-hot encoded origin.
4. **Standard Scaling**: Standardizing features to zero mean and unit variance.
5. **Model Benchmarks with 5-Fold Cross-Validation**:

| Model Architecture | Train $R^2$ | 5-Fold CV $R^2$ | Test $R^2$ | Test RMSE | Test MAE | Tuned Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | 0.8555 | **0.8409** | **0.8754** | **2.5878** | **1.9578** | Baseline OLS |
| **Ridge Regression** | 0.8554 | 0.8406 | 0.8756 | 2.5867 | 1.9439 | $\alpha = 0.160$ |
| **Lasso Regression** | 0.8555 | 0.8405 | 0.8756 | 2.5864 | 1.9524 | $\alpha = 0.001$ |
| **ElasticNet** | 0.8554 | 0.8403 | 0.8756 | 2.5860 | 1.9448 | $\alpha = 0.001, l1\_ratio = 0.7$ |

### Unsupervised Vehicle Profiling / Clustering

- **Optimal Cluster Determination**: Evaluated via **Elbow Method** (Within-Cluster Sum of Squares) and **Silhouette Scores** across $k \in [2, 8]$. Selected $k=3$.
- **2D PCA Projection**: Projects 6-dimensional scaled features onto 2 Principal Components capturing **91.8% of total variance** (PC1: 79.7%, PC2: 12.2%).
- **Dynamic Archetype Derivation**: Centroids are evaluated against dataset medians to generate human-readable profile titles:
  - **Cluster 0: *Mid-Heavy Cruiser*** (23.6% fleet, avg weight 3222 lbs, 6 cyl, 19.8 MPG)
  - **Cluster 1: *Heavy High-Power*** (24.4% fleet, avg weight 4144 lbs, 8 cyl V8, 14.5 MPG)
  - **Cluster 2: *Lightweight Efficient*** (52.0% fleet, avg weight 2306 lbs, 4 cyl, 29.4 MPG)

---

## 6. Feature Engineering & Preprocessing

Derived domain features implemented in [`app/utils/preprocessing.py`](app/utils/preprocessing.py):
- $\text{power\_to\_weight} = \frac{\text{horsepower}}{\text{weight}}$ (measures power density and acceleration potential)
- $\text{displacement\_per\_cylinder} = \frac{\text{displacement}}{\text{cylinders}}$ (cylinder displacement capacity)
- $\text{weight\_per\_cylinder} = \frac{\text{weight}}{\text{cylinders}}$ (chassis mass load per cylinder)
- $\text{vehicle\_age} = 82 - \text{model\_year}$ (time delta from reference era)
- $\text{origin\_usa}, \text{origin\_europe}, \text{origin\_japan}$ (one-hot regional categorical flags)

---

## 7. Explainability & Model Diagnostics

Standardized regression coefficients quantify the directional impact of each feature:

```text
Feature                     Coefficient   Direction
-------------------------------------------------------
cylinders                   -11.2366      Decreases MPG
displacement                +10.2520      Increases MPG (interaction with cyl)
displacement_per_cylinder    -5.1917      Decreases MPG
weight_per_cylinder          -2.6068      Decreases MPG
weight                       -1.7807      Decreases MPG
power_to_weight              -1.6463      Decreases MPG
model_year                   +1.5714      Increases MPG
vehicle_age                  -1.5714      Decreases MPG
```

> **Note on Interpretation**: In multivariate regression with interaction terms and collinearity, coefficients reflect partial derivatives conditional on the model specification and scaling, not absolute isolated physical causation.

---

## 8. Database Architecture

Database: `automobile_ml` (configured in [`database/schema.sql`](database/schema.sql)):

- **`predictions`**: Stores timestamped user queries, inputs, predicted MPG, efficiency category, cluster ID, and profile name.
- **`model_metrics`**: Stores benchmark metrics ($R^2$, MAE, MSE, RMSE) across models.
- **`cluster_profiles`**: Stores cluster centroids and dynamic profile archetypes.

All database queries use **parameterized SQL statements** to guarantee SQL-injection immunity. An automatic **SQLite fallback** mechanism ensures the application runs immediately without requiring a running MySQL server.

---

## 9. REST API Documentation

| Method | Route | Description | Input / Output |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | API health and loaded model status | `{"status": "healthy", ...}` |
| `POST` | `/predict` | Predict MPG from vehicle specifications | `VehicleInput` ➔ `PredictionResponse` |
| `POST` | `/profile` | Categorize vehicle into cluster archetype | `VehicleInput` ➔ `ProfileResponse` |
| `POST` | `/analyze` | Unified prediction + profiling + DB persistence | `VehicleInput` ➔ `AnalysisResponse` |
| `GET` | `/predictions` | Filtered, sorted, paginated prediction log | Query params ➔ `PredictionHistoryResponse` |
| `DELETE`| `/predictions` | Clear stored prediction history | `{"status": "success"}` |
| `GET` | `/statistics` | Dashboard KPIs and fleet summary | Aggregated JSON statistics |
| `GET` | `/clusters` | Cluster profiles and PCA coordinates | `ClusterListResponse` |
| `GET` | `/model-info` | Regression comparison and coefficients | `ModelInformationResponse` |

Interactive Swagger documentation is available at:
`http://127.0.0.1:8000/docs`

---

## 10. Web Dashboard & User Interface

The platform provides a responsive, multi-page automotive-themed web application:
- **`index.html` (Dashboard)**: KPI stat cards, quick MPG estimator, fleet archetype doughnut chart, model leaderboard, and recent activity preview.
- **`prediction.html` (Prediction Studio)**: Parameter sliders/inputs, 4 quick presets, animated SVG MPG dial, efficiency badge, cluster badge, comparison deltas, and explainability breakdown cards.
- **`profiles.html` (Vehicle Archetypes)**: Discovered archetype cards, interactive 2D PCA scatter map with centroids, centroid comparison bar charts, and Elbow/Silhouette validation curves.
- **`analytics.html` (Model Diagnostics)**: Supervised benchmark table, feature coefficient impact bar chart, Actual vs. Predicted plot, and Residual diagnostics.
- **`history.html` (Audit Log)**: Live searchable, filterable, sortable historical prediction log with pagination and CSV/JSON export.
- **`about.html` (Documentation)**: Full architecture breakdown, methodology, and API reference.

---

## 11. Installation & Setup Guide

### 1. Clone Repository & Navigate
```bash
git clone <repository_url>
cd automobile-efficiency-profiler
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure MySQL credentials if using MySQL:
```bash
cp .env.example .env
```
*(By default, `DB_TYPE=auto` connects to MySQL if available, or seamlessly uses local SQLite).*

---

## 12. Running Model Training

Run the training pipeline to prepare data, train models, and export artifacts:
```bash
# 1. Download and clean dataset
python training/download_data.py

# 2. Train supervised regression models & export diagnostic plots
python training/train_regression.py

# 3. Train unsupervised KMeans clustering & PCA
python training/train_clustering.py

# 4. View terminal evaluation summary
python training/evaluate_models.py
```

---

## 13. Running the Application

Start the FastAPI application and dashboard server:
```bash
python run.py
```
Or directly via Uvicorn:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
- **Web Dashboard**: `http://127.0.0.1:8000`
- **Swagger API Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Docs**: `http://127.0.0.1:8000/redoc`

---

## 14. Testing Suite

Execute automated unit and integration tests with `pytest`:
```bash
pytest tests/ -v
```

Tests cover:
- Preprocessing and feature engineering transformers
- Regression model prediction and bounds
- Vehicle profiling, cluster assignment, and centroid distances
- Full FastAPI endpoint contracts (`/health`, `/predict`, `/profile`, `/analyze`, `/predictions`, `/statistics`)
- Validation exception handling on invalid inputs (422 status)

---

## 15. Limitations & Ethical Considerations

- **Historical Scope**: The Auto MPG dataset contains vehicles manufactured between 1970 and 1982. Modern automotive technologies (turbochargers, direct injection, hybrid powertrains, electronic ECUs, electric vehicles) operate outside the empirical parameter distribution of this dataset.
- **Correlation vs. Causation**: Feature coefficients capture empirical statistical relationships within the training sample and must not be confused with pure mechanical causation in arbitrary engine designs.

---

## License

This project is licensed under the MIT License.
>>>>>>> 6f151690ed966800cf2e22383f5fb50b383934f9
