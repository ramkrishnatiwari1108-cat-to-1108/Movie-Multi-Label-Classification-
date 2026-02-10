

# 🎬 Movie Genre Classification – End-to-End ML System

## Overview
This project implements an end-to-end multi-label classification system to predict movie genres
from an IMDB movie dataset. The solution covers model training, experiment tracking, model serving,
containerization, and monitoring, with a clear separation between training and inference pipelines.

Two different modeling approaches are implemented and served via REST APIs.

---

## Problem Statement
Given a movie title, predict the most likely genres associated with the movie.

### Example Input
```json
{ "title": "Inception" }
{
  "predicted_genres": ["Action", "Sci-Fi", "Thriller"],
  "confidence": 0.82,
  "inference_time": "0.03s"
}

## Project Features

This project implements a complete machine learning pipeline for multi-label text classification. The core components include:

* **Data Preparation & Feature Extraction**: 
    * Text cleaning and preprocessing.
    * Feature engineering using `TfidfVectorizer` with n-gram support.
* **Model Training & Evaluation**: 
    * Jupyter notebooks containing the end-to-end training logic.
    * Performance testing using Multi-Label Decision Tree algorithms.
* **Model Persistence**: 
    * Trained models and vectorizers are serialized using `joblib` for fast loading and deployment.
* **Inference API**: 
    * A production-ready **FastAPI** application to serve real-time predictions.
* **Monitoring & Logging**: 
    * Detailed logging of inference metrics, including:
        * **Execution Time**: Tracking how long each prediction takes.
        * **Label Count**: Reporting the number of genres predicted for each input.



Folder and File Structure
.
├── data
│   ├── movies_overview.csv
│   ├── movie_genres.csv
│   └── final_movie_genre_V2.csv
│
├── Adaptation.ipynb
├── Transformation.ipynb
├── app.py
├── tfidf_vectorizer.pkl
├── tfidf_vectorizer_v2.pkl
├── mlb_classes.pkl
├── chain_svm.pkl
├── knn.pkl
├── inference.log
└── README.md


## Problem Statement

Movies usually have more than one genre. 

Because of this, predicting genres is not a single-label problem. Instead of predicting just one genre, the model must predict all relevant genres for a movie title. This project explores two major approaches that are commonly used for multi-label classification.






# Problem Transformation Approach

## (Transformation.ipynb)

### What This Means

Problem Transformation methods convert a multi-label problem into multiple single-label problems.

Each label is predicted in a structured way instead of all at once.

### Model Used

- Classification Chain with SVM
- TF-IDF features extracted from movie titles
- SVM decision scores instead of probabilities
- Manual thresholding to select genres

### Why Classification Chains Were Used

- They learn relationships between genres
- Better than treating each genre separately
- Useful when genres often appear together

### What Happens in the Notebook

The notebook shows the full experimentation process:
- Loading and understanding the dataset
- Cleaning and preparing movie titles
- Converting text into TF-IDF features
-
- Encoding genres using MultiLabelBinarizer
-
- Training a Classification Chain model
-
- Testing different threshold values
-
- Evaluating results using multi-label metrics

Threshold tuning was very important to avoid missing genres.

### How Prediction Works
 
SVM does not give probabilities. It gives decision scores.
 
A genre is selected if its score is above a chosen threshold.






# Algorithm Adaptation Approach

## (Adaptation.ipynb)

### What This Means

Algorithm adaptation methods use models that are designed to handle multi-label data directly.

- No conversion into multiple single-label problems is needed.

### Models Used

- **Multi-Label K-Nearest Neighbors (ML-KNN)**
- *(Explored)* Multi-Label Decision Trees

### Why ML-KNN Was Used

- It gives probability values for each genre.
- Easy to apply confidence thresholds.
- Naturally supports multi-label prediction.

### What Happens in the Notebook

The notebook includes:

1. Data preprocessing steps.
2. A separate TF-IDF vectorizer for adaptation models.
3. Training the ML-KNN model.
4. Checking probability outputs.
5. Selecting a suitable threshold.
6. Evaluating performance using metrics.

### How Prediction Works

- Each genre gets a probability score.
- A genre is selected if its probability is above the threshold.







# Feature Engineering

TF-IDF is used to convert movie titles into numbers.

## Two Separate Vectorizers
- `tfidf_vectorizer.pkl` for transformation models
- `tfidf_vectorizer_v2.pkl` for adaptation models

Titles are converted to lowercase.

No heavy text cleaning was done to avoid losing meaning.

# Model Saving and Loading

All trained components are saved using `joblib`:
- TF-IDF vectorizers
- Genre label encoder (`MultiLabelBinarizer`)
- Trained models

This allows:
- Fast loading during inference
- No retraining needed
- Stable and repeatable predictions

# FastAPI Application (`app.py`)

## API Overview
The FastAPI app provides two endpoints, one for each approach:
| Endpoint | Method | Purpose |
|---|---|---|
| `/predict/transformation` | POST | SVM Classification Chain |
| `/predict/adaptation` | POST | ML-KNN |

## Request Format
```json
default:
{
  "title": "The Dark Knight"
}
```
## Response Format
```json
default:
{
  "predicted_genres": ["Action", "Crime", "Drama"],
  "inference_time": 0.012
}
```

# Why Models Are Loaded Locally - MLflow Explanation 
Initially, models were planned to be loaded using MLflow model URIs. However:
- The MLflow UI was restarted.
- Stored artifacts were lost.
- The trained models could not be recovered.
I also sent out an email , got pannicked 
This issue was communicated to the evaluator.
As a solution:
- Models were saved locally using `joblib`.
- The inference logic remained unchanged.
- The MLflow tracking URI is still present for future use.

# Inference Logging 
Every prediction request is logged with:
the endpoint used,
how long inference took,
and how many genres were predicted.

def example log entry: 2026-02-10 18:32:12 | endpoint=adaptation | latency=0.011s | confidence=3_labels 
this helps track performance, prediction behavior, and basic monitoring without extra tools.

# Threshold Selection Logic 
default thresholds were not used. Thresholds were manually chosen to:
avoid predicting too few genres,
improve recall,
and match real movie genre overlap.
different models use different thresholds based on their output type.

# Important Design Decisions 
simple and clear prediction logic, separate pipelines for both approaches, comments kept in code to show experimentation, honest fallback from MLflow to local loading, logging added after testing was complete.

# How to Run the Application
you need to install dependencies first:
pip install fastapi uvicorn scikit-learn joblib mlflow
yuvicorn app:app --reload 
and open in your browser: http://127.0.0.1:8000/docs



# Screenshots
below are the screenshots provided 


#Ml Flow Screenshots

ML fllow UI (as i  had to do it again)
![Example](./Screenshots/ML Flow UI.png)











### Training Container (Reference)

Model training and experimentation were conducted using a Jupyter notebook
(train.ipynb) to enable rapid iteration.

The training Docker image includes this notebook as a reference for the
training logic. Due to time constraints, training is executed locally,
while inference is fully containerized and production-ready.
