import time
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


# beflow a snippet for loggint

import logging
logging.basicConfig(
    filename="inference.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)



import joblib

"""
Note ignore the commmentts , but i am keeping it , to show that i had to change and test a lot of things in it 



"""


# Loading  the saved components
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_v2_adap = joblib.load('tfidf_vectorizer_v2.pkl')

# chain_svm = joblib.load('chain_svm_model.pkl')

mlb = joblib.load('mlb_classes.pkl')

# importing the model locally , i had no choice , (sad)




model_transform = joblib.load('chain_svm.pkl')
model_adapt = joblib.load('knn.pkl')





# Now you can call your predict_genres function


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"   # change if needed

"""

initially planned this , but as i relaunched my entire ml flow ui .i lost the models( i notified this on the mail also to
                                                                                     the sir ) , 
i didnt knew how to ge it back , but i
decieded to make it locally ,  i will be including screenshot , of it , as i added it later,

"""

# # Model URIs from MLflow
# MODEL_URIS = {
#     "transformation_1": "mlflow-artifacts:/5/35b92a4fc0e44576ae42ea63376d9098/artifacts/model.pkl",
#     "transformation_2": "mlflow-artifacts:/5/9f2aef338e3f4d0e9948b4d8ab9591a7/artifacts/model.pkl",
#     "adaptation_1": "mlflow-artifacts:/5/83ec76336e4b47fc8b8dbd7bb77de16b/artifacts/model.pkl",
#     "adaptation_2": "mlflow-artifacts:/5/7dee94a349be48ba95be7d634353d46b/artifacts/model.pkl",
# }

# App Initialization


app = FastAPI(title="Movie Genre Prediction API")


# -----------------------------
# Request / Response Schemas
# -----------------------------

class MovieInput(BaseModel):
    title: str

class PredictionResponse(BaseModel):
    predicted_genres: List[str]
    inference_time: float

# 
# Prediction Functions

# 


# using th similiar aporach i used for predictng in my jupytter notebook 
def predict_transformation(title, threshold=-0.3):
   
    sample_X = tfidf.transform([title])


    scores = model_transform.decision_function(sample_X)[0]

    #  threshold logic to map scores to class labels
    predicted_genres = [
        mlb.classes_[i]
        for i, score in enumerate(scores)
        if score >= threshold
    ]

    return predicted_genres





























def predict_adaptation(title: str) -> List[str]:
    """
    Uses adaptation models
    """
    title_clean = title.lower()
    title_vec = tfidf_v2_adap.transform([title_clean])
    
    THRESHOLD = 0.4  
    
    probs = model_adapt.predict_proba(title_vec)[0]
    
    labels = [
        mlb.classes_[i]
        for i, p in enumerate(probs)
        if p >= THRESHOLD
    ]
    
    return labels

# -----------------------------
# API Endpoints
# -----------------------------

@app.post("/predict/transformation", response_model=PredictionResponse)
def predict_transformation_api(data: MovieInput):
    start_time = time.time()

    genres = predict_transformation(data.title)

    inference_time = round(time.time() - start_time, 3)
    logging.info(
        f"endpoint=transformation | "
        f"latency={inference_time}s | "
        f"confidence={len(genres)}_labels"  # edited this later from confidence N/a to this , 
    )



    return {
        "predicted_genres": genres,
        "inference_time": inference_time
    }


@app.post("/predict/adaptation", response_model=PredictionResponse)
def predict_adaptation_api(data: MovieInput):
    start_time = time.time()

    genres = predict_adaptation(data.title)

    inference_time = round(time.time() - start_time, 3)
    logging.info(
        f"endpoint=adaptation | "
        f"latency={inference_time}s | "
        f"confidence= {len(genres)}_labels"
    )
    return {
        "predicted_genres": genres,
        "inference_time": inference_time
    }









































# def predict_genres_proba(title, threshold=0.3):

#     sample_X = tfidf.transform([title])

#     # [0] accesses the probabilities for our single input title


#     predicted_genres = [
#         mlb.classes_[i]
#         for i, p in enumerate(probs)
#         if p >= threshold
#     ]

#     return predicted_genres

