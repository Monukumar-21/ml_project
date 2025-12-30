import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging  # Added this to support your logging.info call

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {}) 

            logging.info(f"Started Hyperparameter Tuning for: {model_name}")
            
            # GridSearchCV helps find the best combination of parameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Re-initialize the model with the best parameters found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)


            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            
            logging.info(f"Completed {model_name} with R2 Score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)   