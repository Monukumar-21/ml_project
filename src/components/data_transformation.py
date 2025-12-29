import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Standardize case: Ensure these match your CSV headers exactly
            numerical_columns = ['writing_score', 'reading_score'] 
            categorical_columns = [
                'gender', 'race_ethnicity', 'lunch', 'parental_level_of_education',
                'test_preparation_course'
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)), # Added sparse_output=False
                    ("scaler", StandardScaler(with_mean=False)) # with_mean=False helps with encoded data
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns) # Fixed typo 'cat_pipelinr'
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def intiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Separate Features and Target
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # FIXED: Now correctly using test_df
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Applying transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Concatenate arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path, # Fixed fil_path typo
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, self.data_tranformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)

