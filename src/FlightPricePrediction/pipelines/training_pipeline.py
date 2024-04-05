from src.FlightPricePrediction.components.data_ingestion import DataIngestion

from src.FlightPricePrediction.components.data_transformation import DataTransformation

from src.FlightPricePrediction.components.model_trainer import ModelTrainer

from src.FlightPricePrediction.components.model_evaluation import ModelEvaluation

from src.FlightPricePrediction.utils.utils import load_object



import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.FlightPricePrediction.utils.utils import load_object


import os
import sys
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.exception import customexception
import pandas as pd
class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion()
            train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
            return train_data_path,test_data_path
        except Exception as e:
            raise customexception(e,sys)
        
    def start_data_transformation(self,train_data_path,test_data_path):
        
        try:
            data_transformation = DataTransformation()
            train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)
            return train_arr,test_arr
        except Exception as e:
            raise customexception(e,sys)
    
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer=ModelTrainer()
            model_trainer.initate_model_training(train_arr,test_arr)
        except Exception as e:
            raise customexception(e,sys)


    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))# here is RMSE
        mae = mean_absolute_error(actual, pred)# here is MAE
        r2 = r2_score(actual, pred)# here is r3 value
        return rmse, mae, r2


    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

        

            #mlflow.set_registry_uri("https://dagshub.com/sunny.savita/fsdsmendtoend.mlflow")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)



            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)


                # this condition is for the dagshub
                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                # it is for the local 
                else:
                    mlflow.sklearn.log_model(model, "model")


                

            
        except Exception as e:
            raise e

                
    def start_trainig(self):
        try:
            train_data_path,test_data_path=self.start_data_ingestion()
            train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_arr,test_arr)
            self.initiate_model_evaluation(train_arr,test_arr)
        except Exception as e:
            raise customexception(e,sys)



model_trainer_obj=TrainingPipeline()
model_trainer_obj.start_trainig()

#model_trainer_obj=ModelTrainer()
#model_trainer_obj.initate_model_training(train_arr,test_arr)

#model_eval_obj = ModelEvaluation()
#model_eval_obj.start_evaluation()