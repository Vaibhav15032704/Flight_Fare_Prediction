import os
import sys
import pandas as pd
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
    
    
class CustomData:
    def __init__(self,
                 
                 Airline:str,
                 Source:str,
                 Destination:str,
                 Total_Stops:str,
                 Month_of_Journey:int,
                 Day_of_Journey:int,
                 Duration_in_minute:int):
        
        
        self.Airline=Airline
        self.Source=Source
        self.Destination=Destination
        self.Total_Stops=Total_Stops
        self.Month_of_Journey=Month_of_Journey
        self.Day_of_Journey = Day_of_Journey
        self.Duration_in_minute=Duration_in_minute  
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    
                    'Airline':[self.Airline],
                    'Source':[self.Source],
                    'Destination':[self.Destination],
                    'Total_Stops':[self.Total_Stops],
                    'Month_of_Journey':[self.Month_of_Journey],
                    'Day_of_Journey':[self.Day_of_Journey],
                    'Duration_in_minute':[self.Duration_in_minute]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)