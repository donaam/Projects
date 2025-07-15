import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info
            ('Error occurred in predict function of prediction pipeline')
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        cc_num: str,
        category: str,
        first: str,
        last: str,
        gender: str,
        job: str,
        dob: str,
        date: str,
        time: str
    ):
        self.cc_num = cc_num
        self.category = category
        self.first = first
        self.last = last
        self.gender = gender
        self.job = job
        self.dob = dob
        self.date = date
        self.time = time

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'cc_num': [self.cc_num],
                'category': [self.category],
                'first': [self.first],
                'last': [self.last],
                'gender': [self.gender],
                'job': [self.job],
                'dob': [self.dob],
                'date': [self.date],
                'time': [self.time]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Input DataFrame created successfully')
            return df

        except Exception as e:
            logging.info
            ('Error occurred in get_data_as_dataframe method of CustomData')
            raise CustomException(e, sys)
