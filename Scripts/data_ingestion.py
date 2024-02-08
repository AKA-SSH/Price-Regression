import sys
import pandas as pd

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

from Scripts.feature_engineering import FeatureEngineering
from Scripts.encoding import Encoding

class DataIngestion:
    def __init__(self) -> None:
        pass

    def data_ingestion(self, raw_data_file_path='Data\\raw_data.csv'):
        """
        Performs the data ingestion process for space object data.

        Parameters:
        - raw_data_file_path (str): The file path to the raw data in CSV format.

        Raises:
        - CustomException: If an error occurs during the data ingestion process.

        Example Usage:
        ```python
        data_ingestion_object = DataIngestion()
        data_ingestion_object.data_ingestion(raw_data_file_path='Data\\raw_data.csv')
        ```
        """
        try:
            logging.info('Data ingestion initiated')

            logging.info(f'Loading raw data from {raw_data_file_path}')
            raw_dataframe = pd.read_csv(raw_data_file_path)

            selected_columns = ['stars', 'reviews', 'price', 'category', 'isBestSeller', 'boughtInLastMonth']
            dataframe = raw_dataframe[selected_columns].copy()

            logging.info('Data ingestion completed')

            logging.info('Saving features and target data')
            pickle_file(object=dataframe, file_name='dataframe.pkl')
            logging.info('Contents saved')

        except Exception as CE:
            logging.error(f'Error during data ingestion: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)

if __name__ == '__main__':
    # Data ingestion
    logging.info('Performing data ingestion...')
    data_ingestion_object = DataIngestion()
    data_ingestion_object.data_ingestion()
    
    # Feature Engineering
    logging.info('Performing feature engineering...')
    feature_engineering_object = FeatureEngineering()
    feature_engineering_object.engineer_feature('artifacts\\dataframe.pkl')

    # Encoding
    logging.info('Performing encoding...')
    encoding_object= Encoding()
    encoding_object.encode('artifacts\\dataframe.pkl')