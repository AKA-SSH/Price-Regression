import sys
import pandas as pd

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file

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

        Returns:
        - tuple: A tuple containing the cleaned features and target data.

        Example Usage:
        ```python
        data_ingestion_object = DataIngestion()
        data_ingestion_object.data_ingestion(raw_data_file_path='path/to/raw_data.csv')
        ```
        """
        try:
            logging.info('Data ingestion initiated')

            logging.info(f'Loading raw data from {raw_data_file_path}')
            raw_dataframe = pd.read_csv(raw_data_file_path, low_memory=False)
            selected_columns= ['stars', 'reviews', 'price', 'category', 'isBestSeller', 'boughtInLastMonth']
            dataframe= raw_dataframe[selected_columns].copy()

            features, target= dataframe.drop('price', axis=1), dataframe['price']

            logging.info('Data ingestion completed')

            logging.info('Saving features and target data')
            pickle_file(object=features, file_name='features.pkl')
            pickle_file(object=target, file_name='target.pkl')
            logging.info('Contents saved')

        except Exception as CE:
            logging.error(f'Error during data ingestion: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)

if __name__ == '__main__':
    # Data ingestion
    data_ingestion_object = DataIngestion()
    data_ingestion_object.data_ingestion()