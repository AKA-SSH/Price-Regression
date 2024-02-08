import sys
import pandas as pd

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class FeatureEngineering:
    def __init__(self) -> None:
        pass

    def engineer_feature(self, cleaned_dataframe_file_path: str):
        """
        Performs feature engineering.

        Parameters:
        - cleaned_dataframe_file_path (str): File path to the pickled file containing the cleaned DataFrame.

        Raises:
        - CustomException: If an error occurs during the feature engineering process.

        Returns:
        None

        Example Usage:
        ```python
        feature_engineering = FeatureEngineering()
        feature_engineering.engineer_feature('artifacts\\cleaned_dataframe.pkl')
        ```
        """
        try:
            logging.info('Feature engineering initiated')

            logging.info('Loading cleaned DataFrame')
            dataframe = unpickle_file(cleaned_dataframe_file_path)

            logging.info('Creating estimatedSaleCount')
            dataframe.loc[:, 'estimatedSaleCount'] = round(66.67 * dataframe['reviews'])

            logging.info('Creating estimatedRevenue')
            dataframe.loc[:, 'estimatedRevenue'] = dataframe['price'] * dataframe['estimatedSaleCount']
            dataframe = dataframe.drop('estimatedSaleCount', axis=1)

            logging.info('Creating priceOptimality')
            bin_edges = [0, 10, 100, 1000, 10_000, 100_000, float('inf')]
            bin_labels = [0, 1, 2, 3, 4, 5]
            dataframe['priceOptimality'] = pd.cut(dataframe.reviews, bins= bin_edges, labels= bin_labels, right=False)
            
            logging.info('Saving DataFrame with engineered features')
            pickle_file(object=dataframe, file_name='dataframe.pkl')

            logging.info('Feature engineering completed')

        except Exception as CE:
            logging.error(f'Error during feature engineering: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)
