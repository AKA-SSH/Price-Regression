import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class DataManipulation:
    def __init__(self, encoded_dataframe_file_path:str):
        """
        Initialize DataManipulation object.

        Parameters:
        - encoded_dataframe_file_path (str): File path to the pickled encoded DataFrame.
        """
        dataframe = unpickle_file(encoded_dataframe_file_path)
        train_data, test_data = train_test_split(dataframe, test_size=0.2, random_state=42)
        self.train_features, self.train_target = train_data.drop('price', axis=1), train_data['price']
        self.test_features, self.test_target = test_data.drop('price', axis=1), test_data['price']

    def data_transformation(self):
        """
        Perform data transformation using PowerTransformer and save the transformer.
        """
        try:
            logging.info('Data transformation initiated')

            YJ = PowerTransformer(method='yeo-johnson', standardize=True)
            self.transformed_train_features = YJ.fit_transform(self.train_features)
            self.transformed_test_features = YJ.transform(self.test_features)
            # Pickle the transformer 
            pickle_file(object=YJ, file_name='yeo-johnson-transformer.pkl')

            logging.info('Data transformation completed')

        except Exception as e:
            logging.error(f'Error during data transformation: {str(e)}', exc_info=True)
            raise CustomException(e, sys)

    def data_scaling(self):
        """
        Perform data scaling using MinMaxScaler and save the scaler.
        """
        try:
            logging.info('Data scaling initiated')

            MM = MinMaxScaler()
            self.scaled_train_features = MM.fit_transform(self.transformed_train_features)
            self.scaled_test_features = MM.transform(self.transformed_test_features)
            # Pickle the scaler 
            pickle_file(object=MM, file_name='min-max-scaler.pkl')

            self.train_features = pd.DataFrame(data=self.scaled_train_features, index=self.train_features.index, columns=self.train_features.columns)
            self.test_features = pd.DataFrame(data=self.scaled_test_features, index=self.test_features.index, columns=self.test_features.columns)

            # Pickle the transformed/scaled features and the targets
            pickle_file(object=self.train_features, file_name='train-features.pkl')
            pickle_file(object=self.train_target, file_name='train-target.pkl')
            pickle_file(object=self.test_features, file_name='test-features.pkl')
            pickle_file(object=self.test_target, file_name='test-target.pkl')

            logging.info('Data scaling completed')

        except Exception as e:
            logging.error(f'Error during data scaling: {str(e)}', exc_info=True)
            raise CustomException(e, sys)
