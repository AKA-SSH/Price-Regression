import sys
from catboost import CatBoostRegressor

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class ModelTrainer:
    def __init__(self) -> None:
        pass

    def train_model(self, train_features_file_path: str, train_target_file_path: str):
        """
        Train a CatBoostRegressor model using the provided features and target data.

        Parameters:
        - train_features_file_path (str): File path to the pickled file containing the training features.
        - train_target_file_path (str): File path to the pickled file containing the training target.

        Raises:
        - CustomException: If an error occurs during the model training process.
        """
        try:
            logging.info('Model training initiated')

            train_features = unpickle_file(train_features_file_path)
            train_target = unpickle_file(train_target_file_path)

            selected_parameters= {'iterations': 1682, 
                                  'depth': 4, 
                                  'learning_rate': 0.17464278943281045, 
                                  'random_strength': 0.5187807297310542, 
                                  'bagging_temperature': 0.9983467903906863, 
                                  'border_count': 229}

            CBR = CatBoostRegressor(**selected_parameters, verbose=0)
            CBR.fit(train_features, train_target)

            pickle_file(object=CBR, file_name='model.pkl')

            logging.info('Model training completed')

        except Exception as e:
            logging.error(f'Error during model training: {str(e)}', exc_info=True)
            raise CustomException(e, sys)
