import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.logger import logging
from utils.exception import CustomException
from utils.pickle_file import pickle_file
from utils.unpickle_file import unpickle_file

class Encoding:
    def __init__(self) -> None:
        pass

    def encode(self, engineered_dataframe_file_path:str):
        """
        Encodes categorical variables using Label Encoding.

        Parameters:
        - engineered_dataframe_file_path (str): File path to the pickled file containing the engineered DataFrame.

        Raises:
        - CustomException: If an error occurs during the encoding process.
        """
        try:
            logging.info('Encoding initiated')

            logging.info(f'Loading engineered DataFrame from {engineered_dataframe_file_path}')
            dataframe = unpickle_file(engineered_dataframe_file_path)

            LE = LabelEncoder()
            dataframe['isBestSeller'] = dataframe['isBestSeller'].astype(int)
            dataframe['category'] = LE.fit_transform(dataframe['category'])

            logging.info('Encoding completed')
            
            logging.info('Saving encoded dataFrame and encoder object')
            pickle_file(object=LE, file_name='label_encoder.pkl')
            pickle_file(object=dataframe, file_name='dataframe.pkl')

        except Exception as CE:
            logging.error(f'Error during data preprocessing: {str(CE)}', exc_info=True)
            raise CustomException(CE, sys)
