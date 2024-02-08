from tabulate import tabulate
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def evaluate_model(true, pred):
    '''
    This function returns an evaluation table with accuracy, precision, recall and F1 score

    Input:
        - True Value
        - Predicted Value
    
    Output:
        - Evaluation Table
    '''
    MAE= f'{mean_absolute_error(true, pred):.3f}'
    RMSE= f'{root_mean_squared_error(true, pred):.3f}'
    R2= f'{r2_score(true, pred):.3f}'
    table= [['RMSE', RMSE],
            ['MAE', MAE],
            ['R2', R2]]
    
    evaluation= tabulate(table, headers= ['METRIC', 'SCORE'], tablefmt= 'grid')
    return evaluation