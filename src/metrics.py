import pandas as pd 
import numpy as np

def mape_scoring(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def MAPE(data, actual, prediction):
    """
    Compute MAPE over test set
    """
    low = data[data[actual] < 1000]
    high = data[data[actual] > 1000]
    
    
    s = np.absolute(data[actual] - data[prediction])/data[actual]
    s_low = np.absolute(low[actual] - low[prediction])/low[actual]
    s_high = np.absolute(high[actual] - high[prediction])/high[actual]
    
    met = [(np.sum(elem)/elem.size)*100 for elem in [s, s_low, s_high]]
    
    print('MAPE overall is {:.4f} \n MAPE under 1000$ is {:.4f} \n MAPE over 1000$ is {:.4f}'.format(met[0], met[1], met[2]))
    return None