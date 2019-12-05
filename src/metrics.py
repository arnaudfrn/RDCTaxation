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


def MAE(data, actual, prediction):
    """
    Compute MAE over test set
    """
    low = data[data[actual] < 1000]
    high = data[data[actual] > 1000]

    s = np.absolute(data[actual] - data[prediction])
    s_low = np.absolute(low[actual] - low[prediction])
    s_high = np.absolute(high[actual] - high[prediction])

    met = [(np.sum(elem) / elem.size) for elem in [s, s_low, s_high]]

    print('MAE overall is {:.1f} \n MAE under 1000$ is {:.1f} \n MAE over 1000$ is {:.1f}'\
          .format(met[0], met[1], met[2]))
    return None

def group(series):
    if series < 500:
        return 0
    elif 500 <= series < 1000:
        return 1
    elif 1000 <= series < 2000:
        return 2
    elif 2000 <= series:
        return 3

def group_error(series):
    if series < 500:
        return 0
    elif 500 <= series < 1000:
        return 1
    elif 1000 <= series < 2000:
        return 2
    elif 2000 <= series < 5000:
        return 3
    elif 5000 <= series < 10000:
        return 4
    elif 10000 <= series < 25000:
        return 5
    elif 25000 <= series:
        return 6