import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE

def impute_basic(df):
    return df.replace(np.nan, -1)

def feature_preselection(df):
    """
    prepare dataset for train test split by removing highly correlated feature and id columns
    
    :df pd.DataFrame of input data
    output: 
    X -> pd.dataFrame with 1-1 95% Correlated columns removed and filtering for id columns 'a7', 'code', 'training'
    y -> np.Array of Value 
    """
    #Split target/train variable
    y = df[['value']].values.ravel()
    
    df_x = df[[i for i in list(df.columns) if i not in ['value','a7','code','training']]]
    
    # Create correlation matrix
    corr_matrix = df_x.corr()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
    
    # Subselect out very correlated variables
    X = df_x[[var for var in list(df_x.columns) if var not in to_drop ]]
    
    return X,y

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


def smote_apply(X_train, y_train, threshold = 5000):   
    """
    Create balanced dataset for value > and < to treshold. Aims at imprving accuracy on high value houses

    :X_train pd.DataFrame of trainisng set 
    :y_train pd.DataFrame of traget variable
    :threshold value separating the two class to balance

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    """
    X_train.loc[:,'target'] = y_train
    col_names = list(X_train.columns)
    bool_target = np.where(X_train['target'] > threshold, 1, 0)

    #Apply SMOTE Method
    smote = SMOTE(random_state=12, ratio = 1.0)
    X_train_new, bool_target = smote.fit_sample(X_train, bool_target)
    X_train = pd.DataFrame(X_train_new, columns = col_names)
    y_train = X_train['target'].values.ravel()
    
    return X_train.drop(['target'], axis = 1), y_train


def scaling_apply(X_train, X_test):
    #Apply standard scaler
    col_names = list(X_train.columns)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), columns = col_names)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = col_names)
    return X_train, X_test

def create_mlp(dim):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(8, input_dim=dim, activation="relu"))
    #model.add(tf.keras.layers.Dense(4, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return model

def mape_scoring(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

