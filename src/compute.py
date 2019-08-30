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
    s = np.absolute(data[actual] - data[prediction])/data[actual]
    mape = (np.sum(s)/s.size)*100
    return(mape)


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
    bool_target = np.where(X_train['target'] > 5000, 1, 0)

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


