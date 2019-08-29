import pandas as pd 
import numpy as np

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
    s = np.absolute(data[actual] - data[prediction])/data[actual]
    mape = (np.sum(s)/s.size)*100
    return(mape)

    