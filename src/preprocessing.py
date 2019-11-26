import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin


def smote_apply(X_train, y_train, threshold=5000):
    """
    Create balanced dataset for value > and < to treshold. Aims at imprving accuracy on high value houses

    :X_train pd.DataFrame of trainisng set
    :y_train pd.DataFrame of traget variable
    :threshold value separating the two class to balance

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    """
    X_train.loc[:, 'target'] = y_train
    col_names = list(X_train.columns)
    bool_target = np.where(X_train['target'] > threshold, 1, 0)

    # Apply SMOTE Method
    smote = SMOTE(random_state=12, ratio=1.0)
    X_train_new, bool_target = smote.fit_sample(X_train, bool_target)
    X_train = pd.DataFrame(X_train_new, columns=col_names)
    y_train = X_train['target'].values.ravel()

    return X_train.drop(['target'], axis=1), y_train


class SmoteRegression(BaseEstimator, TransformerMixin):
    """
    Create balanced dataset for value > and < to treshold. Aims at imprving accuracy on high value houses
    We develop a method similar to SMOGN
    Based on: http://proceedings.mlr.press/v74/branco17a/branco17a.pdf

    Example:

        sm = SmoteRegression(threshold=2000, ratio=1.0)
        X, y = sm.fit(X=X_train, y=y_train)  ## Check if Threshold is fine at the given level
        X, y = sm.fit_sample(X=X_train, y=y_train)

    """
    def __init__(self, threshold=2000, ratio=1.0, random_state=0):
        """

        :param threshold: Threshold above which you over-sample your individuals
        :param ratio: Ratio of (individual below threshold)/ (individual over threshold)
        :param random_state:

        """
        self.threshold = threshold
        self.ratio = ratio
        self.random_state = random_state
        self.smote_ = None

    def fit(self, X, y=None):
        """
        Fit the model to check that  SMOTE class is instanciated correctly - Can be some problem with threshold if not
        correctly set (like (individual below threshold)/ (individual over threshold) > ratio )

        :param X: pd.DataFrame of input data
        :param y: vector np.Array of pd.Series of taregt variable
        :return: self
        """
        # Instantiate SMOTE from ImbLearn
        self.smote_ = SMOTE(random_state=self.random_state, ratio=self.ratio)

        return self

    def fit_sampple(self, X, y):
        """
        Fit and resample the data to have a new dataset of size
        (individual below threshold) + (individual below threshold) * ratio

        :param X: pd.DataFrame of input data
        :param y: vector np.Array of pd.Series of taregt variable
        :return: X : pd.DataFrame of resampled data
                 y : vector np.Array of pd.Series of target variable
        """

        self.fit(X=X, y=y)
        # Add assertion

        X.loc[:, 'target'] = y
        col_names = list(X.columns)
        bool_target = np.where(X['target'] > self.threshold, 1, 0)

        X, bool_target = self.smote_.fit_sample(X, bool_target)
        X = pd.DataFrame(X, columns=col_names)
        y = X['target'].values.ravel()
        return X, y





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




def scaling_apply(X_train, X_test):
    #Apply standard scaler
    col_names = list(X_train.columns)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), columns = col_names)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = col_names)
    return X_train, X_test

