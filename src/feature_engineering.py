import pandas as pd 
import numpy as np
from src.metrics import  MAPE
from src.preprocessing import impute_basic, feature_preselection, smote_apply
from sklearn import base
from sklearn.model_selection import KFold
from haversine import haversine, Unit


def dist_feats(df, feat_name):   
    df['sum_dist'] = df[dist_feats_int].sum(axis=1)
    return df


class KFoldTargetEncoder(base.BaseEstimator, base.TransformerMixin):
"""
Bugged
"""
    def __init__(self, n_fold=5, verbosity=False, discardOriginal_col=False):
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
        self.colnames = None
        self.targetName = None
        self._y = None
        self._X = None
        self.encodedName = None


    def fit(self, X, y, colnames):
        self._y = y
        self.colnames = colnames

        assert (type(self.colnames) == str)
        assert (self.colnames in X.columns)

        self.targetName = 'target_'
        X = pd.concat([X, pd.Series(self._y, name='target_')], axis=1)
        mean_of_target = X[self.targetName].mean()
        max_of_target = X[self.targetName].max()
        min_of_target = X[self.targetName].min()

        kf = KFold(n_splits=self.n_fold, shuffle=False, random_state=2019)

        col_mean_name = self.colnames + '_' + 'mean'
        col_max_name = self.colnames + '_' + 'max'
        col_min_name = self.colnames + '_' + 'min'
        X[col_mean_name] = np.nan
        X[col_max_name] = np.nan
        X[col_min_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].mean())
            X.loc[X.index[val_ind], col_max_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].max())
            X.loc[X.index[val_ind], col_min_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].min())

        X[col_mean_name].fillna(mean_of_target, inplace=True)
        X[col_max_name].fillna(max_of_target, inplace=True)
        X[col_min_name].fillna(min_of_target, inplace=True)
        X.drop(self.targetName, axis=1, inplace=True)

        self._X = X
        self.encodedName = [col_mean_name, col_min_name, col_max_name]

        return self


    def transform(self, X, y=None):
        if y is None:
            for encod in self.encodedName:
                mean = self._X[[self.colnames, encod]].groupby(self.colnames).mean().reset_index()

                dd = {}
                for index, row in mean.iterrows():
                    dd[row[self.colnames]] = row[encod]

                X[encod] = X[self.colnames]
                X = X.replace({encod: dd})
            return X

        else:

            X = self._X
            if self.verbosity:
                encoded_feature = X[col_mean_name].values
                print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values,
                                                                                                  encoded_feature)[0][1]))
            if self.discardOriginal_col:
                X = X.drop(self.colnames, axis=1)

            return X


class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train, colNames, encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def fit(self, X, y=None):
        return self    

    def transform(self,X):        
        mean =  self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]        
            X[self.encodedName] = X[self.colNames]
        X = X.applymap(dd.get)  
        return X



def NeightborLessThan(arr, n, thresh): 
    
    ind = []
    for i in range(0, n):  
        if arr[i] < thresh and arr[i] != 0:
            ind.append(i)
       
    return ind


def Train_Dist_N(df, thresh):
    arr_l = data[['longitude','latitude']].to_numpy()
    
    empty = np.zeros((arr_l.shape[0],arr_l.shape[0]))
    #### change this for test set 
    for i in range(arr_l.shape[0]):
        for j in range(arr_l.shape[0]):
            empty[i,j] = haversine(arr_l[i], arr_l[j]) #long lat of 1 point
            
    mean_value = []
    for k in range(arr_l.shape[0]):
        ind = NeightborLessThan(arr = empty[k], n = len(empty[k]), thresh = thresh)
        mean_value.append(data.loc[ind , 'value'].mean())
    
    return mean_value

def Get_Dist_N(X_train, y_train, thresh, X_test=None, y_test=None):
    """
    function computing the average price of the houses in a given radius for train and test set
    Input train: df with longitude, latitude and value 
    Input thresh: max distance to houses to consider in KM
    Input test: not mandatory, if nothing passed compute only on the traning, if dataframe passed, return computation for the test set 
                based on value and distance from the training set
    
    output mean value: list of value to be concatenated on datasets
    """
    y_train = pd.DataFrame(y_train).rename(columns={0: "value"})
    train = pd.concat([X_train, y_train], axis=1)

    arr_l = train[['longitude', 'latitude']].to_numpy()
    
    if X_test is None:
        empty = np.zeros((arr_l.shape[0],arr_l.shape[0]))

        for i in range(arr_l.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i, j] = haversine(arr_l[i], arr_l[j])  # long lat of 1 point
        
        mean_value = []
        for k in range(arr_l.shape[0]):
            ind = NeightborLessThan(arr = empty[k], n = len(empty[k]), thresh = thresh)
            mean_value.append(train.reset_index(drop = True).loc[ind , 'value'].mean())
                
    else:
        y_test = pd.DataFrame(y_test).rename(columns={0 : "value"})
        test = pd.concat([X_test.reset_index(drop=True), y_test], axis=1)

        arr_test = test[['longitude', 'latitude']].to_numpy()
        empty = np.zeros((arr_test.shape[0], arr_l.shape[0]))

        for i in range(arr_test.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i, j] = haversine(arr_test[i], arr_l[j])  # long lat of 1 point
                
        mean_value = []
        for k in range(arr_test.shape[0]):
            ind = NeightborLessThan(arr=empty[k], n=len(empty[k]), thresh=thresh)
            mean_value.append(train.reset_index(drop=True).loc[ind, 'value'].mean())
        
    return mean_value



def group(series):
    if series < 500:
        return 0
    elif 500 <= series < 1000:
        return 1
    elif 1000 <= series < 2000:
        return 2
    elif 2000 <= series:
        return 3




class AvgValueKNeighbors(BaseEstimator, TransformerMixin):
"""
Class to compute mean price of K nearest houses in terms of geographical positioning
The class implements cKDTree which are binary tree optimised for nearest neightboor lookup:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

Example:

test = AvgValueKNeighbors(k=2)
test.fit_transform(X, y)

"""
    def __init__(self, k=3):
        """
        :param k: Number of neighbors to consider in the computation
        """
        self.k = k
        self.fitted = False
        self.tree = None
        self.col = None
        self._y = None
        self.arr = None

    def fit(self, X, y, col=['longitude','latitude']):

        """
        Sub-select the column
        Fit the KDTree and save vector y for usage in testing mode if necessary

        :param X: pd.DataFrame of shape [n_individual]x[nb_features]
        :param y: np.array of shape [n_individual]x1
        :param col: str, column name to be selected for nearest neighbors
        :return: self
        """

        self._y = y
        self.col = col
        self.arr = X[self.col].to_numpy()
        self.tree = scipy.spatial.cKDTree(self.arr)
        return self

    def transform(self, X, y=None):
        """
        Query tree for k nearest neighbors, compute mean value of selected indexes and return mean house value for
        closest house

        If y is passed, the X will be considered same as training, if no y is passed X will be considered testing data
        :param X: pd.DataFrame, either training data or testing data
        :param y: np.Array
        :return:
        """
        mean_value = []
        if y is not None:
            _, indexes = self.tree.query(self.arr, k=self.k)
            #find points and do averages in arr
            for row in range(indexes.shape[0]):
                mean_value.append(np.mean(self._y[indexes[row]]))
        else:
            arr_test = X[self.col].to_numpy()
            _, indexes = self.tree.query(arr_test, k=self.k)
            # find points and do averages in arr
            for row in range(indexes.shape[0]):
                mean_value.append(np.mean(self._y[indexes[row]]))
        return mean_value


