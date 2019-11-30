import pandas as pd 
import numpy as np
import scipy.spatial
from src.metrics import  MAPE
from src.preprocessing import impute_basic, feature_preselection, smote_apply
from sklearn import base
from sklearn.model_selection import KFold
from haversine import haversine, Unit
from sklearn.base import BaseEstimator, TransformerMixin



def dist_feats(df, feat_name):   
    df['sum_dist'] = df[feat_name].sum(axis=1)
    return df


class KFoldTargetEncoder(base.BaseEstimator, base.TransformerMixin):
    """
    Use K-Fold target encoding for categorical variable to reduce overfitting in target encoding
    Follows SKLearn API

    Example:
    For test set application
    test = KFoldTargetEncoder(n_fold=5)
    test.fit(X, y).transform(X_test)

    or

    for train set application
    test = KFoldTargetEncoder(n_fold=5)
    test.fit(X, y).transform(X, y)
    """

    def __init__(self, n_fold=5, verbosity=False, discardOriginal_col=False):
        """

        :param n_fold: Number of folds to use in the CV, default is 5
        :param verbosity: Give info at the end of training on correlation between new features and target
        Default is False
        :param discardOriginal_col: Remove original column in the Df - Default is False
        """
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
        self.colname = None
        self.targetName = None
        self._y = None
        self._X = None
        self.encodedName = None
        self.dict_target = None

    def fit(self, X, y, colname=None):
        """
        Fit the encoder in a out-of-fold manner, when fold does not contain the desired category then it impute by the
         mean across all categories

        :param X: pd.DataFrame of shape [n_individual]x[nb_features]
        :param y: np.array of shape [n_individual]x1
        :param colname: Name of column to target encode, must be string
        :return: Self
        """
        self._y = y
        self.colname = colname

        assert (type(self.colname) == str)
        assert (self.colname in X.columns)

        self.targetName = 'target_'
        X = pd.concat([X.reset_index(drop=True), pd.Series(self._y, name='target_')], axis=1)
        mean_of_target = X[self.targetName].mean()
        max_of_target = X[self.targetName].max()
        min_of_target = X[self.targetName].min()

        kf_ = KFold(n_splits=self.n_fold, shuffle=False, random_state=2019)

        col_mean_name = self.colname + '_' + 'mean'
        col_max_name = self.colname + '_' + 'max'
        col_min_name = self.colname + '_' + 'min'

        self.dict_target = {col_mean_name: mean_of_target,
                            col_max_name: max_of_target,
                            col_min_name: min_of_target}

        for colname in [col_mean_name, col_max_name, col_min_name]:
            X[colname] = np.nan

        for tr_ind, val_ind in kf_.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colname].map(
                X_tr.groupby(self.colname)[self.targetName].mean())
            X.loc[X.index[val_ind], col_max_name] = X_val[self.colname].map(
                X_tr.groupby(self.colname)[self.targetName].max())
            X.loc[X.index[val_ind], col_min_name] = X_val[self.colname].map(
                X_tr.groupby(self.colname)[self.targetName].min())

        X[col_mean_name].fillna(mean_of_target, inplace=True)
        X[col_max_name].fillna(max_of_target, inplace=True)
        X[col_min_name].fillna(min_of_target, inplace=True)
        X.drop(self.targetName, axis=1, inplace=True)

        self._X = X
        self.encodedName = [col_mean_name, col_min_name, col_max_name]

        return self

    @staticmethod
    def _list_duplicates(d):
        """
        For debugging the dict dd in self.transform(X)
        """
        seen = set()
        duplicates = set(x for x in list(d.keys()) if x in seen or seen.add(x))
        return list(duplicates)

    def transform(self, X, y=None):
        """
        If train set is passed, MUST pass y vector as well.
        If test set is passed, y is None

        :param X: pd.DataFrame of shape [n_individual]x[nb_features]
        :param y: np.array of shape [n_individual]x1
        :return: DataFrame of Xf
        """
        if y is None:
            for encod in self.encodedName:
                mean = self._X[[self.colname, encod]].groupby(self.colname).mean().reset_index()

                dd = {}
                for index, row in mean.iterrows():
                    dd[row[self.colname]] = row[encod]

                X[encod] = X[self.colname]
                # Fill by overall mean
                X.loc[:, encod] = X[encod].map(dd).fillna(self.dict_target[encod])
            return X

        else:
            X = self._X

            if self.verbosity:  ## Bug
                raise NotImplementedError

                #for encod in self.encodedName:
                 #   encoded_feature = X[encod].values
                 #   print('Correlation between the new feature, {} and, {} is {}.'.format(encod,
                 #                                                                     self.targetName,
                 #                                                                     np.corrcoef(
                 #                                                                         X[self.targetName].values,
                 #                                                                         encoded_feature)[0][1]))

            if self.discardOriginal_col:
                X = X.drop(self.colname, axis=1)

            return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Encoders that utilize the target must make sure that the training data are transformed with:
            transform(X, y)
        and not with:
            transform(X)
        """
        return self.fit(X, y, **fit_params).transform(X, y)


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


class AvgValueDistNeighbors(BaseEstimator, TransformerMixin):
    """
    Class to compute mean price of houses within a gisven distance
    The class implements cKDTree which are binary tree optimised for nearest neightboor lookup:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

    Example:

    test = AvgValueDistNeighbors(d=500)
    test.fit_transform(X, y)

    """

    def __init__(self, d=500):
        """
        :param d: radius distance to neighbors to consider in the computation
        """
        self.d = d
        self.r = None
        self.fitted = False
        self.tree = None
        self.col = None
        self._y = None
        self.arr = None

    @staticmethod
    def convert_d_to_r(d):
        """
        cKDTree take r as input, but its not user friedly.
        At Kanaga position on earth, 0.00144555 = 160.35 m

        This finction takes for input a meter distance and turn into a radian based measure
        :param d: meter distance
        :return: radian based distance
        """
        return (d / 160.35) * 0.00144555

    def fit(self, X, y, col=['longitude', 'latitude']):

        """
        Sub-select the column
        Fit the KDTree and save vector y for usage in testing mode if necessary

        Conversion from radius to distance at Kanaga location: 0.00144555 = 160.35 m

        :param X: pd.DataFrame of shape [n_individual]x[nb_features]
        :param y: np.array of shape [n_individual]x1
        :param col: str, column name to be selected for nearest neighbors
        :return: self
        """
        self.r = self.convert_d_to_r(self.d)
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
        :return: X
        """
        overall_mean = np.mean(self._y)
        mean_value = []
        if y is not None:
            indexes = self.tree.query_ball_point(self.arr, r=self.r)
            # find points and do averages in arr
            for row in range(indexes.shape[0]):
                # print(indexes[row])
                if len(indexes[row]) == 0:
                    mean_value.append(overall_mean)  # If no house close, append the overall mean price of all houses
                else:
                    mean_value.append(np.mean(self._y[indexes[row]]))


        else:
            arr_test = X[self.col].to_numpy()
            indexes = self.tree.query_ball_point(arr_test, r=self.r)

            # find points and do averages in arr
            for row in range(indexes.shape[0]):
                if len(indexes[row]) == 0:
                    mean_value.append(overall_mean)
                else:
                    mean_value.append(np.mean(self._y[indexes[row]]))

        # Create col in the original Df
        X.loc[:, 'avg_' + str(self.d) + '_nb'] = mean_value
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Encoders that utilize the target must make sure that the training data are transformed with:
            transform(X, y)
        and not with:
            transform(X)
        """
        return self.fit(X, y, **fit_params).transform(X, y)


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

    def fit(self, X, y, col=['longitude', 'latitude']):

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
        :return: X
        """
        mean_value = []
        if y is not None:
            _, indexes = self.tree.query(self.arr, k=self.k)
            # find points and do averages in arr
            for row in range(indexes.shape[0]):
                mean_value.append(np.mean(self._y[indexes[row]]))

        else:
            arr_test = X[self.col].to_numpy()
            _, indexes = self.tree.query(arr_test, k=self.k)

            # find points and do averages in arr
            for row in range(indexes.shape[0]):
                mean_value.append(np.mean(self._y[indexes[row]]))

        X.loc[:, 'avg_' + str(self.k) + '_nb'] = mean_value
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Encoders that utilize the target must make sure that the training data are transformed with:
            transform(X, y)
        and not with:
            transform(X)
        """
        return self.fit(X, y, **fit_params).transform(X, y)


