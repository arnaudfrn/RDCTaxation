import pandas as pd 
import numpy as np
from src.compute import impute_basic, MAPE, feature_preselection, smote_apply
from sklearn import base
from sklearn.model_selection import KFold



def dist_feats(df, feat_name):   
    df['sum_dist'] = df[dist_feats_int].sum(axis=1)
    return df


def preprocessing_master(df, feats_list, val_set = True):
    df = impute_basic(df)
    #3 - Feature engineering
    df_a7 = df.groupby('a7').agg(a_mean = ('value', 'mean'),
                                  a_max = ('value', 'max'),
                                  a_min = ('value', 'min')).reset_index()

    df_commune = df.groupby('commune').agg(com_mean = ('value', 'mean'),
                                  com_max = ('value', 'max'),
                                  com_min = ('value', 'min')).reset_index()

    df_geostrat = df.groupby('geo_stratum').agg(geo_mean = ('value', 'mean'),
                                  geo_max = ('value', 'max'),
                                  geo_min = ('value', 'min')).reset_index()

    df = dist_feats(df)
    #Merge all
    df = (df
     .merge(df_a7, on = ['a7'])
     .merge(df_geostrat, on = ['geo_stratum'])
     .merge(df_commune, on = ['commune']))
    
    
    y = df['value'].values.ravel()
    X = df[feats_list]
    return X,y




class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames,targetName,n_fold=5,verbosity=False,discardOriginal_col=False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self


    def transform(self,X):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        max_of_target = X[self.targetName].max()
        min_of_target = X[self.targetName].min()
        
        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=2019)

        col_mean_name = self.colnames + '_' + 'mean'
        col_max_name = self.colnames + '_' + 'max'
        col_min_name = self.colnames + '_' + 'min'
        X[col_mean_name] = np.nan
        X[col_max_name] = np.nan
        X[col_min_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
            X.loc[X.index[val_ind], col_max_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].max())
            X.loc[X.index[val_ind], col_min_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].min())

        X[col_mean_name].fillna(mean_of_target, inplace = True)
        X[col_max_name].fillna(max_of_target, inplace = True)
        X[col_min_name].fillna(min_of_target, inplace = True)

        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
            

        return X



def NeightborLessThan(arr, n, thresh): 
    
    ind = []
    for i in range(0, n):  
        if arr[i] < thresh and arr[i] != 0:
            ind.append(i)
       
    return ind



def Find3Smallest(arr, n): 
    MAX. = 1000000

    firstmin = MAX
    secmin = MAX
    thirdmin = MAX
    ind_fist = 0 
    ind_sec = 0 
    ind_third = 0
  
    for i in range(0, n):  
        # Check if current element 
        # is less than firstmin,  
        # then update first,second 
        # and third 
  
        if arr[i] < firstmin and arr[i] != 0:
            thirdmin = secmin 
            secmin = firstmin 
            firstmin = arr[i]
            ind_fist = i
  
        # Check if current element is 
        # less than secmin then update 
        # second and third 
        elif arr[i] < secmin and arr[i] != 0: 
            thirdmin = secmin 
            secmin = arr[i]
            ind_sec = i
  
        # Check if current element is 
        # less than,then upadte third 
        elif arr[i] < thirdmin and arr[i] != 0: 
            thirdmin = arr[i] 
            ind_third = i
  
    #print("First min = ", firstmin) 
    #print("Second min = ", secmin) 
    #print("Third min = ", thirdmin) 
    return [ind_fist, ind_sec, ind_third]


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


def Get_3_NN(train, test = None):
    """
    function computing the average price of the 3 closest houses for train and test set
    Input train: df with longitude, latitude and value 
    Input test: not mandatory, if nothing passed compute only on the traning, if dataframe passed, return computation for the test set 
                based on value and distance from the training set
    
    output mean value: list of value to be concatenated on datasets
    """
    arr_l = train[['longitude','latitude']].to_numpy()
    
    if test is None:
        empty = np.zeros((arr_l.shape[0],arr_l.shape[0]))

        for i in range(arr_l.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_l[i], arr_l[j]) #long lat of 1 point
        
        mean_value = []
        for k in range(arr_l.shape[0]):
            ind = Find3Smallest(arr = empty[k], n = len(empty[k]))
            mean_value.append(train.loc[ind , 'value'].mean())
                
    else:
        arr_test = test[['longitude','latitude']].to_numpy()
        empty = np.zeros((arr_test.shape[0],arr_l.shape[0]))

        for i in range(arr_test.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_test[i], arr_l[j]) #long lat of 1 point
                
        mean_value = []
        for k in range(arr_test.shape[0]):
            ind = Find3Smallest(arr = empty[k], n = len(empty[k]))
            mean_value.append(train.loc[ind , 'value'].mean())

    return mean_value
    

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

