import pandas as pd 
import numpy as np
from src.compute import impute_basic, MAPE, feature_preselection, smote_apply
from sklearn import base
from sklearn.model_selection import KFold
from haversine import haversine, Unit


def dist_feats(df, feat_name):   
    df['sum_dist'] = df[dist_feats_int].sum(axis=1)
    return df



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

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train,colNames,encodedName):
        
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



def Find3Smallest(arr, n): 
    MAX = 1000000

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

def Find5Smallest(arr, n): 
    """
    function takes a 1-d array of size n and return a list of indices of the 3 smallest
    """
    MAX = 100000
    firstmin = MAX
    secmin = MAX
    thirdmin = MAX
    fourthmin = MAX
    fifthmin = MAX
    ind_fist = 0 
    ind_sec = 0 
    ind_third = 0
    ind_fourth = 0
    ind_fifth = 0
  
    for i in range(0, n):  
  
        if arr[i] < firstmin and arr[i] != 0:
            thirdmin = secmin 
            secmin = firstmin
            fourthmin = thirdmin
            fifthmin = fourthmin
            firstmin = arr[i]
            ind_fist = i

        elif arr[i] < secmin and arr[i] != 0: 
            thirdmin = secmin
            fourthmin = thirdmin
            fifthmin = fourthmin
            secmin = arr[i]
            ind_sec = i
  
        elif arr[i] < thirdmin and arr[i] != 0:
            fourthmin = thirdmin
            fifthmin = fourthmin
            thirdmin = arr[i] 
            ind_third = i
            
        elif arr[i] < fourthmin and arr[i] != 0:
            fifthmin = fourthmin 
            fourthmin = arr[i]
            ind_fourth = i
            
        elif arr[i] < fifthmin and arr[i] != 0: 
            fifthmin = arr[i]
            ind_fifth = i
            
            
  
    return [ind_fist, ind_sec, ind_third, ind_fourth, ind_fifth]


def Get_3_NN(X_train, y_train, X_test=None, y_test=None):
    """
    function computing the average price of the 3 closest houses for train and test set
    Input train: df with longitude, latitude and value 
    Input test: not mandatory, if nothing passed compute only on the traning, if dataframe passed, return computation for the test set 
                based on value and distance from the training set
    
    output mean value: list of value to be concatenated on datasets
    """
    y_train = pd.DataFrame(y_train).rename(columns={0 : "value"})
    train = pd.concat([X_train, y_train], axis=1)


    arr_l = train[['longitude','latitude']].to_numpy()
    
    if X_test is None:
        empty = np.zeros((arr_l.shape[0],arr_l.shape[0]))

        for i in range(arr_l.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_l[i], arr_l[j]) #long lat of 1 point
        
        mean_value = []
        for k in range(arr_l.shape[0]):
            ind = Find3Smallest(arr = empty[k], n = len(empty[k]))
            mean_value.append(train.loc[ind , 'value'].mean())
                
    else:

        y_test = pd.DataFrame(y_test).rename(columns={0 : "value"})
        test = pd.concat([X_test.reset_index(drop=True), y_test], axis=1)


        arr_test = test[['longitude','latitude']].to_numpy()
        empty = np.zeros((arr_test.shape[0],arr_l.shape[0]))

        for i in range(arr_test.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_test[i], arr_l[j]) #long lat of 1 point
                
        mean_value = []
        for k in range(arr_test.shape[0]):
            ind = Find3Smallest(arr = empty[k], n = len(empty[k]))
            mean_value.append(train.reset_index(drop = True).loc[ind , 'value'].mean())

    return mean_value
    



def Get_5_NN(X_train, y_train, X_test=None, y_test=None):
    """
    function computing the average price of the 5 closest houses for train and test set
    Input train: df with longitude, latitude and value 
    Input test: not mandatory, if nothing passed compute only on the traning, if dataframe passed, return computation for the test set 
                based on value and distance from the training set
    
    output mean value: list of value to be concatenated on datasets
    """
    y_train = pd.DataFrame(y_train).rename(columns={0 : "value"})
    train = pd.concat([X_train, y_train], axis=1)


    arr_l = train[['longitude','latitude']].to_numpy()
    
    if X_test is None:
        empty = np.zeros((arr_l.shape[0],arr_l.shape[0]))

        for i in range(arr_l.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_l[i], arr_l[j]) #long lat of 1 point
        
        mean_value = []
        for k in range(arr_l.shape[0]):
            ind = Find5Smallest(arr = empty[k], n = len(empty[k]))
            mean_value.append(train.loc[ind , 'value'].mean())
                
    else:

        y_test = pd.DataFrame(y_test).rename(columns={0 : "value"})
        test = pd.concat([X_test.reset_index(drop=True), y_test], axis=1)


        arr_test = test[['longitude','latitude']].to_numpy()
        empty = np.zeros((arr_test.shape[0],arr_l.shape[0]))

        for i in range(arr_test.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_test[i], arr_l[j]) #long lat of 1 point
                
        mean_value = []
        for k in range(arr_test.shape[0]):
            ind = Find5Smallest(arr = empty[k], n = len(empty[k]))
            mean_value.append(train.reset_index(drop = True).loc[ind , 'value'].mean())

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
    y_train = pd.DataFrame(y_train).rename(columns={0 : "value"})
    train = pd.concat([X_train, y_train], axis=1)

    arr_l = train[['longitude','latitude']].to_numpy()
    
    if X_test is None:
        empty = np.zeros((arr_l.shape[0],arr_l.shape[0]))

        for i in range(arr_l.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_l[i], arr_l[j]) #long lat of 1 point
        
        mean_value = []
        for k in range(arr_l.shape[0]):
            ind = NeightborLessThan(arr = empty[k], n = len(empty[k]), thresh = thresh)
            mean_value.append(train.reset_index(drop = True).loc[ind , 'value'].mean())
                
    else:
        y_test = pd.DataFrame(y_test).rename(columns={0 : "value"})
        test = pd.concat([X_test.reset_index(drop=True), y_test], axis=1)


        arr_test = test[['longitude','latitude']].to_numpy()
        empty = np.zeros((arr_test.shape[0],arr_l.shape[0]))

        for i in range(arr_test.shape[0]):
            for j in range(arr_l.shape[0]):
                empty[i,j] = haversine(arr_test[i], arr_l[j]) #long lat of 1 point
                
        mean_value = []
        for k in range(arr_test.shape[0]):
            ind = NeightborLessThan(arr = empty[k], n = len(empty[k]), thresh = thresh)
            mean_value.append(train.reset_index(drop = True).loc[ind , 'value'].mean())
        
    return mean_value