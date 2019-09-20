import pandas as pd 
import numpy as np
from src.compute import impute_basic, MAPE, feature_preselection, smote_apply


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
