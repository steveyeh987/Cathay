# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:38:41 2017

@author: dadayeh
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

#cctxn
cctxn = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_cctxn.csv', index_col=0)
#le = LabelEncoder()
#c = cctxn.apply(le.fit_transform)
cc = MultiColumnLabelEncoder().fit_transform(cctxn)
cc.to_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_cctxn_numeric.csv',header=False,index=False)

#atm
atm = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_atm.csv', index_col=0)
a = MultiColumnLabelEncoder().fit_transform(atm)
a.to_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_atm_numeric.csv',header=False,index=False)

#cti
cti = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_cti.csv', index_col=0)
ct = MultiColumnLabelEncoder().fit_transform(cti)
ct.to_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_cti_numeric.csv',header=False,index=False)

#mybank
mybank = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_mybank.csv', index_col=0)
my = MultiColumnLabelEncoder().fit_transform(mybank)
my.to_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_mybank_numeric.csv',header=False,index=False)

#customer_profile
profile1 = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_profile_partition_time=3696969600.csv', index_col=0)
pr1 = MultiColumnLabelEncoder().fit_transform(profile1)
pr1.to_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_profile_partition_time=3696969600_numeric.csv',header=False,index=False)

profile2 = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_profile_partition_time=3699475200.csv', index_col=0)
pr2 = MultiColumnLabelEncoder().fit_transform(profile2)
pr2.to_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_profile_partition_time=3699475200_numeric.csv',header=False,index=False)
