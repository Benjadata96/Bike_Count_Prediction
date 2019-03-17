#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:45:17 2019

@author: benjaminsalem
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100
import matplotlib.pyplot as plt

mpl.style.use('seaborn')
import seaborn as sns

class Model():

    def __init__(self, namee, train_test_split, cross_valid_fold, is_reg):
        self.tt_split = train_test_split
        self.cv_fold = cross_valid_fold
        self.is_reg = is_reg
        self.name = namee
        self.model = None
        self.inputs = None
        self.outputs = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None
        print('..Model '+str(self.name)+' initialized..')
        print('..Test set will be a '+ str(self.tt_split) +' ratio of the dataset..')
        print('..There will be '+ str(self.cv_fold) +' folds in the cross-validation..')
        
    def prepare_inputs_outputs(self,df):
        
        if self.is_reg == 0:
            col = ['season','holiday','workingday','weather','humidity', 'year','month','day',
            'hour','temp','atemp','windspeed']
            print('..Gradient Boosting inputs initialized..')
        
        if self.is_reg == 1:
            col=['season1','season2','season3','season4','holiday','workingday',
                 'weather1','weather2','weather3','weather4','humidity','year',
                 'month','day','hour','temp','atemp','windspeed']
            print('..Regression inputs initialized..')

        self.inputs = df[col]
        self.outputs = df['count'].values

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.inputs, self.outputs, random_state = 0, test_size =self.tt_split) 
        print('..Inputs & outputs computed..')
        
    def training_model(self,param_grid):
        
        if self.is_reg ==0:
            self.model = GridSearchCV(GBR(random_state=0),param_grid = param_grid, cv=self.cv_fold)
            print('..Gradient Boosting model initialized..')
        if self.is_reg == 1:
            self.model = GridSearchCV(Ridge(random_state=0),param_grid = param_grid, cv=self.cv_fold)
            print('..Regression model initialized..')
        
        self.model.fit(self.X_train, self.Y_train)
        print('..Model is trained..')
        print("Best cross-validation score: {:.2f}".format(self.model.best_score_))
        print("Best hyperparameters: ", self.model.best_params_)
        print("Best model : ", self.model.best_estimator_)
        print("Test set score: {:.2f}".format(self.model.score(self.X_test, self.Y_test)))
        
    def plot_feature_importances(self):
        
        print('..Plotting features importance in the model..')
        n_features = self.inputs.shape[1]
        if self.is_reg == 0:
            plt.barh(np.arange(n_features), self.model.best_estimator_.feature_importances_, align='center')
            plt.xlabel("Feature importance")
        else : 
            plt.barh(np.arange(n_features), self.model.best_estimator_.coef_, align='center')
            plt.xlabel("Feature coefficients")
        plt.yticks(np.arange(n_features), self.inputs.columns) 
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        
    def main(self,df, param_grid):
        self.prepare_inputs_outputs(df)
        self.training_model(param_grid)
        self.plot_feature_importances()
        print('..Features importance plotted..')






