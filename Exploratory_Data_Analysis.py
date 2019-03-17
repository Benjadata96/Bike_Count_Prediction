#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:18:23 2019

@author: benjaminsalem
"""
import pandas as pd
import numpy as np
import os

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100
import matplotlib.pyplot as plt

mpl.style.use('seaborn')
import seaborn as sns

class Features_Plotting():
    
    def __init__(self, training_table, continuous_features, categorical_features):
        
        self.training_table = training_table
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        
        print('..Features plotter initialized..')     
        
    def plot_categorical_instance(self, feature):
        
        self.training_table[feature].value_counts(normalize = True, dropna = False).plot(kind = 'bar')
        plt.title(feature + ' feature instances across our dataset.')
        plt.show()
        
    def plot_continuous_instance(self, feature):
        
        self.training_table[feature].hist(bins = 50)
        plt.title(feature + ' feature histogram distribution across our dataset.')
        plt.show()
    
    def plot_feature_instances(self, feature):
        
        if feature in self.categorical_features:
            self.plot_categorical_instance(feature)
        elif feature in self.continuous_features:
            self.plot_continuous_instance(feature)


class Plotter(object):
    
    def __init__(self, training_table):
        
        self.training_table = training_table
        print('..Plotter initialized..')
      
    def autolabel(self, rects, ax):
        
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
            
    def plot_bar(self, df):
        
        N = len(df.index)
        means = df['mean']
        std = df['std']

        ind = np.arange(N) 
        width = 0.35    

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, means, width, color='r', yerr=2*std)

        ax.set_title(df.index.name)
        ax.set_xticks(ind)
        ax.set_xticklabels(df.index)
        
        self.autolabel(rects1, ax)
        
        plt.show()
        
    def plot_correlation_matrix(self, size = 9):

        print('..Plotting correlation matrix..')
        corr = self.training_table.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
        
        plt.xticks(range(len(corr.columns)), corr.columns);
        plt.yticks(range(len(corr.columns)), corr.columns);
        plt.title('Correlation matrix')
        
        plt.show()
        print('..Correlation matrix plotted..')       


class DF_Modifier():
    
    def __init__(self, df_initial):
        self.df_initial = df_initial
        self.df_final = df_initial.copy()
        print('..Dataframe modifier initialized..')
    
    def convert_date(self):
        self.df_final['year'] = pd.to_datetime(self.df_initial['datetime']).dt.year
        self.df_final['month'] = pd.to_datetime(self.df_initial['datetime']).dt.month
        self.df_final['day'] = pd.to_datetime(self.df_initial['datetime']).dt.day
        self.df_final['hour'] = pd.to_datetime(self.df_initial['datetime']).dt.hour
        
        print('..Date converted to feature..')
        
    def keep_useful_columns(self):
        col = ['season','holiday','workingday','weather','year','month','day','hour','temp','atemp','windspeed','humidity','count']
        self.df_final = self.df_final[col]
        
        print('..Useless columns deleted from the dataframe..')
        
    
    def main(self):
        self.convert_date()
        self.keep_useful_columns()
        
        print('..New dataframe ready..')
        return(self.df_final)
        
        
