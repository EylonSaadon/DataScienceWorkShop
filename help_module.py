import pandas as pd
import numpy as np
import seaborn as sns
import os
import sklearn
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from  sklearn import feature_selection
from sklearn import linear_model
from sklearn import metrics
from pandas.stats.api import ols
import pickle


def create_wageGap_graph_tuple(data_no_na,countries_count,remaining_features,x):
    # create x vector
    plot_data = data_no_na
    years = set(plot_data['Year'])
    years = list(years)
    country_colums_list = list(x)[0:countries_count]
    y_graphs_dict = {}
    # split to x axis (years) and y axis(wage gap)
    for country in country_colums_list:
        y_vector = plot_data[plot_data[country] == 1]
        y_vector = pd.DataFrame(y_vector, columns=['Year', 'WageGaP'])
        y_graphs_dict[country] = y_vector

    # add average
    aggregated = plot_data.groupby('Year').mean()['WageGaP']
    aggregated_wg = aggregated.to_frame(name='WageGaP_Avg')
    np_array_gap_average = aggregated_wg['WageGaP_Avg'].values
    y_graphs_dict['WageGapAvg'] = np_array_gap_average.tolist()


    plot_tuple = ()
    feature_countries = [x for x in remaining_features if x.startswith('country')]

    # plotting the graphs with relevent colors
    for country in country_colums_list:
        color = 'r'
        if country in remaining_features:
            color = 'g'
        plot_tuple = plot_tuple + (y_graphs_dict[country].Year, y_graphs_dict[country].WageGaP, color)

    # ploting average in blue color
    plot_tuple = plot_tuple + (years, y_graphs_dict['WageGapAvg'], 'b')
    return plot_tuple, country_colums_list,feature_countries
