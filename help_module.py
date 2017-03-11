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

def create_graph_test_prediction_vs_data(data_no_na,traintest_threshold,y_hat_test_sklearn,
                                         with_statsmodel = False,y_hat_test_stats=None):

    test = data_no_na[data_no_na['Year'] >= traintest_threshold]

    plot_data = test[['Year', 'WageGaP']]
    plot_data = plot_data.groupby('Year').mean()

    cols = test.columns.tolist()
    test_2 = test.reset_index(drop=True)
    test_2.columns = cols

    df_y_hat_sklearn = pd.DataFrame(data=y_hat_test_sklearn, columns=['WageGaPPredict'])

    df_prediction_sklearn_only_test = pd.concat([test_2, df_y_hat_sklearn], axis=1, join='inner')
    plot_Prediction_sklearn_only_test = df_prediction_sklearn_only_test[['Year', 'WageGaPPredict']]
    plot_Prediction_sklearn_only_test = plot_Prediction_sklearn_only_test.groupby('Year').mean()

    plot_Prediction_stats_only_test= None
    if with_statsmodel:
        df_y_hat_stats = pd.DataFrame(data=y_hat_test_stats,columns=['WageGaPPredict'])

        df_prediction_stats_only_test = pd.concat([test_2,df_y_hat_stats], axis=1, join='inner')
        plot_Prediction_stats_only_test = df_prediction_stats_only_test[['Year','WageGaPPredict']]
        plot_Prediction_stats_only_test = plot_Prediction_stats_only_test.groupby('Year').mean()

    lines = plt.plot(plot_data.index, plot_data.WageGaP, color='r')
    if with_statsmodel:
        lines = plt.plot(plot_Prediction_stats_only_test.index,plot_Prediction_stats_only_test.WageGaPPredict,color='b')
        lines = plt.plot(plot_Prediction_sklearn_only_test.index,plot_Prediction_sklearn_only_test.WageGaPPredict,color='g')
    else:
        lines = plt.plot(plot_Prediction_sklearn_only_test.index, plot_Prediction_sklearn_only_test.WageGaPPredict,
                     color='b')

    plt.ylabel('Average Wage Gap ratio')
    plt.xlabel('Years')
    if with_statsmodel:
        plt.title('Average Wage Gap prediction (green-sklearn; blue-modelstats) vs data(red)-only test')
    else:
        plt.title('Average Wage Gap prediction (blue-sklearn) vs data(red)-only test')
    plt.ylim(0, 30)
    plt.xlim(traintest_threshold, 2015)
    plt.ticklabel_format(useOffset=False)
    plt.show()

def create_graph_prediction_vs_data(data_no_na,y_hat_sklearn):

    plot_data = data_no_na[['Year', 'WageGaP']]
    plot_data = plot_data.groupby('Year').mean()


    df_y_hat_sklearn = pd.DataFrame(data=y_hat_sklearn, columns=['WageGaPPredict'])

    df_prediction_and_data_sklearn = pd.concat([data_no_na, df_y_hat_sklearn], axis=1, join='inner')
    plot_Prediction_sklearn = df_prediction_and_data_sklearn[['Year', 'WageGaPPredict']]
    plot_Prediction_sklearn = plot_Prediction_sklearn.groupby('Year').mean()

    lines = plt.plot(plot_data.index, plot_data.WageGaP, color='r')
    lines = plt.plot(plot_Prediction_sklearn.index, plot_Prediction_sklearn.WageGaPPredict, color='b')

    plt.ylabel('Average Wage Gap ratio')
    plt.xlabel('Years')
    plt.title('Average Wage Gap prediction(blue) vs data(red)')
    plt.ylim(0, 100)
    plt.show()

def create_error_graph_test_pred_vs_data(data_no_na,traintest_threshold,y_hat_test_sklearn):

    test = data_no_na[data_no_na['Year'] >= traintest_threshold]

    cols = test.columns.tolist()
    test_2 = test.reset_index(drop=True)
    test_2.columns = cols

    df_y_hat_sklearn_only_test = pd.DataFrame(data=y_hat_test_sklearn, columns=['WageGaPPredict'])

    df_prediction_and_data_sklearn = pd.concat([test_2, df_y_hat_sklearn_only_test], axis=1, join='inner')

    df_data_sklearn = df_prediction_and_data_sklearn[['Year', 'WageGaPPredict']]
    df_prediction_sklearn = df_prediction_and_data_sklearn[['Year', 'WageGaP']]

    df_data_sklearn_mean = df_data_sklearn.groupby('Year').mean()
    df_prediction_sklearn_mean = df_prediction_sklearn.groupby('Year').mean()

    df_prediction_and_data_sklearn_mean = pd.concat([df_data_sklearn_mean, df_prediction_sklearn_mean], axis=1,
                                                    join='inner')

    precentage_error_sklearn = ((abs(
        df_prediction_and_data_sklearn_mean['WageGaPPredict'] - df_prediction_and_data_sklearn_mean['WageGaP']) /
                                 df_prediction_and_data_sklearn_mean['WageGaP']) * 100)
    df_precentage_error_sklearn = pd.DataFrame(data=precentage_error_sklearn, columns=['PrecentageErr'])

    plot_precentageErr_sklearn = pd.concat([df_prediction_and_data_sklearn_mean, df_precentage_error_sklearn], axis=1,
                                           join='inner')

    lines = plt.plot(plot_precentageErr_sklearn.index, plot_precentageErr_sklearn.PrecentageErr, color='g')

    plt.ylabel('Error in Precentage over the years')
    plt.xlabel('Years')
    plt.title('Average Precentage Error ')
    plt.ylim(0, 40)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    mean_err = np.mean(plot_precentageErr_sklearn.PrecentageErr)
    max_err = np.max(plot_precentageErr_sklearn.PrecentageErr)

    print('average error in precentage: %.2f' % mean_err)
    print('max error in precentage: %.2f' % max_err)


def plot_average_error(data_no_na,y_hat_test_sklearn,y_hat_sklearn):

    df_y_hat_sklearn_only_test = pd.DataFrame(data=y_hat_test_sklearn, columns=['WageGaPPredict'])

    df_y_hat_sklearn = pd.DataFrame(data=y_hat_sklearn, columns=['WageGaPPredict'])

    df_prediction_and_data_sklearn = pd.concat([data_no_na, df_y_hat_sklearn], axis=1, join='inner')

    df_data_sklearn = df_prediction_and_data_sklearn[['Year', 'WageGaPPredict']]
    df_prediction_sklearn = df_prediction_and_data_sklearn[['Year', 'WageGaP']]

    df_data_sklearn_mean = df_data_sklearn.groupby('Year').mean()
    df_prediction_sklearn_mean = df_prediction_sklearn.groupby('Year').mean()

    df_prediction_and_data_sklearn_mean = pd.concat([df_data_sklearn_mean, df_prediction_sklearn_mean], axis=1,
                                                    join='inner')

    precentage_error_sklearn = ((abs(
        df_prediction_and_data_sklearn_mean['WageGaPPredict'] - df_prediction_and_data_sklearn_mean['WageGaP']) /
                                 df_prediction_and_data_sklearn_mean['WageGaP']) * 100)
    df_precentage_error_sklearn = pd.DataFrame(data=precentage_error_sklearn, columns=['PrecentageErr'])

    plot_precentageErr_sklearn = pd.concat([df_prediction_and_data_sklearn_mean, df_precentage_error_sklearn], axis=1,
                                           join='inner')

    lines = plt.plot(plot_precentageErr_sklearn.index, plot_precentageErr_sklearn.PrecentageErr, color='g')

    plt.ylabel('Error in Precentage over the years')
    plt.xlabel('Years')
    plt.title('Average Precentage Error ')
    plt.ylim(0, 20)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    mean_err = np.mean(plot_precentageErr_sklearn.PrecentageErr)
    max_err = np.max(plot_precentageErr_sklearn.PrecentageErr)

    print('average error in precentage: %.2f' % mean_err)
    print('max error in precentage: %.2f' % max_err)
