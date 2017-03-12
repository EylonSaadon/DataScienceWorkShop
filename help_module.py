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

    print ('Total Countries {0}'.format(len(country_colums_list)))
    print ('Countries in feature list {0}'.format(len(feature_countries)))
    plt.show()

    for i in range(0, len(plot_tuple), 3):
        plt.plot(plot_tuple[i], plot_tuple[i + 1], plot_tuple[i + 2])

    plt.ylabel('Wage Gap ratio')
    plt.xlabel('Years')
    plt.title('Wage Gap VS Years for all countries')
    plt.xlim(1975, 2015)
    plt.ylim(-40, 70)
    plt.show()

def create_graph_test_prediction_vs_data(data_no_na,traintest_threshold,y_hat_test_sklearn=None,
                                         with_statsmodel = False,y_hat_test_stats=None, stats_only=False):

    test = data_no_na[data_no_na['Year'] >= traintest_threshold]

    plot_data = test[['Year', 'WageGaP']]
    plot_data = plot_data.groupby('Year').mean()

    cols = test.columns.tolist()
    test_2 = test.reset_index(drop=True)
    test_2.columns = cols

    if not stats_only:
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
        if not stats_only:
            lines = plt.plot(plot_Prediction_sklearn_only_test.index,plot_Prediction_sklearn_only_test.WageGaPPredict,color='g')
    else:
        lines = plt.plot(plot_Prediction_sklearn_only_test.index, plot_Prediction_sklearn_only_test.WageGaPPredict,
                     color='b')

    plt.ylabel('Average Wage Gap ratio')
    plt.xlabel('Years')
    if stats_only:
        plt.title('Average Wage Gap prediction (green - modelStats) vs data(red)')
    elif  with_statsmodel:
        plt.title('Average Wage Gap prediction (green-sklearn; blue-modelstats) vs data(red)-only test')
    else:
        plt.title('Average Wage Gap prediction (blue-sklearn) vs data(red)-only test')
    plt.ylim(0, 30)
    plt.xlim(traintest_threshold, 2015)
    plt.ticklabel_format(useOffset=False)
    plt.show()

def create_graph_prediction_vs_data(data_no_na,y_hat_sklearn=None ,with_statsmodel = False, y_hat_stats=None,stats_only=False):

    plot_data = data_no_na[['Year', 'WageGaP']]
    plot_data = plot_data.groupby('Year').mean()

    plot_Prediction_stats=None
    df_y_hat_stats=None
    plot_Prediction_sklearn=None

    if with_statsmodel:
        df_y_hat_stats = pd.DataFrame(data=y_hat_stats, columns=['WageGaPPredict'])
        df_prediction_stats = pd.concat([data_no_na, df_y_hat_stats], axis=1, join='inner')
        plot_Prediction_stats = df_prediction_stats[['Year', 'WageGaPPredict']]
        plot_Prediction_stats = plot_Prediction_stats.groupby('Year').mean()

    if not stats_only:
        df_y_hat_sklearn = pd.DataFrame(data=y_hat_sklearn, columns=['WageGaPPredict'])

        df_prediction_and_data_sklearn = pd.concat([data_no_na, df_y_hat_sklearn], axis=1, join='inner')
        plot_Prediction_sklearn = df_prediction_and_data_sklearn[['Year', 'WageGaPPredict']]
        plot_Prediction_sklearn = plot_Prediction_sklearn.groupby('Year').mean()

    lines = plt.plot(plot_data.index, plot_data.WageGaP, color='r')
    if with_statsmodel:
        lines = plt.plot(plot_Prediction_stats.index, plot_Prediction_stats.WageGaPPredict, color='b')
        if not stats_only:
            lines = plt.plot(plot_Prediction_sklearn.index, plot_Prediction_sklearn.WageGaPPredict, color='g')
    else:
        lines = plt.plot(plot_Prediction_sklearn.index, plot_Prediction_sklearn.WageGaPPredict, color='b')

    plt.ylabel('Average Wage Gap ratio')
    plt.xlabel('Years')
    if stats_only:
        plt.title('Average Wage Gap prediction (blue - modelStats) vs data(red)')
    elif with_statsmodel:
        plt.title('Average Wage Gap prediction (green-sklearn; blue-modelstats) vs data(red)')
    else:
        plt.title('Average Wage Gap prediction(blue) vs data(red)')
    plt.ylim(0, 100)
    plt.show()
    return df_y_hat_stats


def create_error_graph_test_pred_vs_data(data_no_na, traintest_threshold, y_hat_test_sklearn=None,
                                         stats_only=False,y_hat_reduced_test_stats=None):

    test = data_no_na[data_no_na['Year'] >= traintest_threshold]

    cols = test.columns.tolist()
    test_2 = test.reset_index(drop=True)
    test_2.columns = cols

    if stats_only:
        df_y_hat_stats = pd.DataFrame(data=y_hat_reduced_test_stats, columns=['WageGaPPredict'])

        df_prediction_and_data_stats = pd.concat([test_2, df_y_hat_stats], axis=1, join='inner')

        df_data_stats = df_prediction_and_data_stats[['Year', 'WageGaP']]
        df_prediction_stats = df_prediction_and_data_stats[['Year', 'WageGaPPredict']]

        df_data_stats_mean = df_data_stats.groupby('Year').mean()
        df_prediction_stats_mean = df_prediction_stats.groupby('Year').mean()
        df_prediction_and_data_stats_mean = pd.concat([df_data_stats_mean, df_prediction_stats_mean], axis=1,
                                                      join='inner')

        precentage_error_stats = ((abs(
            df_prediction_and_data_stats_mean['WageGaPPredict'] - df_prediction_and_data_stats_mean['WageGaP']) /
                                   df_prediction_and_data_stats_mean['WageGaP']) * 100)
        df_precentage_error_stats = pd.DataFrame(data=precentage_error_stats, columns=['PrecentageErr'])

        plot_precentageErr_stats = pd.concat([df_prediction_and_data_stats_mean, df_precentage_error_stats], axis=1,
                                             join='inner')
        lines = plt.plot(plot_precentageErr_stats.index, plot_precentageErr_stats.PrecentageErr, color='b')
    else:
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
    if stats_only:
        plt.ylim(0, 20)
    else:
        plt.ylim(0, 40)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    if stats_only:
        err_mean_stats = np.mean(plot_precentageErr_stats.PrecentageErr)
        err_max_stats = np.max(plot_precentageErr_stats.PrecentageErr)

        print('average error using modelstats in precentage: %.2f' % err_mean_stats)
        print('max error using modelstats in precentage: %.2f' % err_max_stats)
    else:

            mean_err = np.mean(plot_precentageErr_sklearn.PrecentageErr)
            max_err = np.max(plot_precentageErr_sklearn.PrecentageErr)

            print('average error in precentage: %.2f' % mean_err)
            print('max error in precentage: %.2f' % max_err)


def plot_average_error(data_no_na,y_hat_test_sklearn=None,y_hat_sklearn=None, with_statsmodel = False ,
                       y_hat_stats=None,stats_only=False):

    if not stats_only:
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

    if with_statsmodel:
        # preparing data for stats
        df_y_hat_stats = pd.DataFrame(data=y_hat_stats, columns=['WageGaPPredict'])

        df_prediction_and_data_stats = pd.concat([data_no_na, df_y_hat_stats], axis=1, join='inner')

        df_data_stats = df_prediction_and_data_stats[['Year', 'WageGaP']]
        df_prediction_stats = df_prediction_and_data_stats[['Year', 'WageGaPPredict']]

        df_data_stats_mean = df_data_stats.groupby('Year').mean()
        df_prediction_stats_mean = df_prediction_stats.groupby('Year').mean()

        df_prediction_and_data_stats_mean = pd.concat([df_data_stats_mean, df_prediction_stats_mean], axis=1,
                                                      join='inner')

        precentage_error_stats = ((abs(
            df_prediction_and_data_stats_mean['WageGaPPredict'] - df_prediction_and_data_stats_mean['WageGaP']) /
                                   df_prediction_and_data_stats_mean['WageGaP']) * 100)
        df_precentage_error_stats = pd.DataFrame(data=precentage_error_stats, columns=['PrecentageErr'])

        plot_precentageErr_stats = pd.concat([df_prediction_and_data_stats_mean, df_precentage_error_stats], axis=1,
                                             join='inner')

    if with_statsmodel:
        lines = plt.plot(plot_precentageErr_stats.index, plot_precentageErr_stats.PrecentageErr, color='b')

    if not stats_only:
        lines = plt.plot(plot_precentageErr_sklearn.index, plot_precentageErr_sklearn.PrecentageErr, color='g')

    if with_statsmodel:
        plt.ylabel('Error in Precentage over the years (green-sklearn; blue-modelstats)')
    else:
        plt.ylabel('Error in Precentage over the years')

    plt.xlabel('Years')
    plt.title('Average Precentage Error ')
    plt.ylim(0, 20)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    if not stats_only:
        err_mean_sklearn = np.mean(plot_precentageErr_sklearn.PrecentageErr)
        err_max_sklearn = np.max(plot_precentageErr_sklearn.PrecentageErr)
        print('average error in precentage: %.2f' % err_mean_sklearn)
        print('max error in precentage: %.2f' % err_max_sklearn)

    if with_statsmodel:
        err_mean_stats = np.mean(plot_precentageErr_stats.PrecentageErr)
        err_max_stats = np.max(plot_precentageErr_stats.PrecentageErr)

        print('average error using modelstats in precentage: %.2f' % err_mean_stats)
        print('max error using modelstats in precentage: %.2f' % err_max_stats)
    if stats_only:
        return plot_precentageErr_stats

def create_correlation_matrix(data_no_na,remaining_features):
    from pandas.tools.plotting import scatter_matrix
    # help df for renaming the indicator code with meaningfull names
    temp_data_row = pd.read_csv('name_to_code.csv', header=None, skiprows=1)
    temp_data_row.columns = ['index', 'Indicator Name', 'Indicator Code', 'meaning']

    correlation_cols = [x for x in remaining_features if 'country' not in x] + ['Year'] + ['WageGaP']
    features_df = data_no_na[correlation_cols]
    # replace the columns names with codes
    # create dict old-new
    names_dict = {}
    for index, row in temp_data_row.iterrows():
        names_dict[row['Indicator Name']] = row['meaning']

    features_df = features_df.rename(columns=names_dict)
    # output correlations per groups
    print ('Selecting correlation groups manually')
    group1 = ['Age dependency', 'Age population0', 'Age population1', 'Age population5', 'WageGaP']

    group2 = ['Adolescent fertility', 'Fertility rate', 'Death rate', 'WageGaP']
    group3 = ['Employment ratio f', 'Employment ratio m', 'Employment ratio', 'Employment ratio 24', 'WageGaP']
    group4 = ['GDP', 'GDP growth', 'GDP per capita', 'GNI per capita', 'GNI, Atlas', 'WageGaP']
    group5 = ['Immunization, measles', 'Life expectancy f', 'Life expectancy m', 'Mortality  infant ', 'WageGaP']
    group6 = ['Labor participation f', 'Labor participation m', 'Unemployment, youth female', 'Unemployment, youth',
              'WageGaP']

    # print scatter matrix
    corr_df1 = features_df[group1]
    scatter_matrix(corr_df1, alpha=0.7, figsize=(11, 11), diagonal='kde')

    corr_df2 = features_df[group2]
    scatter_matrix(corr_df2, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df3 = features_df[group3]
    scatter_matrix(corr_df3, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df4 = features_df[group4]
    scatter_matrix(corr_df4, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df5 = features_df[group5]
    scatter_matrix(corr_df5, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df6 = features_df[group6]
    scatter_matrix(corr_df6, alpha=0.6, figsize=(12, 12), diagonal='kde')
    return features_df


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

    print ('Total Countries {0}'.format(len(country_colums_list)))
    print ('Countries in feature list {0}'.format(len(feature_countries)))
    plt.show()

    for i in range(0, len(plot_tuple), 3):
        plt.plot(plot_tuple[i], plot_tuple[i + 1], plot_tuple[i + 2])

    plt.ylabel('Wage Gap ratio')
    plt.xlabel('Years')
    plt.title('Wage Gap VS Years for all countries')
    plt.xlim(1975, 2015)
    plt.ylim(-40, 70)
    plt.show()

def create_graph_test_prediction_vs_data(data_no_na,traintest_threshold,y_hat_test_sklearn=None,
                                         with_statsmodel = False,y_hat_test_stats=None, stats_only=False):

    test = data_no_na[data_no_na['Year'] >= traintest_threshold]

    plot_data = test[['Year', 'WageGaP']]
    plot_data = plot_data.groupby('Year').mean()

    cols = test.columns.tolist()
    test_2 = test.reset_index(drop=True)
    test_2.columns = cols

    if not stats_only:
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
        if not stats_only:
            lines = plt.plot(plot_Prediction_sklearn_only_test.index,plot_Prediction_sklearn_only_test.WageGaPPredict,color='g')
    else:
        lines = plt.plot(plot_Prediction_sklearn_only_test.index, plot_Prediction_sklearn_only_test.WageGaPPredict,
                     color='b')

    plt.ylabel('Average Wage Gap ratio')
    plt.xlabel('Years')
    if stats_only:
        plt.title('Average Wage Gap prediction (green - modelStats) vs data(red)')
    elif  with_statsmodel:
        plt.title('Average Wage Gap prediction (green-sklearn; blue-modelstats) vs data(red)-only test')
    else:
        plt.title('Average Wage Gap prediction (blue-sklearn) vs data(red)-only test')
    plt.ylim(0, 30)
    plt.xlim(traintest_threshold, 2015)
    plt.ticklabel_format(useOffset=False)
    plt.show()

def create_graph_prediction_vs_data(data_no_na,y_hat_sklearn=None ,with_statsmodel = False, y_hat_stats=None,stats_only=False):

    plot_data = data_no_na[['Year', 'WageGaP']]
    plot_data = plot_data.groupby('Year').mean()

    plot_Prediction_stats=None
    df_y_hat_stats=None
    plot_Prediction_sklearn=None

    if with_statsmodel:
        df_y_hat_stats = pd.DataFrame(data=y_hat_stats, columns=['WageGaPPredict'])
        df_prediction_stats = pd.concat([data_no_na, df_y_hat_stats], axis=1, join='inner')
        plot_Prediction_stats = df_prediction_stats[['Year', 'WageGaPPredict']]
        plot_Prediction_stats = plot_Prediction_stats.groupby('Year').mean()

    if not stats_only:
        df_y_hat_sklearn = pd.DataFrame(data=y_hat_sklearn, columns=['WageGaPPredict'])

        df_prediction_and_data_sklearn = pd.concat([data_no_na, df_y_hat_sklearn], axis=1, join='inner')
        plot_Prediction_sklearn = df_prediction_and_data_sklearn[['Year', 'WageGaPPredict']]
        plot_Prediction_sklearn = plot_Prediction_sklearn.groupby('Year').mean()

    lines = plt.plot(plot_data.index, plot_data.WageGaP, color='r')
    if with_statsmodel:
        lines = plt.plot(plot_Prediction_stats.index, plot_Prediction_stats.WageGaPPredict, color='b')
        if not stats_only:
            lines = plt.plot(plot_Prediction_sklearn.index, plot_Prediction_sklearn.WageGaPPredict, color='g')
    else:
        lines = plt.plot(plot_Prediction_sklearn.index, plot_Prediction_sklearn.WageGaPPredict, color='b')

    plt.ylabel('Average Wage Gap ratio')
    plt.xlabel('Years')
    if stats_only:
        plt.title('Average Wage Gap prediction (blue - modelStats) vs data(red)')
    elif with_statsmodel:
        plt.title('Average Wage Gap prediction (green-sklearn; blue-modelstats) vs data(red)')
    else:
        plt.title('Average Wage Gap prediction(blue) vs data(red)')
    plt.ylim(0, 100)
    plt.show()
    return df_y_hat_stats


def create_error_graph_test_pred_vs_data(data_no_na, traintest_threshold, y_hat_test_sklearn=None,
                                         stats_only=False,y_hat_reduced_test_stats=None):

    test = data_no_na[data_no_na['Year'] >= traintest_threshold]

    cols = test.columns.tolist()
    test_2 = test.reset_index(drop=True)
    test_2.columns = cols

    if stats_only:
        df_y_hat_stats = pd.DataFrame(data=y_hat_reduced_test_stats, columns=['WageGaPPredict'])

        df_prediction_and_data_stats = pd.concat([test_2, df_y_hat_stats], axis=1, join='inner')

        df_data_stats = df_prediction_and_data_stats[['Year', 'WageGaP']]
        df_prediction_stats = df_prediction_and_data_stats[['Year', 'WageGaPPredict']]

        df_data_stats_mean = df_data_stats.groupby('Year').mean()
        df_prediction_stats_mean = df_prediction_stats.groupby('Year').mean()
        df_prediction_and_data_stats_mean = pd.concat([df_data_stats_mean, df_prediction_stats_mean], axis=1,
                                                      join='inner')

        precentage_error_stats = ((abs(
            df_prediction_and_data_stats_mean['WageGaPPredict'] - df_prediction_and_data_stats_mean['WageGaP']) /
                                   df_prediction_and_data_stats_mean['WageGaP']) * 100)
        df_precentage_error_stats = pd.DataFrame(data=precentage_error_stats, columns=['PrecentageErr'])

        plot_precentageErr_stats = pd.concat([df_prediction_and_data_stats_mean, df_precentage_error_stats], axis=1,
                                             join='inner')
        lines = plt.plot(plot_precentageErr_stats.index, plot_precentageErr_stats.PrecentageErr, color='b')
    else:
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
    if stats_only:
        plt.ylim(0, 20)
    else:
        plt.ylim(0, 40)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    if stats_only:
        err_mean_stats = np.mean(plot_precentageErr_stats.PrecentageErr)
        err_max_stats = np.max(plot_precentageErr_stats.PrecentageErr)

        print('average error using modelstats in precentage: %.2f' % err_mean_stats)
        print('max error using modelstats in precentage: %.2f' % err_max_stats)
    else:

            mean_err = np.mean(plot_precentageErr_sklearn.PrecentageErr)
            max_err = np.max(plot_precentageErr_sklearn.PrecentageErr)

            print('average error in precentage: %.2f' % mean_err)
            print('max error in precentage: %.2f' % max_err)


def plot_average_error(data_no_na,y_hat_test_sklearn=None,y_hat_sklearn=None, with_statsmodel = False ,
                       y_hat_stats=None,stats_only=False):

    if not stats_only:
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

    if with_statsmodel:
        # preparing data for stats
        df_y_hat_stats = pd.DataFrame(data=y_hat_stats, columns=['WageGaPPredict'])

        df_prediction_and_data_stats = pd.concat([data_no_na, df_y_hat_stats], axis=1, join='inner')

        df_data_stats = df_prediction_and_data_stats[['Year', 'WageGaP']]
        df_prediction_stats = df_prediction_and_data_stats[['Year', 'WageGaPPredict']]

        df_data_stats_mean = df_data_stats.groupby('Year').mean()
        df_prediction_stats_mean = df_prediction_stats.groupby('Year').mean()

        df_prediction_and_data_stats_mean = pd.concat([df_data_stats_mean, df_prediction_stats_mean], axis=1,
                                                      join='inner')

        precentage_error_stats = ((abs(
            df_prediction_and_data_stats_mean['WageGaPPredict'] - df_prediction_and_data_stats_mean['WageGaP']) /
                                   df_prediction_and_data_stats_mean['WageGaP']) * 100)
        df_precentage_error_stats = pd.DataFrame(data=precentage_error_stats, columns=['PrecentageErr'])

        plot_precentageErr_stats = pd.concat([df_prediction_and_data_stats_mean, df_precentage_error_stats], axis=1,
                                             join='inner')

    if with_statsmodel:
        lines = plt.plot(plot_precentageErr_stats.index, plot_precentageErr_stats.PrecentageErr, color='b')

    if not stats_only:
        lines = plt.plot(plot_precentageErr_sklearn.index, plot_precentageErr_sklearn.PrecentageErr, color='g')

    if with_statsmodel:
        plt.ylabel('Error in Precentage over the years (green-sklearn; blue-modelstats)')
    else:
        plt.ylabel('Error in Precentage over the years')

    plt.xlabel('Years')
    plt.title('Average Precentage Error ')
    plt.ylim(0, 20)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    if not stats_only:
        err_mean_sklearn = np.mean(plot_precentageErr_sklearn.PrecentageErr)
        err_max_sklearn = np.max(plot_precentageErr_sklearn.PrecentageErr)
        print('average error in precentage: %.2f' % err_mean_sklearn)
        print('max error in precentage: %.2f' % err_max_sklearn)

    if with_statsmodel:
        err_mean_stats = np.mean(plot_precentageErr_stats.PrecentageErr)
        err_max_stats = np.max(plot_precentageErr_stats.PrecentageErr)

        print('average error using modelstats in precentage: %.2f' % err_mean_stats)
        print('max error using modelstats in precentage: %.2f' % err_max_stats)
    if stats_only:
        return plot_precentageErr_stats

def create_correlation_matrix(data_no_na,remaining_features):
    from pandas.tools.plotting import scatter_matrix
    # help df for renaming the indicator code with meaningfull names
    temp_data_row = pd.read_csv('name_to_code.csv', header=None, skiprows=1)
    temp_data_row.columns = ['index', 'Indicator Name', 'Indicator Code', 'meaning']

    correlation_cols = [x for x in remaining_features if 'country' not in x] + ['Year'] + ['WageGaP']
    features_df = data_no_na[correlation_cols]
    # replace the columns names with codes
    # create dict old-new
    names_dict = {}
    for index, row in temp_data_row.iterrows():
        names_dict[row['Indicator Name']] = row['meaning']

    features_df = features_df.rename(columns=names_dict)
    # output correlations per groups
    print ('Selecting correlation groups manually')
    group1 = ['Age dependency', 'Age population0', 'Age population1', 'Age population5', 'WageGaP']

    group2 = ['Adolescent fertility', 'Fertility rate', 'Death rate', 'WageGaP']
    group3 = ['Employment ratio f', 'Employment ratio m', 'Employment ratio', 'Employment ratio 24', 'WageGaP']
    group4 = ['GDP', 'GDP growth', 'GDP per capita', 'GNI per capita', 'GNI, Atlas', 'WageGaP']
    group5 = ['Immunization, measles', 'Life expectancy f', 'Life expectancy m', 'Mortality  infant ', 'WageGaP']
    group6 = ['Labor participation f', 'Labor participation m', 'Unemployment, youth female', 'Unemployment, youth',
              'WageGaP']

    # print scatter matrix
    corr_df1 = features_df[group1]
    scatter_matrix(corr_df1, alpha=0.7, figsize=(11, 11), diagonal='kde')

    corr_df2 = features_df[group2]
    scatter_matrix(corr_df2, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df3 = features_df[group3]
    scatter_matrix(corr_df3, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df4 = features_df[group4]
    scatter_matrix(corr_df4, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df5 = features_df[group5]
    scatter_matrix(corr_df5, alpha=0.6, figsize=(12, 12), diagonal='kde')

    corr_df6 = features_df[group6]
    scatter_matrix(corr_df6, alpha=0.6, figsize=(12, 12), diagonal='kde')
    return features_df

def prepare_data(data_no_na, y_hat_test, traintest_threshold):
    test = data_no_na[data_no_na['Year'] >= traintest_threshold]

    cols = test.columns.tolist()
    test_2 = test.reset_index(drop=True)
    test_2.columns = cols

    df_y_hat_without_countries = pd.DataFrame(data=y_hat_test,columns=['WageGaPPredict'])

    df_prediction_and_data_without_countries = pd.concat([test_2,df_y_hat_without_countries], axis=1, join='inner')

    df_data_without_countries = df_prediction_and_data_without_countries[['Year','WageGaP']]
    df_prediction_without_countries = df_prediction_and_data_without_countries[['Year','WageGaPPredict']]

    df_data_without_countries_mean = df_data_without_countries.groupby('Year').mean()
    df_prediction_without_countries_mean = df_prediction_without_countries.groupby('Year').mean()
    df_prediction_and_data_without_countries_mean = pd.concat([df_data_without_countries_mean,df_prediction_without_countries_mean], axis=1, join='inner')

    precentage_error_without_countries = ((abs(df_prediction_and_data_without_countries_mean['WageGaPPredict']-df_prediction_and_data_without_countries_mean['WageGaP'])/df_prediction_and_data_without_countries_mean['WageGaP'])*100)
    return pd.DataFrame(data=precentage_error_without_countries,columns=['PrecentageErr'])

def display_average_error(plot_precentageErr_without_countries, plot_precentageErr_only_countries, plot_final_model_precentageErr):
    lines = plt.plot(plot_precentageErr_without_countries.index,plot_precentageErr_without_countries.PrecentageErr,color='b')
    lines = plt.plot(plot_precentageErr_only_countries.index,plot_precentageErr_only_countries.PrecentageErr,color='r')
    lines = plt.plot(plot_final_model_precentageErr.index,plot_final_model_precentageErr.PrecentageErr,color='g')

    plt.ylabel('Error in Precentage over the years')
    plt.xlabel('Years')
    plt.title('Average Precentage Error  (b w/o countries, r only countries, g original)')
    plt.ylim(0,20)
    plt.xlim(2010,2016)
    plt.ticklabel_format(useOffset=False)
    plt.show()

    mean_err = np.mean(plot_precentageErr_without_countries.PrecentageErr)
    max_err = np.max(plot_precentageErr_without_countries.PrecentageErr)

    print('average error (without countries) in precentage: %.2f' %mean_err)
    print('max error (without countries) in precentage: %.2f' %max_err)

    mean_err = np.mean(plot_precentageErr_only_countries.PrecentageErr)
    max_err = np.max(plot_precentageErr_only_countries.PrecentageErr)

    print('average error (only countries) in precentage: %.2f' %mean_err)
    print('max error (only countries) in precentage: %.2f' %max_err)

    mean_err = np.mean(plot_final_model_precentageErr.PrecentageErr)
    max_err = np.max(plot_final_model_precentageErr.PrecentageErr)

    print('average error in precentage our model: %.2f' %mean_err)
    print('max error in precentage our model: %.2f' %max_err)

def create_cross_validation_table(data_no_na,remaining_features_new):
    df = data_no_na.copy()
    feats_used = remaining_features_new[:]

    full_features_list = df.columns.tolist()
    countries = [x for x in full_features_list if x.startswith("country-full-name")]
    feats_used = [x for x in feats_used if not x.startswith("country-full-name")]
    feats_not_used = [x for x in full_features_list if (x not in feats_used) and x.startswith("country-full-name")]

    df_country_to_mse = pd.DataFrame(
        columns=['country', 'MSE', 'delta from avg', 'mean Error in precentage', 'max error in precentage'])
    country_index = 0
    errors_sum = 0

    for i in range(len(countries)):
        country = countries[i]
        ######normalize#######

        # split y to train and test by country
        y_train = df[df[country] == 0].copy()
        y_train = y_train['WageGaP']

        y_test = df[df[country] == 1].copy()
        y_test = y_test['WageGaP']

        # normelized values in x will be in range(0,1)
        l = list(df)
        x_normelize = pd.DataFrame(sp.MinMaxScaler().fit_transform(df), columns=l)
        x_normelize.head()

        # remove the prediction column from x
        x_normelize.drop('WageGaP', axis=1, inplace=True)

        x_train = x_normelize[x_normelize[country] == 0].copy()
        x_test = x_normelize[x_normelize[country] == 1].copy()

        for feat in feats_not_used:
            x_normelize.drop(feat, axis=1, inplace=True)

        ######normalize#######

        stats_model = sm.OLS(y_train, x_train)
        stats_model_results = stats_model.fit()

        y_hat = stats_model_results.predict(x_test)
        err = metrics.mean_squared_error(y_test, y_hat)
        errors_sum += err

        test = df[df[country] == 1].copy()

        cols = test.columns.tolist()
        test_2 = test.reset_index(drop=True)
        test_2.columns = cols

        df_y_hat = pd.DataFrame(data=y_hat, columns=['WageGaPPredict'])

        df_prediction_and_data = pd.concat([test_2, df_y_hat], axis=1, join='inner')

        df_data = df_prediction_and_data[['Year', 'WageGaP']]
        df_prediction = df_prediction_and_data[['Year', 'WageGaPPredict']]

        df_prediction_and_data_mean = pd.concat([df_data, df_prediction], axis=1, join='inner')

        precentage_error = ((abs(
            df_prediction_and_data_mean['WageGaPPredict'] - df_prediction_and_data_mean['WageGaP']) /
                             df_prediction_and_data_mean['WageGaP']) * 100)
        df_precentage_error = pd.DataFrame(data=precentage_error, columns=['PrecentageErr'])
        err_mean = np.mean(precentage_error)
        err_max = np.max(precentage_error)

        df_country_to_mse.loc[country_index] = [country, err, 0, err_mean, err_max]

        country_index += 1

    print('average mse is : %.2f' % (errors_sum / len(countries)))

    for i in range(country_index - 1):
        df_country_to_mse.ix[i, 'delta from avg'] = abs(df_country_to_mse.iloc[i]['MSE'] - errors_sum / len(countries))

    df_country_to_mse.sort_values(['delta from avg'], ascending=False).head(29)

def prepare_data_future_for_prediction(full_df,spline_order):
    # from outside
    df_with2016 = full_df.copy()

    # all local variable
    features_list = df_with2016.columns.tolist()
    countries = [x for x in features_list if x.startswith("country-full-name")]
    feats_no_countries = [x for x in features_list if not x.startswith("country-full-name")]

    # prepare extraoplate
    # Function to curve fit to the data
    def func(x, a, b):
        return a * x + b

    # Initial parameter guess, just to kick off the optimization
    guess = (0.5, 0.5)

    for i in range(len(countries)):
        country = countries[i]
        df_country = full_df[full_df[country] == 1].copy()

        df_country = df_country.sort_values(['Year'])
        cols = df_country.columns.tolist()
        df_country_with2016 = df_country.reset_index(drop=True)
        df_country_with2016.columns = cols

        df_country_with2016 = df_country_with2016.sort_values(['Year'])

        last_country_year = df_country_with2016.loc[len(df_country_with2016) - 1, 'Year']
        if (last_country_year < 2016):
            num_of_years_to_fill = 2016 - last_country_year
            for j in range(num_of_years_to_fill.astype(np.integer)):
                current_row_in_country_df = len(df_country_with2016)
                df_country_with2016.loc[current_row_in_country_df] = [np.nan] * len(features_list)
                df_country_with2016.loc[current_row_in_country_df, country] = 1.0
                df_country_with2016.loc[current_row_in_country_df, 'Year'] = last_country_year + j + 1

                # fill other countries with  0
                for other_country in countries:
                    if (not other_country == country):
                        df_country_with2016.loc[current_row_in_country_df, other_country] = 0.0

                df_country_with2016 = df_country_with2016.interpolate(limit_direction='forward', method='spline',
                                                                      order=spline_order, limit=None)
                row_to_add = df_country_with2016.loc[current_row_in_country_df]

                # add line to main DF
                current_row = len(df_with2016)
                df_with2016.loc[current_row] = row_to_add
    return df_with2016

def future_prediction(df_with2016,stats_final_model):
    # from outside
    df_future_prediction = df_with2016.copy()

    # all local variables from this point on

    # split to X and y
    x_future_prediction = df_future_prediction[list(df_future_prediction)[:-1]]

    # Normelize all values in x
    l = list(df_future_prediction)
    x_future_prediction_normelize = pd.DataFrame(sp.MinMaxScaler().fit_transform(df_future_prediction), columns=l)
    x_future_prediction_normelize.drop('WageGaP', axis=1, inplace=True)

    # split to train and test data
    min_year = min(x_future_prediction_normelize['Year']);
    max_year = max(x_future_prediction_normelize['Year']);
    normalized_2017 = ((2017 - min_year) / (max_year - min_year))

    y_hat_future_prediction = stats_final_model.predict(x_future_prediction_normelize)
    df_y_hat_future_prediction = pd.DataFrame(data=y_hat_future_prediction, columns=['WageGaPPredict'])

    df_future_prediction_data_and_predict = pd.concat([df_future_prediction, df_y_hat_future_prediction], axis=1,
                                                      join='inner')

    features_list = df_future_prediction.columns.tolist()
    countries = [x for x in features_list if x.startswith("country-full-name")]
    feats_no_countries = [x for x in features_list if not x.startswith("country-full-name")]

    df_future_prediction_data_and_predict_mean = df_future_prediction_data_and_predict[['Year', 'WageGaPPredict']]

    df_future_prediction_data_and_predict_mean = df_future_prediction_data_and_predict_mean.groupby('Year').mean()

    print("WageGap Prediction For 2016 using feature prediction is:")
    print(df_future_prediction_data_and_predict_mean[df_future_prediction_data_and_predict_mean.index == 2016].iloc[0, 0])