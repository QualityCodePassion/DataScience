# -*- coding: utf-8 -*-

import sys

sys.path.append('D:\\Dropbox\\dev\\GitHub\\xgboost\\python-package')

import os
import numpy as np
import pandas as pd
from functools import wraps
import xgboost as xgb

from time import strftime

from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.cross_validation import KFold

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error

DEBUG_MODE = False

gen_submission = True


def get_cache(var_name, data_path, reload=False):
    '''
    A function for caching data in order to reduce IO

    Parameters
    ----------
    var_name: str
        name of variable

    data_path: str
        path of data (csv file)

    reload: bool (optional, default False)
        reload the data set

    Returns
    -------
    data: pandas.core.frame.DataFrame
        data frame
    '''
    global data
    if not reload and var_name in globals():
        return globals()[var_name]
    else:
        data_path = os.path.expanduser(data_path)
        data = pd.read_csv(data_path, index_col=0)
        globals()[var_name] = data
        return data


def ignore_err_decorator(f):
    '''
    A decorator for ignoring errors
    '''

    @wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            print('An error occurred when run %s: %s' % (f.__name__, ex))

    return func


# @ignore_err_decorator
def grid_search_learning(test_data_path, output_data_path, model, X, y, name, tune_param_grid, k, test_percentage):
    '''
    A function for applying ML model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator-like object
        the model for learning
    X: array
        input matrix
    y: array
        output vector
    tune_param_grid: sklearn grid-search param
        the tuning parameter matrix
    k: int
        k-fold cross validation
    '''
    print('', name, sep='\n', end='\n--------------------\n')

    # based on http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#example-model-selection-grid-search-digits-py

    print("# Tuning hyper-parameters for", name, "with", k, "CV folds with test size =", test_percentage)
    print("with tune grid: ", tune_param_grid)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percentage, random_state=42)

    # Tune the model based on all combinations of the tuning parameters
    # provided in the tune_param_grid
    print("started GridSearchCV for model at", strftime("%Y-%m-%d %H:%M:%S"))

    clf = GridSearchCV(model, tune_param_grid, cv=k, scoring='mean_absolute_error', n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Ended GridSearchCV for model at", strftime("%Y-%m-%d %H:%M:%S"))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)

    print('test error before adjustig = ', mean_absolute_error(y_true, y_pred))
    print()

    # remove any predictions above max_expected
    max_expected = 300
    corrected_min = 0.5

    y_pred = (pd.DataFrame(np.where(y_pred > max_expected, max_expected, y_pred)))  # .drop('Expected', 1)

    y_pred = (pd.DataFrame(np.where(y_pred < 0, corrected_min, y_pred)))  # .drop('Expected', 1)

    print('test error after adjusting = ', mean_absolute_error(y_true, y_pred))
    print()

    # print( fitted_model.feature_importances_ )
    # print( clf.get_params() ) #..feature_importances_)


    if gen_submission:
        print('loading test data', test_data_path)
        X_actual_test = np.asarray(get_cache('data', test_data_path, True))
        # X_actual_test = np.asarray(data.iloc[:, :-1])

        y_pred = pd.DataFrame(clf.predict(X_actual_test))

        # output_data = pd.concat([X_actual_test.loc[:,0], y_pred], axis=1).reindex_like(X_actual_test)

        output_data_path += str(mean_score)

        expected_col = y_pred[0]
        print('max predict value of: ', expected_col.max())  # , 'being capped at:', max_expected )
        print('Prediction: mean =', expected_col.mean(), ', std =', expected_col.std(),
              ', median = ', expected_col.median())

        y_pred = (pd.DataFrame(np.where(y_pred > max_expected, max_expected, y_pred),
                               columns=(y_pred.columns)))  # .drop('Expected', 1)

        y_pred = (pd.DataFrame(np.where(y_pred < 0, corrected_min, y_pred),
                               columns=(y_pred.columns)))  # .drop('Expected', 1)

        y_pred.index += 1

        print('outputting submission file', output_data_path)
        y_pred.to_csv(os.path.expanduser(output_data_path), header=True)


def main(data_path, test_data_path, output_data_path):
    '''
    use different models to fit data

    models are
    1. Naive Bayes
    2. Logistic regression
    3. LDA
    4. QDA
    5. Decision Tree
    6. KNN
    7. SVM
    8. AdaBoost
    9. Random Forest
    10. Gradient Boost

    Parameter
    ---------
    data_path: str
        path of cleaned data
    '''
    print('loading training data', data_path)
    data = get_cache('data', data_path)
    X = np.asarray(data.iloc[:, :-1])
    y = np.asarray(data.Expected)

    if gen_submission:
        print('Making sure test data exists and the submission file is writable')
        f = open(test_data_path, 'r')
        f.close()
        w = open(output_data_path, 'w')
        w.close()

    # It takes a long timme, so probably better to uncomment just the one model you want to test
    models = [  # (SGDRegressor(), 'SGD Regressor', 5, 0.2),
                # (GaussianNB(), 'Naive Bayes', 3),
                # (DecisionTreeRegressor(max_depth=5), 'Decision Tree', 2),
                # (KNeighborsRegressor(n_neighbors=5), 'KNN', 2),
                # (RandomForestRegressor(), 'Random Forest', 3, 0.05),
                # (LogisticRegression(), 'Logistic regression', 2),
                # (AdaBoostRegressor(), 'AdaBoost', 2),
                (GradientBoostingRegressor(), 'Gradient Boost', 3),
                # (xgb.XGBRegressor(), 'X Gradient Boost', 2, 0.5),
                # (SVC(), 'SVM', 2)
                ]


    # TODO: Tidy up this Mess!

    # Make sure that this only has the arugments relavent to the model you are testing.
    # Put in the different parameters you want to try as a list for each arugments as shown below:


    tune_param_grid = [{'loss': ['lad'],
                        'n_estimators': [500],
                        'max_depth': [20]
                        }]

    tune_param_grid = [{'loss': ['epsilon_insensitive'],
                        'penalty': ['l1'],
                        'random_state': [42],
                        'epsilon': [0.1],
                        'n_iter': [200, 300, 500]
                        }]

    tune_param_grid = [{'loss': ['lad'],
                        'learning_rate': [0.001],
                        'max_depth': [10],
                        'subsample': [0.25],
                        'n_estimators': [10000]}]  # 1000

    for model, name, k in models:
        test_size = 0.1
        grid_search_learning(test_data_path, output_data_path, model, X, y, name, tune_param_grid, k, test_size)

    tune_param_grid = [{'loss': ['lad'],
                        'n_estimators': [250],
                        'max_depth': [3, 5]
                        }]

    tune_param_grid = [{'loss': ['epsilon_insensitive'],
                        'penalty': ['l1'],
                        'random_state': [42],
                        'epsilon': [0.1],
                        'n_iter': [30, 50, 70, 90, 110]
                        }]

    tune_param_grid = [{'loss': ['lad'],
                        'learning_rate': [0.0001],
                        'max_depth': [10],
                        'subsample': [0.5],
                        'n_estimators': [10000]}]  # 1000

    for model, name, k in models:
        test_size = 0.5
        grid_search_learning(test_data_path, output_data_path, model, X, y, name, tune_param_grid, k, test_size)

    tune_param_grid = [{'loss': ['lad'],
                        'n_estimators': [100],
                        'penalty': [3, 5]
                        }]

    tune_param_grid = [{'loss': ['epsilon_insensitive'],
                        'penalty': ['l1'],
                        'random_state': [42],
                        'epsilon': [0.1],
                        'n_iter': [5, 6, 7, 8, 9, 10, 15, 20]
                        }]

    tune_param_grid = [{'loss': ['lad'],
                        'learning_rate': [0.001],
                        'max_depth': [15],
                        'subsample': [0.25],
                        'n_estimators': [5000]}]  # 1000

    for model, name, k in models:
        test_size = 0.5
        grid_search_learning(test_data_path, output_data_path, model, X, y, name, tune_param_grid, k, test_size)


if __name__ == '__main__':
    print("program started at", strftime("%Y-%m-%d %H:%M:%S"))

    #postfix = 'pre_groupby_med_with_sample_no_outliers_mins_or_null_Refs_exp_LT_69.csv'
    #postfix = 'pre_groupby_med_with_sample_no_outliers_mins_or_null_Refs_exp_LT_90.csv'
    postfix = 'pre_groupby_med_with_sample_no_mins_or_null_Refs_WITH_outliers_exp_LT_69_cnt.csv'

    if DEBUG_MODE:
        train_data_path = '../Data/debug_train_groupby_median.csv_head_of_size_1000.csv'
        test_data_path = '../Data/debug_test_groupby_median.csv_head_of_size_1000.csv'
        output_data_path = '../Data/Submissions/Debug/submission_debug.csv'
    else:
        train_data_path = '../Data/train_' + postfix
        # train_data_path = '../Data/train_groupby_median.csv_head_of_size_1000.csv'

        test_data_path = '../Data/test_' + postfix
        output_data_path = '../Data/Submissions/submission_' + postfix

    main(train_data_path, test_data_path, output_data_path)

    print("program finished at", strftime("%Y-%m-%d %H:%M:%S"))
