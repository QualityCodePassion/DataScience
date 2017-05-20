# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from datetime import datetime

DEBUG_MODE = False
TEST_DATASET = True

INCLUDE_REG_COUNT = True

USE_MEDIAN = True
USE_NEGATIVE_MEAN = False


def get_cache(var_name, data_path, reload=False, index_column = - 1):
    '''
    A function for caching data in order to reduce IO

    Parameters:
    -----------
    var_name: str
        name of variable
    data_path: str
        path of data (csv file)
    reload: bool (optional, default False)
        reload the data set
    '''
    global data
    if not reload and var_name in globals():
        return globals()[var_name]
    else:
        data_path = os.path.expanduser(data_path)
        if( index_column < 0 ):
            data = pd.read_csv(data_path)   # , index_col=0)
        else:
            data = pd.read_csv(data_path, index_col=index_column)

        globals()[var_name] = data
        return data


def convert_time(time_str):
    '''
    A function for converting time string to its time stamp

    Parameters
    ----------
    time_str: str or numpy.nan
        string of time

    Returns
    -------
    timestamp float or numpy.nan
        time stamp of given string
    '''
    if type(time_str) is not str and np.isnan(time_str):
        return time_str
    else:
        time = datetime.strptime(time_str, '%d%b%y:%H:%M:%S')
        return time.timestamp()


def clean(data_path, output_path=None, sample_path = None, reload=False, expected_max_threshold = 69) :
    '''
    A function for clean data

    Parameters
    ----------
    data_path: str
       path of data (csv file)
    random_state: int (optional)
       set random seed for generating random varibales filling the missing data

    Explanation
    -----------
    1. The features with none or one available values
       or has more than 100000 NaN data will be removed

    2. `Datetime` will be converted into `int64`.

    3. Missing data in categorical features will be treated as a category.
       Then categorical data will be processed as continuous integers.

    4. Missing data in numerical feature will be replaced with mean
    '''

    print('loading data file',data_path)
    data = get_cache('data', data_path, reload)


    # Drop the minutes past
    data = data.drop('minutes_past', 1)

    if not TEST_DATASET:
        # for the training data, remove rows with no ref values at all
        data = data.dropna(axis=0, thresh = 5)

    n_features = data.shape[1] - 1


    # do some intial tidying up


    def log(i, index, msg):
        print('[%i/%i] %s: %s' % (i, n_features, index, msg))


    # first remove outliers for particular features
    print("Removing outliers")
    for i, index in enumerate(data.columns[:]):
        if index == 'Id':
            # Don't process the Id
            continue

        #feature = data.loc[:,index]

        #if (index == 'RhoHV') or (index == 'RhoHV_5x5_90th') or (index == 'RhoHV_5x5_50th'):
        #    data.loc[:,index] = data.loc[:,index][ data.loc[:,index] < 1.05]
        #elif (index == 'Zdr') or (index == 'Zdr_5x5_90th') or (index == 'Zdr_5x5_50th'):
        #    data.loc[:,index] = data.loc[:,index][ data.loc[:,index] < 7.8]
        if (index == 'Expected') and not TEST_DATASET:
            # remove the rows of data that have expected rainfall above expected_max_threshold
            expected_max_percentile = 98
            #top_percentile = np.percentile(data.loc[:,index], expected_max_percentile)
            #print('Before capping, the top', expected_max_percentile, 'percentile =', top_percentile)
            #print('Capping it at', expected_max_threshold)
            #print('Before capping the expected: mean =', data.loc[:,index].mean(), ', std =', data.loc[:,index].std(), \
            #      ', median = ', data.loc[:,index].median())
            data.loc[:,index] = data.loc[:,index][ data.loc[:,index] < expected_max_threshold] #300
            #print('After capping capping the expected: mean =', data.loc[:,index].mean(), ', std =', data.loc[:,index].std(), \
            #      ', median = ', data.loc[:,index].median())

        # remove other outliers
        #feature = data.loc[:,index]
        #if (index != 'Expected'):
        #    three_std = 3*data.loc[:,index].std()
        #    if USE_MEDIAN:
        #        feature_median = data.loc[:,index].median()
        #        data.loc[:,index] = data.loc[:,index][ np.abs(data.loc[:,index] - feature_median ) <  three_std]
        #        #data.loc[:,index] = data.loc[:,index][ (data.loc[:,index] - feature_median ) <  three_std]
        #        #data.loc[:,index] = data.loc[:,index][ (data.loc[:,index] - feature_median ) > -three_std]
        #        #data[index] = data.groupby('Id')[index].median()
        #        #data.loc[:,index] = data.loc[:,index][ np.abs(data.loc[:,index] - data.loc[:,index].mean() ) <= 3*data.loc[:,index].std() ]
        #    else:
        #        # default is to use the mean
        #        feature_mean = data.loc[:,index].mean()
        #        data.loc[:,index] = data.loc[:,index][ np.abs(data.loc[:,index] - feature_mean ) < three_std ]

    print("Successfully removed outliers")

    # replace the outliers with NA (but don't change the expected value in the train set)
    #TODO find a more efficient way of doing this!
    if not TEST_DATASET:
        expected_value = data.loc[:,('Id','Expected')]
        data = data.drop('Expected', 1)

    # removed data that were 3 times the standard deviation
    #print("Droppping data that was 3 times greater than the standard deviation")
    #data = (pd.DataFrame( np.where( np.abs(data - data.mean() ) > 3*data.std(), np.nan , data ),
    #                      columns= data.columns ) )   #.drop('Expected', 1)

    if INCLUDE_REG_COUNT:
        print("Counting how many Ref are valid for each ID")
        # Count how many valid Ref values there are for each 'Id'
        ref_data =  data.loc[:,('Id','Ref')]
        ref_data = (ref_data.groupby('Id').count()).rename(columns={'Ref': 'Ref count'})
        #ref_data.to_csv(os.path.expanduser(output_path) )

        # from: http://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-in-a-group-in-pandas-group-by-object
        #gbobj = df[['col1', 'col2', 'col3', 'col4']].groupby(['col1', 'col2'])
        #groupby_id = data.groupby('Id')
        #data = pd.concat([groupby_id.agg('median'), groupby_id.size()], axis=1).rename(columns={0:'count'})

    # Replace each sample with the median
    data = data.groupby('Id').median()
    #data.to_csv(os.path.expanduser(output_path) )


    if INCLUDE_REG_COUNT:
        #TODO better with this or not?
        data = data.join(ref_data)

    # Load the sample data and append the expected value from the sample expected
    print('loading sample file',sample_path)
    sample_data = (get_cache('sample_data', sample_path, reload, 0)).rename(columns={'Expected': 'Sample_Exp'})
    #sample_data = sample_data.rename(columns={'Expected': 'Sample_Exp'})
    print('max sample expected value =', sample_data.loc[:,('Sample_Exp')].max() ) #, ', capped at ', expected_max_threshold )
    sample_data.loc[:,'Sample_Exp'] = sample_data.loc[:,'Sample_Exp'].fillna(sample_data.loc[:,('Sample_Exp')].median)
    sample_data.loc[:,('Sample_Exp')] = sample_data.loc[:,('Sample_Exp')][ sample_data.loc[:,('Sample_Exp')] < expected_max_threshold]
    data = data.join(sample_data)


    if not TEST_DATASET:
        expected_value = expected_value.groupby('Id').median()
        data = data.join(expected_value)

        # As per the new kaggle rules, samples ('Ids') that don't have any 'Ref' value are not used:
        # https://www.kaggle.com/c/how-much-did-it-rain-ii/forums/t/16572/38-missing-data/97181#post97181
        data = data.dropna(subset=['Ref'])
        data = data.dropna(subset=['Expected'])
        data = data.dropna(subset=['Sample_Exp'])
    else:
        # For the test set only
        # Fill the 'Sample_Exp' that were removed above for being about the threshold with the threshold value
        data.loc[:,'Sample_Exp'] = data.loc[:,'Sample_Exp'].fillna(expected_max_threshold)


    # then fill in any NA
    for i, index in enumerate(data.columns[:]):
        if index == 'Id':
            # Don't process the Id
            continue

        feature = data.loc[:,index] #data[index]

        if feature.dtype == np.object:
            print('ERROR: not expecting this data type!')
        else:
            if USE_MEDIAN:
                data.loc[:,index] = feature.fillna(feature.median())
                #data[index] = data.groupby('Id')[index].median()
            elif USE_NEGATIVE_MEAN:
                data.loc[:,index] = feature.fillna( -10*feature.mean() )
            else:
                # default is to use the mean
                data.loc[:,index] = feature.fillna(feature.mean())

            log(i+1, index, 'numerical data')


    if output_path:
        print('outputting csv file', output_path)
        data.to_csv(os.path.expanduser(output_path) ) #, index=False)

if __name__ == '__main__':
    #data_path = '~/downloads/train.csv'
    #output_path = '~/downloads/train_cleaned.csv'

    expected_max_threshold = 69


    postfix = 'pre_groupby_med_with_sample_no_mins_or_null_Refs_WITH_outliers_exp_LT_' + \
              str(expected_max_threshold) \

    if INCLUDE_REG_COUNT:
        postfix += '_cnt'

    postfix += '.csv'

    if DEBUG_MODE:
        data_path = '../Data/debug_train.csv_head_of_size_1000.csv'
        output_path = '../Data/debug_clean.csv'
        sample_path = '../Data/sample_solution.csv_head_of_size_1000.csv'
    elif TEST_DATASET:
        data_path = '../Data/test.csv'
        output_path = '../Data/test_' +  postfix
        sample_path = '../Data/sample_test_solution.csv'
    else:
        data_path = '../Data/train.csv'
        output_path = '../Data/train_' + postfix
        sample_path = '../Data/sample_train_solution.csv'

    clean(data_path, output_path, sample_path, True, expected_max_threshold )
