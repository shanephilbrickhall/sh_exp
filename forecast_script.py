import numpy as np
import pandas as pd
from numpy import loadtxt, genfromtxt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error
import datetime
from datetime import datetime, timedelta
from collections import OrderedDict
import matplotlib.pylab as plt
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
import os

# The following contains the ISO-NE Real-Time pricing prediction model experiment created by Shane Hall for RESurety
# All code is original and was created for the purpose of this demonstration/review. Thank you and lets get started!

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Below is a simple break out of functions for use in script run;
# mainly including utility functions for loading data running DF test month lists etc.

# Un-used
def test_stationarity (timeseries):
    # Determine rolling statistics
    rolmean = timeseries.rolling(window=52, center=False).mean()
    rolstd = timeseries.rolling(window=52, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries.index, timeseries.values, color='blue', label='Original')
    mean = plt.plot(rolmean.index, rolmean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolstd.index, rolstd.values, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def pd_get_month_list(start_date=None, end_date=None):
    # Function expects start date and end date in the format: YYYY-MM-DD
    date1= start_date
    date2= end_date
    month_list = [i.strftime("%b-%y") for i in pd.date_range(start=date1, end=date2, freq='MS')]
    return month_list


def load_data(file_name= None, delim_type= None, header_rows=None,data_type=None,converter=None,use_cols=None):

    if delim_type == 'csv':
        delim = ','
    elif delim_type == 'tab':
        delim = '\t'
    elif delim_type == 'colon':
        delim = ':'
    elif delim_type == 'pipe':
        delim = '|'
    elif delim_type == 'space':
        delim = ' '
    else:
        return "You must include a delimiter type; csv, tab, colon, pipe, space"
    if not header_rows:
        header_skip = 0
    else:
        header_skip = int(header_rows)
    if converter:
        new_array = genfromtxt(str(file_name), delimiter=delim, skip_header=header_skip,
                               dtype=data_type, autostrip=True, converters=converter, usecols=use_cols)
    else:
        new_array = genfromtxt(str(file_name), delimiter=delim, skip_header=header_skip,
                               dtype=data_type, autostrip=True, usecols=use_cols)
    return new_array


def split_data_support_results(ndarray=None):
    # Assumes results are in right most column
    new_array = ndarray
    num_rows = new_array.shape[0]
    num_cols = new_array.shape[1]

    # Split data from results
    support_data = new_array[:,0:num_cols-1]
    results = new_array[:,num_cols-1]
    return support_data, results


def standard_model_run(data=None,train_percentage=None):
    print("Input data set matrix shape, ", data.shape)
    support_set, result_set = split_data_support_results(data)

    if train_percentage:
        train_break = train_percentage
    else:
        train_break = 0.33

    seed = 12

    X_train, X_test, y_train, y_test = train_test_split(support_set, result_set,
                                                        test_size=train_break, random_state=seed)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    model_accuracy = model.score(X_test, y_test)
    return model, model_accuracy


def grid_search_opt(data=None,search_parameters=None,train_percentage=None,):
    # Function expects a ndarray data input
    # Pull in data set break it into testing and training data
    print("Input data set matrix shape", data.shape)
    support_set, result_set = split_data_support_results(data)

    if train_percentage:
        train_break = train_percentage
    else:
        train_break = 0.20

    seed = 12

    X_train, X_test, y_train, y_test = train_test_split(support_set, result_set,
                                                        test_size=train_break, random_state=seed)

    print("Resulting X_train and y_train shapes after input data split: ", X_train.shape, y_train.shape)
    # Initialize model for grid search testing
    model_opt_seed = XGBRegressor()

    if search_parameters:
        grid_search_model_opt = GridSearchCV(model_opt_seed, param_grid=search_parameters, verbose=1)
    else:
        parameters = {
            'learning_rate': [0.001, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
            'max_depth': [1, 3, 4, 5, 6],
            'min_child_weight': [1, 2, 4, 6],
            'silent': [1],
            'subsample': [0.2, 0.25, 0.5, 0.8, 0.90, 1.0],
            'colsample_bytree': [0.5, 0.8, 1.0],
            'n_estimators': [100, 500, 1000],
            'seed': [12]
        }
        grid_search_model_opt = GridSearchCV(model_opt_seed, param_grid=parameters, verbose=1)

    grid_search_model_opt.fit(X_train, y_train)
    # Returns the model object, the test accuracy,
    #  the grid search best run parameters, and the grid search best run accuracy
    return grid_search_model_opt,grid_search_model_opt.score(X_test,y_test), \
           grid_search_model_opt.best_params_, grid_search_model_opt.best_score_


# START OF MAIN SCRIPT:
# First phase is to pull in initial base data sets: ISONE DA, ISONE RT, Boston Precip, Boston Tmp High, Boston Temp Low


# Basic lambda function for removing the '$' from ISONE data
converter_function = lambda x: float(x.decode('UTF-8').strip("$"))

precip_data = load_data(file_name='MA-025-pcp-all-8-2008-2018.csv', delim_type='csv',
                        header_rows=5, data_type=float, use_cols=1)

avg_tmp_data = load_data(file_name='MA-025-tavg-all-8-2008-2018.csv', delim_type='csv',
                         header_rows=5, data_type=float,use_cols=1)

max_tmp_data = load_data(file_name='MA-025-tmax-all-8-2008-2018.csv', delim_type='csv',
                         header_rows=5, data_type=float, use_cols=1)

min_tmp_data = load_data(file_name='MA-025-tmin-all-8-2008-2018.csv', delim_type='csv',
                         header_rows=5, data_type=float,use_cols=1)

iso_day_ahead = load_data(file_name='ISO-NE-MonthlyAvg.DayAhead.csv', delim_type='csv',
                          header_rows=1, data_type=float, use_cols=1, converter={1: converter_function})

iso_real_time = load_data(file_name='ISO-NE-MonthlyAvg.RealTime.csv',delim_type='csv',
                          header_rows=1, data_type=float, use_cols=1, converter={1: converter_function})


dates = pd_get_month_list(start_date='2008-01-01',end_date='2018-08-01')

fig = plt.figure(figsize=(12.8,8))

# Next display these data sets for review allowing for basic observations such as data gaps or other anomalies
ax1 = fig.add_subplot(611)

ax1.plot(dates,iso_day_ahead,'k')
ax1.set_title('ISO-NE Day Ahead LMP NEMA Boston')
ax1.set_ylabel('$/MWh')
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
ax2 = fig.add_subplot(612)

ax2.plot(dates,iso_real_time,'c')
ax2.set_title('ISO-NE Real Time LMP NEMA Boston')
ax2.set_ylabel('$/MWh')
ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
ax3 = fig.add_subplot(613)

ax3.plot(dates,avg_tmp_data,'g')
ax3.set_title('Avg Temp NEMA Boston')
ax3.set_ylabel('Deg F')
ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
ax4 = fig.add_subplot(614)

ax4.plot(dates,max_tmp_data,'b')
ax4.set_title('Max Temp NEMA Boston')
ax4.set_ylabel('Deg F')
ax4.xaxis.set_major_locator(plt.MaxNLocator(10))
ax5 = fig.add_subplot(615)

ax5.plot(dates,min_tmp_data,'r')
ax5.set_title('Min Temp NEMA Boston')
ax5.set_ylabel('Deg F')
ax5.xaxis.set_major_locator(plt.MaxNLocator(10))
ax6 = fig.add_subplot(616)

ax6.plot(dates,precip_data,'m')
ax6.set_title('Avg Precipitation NEMA Boston')
ax6.set_ylabel('Inches')
ax6.set_xlabel('Time')
ax6.xaxis.set_major_locator(plt.MaxNLocator(10))
fig.tight_layout()
plt.show()

# Now that you have loaded all the data and checked for data gaps and other anomalies you can concatenate the data sets
# into a usable matrix for the XGBoost algorithm. Format = [avg temp, max temp, min temp, day ahead, real time
concat_data_set = np.stack((avg_tmp_data,max_tmp_data,min_tmp_data,precip_data,iso_day_ahead, iso_real_time),axis=-1)
print("Stacked input data shape ", concat_data_set.shape," rows 0 through 5: ",concat_data_set[0:5,:])

# Next split the data set into the supporting data set [avg temp, max temp, min temp, day ahead]
# and the result data set [real time]
support_set, result_set = split_data_support_results(concat_data_set)

print("Here is the support and result data set shapes: ", support_set.shape, result_set.shape)

# Set hard seed and split parameters to ensure reproducible results of the experiment
seed = 12
test_train_split = 0.20

# Now split the complete data set into testing and training data for model testing and parameter configuration

X_train, X_test, y_train, y_test = train_test_split(support_set, result_set,
                                                    test_size=test_train_split, random_state=seed)

print("Here is the X_train and y_train shapes: ", X_train.shape, y_train.shape)

print("Start of the non-grid-search optimized model")

# Finally on to the model; in the first phase the model is trained using standard model parameters

model_01 = XGBRegressor()
model_01.fit(X_train, y_train)
std_prediction = model_01.predict(X_test)

# Examine the standard model run note parameter selection and accuracy
print("Here is the standard model summary: ", model_01)

# Determine model accuracy score
print("Pre-Grid Search Model Accuracy: %.2f%%" % (model_01.score(X_test,y_test)))


print("Start of the grid-search optimized model")

# In the second phase the model will be run through a standard parameter grid search style optimization
# in doing this we can investigate/dial in the model to improve accuracy

model_02_opt_seed = XGBRegressor()

# Set the grid of search parameters; using a limited search grid in this example for tme sake
parameters = {
            'learning_rate': [0.1, 0.3, 0.8, 1.0],
            'max_depth': [3, 4,6],
            'min_child_weight': [1, 4, 6],
            'silent': [1],
            'subsample': [0.2, 0.5, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.8, 1.0],
            'n_estimators': [100],
            'seed': [12]
}

grid_search_model_opt = GridSearchCV(model_02_opt_seed, param_grid=parameters, verbose=1)

grid_search_model_opt.fit(X_train,y_train)

# Examine the result of the grid search optimized mode; Note the gird search training did improve accuracy
print("Here are the results of the model grid search: ", grid_search_model_opt.best_params_,
      " the best score is",
      grid_search_model_opt.best_score_, )


opt_model_fin = XGBRegressor(**grid_search_model_opt.best_params_)
opt_model_fin.fit(X_train,y_train)
opt_prediction = opt_model_fin.predict(X_test)
print("Post-Grid Search Model Accuracy: %.2f%%" % (opt_model_fin.score(X_test,y_test)))

# Now plot the results of both models for comparison

dates0 = pd_get_month_list(start_date='2018-08-01',end_date='2020-09-01')

fig0 = plt.figure(figsize=(12.8,8))

ax7 = fig0.add_subplot(211)
ax7.set_title('Standard Model ISO-NE Real-Time Pricing Forecast')
ax7.set_ylabel('$/MWh')
ax7.xaxis.set_major_locator(plt.MaxNLocator(10))

ax7.plot(dates0,std_prediction,'r',label='Standard Model Prediction')
ax7.plot(dates0,y_test,'m',label='Actual ISONE Real Time')
plt.legend()

ax8 = fig0.add_subplot(212)
ax8.set_title('Optimized Model ISO-NE Real-Time Pricing Forecast')
ax8.set_ylabel('$/MWh')
ax8.xaxis.set_major_locator(plt.MaxNLocator(10))

ax8.plot(dates0,opt_prediction,'b',label='Optimized Model Prediction')
ax8.plot(dates0,y_test,'m',label='Actual ISONE Real Time')
plt.legend()
fig0.tight_layout()

plt.show()

# Now that we have constructed and analyzed our data an model lets put it to use
# In phase three we will use 12 Month summary data to build scenarios this is a useful tactic when assessing risk
# and allows to answer many 'what if' scenarios. In this example we will restrict the scenarios to the following:
# Average scenario, Record Heat scenario , and Record Cold
# This can easily be added to for example a Hot Summer or Cold winter scenario could be constructed but was limited
# for simplicity and time sake

# First pull in scenario simulation data

sim_data = load_data('sim_temp_precip_data.csv', delim_type='csv',header_rows=1,data_type=float,use_cols=[1,2,3,4,5,6])

# Plot the loaded data for review and investigation of data gaps or anomalies


# Next add to the weather summary data by pulling summary data from the existing ISONE data
# First Pull in in avg high low for ISO-NE real-time and day ahead data by month
# then Take off most recent 8 months to make annualized data set

annualized_iso_real_time = iso_real_time[8:]
real_time_month_holder = []
for i in range(0,11):
    # Aggregate the data on a monthly basis across the data set
    j = -1
    real_time_month_holder.append(annualized_iso_real_time[j-i::-12])
real_time_month_summary_holder = []
for month in real_time_month_holder:
    # Pull in the avg, max , and min for each month
    m_list = []
    m_list.append((month.average()))
    m_list.append(month.max())
    m_list.append(month.min())
    real_time_month_summary_holder.append(m_list)

# Next concatenate the aggregated results
iso_real_time_summary_data = np.stack(real_time_month_summary_holder)

# Repeat the process for the second ISO-NE data set
annualized_iso_day_ahead = iso_real_time[8:]
day_ahead_month_holder = []
for i in range(0,12):
    j = -1
    day_ahead_month_holder.append(annualized_iso_real_time[j-i::-12])
day_ahead_month_summary_holder = []
for month in day_ahead_month_holder:
    m_list=[]
    m_list.append((month.average()))
    m_list.append(month.max())
    m_list.append(month.min())
    day_ahead_month_summary_holder.append(m_list)
iso_real_time_summary_data = np.stack(day_ahead_month_summary_holder)


print("Pulled in sim data for scenario building: ", sim_data.shape,sim_data[0:6, :])
print(day_ahead_month_summary_holder)

# Next build the Avg, Record Hot, and Record Cold scenarios using the aggregated data

# Avg Scenario
avg_sim_data = load_data('sim_temp_precip_data.csv',delim_type='csv', header_rows=1, data_type=float, use_cols=[1, 2, 3, 6])
print(avg_sim_data.shape, iso_real_time_summary_data[:, 0].shape)
avg_sim_data = np.hstack((avg_sim_data,np.reshape(iso_real_time_summary_data[:, 0], (12, 1))))
print(avg_sim_data.shape, avg_sim_data[0:5, :])

# Record Cold Scenario
record_low_sim = load_data('sim_temp_precip_data.csv',delim_type='csv',header_rows=1,data_type=float,use_cols=[1, 2, 5, 6])
avg_low_sim_data = np.hstack((record_low_sim,np.reshape(iso_real_time_summary_data[:, 2], (12, 1))))
print(avg_low_sim_data.shape, avg_low_sim_data[0:5, :])

# Record Hot Scenario
record_high_sim = load_data('sim_temp_precip_data.csv',delim_type='csv',header_rows=1,data_type=float,use_cols=[1, 4, 3, 6])
avg_high_sim_data = np.hstack((record_high_sim,np.reshape(iso_real_time_summary_data[:, 1], (12, 1))))
print(avg_high_sim_data.shape, avg_high_sim_data[0:5, :])


# Use the optimized model to predict the next 12 Months using the three scenario data sets

avg_prediction = grid_search_model_opt.predict(avg_sim_data)

high_prediction = grid_search_model_opt.predict(avg_high_sim_data)

low_prediction = grid_search_model_opt.predict(avg_low_sim_data)

# Finally plot the three scenarios for review/discussion and further investigation

print(avg_prediction.shape)
print(high_prediction.shape)
print(low_prediction.shape)

dates1 = pd_get_month_list(start_date='2008-01-01',end_date='2019-08-01')
print(dates1.__len__())
fig1 = plt.figure(figsize=(12.8,8))

ax9 = fig1.add_subplot(111)
# Plot historical data
ax9.plot(dates1[:iso_real_time.shape[0]],iso_real_time,'k',label='ISONE NEMA Real Time 10yr History')
# Add connection/interpolation from historical data to sim data
ax9.plot(dates1[iso_real_time.shape[0]-1:iso_real_time.shape[0]+1],
         [iso_real_time[iso_real_time.shape[0]-1],high_prediction[0]],'--r')
ax9.plot(dates1[iso_real_time.shape[0]-1:iso_real_time.shape[0]+1],
         [iso_real_time[iso_real_time.shape[0]-1],avg_prediction[0]],'--g')
ax9.plot(dates1[iso_real_time.shape[0]-1:iso_real_time.shape[0]+1],
         [iso_real_time[iso_real_time.shape[0]-1],low_prediction[0]],'--b')
# Plot simulation results
ax9.plot(dates1[iso_real_time.shape[0]:],avg_prediction,'m',label='Avg Scenario')
ax9.plot(dates1[iso_real_time.shape[0]:],high_prediction,'g',label='Record High(s) Scenario')
ax9.plot(dates1[iso_real_time.shape[0]:],low_prediction,'c',label='Record Low(s) Scenario')
ax9.fill_between(dates1[iso_real_time.shape[0]:],high_prediction,avg_prediction,facecolors='red')
ax9.fill_between(dates1[iso_real_time.shape[0]:],avg_prediction,low_prediction,facecolors='blue')
ax9.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.legend()

plt.show()

# Script complete, thank you for reading!



