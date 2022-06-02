import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import resource
import math

# CAN DUYAR

# preprocessing function was used for operations of "time" feature
def preprocessing(df):
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    return df


# Function for prediction using SARIMAX model based on time series
def predictNumberOfUsers(df,latestHours,nextHours):

    """
    SARIMAX(Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors)
    It is an updated version of the ARIMA model.
    """
    model = SARIMAX(df['users'].tail(latestHours))
    model_fit = model.fit()

    #out-of-sample forecast (it means, future prediction for next 6 hours,24 hours etc.)
    pred_users = model_fit.predict(len(df['users'].tail(latestHours)),len(df['users'].tail(latestHours))+(nextHours-1))

    #original prediction values (they were fractional values so I rounded them in following part)
    print("pred_users (original)")
    print(pred_users)

    #rounded prediction values (The number of users should not be fractional!)
    pred_users = pred_users.round()
    print("pred_users (rounded values)")
    print(pred_users)

    # PERFORMANCE METRICS -> I used Root Mean Square Error (RMSE) and Mean Absolute Error(MAE)
    print("\nPERFORMANCE METRICS:")
    """ predPerformanceEval was used for measurement of performance.
    Since there is no ground truth in performance measurements based on future predictions,
    I predicted the past data with the model and compared it with the values in the original data I have"""
    predPerformanceEval = model_fit.predict(1,len(df['users'].tail(latestHours)))
    mse = mean_squared_error(df['users'].tail(latestHours),predPerformanceEval)
    rmse = math.sqrt(mse)
    print("Root Mean Square Error (RMSE) = {:.2f} ".format(rmse))
    mae = mean_absolute_error(df['users'].tail(latestHours), predPerformanceEval)
    print("Mean Absolute Error(MAE) = {:.2f} ".format(mae))

    print("*****************************************************************")

    # Data Visualization
    plt.figure(figsize=(20, 3))
    plt.plot(df['users'].tail(latestHours), color='blue',label='User count')
    plt.plot(pred_users, '--bo',color='red', label='Prediction')
    plt.legend(loc='best')
    plt.show()

    return df




# MAIN FUNCTION
if __name__ == "__main__":

    time_start = time.perf_counter()

    # reading csv
    df = pd.read_csv("app.csv",sep = ';')

    """I applied pipelining operation for preprocessing and prediction steps. I used '24' and '6' parameters for
    Prediction of the number of users for the next 6 hours using the latest 24 hour’s data and also I used
    '24*5' (it means 5 days) and '24' parameters to find total number of users will be observed 24 hours later
     than the latest data point using the latest 5 day’s data. (The exact date is 30.12.18 09:00) """
    pipeline = df.pipe(preprocessing).pipe(predictNumberOfUsers, latestHours = 24, nextHours = 6).pipe(predictNumberOfUsers,
     latestHours = (24*5), nextHours = 24)

    time_execution = (time.perf_counter() - time_start)

    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0

    print("\nTotal duration and the memory usage of the overall system\n")
    print ("Total Duration: %5.1f secs / Memory Usage: %5.1f MByte" % (time_execution,memory_usage))
