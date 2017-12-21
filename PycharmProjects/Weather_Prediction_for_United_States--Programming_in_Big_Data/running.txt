Instruction to execute the project:

There are 2 versions of this project in folders named "Local Version" and "Cluster Version". 

1) In the repository we have 6 python files named 'preprocess_44201','preprocess_42401','preprocess_42101','preprocess_42602', 'preprocess_88101', 'preprocess_88502' which preprocesses massive dataset for each of the 6 criteria gases. For each criteria gas we read read and aggragate 4 support metrological features that are combined with these gases.

2) We perform feature selection, feature engineering in the preprocessing phase.

3) Next we use Spark ML (Machine Learning) to predict the values of 4 support metrological features that is temperature, pressure, RH_DP and wind for year 2018 for each state for every month.

4) To do this we we execute 4 python files named 'predicted_temperature.py', 'predicted_wind.py', 'predicted_rh.py' and 'predicted_pressure.py'.

5) Using these 4 predicted values for each state/county for each month of year 2018 we find out the max_value_xxxxxx correspoding to each of the criteria gases '44201','42401','42101','42602', '88101','88502' by using various regression models and cross validation. Python file executed for this is max_value_aqi.py

6) Next we combined the max_value_xxxxxx for all critieria gases in one single csv file or cassandra table using max_value_combined.py

7) Now we have 6 values of max_value_xxxxxx for different pollutants in one table which are used to calculate global AQI using predicted_aqi.py.

8) We also find global warming trend by using temperature values from two consecutive years. File executed for this global_warming_trend.py.

9) The same hierarchy and structure of execution is followed for county level weather forecasting.
