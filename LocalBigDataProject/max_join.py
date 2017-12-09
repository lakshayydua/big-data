import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark import SparkConf, SparkContext
import sys
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
    .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

# def df_for(keyspace, table, split_size=100):
#     df = spark.createDataFrame(sc.cassandraTable(keyspace, table, split_size=split_size).setName(table))
#     df.createOrReplaceTempView(table)
#     return df

# Table1 = df_for(keyspace, 'predicted_temp', split_size=None)
# Table2 = df_for(keyspace, 'predicted_pressure', split_size=None)
# Table3 = df_for(keyspace, 'predicted_wind', split_size=None)
# Table4 = df_for(keyspace, 'predicted_rh', split_size=None)

train_f = {}
for j in ['44201','42401','42101','42602']:
    schema = types.StructType([
        types.StructField('month', types.IntegerType(), True),
        types.StructField('year', types.IntegerType(), True),
        types.StructField('am_temp', types.DoubleType(), True),
        types.StructField('am_press', types.DoubleType(), True),
        types.StructField('am_wind', types.DoubleType(), True),
        types.StructField('am_rh', types.DoubleType(), True),
        types.StructField('max_value_pred_' + str(j), types.StringType(), True),
        types.StructField('state_code', types.StringType(), True)])

    train_f[j] = spark.createDataFrame(sc.emptyRDD(), schema=schema)

    for i in [0, 1]:
        training = spark.read.csv("/home/ldua/Desktop/FinalBig/max_value/predicted_max_value_"+ str(j) + "_" + str(i) + "/" + str(i) + ".csv", header = True, schema = schema)
        train_f[j] = train_f[j].union(training)
        train_f[j].show()
    train_f[j].coalesce(1).write.csv('/home/ldua/Desktop/FinalBig/max_value_combined/', sep=',', header=True)


schema2 = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

test = spark.read.csv("/home/ldua/Desktop/BigDataProject/test.csv", header = True, schema = schema2)

train_f['44201'].createOrReplaceTempView('g44201')
train_f['42401'].createOrReplaceTempView('g42401')
train_f['42101'].createOrReplaceTempView('g42101')
train_f['42602'].createOrReplaceTempView('g42602')
test.createOrReplaceTempView('test')

max_value = spark.sql('''SELECT t.state_code, t.month, t.year, a.max_value_pred_44201, b.max_value_pred_42401, c.max_value_pred_42101, d.max_value_pred_42602
FROM test t
FULL OUTER JOIN g44201 a
ON t.state_code=a.state_code AND t.month=a.month AND t.year=a.year
FULL OUTER JOIN g42401 b
ON t.state_code=b.state_code AND t.month=b.month AND t.year=b.year
FULL OUTER JOIN g42101 c
ON t.state_code=c.state_code AND t.month=c.month AND t.year=c.year
FULL OUTER JOIN g42602 d
ON t.state_code=d.state_code AND t.month=d.month AND t.year=d.year''')


max_value.show()
max_value.coalesce(1).write.csv('/home/ldua/Desktop/FinalBig/max_value_combined/', sep=',', header=True)

        #################

# predicted_temp.createOrReplaceTempView('predicted_temp')
# predicted_pressure.createOrReplaceTempView('predicted_pressure')
# predicted_wind.createOrReplaceTempView('predicted_wind'
# predicted_rh.createOrReplaceTempView('predicted_rh')
#
# support = spark.sql('''SELECT a.state_code, a.month, a.year, a.predicted_temp as am_temp, b.predicted_pressure as am_press, c.predicted_wind as am_wind, d.predicted_rh as am_rh
# FROM predicted_temp a
# JOIN predicted_pressure b
# ON a.state_code=b.state_code AND a.month=b.month AND a.year=b.year
# JOIN predicted_wind c
# ON b.state_code=c.state_code AND b.month=c.month AND b.year=c.year
# JOIN predicted_rh d
# ON c.state_code=d.state_code AND c.month=d.month AND c.year=d.year''')


    # train_final = train_f[j].join(support_g, [(train_g['State Code'] == support_g['sc']) & (train_g['Year'] == support_g['y']) & (train_g['Month'] == support_g['m'])]).drop('sc', 'm', 'y').select('*').sort('State Code', 'Year', 'Month')
    #
    #
    # train_final[j]

#       train = training[['State Code', 'Date Local', 'Observation Count', 'Observation Percent', '1st Max Value', '1st Max Hour','Arithmetic Mean']]
#
#
#         split_col = functions.split(train['Date Local'], '-')
#         train = train.withColumn('Year', split_col.getItem(0))
#         train = train.withColumn('Month', split_col.getItem(1))
#
#         train = train.drop('Date Local')
#         train.createOrReplaceTempView('train')
#         train_g = train.groupBy(train['State Code'],train['Month'],train['Year']).agg(functions.avg(train['Observation Count']).alias('Observation Count'),functions.avg(train['Observation Percent']).alias('Observation Percent'),
#            functions.avg(train['1st Max Value']).alias('1st Max Value'),functions.avg(train['1st Max Hour']).alias('1st Max Hour'),
#            functions.avg(train['Arithmetic Mean']).alias('Arithmetic Mean'))
#         supportl = ['WIND', 'TEMP', 'RH_DP', 'PRESS']
#         for i in supportl:
#
#              support = spark.read.csv("/home/ldua/Desktop/BigDataProject/support/daily_" + str(i) + "_" + str(year) + ".csv", header=True)
#              #support = spark.read.csv("support/daily_" + str(i) + "_" + str(year) + ".csv", header=True)
#              support_f = support.select('State Code', 'Date Local', 'Arithmetic Mean')
#              split_col = functions.split(support_f['Date Local'], '-')
#              support_f = support_f.withColumn('Year', split_col.getItem(0))
#              support_f = support_f.withColumn('Month', split_col.getItem(1))
#              support_f = support_f.drop('Date Local')
#              support_t = support_f.groupBy([support_f['State Code'],support_f['Month'],support_f['Year']]).agg(
#                          functions.avg(support_f['Arithmetic Mean']).alias('AM'))
#              support_g = support_t.select(support_t['State Code'].alias('sc'),support_t['Month'].alias('m'),support_t['Year'].alias('y'),support_t['AM'])
#              train_g = train_g.join(support_g,[(train_g['State Code'] == support_g['sc']) & (train_g['Year'] == support_g['y']) & (train_g['Month']== support_g['m'])
#                                     ]).drop('sc','m','y').select('*').sort('State Code','Year','Month')
#
#         train_final[itr] = train_final[itr].union(train_g)
#
#
#
#     train_final[itr].coalesce(1).write.csv('/home/ldua/Desktop/FinalBig/FinalOutput/' + str(j), sep=',', header=True)
#
#     itr += 1
# predicted_temp = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_temp/predicted_temp.csv", header=True)
# predicted_temp.createOrReplaceTempView('predicted_temp')
# predicted_pressure = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_press/predicted_press.csv",
#                                     header=True)
# predicted_pressure.createOrReplaceTempView('predicted_press')
# predicted_wind = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_wind/predicted_wind.csv", header=True)
# predicted_wind.createOrReplaceTempView('predicted_wind')
# predicted_rh = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_rh/predicted_rh.csv", header=True)
# predicted_rh.createOrReplaceTempView('predicted_rh')
#
# support = spark.sql('''SELECT a.state_code, a.month, a.year, a.predicted_temp as am_temp, b.predicted_press as am_press, c.predicted_wind as am_wind, d.predicted_rh as am_rh
# FROM predicted_temp a
# JOIN predicted_press b
# ON a.state_code=b.state_code AND a.month=b.month AND a.year=b.year
# JOIN predicted_wind c
# ON b.state_code=c.state_code AND b.month=c.month AND b.year=c.year
# JOIN predicted_rh d
# ON c.state_code=d.state_code AND c.month=d.month AND c.year=d.year''')
#
# support.createOrReplaceTempView('support')
#
# # support.show()
#
# schema = types.StructType([types.StructField('state_code', types.IntegerType(), True),
#                            types.StructField('month', types.IntegerType(), True),
#                            types.StructField('year', types.IntegerType(), True),
#                            types.StructField('observation_count', types.DoubleType(), True),
#                            types.StructField('observation_percent', types.DoubleType(), True),
#                            types.StructField('max_value', types.DoubleType(), True),
#                            types.StructField('max_hour', types.DoubleType(), True),
#                            types.StructField('arithmetic_mean', types.DoubleType(), True),
#                            types.StructField('am_wind', types.DoubleType(), True),
#                            types.StructField('am_temp', types.DoubleType(), True),
#                            types.StructField('am_rh', types.DoubleType(), True),
#                            types.StructField('am_press', types.DoubleType(), True)])
# #
# #
# # # schema2 = types.StructType([types.StructField('state_code', types.IntegerType(), True),
# # #                             types.StructField('month', types.IntegerType(), True),
# # #                             types.StructField('year', types.IntegerType(), True)])
# #
# # # testing = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/test.csv", header=True, schema=schema2)
# # ###############testing = spark.read.csv("test.csv", header=True, schema=schema2)
# support = support.withColumn('state_code', support['state_code'].cast('Integer'))
# support = support.withColumn('month', support['month'].cast('Integer'))
# support = support.withColumn('year', support['year'].cast('Integer'))
# support = support.withColumn('am_temp', support['am_temp'].cast('Double'))
# support = support.withColumn('am_press', support['am_press'].cast('Double'))
# support = support.withColumn('am_wind', support['am_wind'].cast('Double'))
# support = support.withColumn('am_rh', support['am_rh'].cast('Double'))
#
# testing = support
# testing.createOrReplaceTempView('testing')
#
# predictions = {}
# i = 0
#
# for criteria_gas in ['44201','42401','42101','42602']:
#     j = 0
#     itr = 0
#     schema2 = types.StructType([types.StructField('month', types.IntegerType(), True),
#                                 types.StructField('year', types.IntegerType(), True),
#                                 types.StructField('am_temp', types.DoubleType(), True),
#                                 types.StructField('am_press', types.DoubleType(), True),
#                                 types.StructField('am_wind', types.DoubleType(), True),
#                                 types.StructField('am_rh', types.DoubleType(), True),
#                                 types.StructField('max_value_pred_' + criteria_gas, types.IntegerType(), True),
#                                 types.StructField('state_code', types.IntegerType(), True)])
#
#     predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=schema2)
#
#     training = spark.read.csv(
#         "/home/ldua/Desktop/FinalBig/First/" + str(criteria_gas) + "/" + str(criteria_gas) + ".csv", header=True,
#         schema=schema)
#     # training.show()
#     # training = training.withColumn('state_code', training['state_code'].cast('Integer'))
#     # training = training.withColumn('month', training['month'].cast('Integer'))
#     # training = training.withColumn('year', training['year'].cast('Integer'))
#     # training = training.withColumn('am_temp', training['am_temp'].cast('Double'))
#     # training = training.withColumn('am_press', training['am_press'].cast('Double'))
#     # training = training.withColumn('am_wind', training['am_wind'].cast('Double'))
#     # training = training.withColumn('am_rh', training['am_rh'].cast('Double'))
#
#     # training = df_for(keyspace, table_name, split_size=None)
#
#     state = training.select('state_code').distinct()  #.where(training['year']>=2015)
#     stateval = state.collect()
#     training.createOrReplaceTempView('training')
#
#     # print(monthval)
#
#     for i in stateval:
#         print(i['state_code'])
#         itr+=1
#         df = spark.sql(
#             '''select month,year,am_temp,am_press,am_wind,am_rh,max_value from training where state_code=''' + str(
#                 i['state_code'])) #+ ''' and year>=2015''')
#         #        df = training.select('month','year', 'am_temp', 'am_press', 'am_wind', 'am_rh', 'max_value').where(
#         #           (training['state_code'] == i["state_code"]))
#         df_test = spark.sql('''select month,year,am_temp,am_press,am_wind,am_rh from testing where state_code=''' + str(
#             i['state_code']))
#         # df_test = testing.select('month','year','am_temp', 'am_press', 'am_wind', 'am_rh').where(
#         #    (testing['state_code'] == i["state_code"]))
#
#         prediction_Col_Name = "max_value_pred_" + str(criteria_gas)
#
#         vecAssembler = VectorAssembler(inputCols=["month", "year", "am_temp", "am_press", "am_wind", "am_rh"],
#                                        outputCol="features")
#         # lr = LinearRegression(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#         # rfr = RandomForestRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#         # dtr = DecisionTreeRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#         gbtr = GBTRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#
#         # Linear_Regressor = [vecAssembler, lr]
#         # Random_Forest = [vecAssembler, rfr]
#         # DecisionTree_Regressor = [vecAssembler, dtr]
#         GBT_Regressor = [vecAssembler, gbtr]
#
#         models = [
#             # ('Linear Regressor', Pipeline(stages=Linear_Regressor)),
#             # ('Random Forest Regressor', Pipeline(stages=Random_Forest)),
#             # ('Decision Tree Regressor', Pipeline(stages=DecisionTree_Regressor)),
#             ('GBT Regressor', Pipeline(stages=GBT_Regressor)),
#         ]
#
#         evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="max_value",
#                                         metricName="mse")
#
#         # split = df.randomSplit([0.80, 0.20])
#         # train = split[0]
#         # test = split[1]
#         # train = train.cache()
#         # test = test.cache()
#
#         # min = 1000
#         # for label, pipeline in models:
#         #     model = pipeline.fit(df)
#         #     pred = model.transform(df)
#         #     score = evaluator.evaluate(pred)
#         #     # print("\nCriteria Gas", criteria_gas)
#         #     print(label, score)
#         #     if min > score:
#         #         min = score
#         #         min_pipe = pipeline
#         # print("\n----Criteria Gas-----", criteria_gas)
#         # print(min_pipe, min)
#
#         for label, pipeline in models:
#             model = pipeline.fit(df)
#             pred = model.transform(df_test)
#             pred = pred.drop("features")  # ,"am_temp", "am_press", "am_wind", "am_rh")
#             pred = pred.withColumn('state_code', functions.lit(i["state_code"]))
#             # pred.show()
#             predictions[criteria_gas] = predictions[criteria_gas].union(pred)
#             # predictions[criteria_gas].show()
#         if(itr==30):
#
#             predictions[criteria_gas].coalesce(1).write.csv(
#                 '/home/ldua/Desktop/FinalBig/max_value/predicted_max_value_'+str(criteria_gas)+'_'+str(j), sep=',',
#                 header=True)
#             predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=schema2)
#             itr = 0
#             j += 1
#
#     predictions[criteria_gas].coalesce(1).write.csv(
#             '/home/ldua/Desktop/FinalBig/max_value/predicted_max_value_'+str(criteria_gas)+'_'+str(j), sep=',',
#             header=True)
#
#
#
#
# predictions[criteria_gas].show()
# print("------------------TRYING TO WRITE------------------")
# print("------------------TRYING TO WRITE------------------")
# print("------------------TRYING TO WRITE------------------")
#
# # predictions[criteria_gas].coalesce(1).write.csv('/home/ldua/Desktop/FinalBig/max_value/predicted_max_value_44201', sep=',',
# #                                      header=True)
#
# # code = ['44201']  # d,'42401']#,'42101','42602']
# #
# # for i in range(len(code)):
# #
# #     pred = predictions[code[i]]
# #     if i == 0:
# #         pred_f = pred.select(pred['state_code'], pred['month'], pred['year'], pred['max_value_pred_' + code[i]])
# #         continue
# #
# #     pred = pred.select(pred['state_code'].alias('sc'), pred['month'].alias('m'), pred['year'].alias('y'),
# #                        pred['max_value_pred_' + code[i]])
# #
# #     pred_f = pred_f.join(pred, [(pred_f['state_code'] == pred['sc']) & (pred_f['year'] == pred['y']) &
# #                                 (pred_f['month'] == pred['m'])]) \
# #         .drop('sc', 'm', 'y') \
# #         .select('*').sort('state_code', 'year', 'month')
#
#
# #pred_f.coalesce(1).write.csv('/home/ldua/Desktop/FinalBig/Output/predicted_max_value_44201', sep=',', header=True)