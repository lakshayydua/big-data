import sys, os
#import pyspark_cassandra
from pyspark import SparkConf
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from cassandra.cluster import Cluster

from cassandra.query import BatchStatement, SimpleStatement

keyspace='ldua'

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
       .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

cluster = Cluster(cluster_seeds)
session = cluster.connect(keyspace)


def load_table(words):

   dict_data = {}

   dict_data['state_code'] = int(words[0])
   dict_data['year'] = int(words[1])
   dict_data['month'] = int(words[2])
   dict_data['am_predicted_44201'] = float(words[3])
   dict_data['am_predicted_42401'] = float(words[4])
   dict_data['am_predicted_42101'] = float(words[5])
   dict_data['am_predicted_42602'] = float(words[6])
   return dict_data

   #state_code | year | month | am_predicted_44201 | am_predicted_42401 | am_predicted_42101 | am_predicted_42602 |


schema = types.StructType([types.StructField('State Code', types.IntegerType(), True),
                          types.StructField('Month', types.IntegerType(), True),
                          types.StructField('Year',types.IntegerType(), True),
                          types.StructField('Observation Count', types.DoubleType(),True),
                          types.StructField('Observation Percent', types.DoubleType(), True),
                          types.StructField('1st Max Value', types.DoubleType(), True),
                          types.StructField('1st Max Hour', types.DoubleType(), True),
                          types.StructField('Arithmetic Mean', types.DoubleType(), True),
                          types.StructField('AM_Wind', types.DoubleType(), True),
                          types.StructField('AM_Temp', types.DoubleType(), True),
                          types.StructField('AM_RH', types.DoubleType(), True),
                          types.StructField('AM_Press', types.DoubleType(), True)])

schema2 = types.StructType([types.StructField('State Code', types.IntegerType(), True),
                          types.StructField('Month', types.IntegerType(), True),
                          types.StructField('Year',types.IntegerType(), True)])

testing = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/test.csv", header=True, schema=schema2)

predictions = {}
i = 0

#'44201','42401','42101','42602'
for criteria_gas in ['44201']:#,'42401','42101','42602']:

   #table_name = "g" + str(criteria_gas)
   scheme = types.StructType([types.StructField('Month', types.IntegerType(), True),
                              types.StructField('Year', types.IntegerType(), True),
                              types.StructField('rh_pred_' + criteria_gas, types.IntegerType(), True),
                              types.StructField('State Code', types.IntegerType(), True),

                              ],
                             )
   predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(),schema=scheme)
   training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/" + str(criteria_gas) + "/" + str(criteria_gas) + ".csv",
                             header=True, schema=schema)

   state = training.select('State Code').distinct()

   stateval = state.collect()

   #print(monthval)

   for i in stateval:
       print(i['State Code'])
       #month = training.select('Month').where(training['State Code'] == i["State Code"]).distinct()
       #monthval = month.collect()
       #for j in monthval:
       #print(j['Month'])
       df = training.select('Month','Year','AM_RH').where((training['State Code'] == i["State Code"])) #& (training['Month'] == j['Month']))
       df.show()
       df_test = testing.select('Month','Year').where(
           (testing['State Code'] == i["State Code"])) #& (testing['Month'] == j['Month']))
       df_test.show()
       prediction_Col_Name = "rh_pred_" + str(criteria_gas)
       #,"Observation Percent", "Observation Count","Arithmetic Mean","AM_Wind", "AM_RH", "AM_RH", "AM_Press"
       vecAssembler = VectorAssembler(inputCols=["Month","Year"], outputCol="features")
       lr = LinearRegression(featuresCol="features", labelCol="AM_RH", predictionCol=prediction_Col_Name)
       rfr = RandomForestRegressor(featuresCol="features", labelCol="AM_RH", predictionCol=prediction_Col_Name)
       gbtr = GBTRegressor(featuresCol="features", labelCol="AM_RH", predictionCol=prediction_Col_Name)
       dtr = DecisionTreeRegressor(featuresCol="features", labelCol="AM_RH", predictionCol=prediction_Col_Name)

       Linear_Regressor = [vecAssembler, lr]
       Random_Forest = [vecAssembler, rfr]
       GBT_Regressor = [vecAssembler, gbtr]
       DecisionTree_Regressor = [vecAssembler, dtr]

       models = [
           #('Linear Regressor', Pipeline(stages=Linear_Regressor)),
           #('Random Forest Regressor', Pipeline(stages=Random_Forest)),
           ('GBT Regressor', Pipeline(stages=GBT_Regressor)),
           ('Decision Tree Regressor', Pipeline(stages=DecisionTree_Regressor)),
       ]

       evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="AM_RH", metricName="mse")

       # split = df.randomSplit([0.80, 0.20])
       # train = split[0]
       # test = split[1]
       # train = train.cache()
       # test = test.cache()

       min = 1000
       for label, pipeline in models:
           model = pipeline.fit(df)
           pred = model.transform(df)
           score = evaluator.evaluate(pred)
           #print("\nCriteria Gas", criteria_gas)
           print(label, score)
           if min > score:
               min = score
               min_pipe=pipeline
       print("\n----Criteria Gas-----", criteria_gas)
       print(min_pipe,min)

       model = min_pipe.fit(df)
       pred = model.transform(df_test)
       pred = pred.drop("features")
       pred = pred.withColumn('State Code',functions.lit(i["State Code"]))
       #pred = pred.withColumn('Month', functions.lit(j["Month"]))
       #pred.show()
       predictions[criteria_gas] = predictions[criteria_gas].uni5on(pred)
       predictions[criteria_gas].show()


code = ['44201']#,'42401','42101','42602']

for i in range(len(code)):

   pred = predictions[code[i]]
   if i == 0:
       pred_f = pred
       pred_f = pred_f.select(pred['State Code'], pred['Year'], pred['Month'], pred['rh_pred_' + code[i]])
       continue

   pred = pred.select(pred['State Code'].alias('sc'), pred['Year'].alias('y'), pred['Month'].alias('m'),
                      pred['rh_pred_' + code[i]])

   pred_f = pred_f.join(pred, [(pred_f['State Code'] == pred['sc']) & (pred_f['Year'] == pred['y']) &
                               (pred_f['Month'] == pred['m'])])\
                               .drop('sc', 'm', 'y')\
                               .select('*').sort('State Code', 'Year', 'Month')

   break

pred_f.show()
pred_f.coalesce(1).write.csv('/home/ldua/Desktop/BigDataProject/Output/rh_predictions', header=True)



