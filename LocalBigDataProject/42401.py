import sys, os
# import pyspark_cassandra
from pyspark import SparkConf
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from cassandra.cluster import Cluster
import warnings

warnings.filterwarnings(action="ignore")

from cassandra.query import BatchStatement, SimpleStatement

keyspace = 'ldua'


os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
    .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+


def load_table(words):
    dict_data = {}
    dict_data['state_name'] = int(words[0])
    dict_data['year'] = int(words[1])
    dict_data['month'] = int(words[2])
    dict_data['max_value_pred_44201'] = float(words[3])
    dict_data['max_value_pred_42401'] = float(words[4])
    dict_data['max_value_pred_42101'] = float(words[5])
    dict_data['max_value_pred_42602'] = float(words[6])
    return dict_data

    # state_code | year | month | am_predicted_44201 | am_predicted_42401 | am_predicted_42101 | am_predicted_42602 |


schema = types.StructType([types.StructField('State Code', types.IntegerType(), True),
                           types.StructField('Month', types.IntegerType(), True),
                           types.StructField('Year', types.IntegerType(), True),
                           types.StructField('Observation Count', types.DoubleType(), True),
                           types.StructField('Observation Percent', types.DoubleType(), True),
                           types.StructField('1st Max Value', types.DoubleType(), True),
                           types.StructField('1st Max Hour', types.DoubleType(), True),
                           types.StructField('Arithmetic Mean', types.DoubleType(), True),
                           types.StructField('AM_Wind', types.DoubleType(), True),
                           types.StructField('AM_Temp', types.DoubleType(), True),
                           types.StructField('AM_RH', types.DoubleType(), True),
                           types.StructField('AM_Press', types.DoubleType(), True)])

schema2 = types.StructType([types.StructField('State Name', types.StringType(), True),
                            types.StructField('Month', types.IntegerType(), True),
                            types.StructField('Year', types.IntegerType(), True)])

testing = spark.read.csv("/home/ldua/Desktop/BigDataProject/test.csv", header=True, schema=schema2)
testing.show()

predictions = {}
i = 0

df1 = spark.read.csv("/home/ldua/Desktop/BigDataProject/pressure_predictions/pressure_predictions.csv", header=True)
df1.createOrReplaceTempView('Table1')
df2 = spark.read.csv("/home/ldua/Desktop/BigDataProject/rh_predictions/rh_predictions.csv", header=True)
df2.createOrReplaceTempView('Table2')
df3 = spark.read.csv("/home/ldua/Desktop/BigDataProject/temp_predictions/temp_predictions.csv", header=True)
df3.createOrReplaceTempView('Table3')
df4 = spark.read.csv("/home/ldua/Desktop/BigDataProject/wind_predictions/wind_predictions.csv", header=True)
df4.createOrReplaceTempView('Table4')

support = spark.sql('''SELECT a.`State Code`, a.Month, a.Year, a.pressure_pred_44201, b.rh_pred_44201, c.temp_pred_44201, d.wind_pred_44201
FROM Table1 a
JOIN Table2 b
ON a.`State Code`=b.`State Code` AND a.Month=b.Month AND a.Year=b.Year
JOIN Table3 c
ON b.`State Code`=c.`State Code` AND b.Month=c.Month AND b.Year=c.Year
JOIN Table4 d
ON c.`State Code`=d.`State Code` AND c.Month=d.Month AND c.Year=d.Year''')
support.createOrReplaceTempView('support')
support.show()
#df_final = spark.sql('''SELECT a.`State Code`, a.Month, a.Year, a.pressure_pred_44201, b.rh_pred_44201 FROM df1 a, (SELECT  b WHERE a.`State Code`=b.`State Code` AND a.Month=b.Month AND a.Year=b.Year''')


#df_final = spark.sql('''SELECT a.`State Code` from df1 a''')
#df_final = spark.sql('''SELECT a.`State Code`, a.Month, a.Year, a.pressure_pred_44201, b.rh_pred_44201 FROM df1 a, df2 b WHERE a.`State Code`=b.`State Code` AND a.Month=b.Month AND a.Year=b.Year''')

#         FROM select_range a
#
#         INNER JOIN select_max_range b
#         ON a.date=b.date AND a.range = b.range
#         ORDER BY date''')
# df_final.show()
#
# #df_final = spark.sql('''SELECT a.State\ Code, a.Year, a.Month, b.pressure_pred_44201 From df1 a ''')

table_name = 'g42401'
training = spark.read.csv("/home/ldua/Desktop/BigDataProject/" + str(42401) + "/" + str(42401) + ".csv",
                             header=True, schema=schema)

training.createOrReplaceTempView(table_name)
training.show()

'''
for criteria_gas in ['44201','42401','42101','42602']:

   #table_name = "g" + str(criteria_gas)
   scheme = types.StructType([types.StructField('Month', types.IntegerType(), True),
                              types.StructField('Year', types.IntegerType(), True),
                              types.StructField('max_value_pred_' + criteria_gas, types.IntegerType(), True),
                              types.StructField('State Name', types.IntegerType(), True),

                              ],
                             )
   predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(),schema=scheme)
   training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output-Final/" + str(criteria_gas) + "/" + str(criteria_gas) + ".csv",
                             header=True, schema=schema)

   state = training.select('State Name').distinct()

   stateval = state.collect()


   for i in stateval:
       #month = training.select('Month').where(training['State Name'] == i["State Name"]).distinct()
       #monthval = month.collect()
       #for j in monthval:
       #print(j['Month'])

       df = training.select('Month','Year','1st Max Value').where((training['State Name'] == i["State Name"])) #& (training['Month'] == j['Month']))
       df_test = testing.select('Month','Year').where((testing['State Name'] == i["State Name"])) #& (testing['Month'] == j['Month']))
       df.show()
       df_test.show()

       prediction_Col_Name = "max_value_pred_" + str(criteria_gas)

       vecAssembler = VectorAssembler(inputCols=["Month","Year"], outputCol="features")
       lr = LinearRegression(featuresCol="features", labelCol="1st Max Value", predictionCol=prediction_Col_Name)
       rfr = RandomForestRegressor(featuresCol="features", labelCol="1st Max Value", predictionCol=prediction_Col_Name)
       gbtr = GBTRegressor(featuresCol="features", labelCol="1st Max Value", predictionCol=prediction_Col_Name)
       dtr = DecisionTreeRegressor(featuresCol="features", labelCol="1st Max Value", predictionCol=prediction_Col_Name)

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

       evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="1st Max Value", metricName="mse")

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
       pred = pred.withColumn('State Name',functions.lit(i["State Name"]))
       #pred = pred.withColumn('Month', functions.lit(j["Month"]))
       #pred.show()
       predictions[criteria_gas] = predictions[criteria_gas].union(pred)
       predictions[criteria_gas].show()


code = ['44201', '42401', '42101', '42602']

for i in range(len(code)):

    pred = predictions[code[i]]
    if i == 0:
        pred_f = pred.select(pred['State Name'], pred['Year'], pred['Month'], pred['max_value_pred_' + code[i]])
        continue

    pred = pred.select(pred['State Name'].alias('sc'), pred['Year'].alias('y'), pred['Month'].alias('m'),
                       pred['max_value_pred_' + code[i]])

    pred_f = pred_f.join(pred, [(pred_f['State Name'] == pred['sc']) & (pred_f['Year'] == pred['y']) &
                                (pred_f['Month'] == pred['m'])]) \
        .drop('sc', 'm', 'y') \
        .select('*').sort('State Name', 'Year', 'Month')

pred_f.show()
pred_f.coalesce(1).write.csv('/home/ldua/Desktop/BigDataProject/Output-X/max_value', header=True)
'''


