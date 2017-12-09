import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark import SparkConf, SparkContext
#import pyspark_cassandra
import sys, re, datetime, uuid
from pyspark import SparkConf
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from cassandra.cluster import Cluster

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
    .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

'''
def load_table(words):
   dict_data = {}
   dict_data['state_code'] = int(words[0])
   dict_data['year'] = int(words[1])
   dict_data['month'] = int(words[2])
   dict_data['predicted_temp'] = float(words[3])
   return dict_data


def df_for(keyspace, table, split_size=100):
    df = spark.createDataFrame(sc.cassandraTable(keyspace, table, split_size=split_size).setName(table))
    df.createOrReplaceTempView(table)
    return df
'''

schema = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                           types.StructField('month', types.IntegerType(), True),
                           types.StructField('year',types.IntegerType(), True),
                           types.StructField('observation_count', types.DoubleType(),True),
                           types.StructField('observation_percent', types.DoubleType(), True),
                           types.StructField('max_value', types.DoubleType(), True),
                           types.StructField('max_hour', types.DoubleType(), True),
                           types.StructField('arithmetic_mean', types.DoubleType(), True),
                           types.StructField('am_wind', types.DoubleType(), True),
                           types.StructField('am_temp', types.DoubleType(), True),
                           types.StructField('am_rh', types.DoubleType(), True),
                           types.StructField('am_press', types.DoubleType(), True)])

schema2 = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                          types.StructField('month', types.IntegerType(), True),
                          types.StructField('year',types.IntegerType(), True)])

testing = spark.read.csv("/home/ldua/Desktop/BigDataProject/test.csv", header=True, schema=schema2)

predictions = {}
i = 0

#'44201','42401','42101','42602'
for criteria_gas in ['44201']:#,'42401','42101','42602']:

    #table_name = "g" + str(criteria_gas)
    scheme = types.StructType([types.StructField('month', types.IntegerType(), True),
                               types.StructField('year', types.IntegerType(), True),
                               types.StructField('predicted_temp', types.IntegerType(), True),
                               types.StructField('state_code', types.IntegerType(), True)
                               ])
    predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=scheme)

    #training = df_for(keyspace, table_name, split_size=None)
    training = spark.read.csv("/home/ldua/Desktop/BigDataProject/1.csv", header=True, schema=schema)

    state = training.select('state_code').distinct()
    stateval = state.collect()


    for i in stateval:
        print(i['state_code'])
        #month = training.select('Month').where(training['State Code'] == i["State Code"]).distinct()
        #monthval = month.collect()
        #for j in monthval:
        #print(j['Month'])
        df = training.select('month','year','am_temp').where((training['state_code'] == i["state_code"])) #& (training['Month'] == j['Month']))

        df_test = testing.select('month','year').where(
           (testing['state_code'] == i["state_code"])) #& (testing['Month'] == j['Month']))

        prediction_Col_Name = "predicted_temp"

        vecAssembler = VectorAssembler(inputCols=["month","year"], outputCol="features")
        #lr = LinearRegression(featuresCol="features", labelCol="am_temp", predictionCol=prediction_Col_Name)
        #rfr = RandomForestRegressor(featuresCol="features", labelCol="am_temp", predictionCol=prediction_Col_Name)
        #dtr = DecisionTreeRegressor(featuresCol="features", labelCol="am_temp", predictionCol=prediction_Col_Name)
        gbtr = GBTRegressor(featuresCol="features", labelCol="am_temp", predictionCol=prediction_Col_Name)

        #Linear_Regressor = [vecAssembler, lr]
        #Random_Forest = [vecAssembler, rfr]
        #DecisionTree_Regressor = [vecAssembler, dtr]
        GBT_Regressor = [vecAssembler, gbtr]

        models = [
           #('Linear Regressor', Pipeline(stages=Linear_Regressor)),
           #('Random Forest Regressor', Pipeline(stages=Random_Forest)),
           #('Decision Tree Regressor', Pipeline(stages=DecisionTree_Regressor)),
            ('GBT Regressor', Pipeline(stages=GBT_Regressor)),
        ]

        evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="am_temp", metricName="mse")

        # split = df.randomSplit([0.80, 0.20])
        # train = split[0]
        # test = split[1]
        # train = train.cache()
        # test = test.cache()

        min = 1000
        for label, pipeline in models:
            model = pipeline.fit(df)
            pred = model.transform(df_test)

            pred = pred.drop("features")
            pred = pred.withColumn('state_code', functions.lit(i["state_code"]))
            predictions[criteria_gas] = predictions[criteria_gas].union(pred)




code = ['44201']#,'42401','42101','42602']

for i in range(len(code)):

   pred = predictions[code[i]]
   if i == 0:
       pred_f = pred
       pred_f = pred_f.select(pred['state_code'], pred['year'], pred['month'], pred['predicted_temp'])
       continue

   pred = pred.select(pred['state_code'].alias('sc'), pred['year'].alias('y'), pred['month'].alias('m'),
                      pred['predicted_temp'])

   pred_f = pred_f.join(pred, [(pred_f['state_code'] == pred['sc']) & (pred_f['year'] == pred['y']) &
                               (pred_f['month'] == pred['m'])])\
                               .drop('sc', 'm', 'y')\
                               .select('*').sort('state_code', 'year', 'month')

   break

print("---------------")
print("---------------")
print("---------------")
print("---------------")
pred_f.show()


pred_f.coalesce(1).write.csv('/home/ldua/Desktop/BigDataProject/OutputLocal37/predicted_temp', sep=',', header=True)

'''
rdd = pred_f.rdd.map(tuple)
words = rdd.map(load_table)
words.saveToCassandra(keyspace, 'predicted_temp')
'''