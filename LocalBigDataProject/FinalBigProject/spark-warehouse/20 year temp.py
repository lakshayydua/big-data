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

schema = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                           types.StructField('month', types.IntegerType(), True),
                           types.StructField('year', types.IntegerType(), True),
                           types.StructField('am_temp', types.DoubleType(), True)])


train_final = spark.createDataFrame(sc.emptyRDD(), schema=schema)

for year in range(1998, 2018):
    support = spark.read.csv("/home/ldua/Desktop/BigDataProject/support/daily_TEMP_" + str(year) + ".csv", header=True)

    support_f = support.select('State Code', 'Date Local', 'Arithmetic Mean')
    split_col = functions.split(support_f['Date Local'], '-')
    support_f = support_f.withColumn('Year', split_col.getItem(0))
    support_f = support_f.withColumn('Month', split_col.getItem(1))
    support_f = support_f.drop('Date Local')
    support_t = support_f.groupBy([support_f['State Code'], support_f['Month'], support_f['Year']]).agg(
        functions.avg(support_f['Arithmetic Mean']).alias('AM'))
    #support_g = support_t.select(support_t['State Code'].alias('sc'), support_t['Month'].alias('m'),
                                # support_t['Year'].alias('y'), support_t['AM'])
    # train_g = train_g.join(support_g,[(train_g['State Code'] == support_g['sc']) & (train_g['Year'] == support_g['y']) & (train_g['Month']== support_g['m'])
    #                         ]).drop('sc','m','y').select('*').sort('State Code','Year','Month')

    train_final = train_final.union(support_t)

train_final.coalesce(1).write.csv('/home/ldua/Desktop/FinalBig/20years/temp', sep=',', header=True)
