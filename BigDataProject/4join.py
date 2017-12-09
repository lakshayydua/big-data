import sys, os
import pyspark_cassandra
from pyspark import SparkConf
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, SimpleStatement

keyspace = 'ldua'

os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
    .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

sc = pyspark_cassandra.CassandraSparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.2'  # make sure we have Spark 2.2+

cluster = Cluster(cluster_seeds)
session = cluster.connect(keyspace)

df_final = spark.sql('''SELECT a.`state_code`, a.month, a.year, a.predicted_temp, b.predicted_pressure, c.predicted_wind, d.predicted_rh
FROM predicted_temp a
JOIN predicted_pressure b
ON a.`state_code`=b.`state_code` AND a.month=b.month AND a.year=b.year
JOIN predicted_wind c
ON b.`state_code`=c.`state_code` AND b.month=c.month AND b.year=c.year
JOIN predicted_rh d
ON c.`state_code`=d.`state_code` AND c.month=d.month AND c.year=d.year''')