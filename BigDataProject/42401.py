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

conf = SparkConf().setAppName('BigDataProject') \
    .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

sc = pyspark_cassandra.CassandraSparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.2'  # make sure we have Spark 2.2+

cluster = Cluster(cluster_seeds)
session = cluster.connect(keyspace)


def load_table(words):
    dict_data = {}
    dict_data['state_code'] = int(words[0])
    dict_data['year'] = int(words[1])
    dict_data['month'] = int(words[2])
    dict_data['max_value_pred_44201'] = float(words[3])
    dict_data['max_value_pred_42401'] = float(words[4])
    #dict_data['max_value_pred_42101'] = float(words[5])
    #dict_data['max_value_pred_42602'] = float(words[6])
    return dict_data

def df_for(keyspace, table, split_size=100):
    df = spark.createDataFrame(sc.cassandraTable(keyspace, table, split_size=split_size).setName(table))
    df.createOrReplaceTempView(table)
    return df

Table1 = df_for(keyspace, 'predicted_temp', split_size=None)
Table2 = df_for(keyspace, 'predicted_pressure', split_size=None)
Table3 = df_for(keyspace, 'predicted_wind', split_size=None)
Table4 = df_for(keyspace, 'predicted_rh', split_size=None)


support = spark.sql('''SELECT a.state_code, a.month, a.year, a.predicted_temp as am_temp, b.predicted_pressure as am_press, c.predicted_wind as am_wind, d.predicted_rh as am_rh
FROM predicted_temp a
JOIN predicted_pressure b
ON a.state_code=b.state_code AND a.month=b.month AND a.year=b.year
JOIN predicted_wind c
ON b.state_code=c.state_code AND b.month=c.month AND b.year=c.year
JOIN predicted_rh d
ON c.state_code=d.state_code AND c.month=d.month AND c.year=d.year''')

support.createOrReplaceTempView('support')

###############################################df_final.show()

'''
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
'''

schema2 = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

# testing = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/test.csv", header=True, schema=schema2)
###############testing = spark.read.csv("test.csv", header=True, schema=schema2)
testing=support


predictions = {}
i = 0
for criteria_gas in ['44201']:#, '42401', '42101', '42602']:

    table_name = "g" + str(criteria_gas)

    scheme = types.StructType([types.StructField('month', types.IntegerType(), True),
                               types.StructField('year', types.IntegerType(), True),
                               types.StructField('max_value_pred_' + criteria_gas, types.IntegerType(), True),
                               types.StructField('state_code', types.IntegerType(), True),
                               ])
    predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=scheme)

    # training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/" + str(criteria_gas) + "/" + str(criteria_gas) + ".csv",
    # training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/" + str(criteria_gas) + "/" + str(criteria_gas) + ".csv",
    # header=True, schema=schema)

    training = df_for(keyspace, table_name, split_size=None)

    state = training.select('state_code').distinct()
    stateval = state.collect()

    # print(monthval)

    for i in stateval:
        print(i['state_code'])
        #month = training.select('month').where(training['state_code'] == i["state_code"]).distinct()
        #monthval = month.collect()
        #for j in monthval:
            #print(j['month'])

        df = training.select('month','year', 'am_temp', 'am_press', 'am_wind', 'am_rh', 'max_value').where(
            (training['state_code'] == i["state_code"])) #& (training['month'] == j['month']))

        df_test = testing.select('month','year','am_temp', 'am_press', 'am_wind', 'am_rh').where(
            (testing['state_code'] == i["state_code"])) #& (testing['month'] == j['month']))

        prediction_Col_Name = "max_value_pred_" + str(criteria_gas)
        # ,"Observation Percent", "Observation Count","Arithmetic Mean","AM_Wind", "AM_Temp", "AM_RH", "AM_Press"
        vecAssembler = VectorAssembler(inputCols=["month","year","am_temp", "am_press", "am_wind", "am_rh"], outputCol="features")
        #lr = LinearRegression(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        #rfr = RandomForestRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        gbtr = GBTRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        #dtr = DecisionTreeRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)

        #Linear_Regressor = [vecAssembler, lr]
        #Random_Forest = [vecAssembler, rfr]
        GBT_Regressor = [vecAssembler, gbtr]
        #DecisionTree_Regressor = [vecAssembler, dtr]

        models = [
            # ('Linear Regressor', Pipeline(stages=Linear_Regressor)),
            # ('Random Forest Regressor', Pipeline(stages=Random_Forest)),
            ('GBT Regressor', Pipeline(stages=GBT_Regressor)),
            #('Decision Tree Regressor', Pipeline(stages=DecisionTree_Regressor)),
        ]

        evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="max_value",
                                        metricName="mse")


        # split = df.randomSplit([0.80, 0.20])
        # train = split[0]
        # test = split[1]
        # train = train.cache()
        # test = test.cache()

        # min = 1000
        # for label, pipeline in models:
        #     model = pipeline.fit(df)
        #     pred = model.transform(df)
        #     score = evaluator.evaluate(pred)
        #     # print("\nCriteria Gas", criteria_gas)
        #     print(label, score)
        #     if min > score:
        #         min = score
        #         min_pipe = pipeline
        # print("\n----Criteria Gas-----", criteria_gas)
        # print(min_pipe, min)
        for label, pipeline in models:
            model = pipeline.fit(df)
            pred = model.transform(df_test)
            pred = pred.drop("features","am_temp", "am_press", "am_wind", "am_rh")
            pred = pred.withColumn('state_code', functions.lit(i["state_code"]))

            predictions[criteria_gas] = predictions[criteria_gas].union(pred)



code = ['44201']#,'42401','42101','42602']

for i in range(len(code)):

    pred = predictions[code[i]]
    if i == 0:
        pred_f = pred.select(pred['state_code'], pred['month'], pred['year'], pred['max_value_pred_' + code[i]])
        continue

    pred = pred.select(pred['state_code'].alias('sc'), pred['month'].alias('m'), pred['year'].alias('y'), pred['max_value_pred_' + code[i]])

    pred_f = pred_f.join(pred, [(pred_f['state_code'] == pred['sc']) & (pred_f['year'] == pred['y']) &
                                (pred_f['month'] == pred['m'])])\
                                .drop('sc', 'm', 'y')\
                                .select('*').sort('state_code', 'year', 'month')

pred_f.show()

rdd = pred_f.rdd.map(tuple)
words = rdd.map(load_table)
words.saveToCassandra(keyspace, 'predictions1')
