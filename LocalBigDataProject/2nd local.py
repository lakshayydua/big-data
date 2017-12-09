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


schema = types.StructType([types.StructField('state_name', types.StringType(), True),
                           types.StructField('month', types.StringType(), True),
                           types.StructField('year', types.StringType(), True),
                           types.StructField('observation_count', types.DoubleType(), True),
                           types.StructField('observation_percent', types.DoubleType(), True),
                           types.StructField('max_value', types.DoubleType(), True),
                           types.StructField('max_hour', types.DoubleType(), True),
                           types.StructField('arithmetic_mean', types.DoubleType(), True),
                           types.StructField('am_wind', types.DoubleType(), True),
                           types.StructField('am_temp', types.DoubleType(), True),
                           types.StructField('am_rh', types.DoubleType(), True),
                           types.StructField('am_press', types.DoubleType(), True)])

schema2 = types.StructType([types.StructField('state_name', types.StringType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

testing = spark.read.csv("/home/ldua/Desktop/BigDataProject/test-X.csv", header=True, schema=schema2)
testing.show()

predictions = {}
i = 0

for criteria_gas in ['44201']:  # ,'42401']:#,'42101','42602']:

    # # table_name = "g" + str(criteria_gas)
    # scheme = types.StructType([types.StructField('month', types.IntegerType(), True),
    #                            types.StructField('year', types.IntegerType(), True),
    #                            types.StructField('max_value_pred_' + criteria_gas, types.IntegerType(), True),
    #                            types.StructField('state_name', types.StringType(), True),
    #
    #                            ],
    #                           )
    # predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=scheme)

    training = spark.read.csv(
        "/home/ldua/Desktop/BigDataProject/Output-Final/" + str(criteria_gas) + "/" + str(criteria_gas) + ".csv",
        header=True, schema=schema)
'''
    state = training.select('state_name').distinct()

    stateval = state.collect()

    for i in stateval:
        # month = training.select('Month').where(training['State Name'] == i["State Name"]).distinct()
        # monthval = month.collect()
        # for j in monthval:
        # print(j['Month'])

        df = training.select('month', 'year', 'max_value').where(
            (training['state_name'] == i["state_name"]))  # & (training['Month'] == j['Month']))
        df_test = testing.select('Month', 'Year').where(
            (testing['state_name'] == i["state_name"]))  # & (testing['Month'] == j['Month']))
        df.show()
        df_test.show()

        prediction_Col_Name = "max_value_pred_" + str(criteria_gas)

        vecAssembler = VectorAssembler(inputCols=["month", "year"], outputCol="features")
        lr = LinearRegression(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        rfr = RandomForestRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        gbtr = GBTRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        dtr = DecisionTreeRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)

        Linear_Regressor = [vecAssembler, lr]
        Random_Forest = [vecAssembler, rfr]
        GBT_Regressor = [vecAssembler, gbtr]
        DecisionTree_Regressor = [vecAssembler, dtr]

        models = [
            # ('Linear Regressor', Pipeline(stages=Linear_Regressor)),
            # ('Random Forest Regressor', Pipeline(stages=Random_Forest)),
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
            # print("\nCriteria Gas", criteria_gas)
            print(label, score)
            if min > score:
                min = score
                min_pipe = pipeline
        print("\n----Criteria Gas-----", criteria_gas)
        print(min_pipe, min)

        model = min_pipe.fit(df)
        pred = model.transform(df_test)
        pred = pred.drop("features")
        pred = pred.withColumn('state_name', functions.lit(i["state_name"]))
        # pred = pred.withColumn('Month', functions.lit(j["Month"]))
        # pred.show()
        predictions[criteria_gas] = predictions[criteria_gas].union(pred)
        predictions[criteria_gas].show()

'''
code = ['44201']  # , '42401', '42101', '42602']

for i in range(len(code)):

    pred = predictions[code[i]]
    if i == 0:
        pred_f = pred.select(pred['state_name'], pred['year'], pred['onth'], pred['max_value_pred_' + code[i]])
        continue

    pred = pred.select(pred['State Name'].alias('sc'), pred['year'].alias('y'), pred['month'].alias('m'),
                       pred['max_value_pred_' + code[i]])

    pred_f = pred_f.join(pred, [(pred_f['state_name'] == pred['sc']) & (pred_f['Year'] == pred['y']) &
                                (pred_f['Month'] == pred['m'])]) \
        .drop('sc', 'm', 'y') \
        .select('*').sort('State Name', 'Year', 'Month')

pred_f.show()
pred_f.coalesce(1).write.csv('/home/ldua/Desktop/BigDataProject/Output-Xxx', header=True)



'''