import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+

def transform(line):

    val = line.split(',')
    return (val[3],val[4])

explicit_schema = types.StructType([types.StructField('State Code', types.IntegerType(), True),
                   types.StructField('Month', types.IntegerType(), True),
                   types.StructField('Year',types.IntegerType(), True),
                   types.StructField('AM_Predicted_44201', types.DoubleType(), True),
                   types.StructField('AM_Predicted_42401', types.DoubleType(), True)])

#State Code,Year,Month,AM_Predicted_44201,AM_Predicted_42401
#Row(State Code=1, Month=2011, Year=1, AM_Predicted_44201=0.02665985549600323, AM_Predicted_42401=1.6022149730848756)
training = sc.textFile("/home/ldua/Desktop/BigDataProject/Output/AQI/part-00000-e88f6806-9bdc-4906-84f7-0647e9a022d8-c000.csv")
#training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/AQI/part-00000-e88f6806-9bdc-4906-84f7-0647e9a022d8-c000.csv", header= True, schema= explicit_schema)
#aqi = training.map(transform)

'''
text = sc.textFile(inputs)
words = text.flatMap(words_once)
sum_Requests_Bytes = words.reduceByKey(calculate_correlation)
final_logs = sum_Requests_Bytes.mapValues(cal_logs)
'''

#temp = training.rdd

aqi = training.map(transform)

print(aqi.collect())
#training.show()

