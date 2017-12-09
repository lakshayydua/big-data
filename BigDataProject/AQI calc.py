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

explicit_schema =[]
schema1 = types.StructType([types.StructField('State Code', types.IntegerType(), True),
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
                           types.StructField('AM_Press', types.DoubleType(), True),
                           types.StructField('AM_Predicted_44201', types.DoubleType(), True)])

schema2 = types.StructType([types.StructField('State Code', types.IntegerType(), True),
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
                           types.StructField('AM_Press', types.DoubleType(), True),
                           types.StructField('AM_Predicted_42401', types.DoubleType(), True)])

schema3 = types.StructType([types.StructField('State Code', types.IntegerType(), True),
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
                           types.StructField('AM_Press', types.DoubleType(), True),
                           types.StructField('AM_Predicted_42101', types.DoubleType(), True)])

explicit_schema.append(schema1)
explicit_schema.append(schema2)
explicit_schema.append(schema3)

code = ['44201','42401','42101']#,'42602']

for i in range(len(code)):
    #schema = explicit_schema+str(i)
    training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/Final/"+str(i)+".csv", header=True, schema=explicit_schema[i])
    if i==0:
        training_f = training
        training_f = training_f.select(training['State Code'], training['Year'], training['Month'],
                                   training['AM_Predicted_'+code[i]])
        continue
    #training1 = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/Final/1.csv", header=True, schema=explicit_schema2)
    #training2 = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/Final/2.csv", header=True, schema=explicit_schema2)


    #training = training.select(training['State Code'],training['Year'],training['Month'],training['AM_Predicted_44201'])
    training = training.select(training['State Code'].alias('sc'),training['Year'].alias('y'),training['Month'].alias('m'),training['AM_Predicted_'+code[i]])

    training_f = training_f.join(training,[(training_f['State Code'] == training['sc']) & (training_f['Year'] == training['y']) & (training_f['Month']== training['m'])
                                        ]).drop('sc','m','y').select('*').sort('State Code','Year','Month')




training_f.show()