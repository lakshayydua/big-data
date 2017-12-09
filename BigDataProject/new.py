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

'''
input_dir = '/home/ldua/Desktop/BigDataProject/Output'#sys.argv[1]

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".csv"):
             (os.path.join(root, file)))
'''
explicit_schema = types.StructType([types.StructField('State Code', types.IntegerType(), True),
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

training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/42401/part-00000-333e0387-d5f5-420a-8f11-2a5f09c4a0fb-c000.csv", header=True, schema=explicit_schema)


vecAssembler = VectorAssembler(inputCols=["State Code", "Month", "Year", "AM_Wind", "AM_Temp", "AM_RH", "AM_Press"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="Arithmetic Mean", predictionCol="AM_Predicted")
rf = RandomForestRegressor(featuresCol="features", labelCol="Arithmetic Mean", predictionCol="AM_Predicted_Ozone")

Linear_Regressor = [vecAssembler, lr]
Random_Forest = [vecAssembler, rf]
#RGB_MLP = [vecAssembler, stringIndexer, mlp]
#LAB_forest = [sqlTrans, rab_vecAssembler, stringIndexer, rf]
#LAB_MLP = [sqlTrans, rab_vecAssembler, stringIndexer, mlp]

models = [
    ('Linear Regressor', Pipeline(stages=Linear_Regressor)),
    ('RandomForestRegressor', Pipeline(stages=Random_Forest)),
    #('LAB-forest', Pipeline(stages=LAB_forest)),
    #('LAB-MLP', Pipeline(stages=LAB_MLP)),
]
evaluator = RegressionEvaluator(predictionCol="AM_Predicted_Ozone", labelCol="Arithmetic Mean", metricName="mse")

split = training.randomSplit([0.75, 0.25])
train = split[0]
test = split[1]
train = train.cache()
test = test.cache()

min = 1000
for label, pipeline in models:
    model = pipeline.fit(train)
    predictions = model.transform(test)
    score = evaluator.evaluate(predictions)
    print(label, score)
    if min > score:
        min = score
        min_pipe=pipeline

print(min_pipe,min)

model = min_pipe.fit(training)
predictions = model.transform(training)

predictions = predictions.drop("features")
predictions.show()
predictions.coalesce(1).write.csv('/home/ldua/Desktop/BigDataProject/Output/Final/Ozone.csv', header=True)

