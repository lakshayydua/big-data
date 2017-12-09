import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"


spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+



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

training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/44201/part-00000-56cf5c5f-2c0f-45fb-897c-a780fdb681a8-c000.csv", header=True, schema=explicit_schema)

'''
def load_table(words):

    #dict_data['am_predicted_42401'] = float(words[0])
    x = float(words[0])
    #a=(SimpleStatement(("INSERT INTO " + table_name + " (am_predicted_42401) VALUES(%s) "), (words[0])))
    #session.execute(a)

    return x
'''

vecAssembler = VectorAssembler(inputCols=["State Code", "Month", "Year", "AM_Wind", "AM_Temp", "AM_RH", "AM_Press"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="Arithmetic Mean", predictionCol="AM_Predicted_44201")
rf = RandomForestRegressor(featuresCol="features", labelCol="Arithmetic Mean", predictionCol="AM_Predicted_44201")

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
evaluator = RegressionEvaluator(predictionCol="AM_Predicted_44201", labelCol="Arithmetic Mean", metricName="mse")

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
##predictions.coalesce(1).write.csv('/home/ldua/Desktop/BigDataProject/Output/Final/44201', header=True)
#rdd = predictions.select(predictions['am_predicted_44201']).rdd.map(tuple)

#words = rdd.map(load_table)
#print(words.collect())




