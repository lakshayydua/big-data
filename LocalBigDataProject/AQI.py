import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark import SparkConf

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
       .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

inputs = '/home/ldua/Desktop/BigDataProject/Output/predictions5/max_value_predicted.csv'
output = '/home/ldua/Desktop/BigDataProject/Output/predictions5/AQI/AQI'

def aqi_cal(val,aqilevel,gaslevel):
    length = len(gaslevel)

    for i in range(len(gaslevel)):
        if (val < gaslevel[i]):
            num1 = val - gaslevel[i - 1]
            num2 = aqilevel[i] - aqilevel[i - 1]
            den = gaslevel[i] - gaslevel[i - 1]
            aqival = ((num1 * num2) / den) + aqilevel[i - 1]
            break
        else:
            if (val >= gaslevel[length-1]):
                aqival = aqilevel[length-1]-1
                break

    return aqival

# def aqi_so2(val):
#
#      return aqi




def transform(line):
    val = line.split(',')
    if val[0] == 'State Code':
        return line+',Global AQI'
    else:
        aqi_level = [0,51,101,151,201,301,401,501]
        ozone_level = [0,.055,.071,.086,.106,.201]
        so_level = [0,36,76,186,305,605,805,1005]
        co_level = [0,4.5,9.5,12.5,15.5,30.5,40.5,50.5]
        no_level = [0,54,101,361,650,1250,1650,2050]
        aqi_oz = aqi_cal(float(val[3]),aqi_level,ozone_level)
        aqi_so = aqi_cal(float(val[4]), aqi_level, so_level)
        aqi_co = aqi_cal(float(val[5]), aqi_level, co_level)
        aqi_no = aqi_cal(float(val[6]), aqi_level, no_level)
        # val[3] = float(val[3])
        # val[4] = float(val[4])
        # val[5] = float(val[5])
        # val[6] = float(val[6])
        # for i in range(len(ozone_level)):
        #     if(val[3]< ozone_level[i]):
        #         num1 = val[3] - ozone_level[i-1]
        #         num2 = aqi_level[i] - aqi_level[i-1]
        #         den = ozone_level[i] - ozone_level[i-1]
        #         aqi_oz = ((num1 * num2)/den)+aqi_level[i-1]
        #         break
        #     else:
        #         if(val[3] >= ozone_level[5]):
        #             aqi_oz = 300
        #             break
        #
        # for i in range(len(so_level)):
        #     if (val[4] < so_level[i]):
        #         num1 = val[4] - so_level[i - 1]
        #         num2 = aqi_level[i] - aqi_level[i - 1]
        #         den = so_level[i] - so_level[i - 1]
        #         aqi_so = ((num1 * num2) / den) + aqi_level[i - 1]
        #         break
        #     else:
        #         if (val[4] > so_level[7]):
        #             aqi_so = 500
        #             break
        #
        # for i in range(len(co_level)):
        #     if (val[5] < co_level[i]):
        #         num1 = val[5] - co_level[i - 1]
        #         num2 = aqi_level[i] - aqi_level[i - 1]
        #         den = co_level[i] - co_level[i - 1]
        #         aqi_co = ((num1 * num2) / den) + aqi_level[i - 1]
        #         break
        #     else:
        #         if (val[5] > co_level[7]):
        #             aqi_co = 500
        #             break
        #
        # for i in range(len(no_level)):
        #     if (val[6] < no_level[i]):
        #         num1 = val[6] - no_level[i - 1]
        #         num2 = aqi_level[i] - aqi_level[i - 1]
        #         den = no_level[i] - no_level[i - 1]
        #         aqi_no = ((num1 * num2) / den) + aqi_level[i - 1]
        #         break
        #     else:
        #         if (val[6] > no_level[7]):
        #             aqi_no = 500
        #             break

        glo = [aqi_no,aqi_so,aqi_oz,aqi_co]

        return (val[0],val[1],val[2],aqi_oz,aqi_so,aqi_co,aqi_no,max(glo))

# explicit_schema = types.StructType([types.StructField('State Code', types.IntegerType(), True),
#                    types.StructField('Month', types.IntegerType(), True),
#                    types.StructField('Year',types.IntegerType(), True),
#                    types.StructField('AM_Predicted_44201', types.DoubleType(), True),
#                    types.StructField('AM_Predicted_42401', types.DoubleType(), True)])

#State Code,Year,Month,AM_Predicted_44201,AM_Predicted_42401
#Row(State Code=1, Month=2011, Year=1, AM_Predicted_44201=0.02665985549600323, AM_Predicted_42401=1.6022149730848756)
#training = sc.textFile("/home/ldua/Desktop/BigDataProject/Output/AQI/part-00000-e88f6806-9bdc-4906-84f7-0647e9a022d8-c000.csv")
#training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/AQI/part-00000-e88f6806-9bdc-4906-84f7-0647e9a022d8-c000.csv", header= True, schema= explicit_schema)
#aqi = training.map(transform)

training = sc.textFile(inputs)

#temp = training.rdd

aqi = training.map(transform)
aqi.coalesce(1).saveAsTextFile(output)
#print(aqi.collect())
#training.show(

