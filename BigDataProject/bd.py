import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark import SparkConf, SparkContext
import pyspark_cassandra
import sys

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
        .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

sc = pyspark_cassandra.CassandraSparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.2'  # make sure we have Spark 2.2+

keyspace = 'ldua'

def load_table(words):

    dict_data = {}

    dict_data['state_code'] = int(words[0])
    dict_data['month'] = int(words[1])
    dict_data['year'] = int(words[2])
    dict_data['observation_count'] = float(words[3])
    dict_data['observation_percent'] = float(words[4])
    dict_data['max_value'] = float(words[5])
    dict_data['max_hour'] = float(words[6])
    dict_data['arithmetic_mean'] = float(words[7])
    dict_data['am_wind'] = float(words[8])
    dict_data['am_temp'] = float(words[9])
    dict_data['am_rh'] = float(words[10])
    dict_data['am_press'] = float(words[11])
    return dict_data


schema = types.StructType([types.StructField('State Code', types.StringType(), True),
                           types.StructField('Month', types.StringType(), True),
                           types.StructField('Year',types.StringType(), True),
                           types.StructField('Observation Count', types.DoubleType(),True),
                           types.StructField('Observation Percent', types.DoubleType(), True),
                           types.StructField('1st Max Value', types.DoubleType(), True),
                           types.StructField('1st Max Hour', types.DoubleType(), True),
                           types.StructField('Arithmetic Mean', types.DoubleType(), True),
                           types.StructField('AM_Wind', types.DoubleType(), True),
                           types.StructField('AM_Temp', types.DoubleType(), True),
                           types.StructField('AM_RH', types.DoubleType(), True),
                           types.StructField('AM_Press', types.DoubleType(), True)])


itr=0
for j in ['44201','42401','42101','42602']:
    train_final={}
    train_final[itr] = spark.createDataFrame(sc.emptyRDD(), schema)

    for year in range(1998, 2018):
        #training = spark.read.csv("/home/ldua/Desktop/BigDataProject/original/daily_"+ str(j) +"_" + str(year) + ".csv", header = True)
        training = spark.read.csv("original/daily_" + str(j) + "_" + str(year) + ".csv", header=True)

        train = training[['State Code', 'Date Local', 'Observation Count', 'Observation Percent', '1st Max Value', '1st Max Hour','Arithmetic Mean']]


        split_col = functions.split(train['Date Local'], '-')
        train = train.withColumn('Year', split_col.getItem(0))
        train = train.withColumn('Month', split_col.getItem(1))

        train = train.drop('Date Local')
        train.createOrReplaceTempView('train')
        train_g = train.groupBy(train['State Code'],train['Month'],train['Year']).agg(functions.avg(train['Observation Count']).alias('Observation Count'),functions.avg(train['Observation Percent']).alias('Observation Percent'),
           functions.avg(train['1st Max Value']).alias('1st Max Value'),functions.avg(train['1st Max Hour']).alias('1st Max Hour'),
           functions.avg(train['Arithmetic Mean']).alias('Arithmetic Mean'))
        supportl = ['WIND', 'TEMP', 'RH_DP', 'PRESS']
        for i in supportl:

             #support = spark.read.csv("/home/ldua/Desktop/BigDataProject/support/daily_" + str(i) + "_" + str(year) + ".csv", header=True)
             support = spark.read.csv("support/daily_" + str(i) + "_" + str(year) + ".csv", header=True)
             support_f = support.select('State Code', 'Date Local', 'Arithmetic Mean')
             split_col = functions.split(support_f['Date Local'], '-')
             support_f = support_f.withColumn('Year', split_col.getItem(0))
             support_f = support_f.withColumn('Month', split_col.getItem(1))
             support_f = support_f.drop('Date Local')
             support_t = support_f.groupBy([support_f['State Code'],support_f['Month'],support_f['Year']]).agg(
                         functions.avg(support_f['Arithmetic Mean']).alias('AM'))
             support_g = support_t.select(support_t['State Code'].alias('sc'),support_t['Month'].alias('m'),support_t['Year'].alias('y'),support_t['AM'])
             train_g = train_g.join(support_g,[(train_g['State Code'] == support_g['sc']) & (train_g['Year'] == support_g['y']) & (train_g['Month']== support_g['m'])
                                    ]).drop('sc','m','y').select('*').sort('State Code','Year','Month')

             #train_f.printSchema()
             #break
             #train_g = pd.merge(train_g, support_g, on=['State Code', 'Year', 'Month'], how='inner')
        train_final[itr] = train_final[itr].union(train_g)

    ##train_final[itr].show()
    #train_final[itr].coalesce(1).write.csv('Output/'+str(j), sep=',', header=True)


    table_name = "g"+str(j)
    rdd = train_final[itr].rdd.map(tuple)
    words = rdd.map(load_table)
    words.saveToCassandra(keyspace, table_name)

    itr += 1





