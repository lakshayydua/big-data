import sys, re, datetime, os, gzip, uuid
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, SimpleStatement
import csv

input_dir = '/home/ldua/Desktop/BigDataProject/Output'#sys.argv[1]
keyspace = 'ldua'#sys.argv[2]
table_name = 'nasalogs4'#sys.argv[3]

cluster = Cluster(['127.0.0.1'])
#cluster = Cluster(['199.60.17.171', '199.60.17.188'])
session = cluster.connect(keyspace)

sc.cassandraTable("keyspace name", "table name")

def read_file(file_data, table_name):
    i = 0
    batch = BatchStatement()
    for line in file_data:
        if (i < 150):
            words = line.split(',')
            x1 = int(words[0])
            x2 = int(words[1])
            x3 = int(words[2])
            x4 = float(words[3])
            x5 = float(words[4])
            x6 = float(words[5])
            x7 = float(words[6])
            x8 = float(words[7])
            x9 = float(words[8])
            x10 = float(words[9])
            x11 = float(words[10])
            x12 = float(words[11])
            batch.add(SimpleStatement(
                "INSERT INTO g" + table_name + "(State_Code, Month, Year, Observation_Count, Observation_Percent, Max_Value, Max_Hour, Arithmetic_Mean, AM_Wind, AM_Temp, AM_RH, AM_Press) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"),
                (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12))
            i = i + 1
        else:
            session.execute(batch)
            batch.clear()
            i = 0



import os
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".csv"):
             file_read = open(os.path.join(root, file))
             table = root.split('/')
             table_name = table[6].split('.')
             read_file(file_read, table_name[0])
