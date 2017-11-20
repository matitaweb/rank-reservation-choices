import rankchoices.commons.load_stage as load_stage
import datetime
import logging
import numpy as np
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import lead, col, sum, count
import pyspark.sql.functions as functions
from pyspark.ml import Pipeline

from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession


import pandas as pd
import os
import shutil
import pyspark.sql.types as types


base                   = "data/light_r100.000"
input_filename         = base+".csv"
output_train_file_name = base+"-train.parquet"
output_test_file_name = base+"-test.parquet"


split= [0.99, 0.01]
random_seed = 1
arguments_col_x = [ 'X_ETA', 'X_SESSO', 'X_GRADO_URG', 'X_PRESTAZIONE']
arguments_col_y = [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
arguments_col = arguments_col_x + arguments_col_y
df = load_stage.load_from_csv (input_filename)

# QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE
my_udf = functions.UserDefinedFunction(load_stage.convert_y_giorni_alla_prenotazione, types.IntegerType())
df = df.withColumnRenamed("Y_GIORNI_ALLA_PRENOTAZIONE", "Y_GIORNI_ALLA_PRENOTAZIONE_OLD")
df = df.withColumn("Y_GIORNI_ALLA_PRENOTAZIONE",my_udf(df["Y_GIORNI_ALLA_PRENOTAZIONE_OLD"]))
df = df.drop('Y_GIORNI_ALLA_PRENOTAZIONE_OLD')


# ONE HOT ENCODING
ohe_col = ["OHE_"+x for x in arguments_col if not x == 'X_ETA']
ohe_col_pair = zip([x for x in arguments_col if not x == 'X_ETA'], ohe_col)
encoders=[OneHotEncoder(inputCol=x[0], outputCol=x[1]) for x in ohe_col_pair]
assemblerOHE = VectorAssembler(inputCols=['X_ETA']+ohe_col, outputCol="features")
pipeline = Pipeline(stages=encoders+[assemblerOHE])
model=pipeline.fit(df)
df_ohe=model.transform(df)
for x in ohe_col:
    df_ohe = df_ohe.drop(x)


# TRAINING / TEST SPLIT
(train_ds, test_ds) = df_ohe.randomSplit(split, random_seed)
train_ds.write.parquet(output_train_file_name, mode="overwrite")
test_ds.write.parquet(output_test_file_name, mode="overwrite")


#K MEANS
kmeans = KMeans().setK(100).setSeed(1).setFeaturesCol("features")
kmeans_model_fitted = kmeans.fit(train_ds)
kmeans_train_ds = kmeans_model_fitted.transform(train_ds)   


#eta = kmeans_train_ds.groupBy("prediction", "X_ETA").agg(count("*").alias("COUNT"))
#eta = eta.where(col("prediction").isin({32}))



# CREATE FREQUENCY DICTIONARY ON arguments_col_y
def build_cluster_freq_dict(kmeans_train_ds, arguments_col_y):

    def getKey(i):i['OCCUR']
    frequency_dict={}
    schema = StructType([
        StructField("IDX", IntegerType(), True),
        StructField("OCCUR", IntegerType(), True),
        StructField("POS", IntegerType(), True)
    ])
    
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    sqlContext = SQLContext(spark)
    
    for ar in arguments_col_y:
        ar_group = kmeans_train_ds.groupBy("prediction", ar).agg(count("*").alias("count")).collect()
        
        # make dictionary
        for r in ar_group :
            c = r['prediction']
            if not c in frequency_dict: frequency_dict[c] = {}
            if not ar in frequency_dict[c]:frequency_dict[c][ar] = []
            frequency_dict[c][ar].append({'IDX': r[ar], 'OCCUR': r['count'], 'POS': -1})
            
    cluster_freq_dict = {}
    # make dataframes
    for c, cgroup in frequency_dict.items():
        for ar, vallist in cgroup.items():
            val_sorted = sorted(vallist, key=getKey, reverse=True)
            curr_pos = -1
            curr_val = -1
            for z in val_sorted :
                if(z['OCCUR'] != curr_val):
                    curr_val = z['OCCUR']
                    curr_pos=curr_pos+1
                z['POS']= curr_pos
                
            df = sqlContext.createDataFrame(val_sorted, schema)
            df = df.orderBy(df.POS.asc())
            cluster_freq_dict[c]={}
            cluster_freq_dict[c][ar]=df
    
    return frequency_dict, cluster_freq_dict




# KMEAN PREDICTION ON TEST SET
kmeans_test_ds = kmeans_model_fitted.transform(test_ds)   


"""
schema = StructType([
    StructField("IDX", IntegerType(), True),
    StructField("OCCUR", IntegerType(), True),
    StructField("POS", IntegerType(), True)
])
data = []
sqlContext.createDataFrame(data, schema)
aaa.orderBy(aaa.OCCUR.desc()).show()
"""

# TEST ACCURACY


"""
stringIndexerX_SES = StringIndexer(inputCol="X_SES", outputCol="CT_X_SES")
stringIndexerX_SESmodel = stringIndexerX_SES.fit(df)
stringIndexerX_SESmodel_labels = stringIndexerX_SESmodel.labels
indexed = stringIndexerX_SESmodel.transform(df)
"""




