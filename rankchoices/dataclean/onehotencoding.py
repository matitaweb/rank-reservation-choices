import datetime
import logging
import numpy as np
import os
import shutil
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.sql.functions import lead, col, sum, count, last, greatest
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

"""
import rankchoices.dataclean.onehotencoding as onehotencoding
df_ohe, kmeans_train_ds, kmeans_test_ds, kmeans_model_fitted, frequency_dict, cluster_freq_dict, accuracyDictList, mean_acc = onehotencoding.start()

dfraw = onehotencoding.load_from_csv (input_filename)
df = onehotencoding.quantize(dfraw)
dfi, indexer_dict  = onehotencoding.add_stringindexer(arguments_col_string, df)
df_ohe = onehotencoding.get_onehotencoding(arguments_col, dfi)
(train_ds, test_ds) = df_ohe.randomSplit(split, random_seed)

"""


# DATA LOADING
def load_from_csv (filename):
    input_schema = StructType([
        StructField("X_ETA", IntegerType()),
        StructField("X_SESSO", IntegerType()),
        StructField("X_GRADO_URG", IntegerType()),
        StructField("STRING_X_PRESTAZIONE", StringType()), 
        StructField("STRING_Y_UE", StringType()),
        StructField("Y_GIORNO_SETTIMANA", IntegerType()),
        StructField("Y_MESE_ANNO", IntegerType()),
        StructField("Y_FASCIA_ORARIA", IntegerType()),
        StructField("Y_GIORNI_ALLA_PRENOTAZIONE", IntegerType())
    ])
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.csv(filename, header=True, mode="DROPMALFORMED",  schema=input_schema, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
    return result

def load_from_parquet (filename):
    
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.parquet(filename)
    return result    
    
def convert_y_giorni_alla_prenotazione(x):
    if x < 2:
        return 0 #"1gg"
    if x < 4:
        return 1 #"3gg"
    if x < 6:
        return 2 #"5gg"
    if x < 8:
        return 3 #"7gg"
    if x < 16:
        return 4 #"15gg"
    if x < 21:
        return 5 #"20gg"
    if x < 31:
        return 6 #"30gg"
    if x < 61:
        return 7 #"60gg"
        
    return 8 #"oltre60gg"    

# CREATE FREQUENCY DICTIONARY ON arguments_col_y
def get_cluster_freq_dict(kmeans_train_ds, arguments_col_y):

    frequency_dict={}

    for ar in arguments_col_y:
        ar_group = kmeans_train_ds.groupBy("prediction", ar).agg(count("*").alias("count")).collect()
        
        # make dictionary
        for r in ar_group :
            c = r['prediction']
            if not c in frequency_dict: frequency_dict[c] = {}
            if not ar in frequency_dict[c]:frequency_dict[c][ar] = []
            frequency_dict[c][ar].append({'IDX': r[ar], 'OCCUR': r['count'], 'POS': -1})
    
    return frequency_dict
    #print("END EXTRACT DATA FROM DATAFRAME")
    
def build_cluster_freq_dict(frequency_dict):
    
    schema = StructType([
        StructField("IDX", IntegerType(), True),
        StructField("OCCUR", IntegerType(), True),
        StructField("POS", IntegerType(), True)
    ])
    
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    sqlContext = SQLContext(spark)
    
    cluster_freq_dict = {}
    # make dataframes
    for c, cgroup in frequency_dict.items():
        cluster_freq_dict[c]={}
        for ar, vallist in cgroup.items():
            val_sorted = sorted(vallist, key=lambda x: x['OCCUR'], reverse=True)
            curr_pos = 0
            curr_val = -1
            cluster_freq_dict[c][ar]={}
            for z in val_sorted :
                if(z['OCCUR'] != curr_val):
                    curr_val = z['OCCUR']
                    curr_pos=curr_pos+1
                z['POS']= curr_pos
                cluster_freq_dict[c][ar][z['IDX']]= z
                
            #df = sqlContext.createDataFrame(val_sorted, schema)
            #df = df.orderBy(df.POS.asc())
            
            cluster_freq_dict[c][ar]['last'] = curr_pos
    
    return cluster_freq_dict

def test_accuracy(kmeans_test_ds, arguments_col_y, cluster_freq_dict):

    accuracyDictList = []
    
    for r in kmeans_test_ds.collect():
        c= r['prediction']
        
        accuracyDict = {}
        accuracyDict['prediction'] = c
        tot_acc=0
        
        for ar in arguments_col_y:
            val= r[ar]
            
            # valori di default
            last_position = None
            position = None
            acc= 0
            accuracyDict[ar] = {}
            
            if ar in cluster_freq_dict[c]:
                last_position = cluster_freq_dict[c][ar]['last']
                if val in cluster_freq_dict[c][ar]:
                    position = cluster_freq_dict[c][ar][val]['POS'] - 1
                    acc = (float(last_position)-float(position))/float(last_position)
                
            accuracyDict[ar]["VAL"]=val
            accuracyDict[ar]["POS"]=position
            accuracyDict[ar]["LAST_POS"]=last_position
                
            accuracyDict[ar]['ACC'] = acc
            tot_acc = tot_acc+acc
            
        accuracyDict['tot_acc'] = tot_acc
        accuracyDict['mean_acc'] = tot_acc/len(arguments_col_y)
        accuracyDictList.append(accuracyDict)
            
    return accuracyDictList

def save_cluster_freq_dict(cluster_freq_dict, file_name_dir):
    if os.path.exists(file_name_dir): shutil.rmtree(file_name_dir)
    os.makedirs(file_name_dir)
    
    for c, cgroup in cluster_freq_dict.items():
        for g, df in cgroup.items():
            fname = file_name_dir+"/"+ str(c) + "." +str(g)+".parquet"
            df.write.parquet(fname)
        
def load_cluster_freq_dict(file_name_dir):
    cluster_freq_dict =  {}
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    for f in os.listdir(file_name_dir) :
        if os.path.isfile(os.path.join(file_name_dir, f)):
            df = spark.read.csv(os.path.join(file_name_dir,f))
            fname_split = f.split(".")
            c= int(fname_split[0])
            col_group_name = str(fname_split[1])
            
            if not c in cluster_freq_dict:
                cluster_freq_dict[c]={}
                
            cluster_freq_dict[c][col_group_name]=df
            
    return cluster_freq_dict

def write_report(filename, tot_col, k_kmeans, arguments_col, accuracyDictList, accuracyMeanList, time_duration_pca, time_duration_kmean, time_duration_test,  k_pca="-", k_pca_perc="-", split="-", split_col="-"):
    file = open(filename, 'w')
    file.write('filename: ' + str(filename)+'\n')
    file.write('k_pca: ' + str(k_pca_perc) + "% " + str(k_pca)+ ' / '+str(tot_col)+'\n')
    file.write('k_kmeans: ' + str(k_kmeans)+'\n')
    
    file.write('time pca: ' + str(datetime.timedelta(seconds=time_duration_pca.total_seconds()))+'  \n')
    file.write('time kmean: ' + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds()))+'  \n')
    file.write('time test: ' + str(datetime.timedelta(seconds=time_duration_test.total_seconds()))+'  \n')
    file.write('train, test: ' + str(split)+'\n')
    file.write('train, test: ' + str(split_col)+'\n\n\n')
    
    mean = np.mean(accuracyMeanList)
    file.write('mean acc.: ' + str(mean)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.mean([e[c]['ACC'] for e in accuracyDictList])
        file.write("MEAN: " + str(c) + ' -> ' + str(m)+'\n')
        
    file.write('\n\n\n')
    
    maxV = np.max(accuracyMeanList)
    file.write('max acc.: ' + str(maxV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.max([e[c]['ACC'] for e in accuracyDictList])
        file.write("MAX: " + str(c) + ' -> ' + str(m)+'\n')
    
    file.write('\n\n\n')

    minV = np.min(accuracyMeanList)
    file.write('min acc.: ' + str(minV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.min([e[c]['ACC'] for e in accuracyDictList])
        file.write("MIN: " + str(c) + ' -> ' + str(m)+'\n')        
    
    file.write('------------------------------------'+'\n')
    
    file.write('\n\n\n')
    
    file.write('------------------------------------'+'\n')
    
    file.close()

def quantize(dfraw):
    my_udf = functions.UserDefinedFunction(convert_y_giorni_alla_prenotazione, types.IntegerType())
    df = dfraw.withColumnRenamed("Y_GIORNI_ALLA_PRENOTAZIONE", "Y_GIORNI_ALLA_PRENOTAZIONE_OLD")
    df = df.withColumn("Y_GIORNI_ALLA_PRENOTAZIONE",my_udf(df["Y_GIORNI_ALLA_PRENOTAZIONE_OLD"]))
    df = df.drop('Y_GIORNI_ALLA_PRENOTAZIONE_OLD')
    return df

def add_stringindexer(string_argument_col, df):
    df_indexed = df
    indexer_dict = {}
    for e in string_argument_col:
        inputCol = e[0]
        outputCol = "INDEX_" + e[1]
        stringindexer = StringIndexer(inputCol=inputCol, outputCol=outputCol) 
        indexer_dict[outputCol] = stringindexer
        stringindexer_model = stringindexer.fit(df_indexed)
        
        df_indexed = stringindexer_model.transform(df_indexed)
        df_indexed = df_indexed.withColumn(e[1], df_indexed[outputCol].cast(IntegerType()))
    return df_indexed, indexer_dict

def get_onehotencoding(arguments_col, df):
    ohe_col = ["OHE_"+x for x in arguments_col if not x == 'X_ETA']
    ohe_col_pair = zip([x for x in arguments_col if not x == 'X_ETA'], ohe_col)
    encoders=[OneHotEncoder(inputCol=x[0], outputCol=x[1]) for x in ohe_col_pair]
    assemblerOHE = VectorAssembler(inputCols=['X_ETA']+ohe_col, outputCol="features")
    pipeline = Pipeline(stages=encoders+[assemblerOHE])
    model=pipeline.fit(df)
    df_ohe=model.transform(df)
    for x in ohe_col: df_ohe = df_ohe.drop(x)
    return df_ohe

def apply_pca(k_pca, train_ds, output_pca_train_filename, test_ds, output_pca_test_filename, pcaInputCol, pcaOutputCol):
    pca = PCA(k=k_pca, inputCol=pcaInputCol, outputCol=pcaOutputCol) #Argument with more than 65535 cols
    pca_model = pca.fit(train_ds)
    
    train_ds_pca = pca_model.transform(train_ds) #train_ds_pca = train_ds_pca.drop(pcaInputCol)
    #train_ds_pca.write.parquet(output_pca_train_filename, mode="overwrite")
    
    test_ds_pca = pca_model.transform(test_ds) #test_ds_pca = test_ds_pca.drop(pcaInputCol)
    #test_ds_pca.write.parquet(output_pca_test_filename, mode="overwrite")
    return pca_model, train_ds_pca, test_ds_pca


def start():

    # INPUT DATA
    base_filename                   = "data/light_r100.000"
    input_filename         = base_filename+".csv"
    output_train_file_name = base_filename+"-train.parquet"
    output_test_file_name  = base_filename+"-test.parquet"
    output_pca_train_filename = base_filename+"-pca-train.parquet"
    output_pca_test_filename  = base_filename+"-pca-test.parquet"
    k_means_num = 100
    
    
    split= [0.8, 0.2]
    random_seed = 1
    arguments_col_string = [('STRING_X_PRESTAZIONE', 'X_PRESTAZIONE'), ('STRING_Y_UE', 'Y_UE')]
    arguments_col_x = [ 'X_ETA', 'X_SESSO', 'X_GRADO_URG', 'X_PRESTAZIONE']
    arguments_col_y = [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
    arguments_col = arguments_col_x + arguments_col_y
            
    k_pca_perc = 10
            
    #LOAD DATA
    
    dfraw = load_from_csv (input_filename)
    
    # QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE
    df = quantize(dfraw)
    
    # STRING INDEXER
    dfi, indexer_dict = add_stringindexer(arguments_col_string, df)
    
    # ONE HOT ENCODING
    df_ohe = get_onehotencoding(arguments_col, dfi)
    
    
    # TRAINING / TEST SPLIT
    (train_ds, test_ds) = df_ohe.randomSplit(split, random_seed)
    #train_ds.write.parquet(output_train_file_name, mode="overwrite")
    #test_ds.write.parquet(output_test_file_name, mode="overwrite")
    
    
    #PCA
    tot_col = len(train_ds.head(1)[0]['features'])
    k_pca = int(tot_col*k_pca_perc/100)
    pcaInputCol="features"
    pcaOutputCol="pca_features"
    t1 = datetime.datetime.now()
    
    pca_model, train_ds_pca, test_ds_pca = apply_pca(k_pca, train_ds, output_pca_train_filename, test_ds, output_pca_test_filename, pcaInputCol, pcaOutputCol)
    
    time_duration_pca = (datetime.datetime.now()-t1)
    
    #K MEANS
    t1 = datetime.datetime.now()
    kmeans = KMeans().setK(k_means_num).setSeed(1).setFeaturesCol(pcaOutputCol)
    kmeans_model_fitted = kmeans.fit(train_ds_pca)
    kmeans_train_ds = kmeans_model_fitted.transform(train_ds_pca)   
    file_name_dir_kmeans = base_filename+".kmeans"
    if os.path.exists(file_name_dir_kmeans): shutil.rmtree(file_name_dir_kmeans)
    kmeans_model_fitted.save(file_name_dir_kmeans)
    time_duration_kmean = (datetime.datetime.now()-t1)
    
    # KMEAN PREDICTION ON TEST SET
    kmeans_test_ds = kmeans_model_fitted.transform(test_ds_pca)   
    
    # KMEAN FREQUENCY DICTIONARY
    frequency_dict = get_cluster_freq_dict(kmeans_train_ds, arguments_col_y)
    cluster_freq_dict = build_cluster_freq_dict(frequency_dict)
    
    
    #TEST  ACCURACY
    t1 = datetime.datetime.now()
    accuracyDictList = test_accuracy(kmeans_test_ds, arguments_col_y, cluster_freq_dict)
    accuracyMeanList = [e['mean_acc'] for e in accuracyDictList]
    mean_acc= np.mean(accuracyMeanList)
    time_duration_test = (datetime.datetime.now()-t1)
    
    # REPORT ACCURACY
    tot_rows = kmeans_train_ds.count() + kmeans_test_ds.count()
    split_col=(kmeans_train_ds.count(), kmeans_test_ds.count())
    
    report_filename = base_filename +".report.txt"
    if os.path.exists(report_filename): os.remove(report_filename)
    write_report(report_filename, tot_col, k_means_num, arguments_col_y, accuracyDictList, accuracyMeanList, time_duration_pca, time_duration_kmean, time_duration_test, k_pca=k_pca, k_pca_perc=k_pca_perc, split=split, split_col=split_col)

    
    return df_ohe, kmeans_train_ds, kmeans_test_ds, kmeans_model_fitted, frequency_dict, cluster_freq_dict, accuracyDictList, mean_acc






