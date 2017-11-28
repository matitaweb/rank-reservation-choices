import datetime
import json, codecs
import logging
import numpy as np
import os
import shutil
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel, IndexToString, VectorAssembler, PCA, PCAModel
from pyspark.ml.clustering import KMeans
from pyspark.ml import PipelineModel
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
import rankchoices.pipeline as pipe
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="KMEANS", stage_stop="KMEANS")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="PCA", stage_stop="PCA")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="LOAD", stage_stop="LOAD")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000.000")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(stage_start="KMEANS")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(stage_start="TEST")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(stage_stop="PCA")

#10.000
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="LOAD", stage_stop="LOAD")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="PCA", stage_stop="TEST")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="DICT", stage_stop="TEST")

#10.000.000
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000.000", stage_start="LOAD", stage_stop="PCA")

#100.000.000
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000.000", stage_start="LOAD", stage_stop="LOAD")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000.000", stage_start="PCA", stage_stop="PCA")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000.000", stage_start="KMEANS", stage_stop="KMEANS")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = onehotencoding.start(base_filename = "data/light_r100.000.000", stage_start="DICT", stage_stop="DICT")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = onehotencoding.start(base_filename = "data/light_r100.000.000", stage_start="TEST", stage_stop="TEST")



"""

def get_input_schema():
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
    return input_schema
    
# DATA LOADING
def load_from_csv (filename, input_schema):
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
    
    #spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    #sqlContext = SQLContext(spark)
    
    cluster_freq_dict = {}
    # make dataframes
    for ci, cgroup in frequency_dict.items():
        c =str(ci)
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
                cluster_freq_dict[c][ar][str(z['IDX'])]= z
                
            cluster_freq_dict[c][ar]['last'] = curr_pos
    
    return cluster_freq_dict

def test_accuracy(kmeans_test_ds, arguments_col_y, cluster_freq_dict):

    accuracyDictList = []
    
    for r in kmeans_test_ds.collect():
        c= str(r['prediction'])
        accuracyDict = get_accuracy(r, cluster_freq_dict, c, arguments_col_y)
        accuracyDictList.append(accuracyDict)
            
    return accuracyDictList

def get_accuracy(r, cluster_freq_dict, c, arguments_col_y):
    accuracyDict = {}
    accuracyDict['prediction'] = c
    tot_acc=0
    for ar in arguments_col_y:
        val= str(r[ar])
        
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
    return accuracyDict
    
def save_cluster_freq_dict(cluster_freq_dict, file_name_dir):
    #if os.path.exists(file_name_dir): shutil.rmtree(file_name_dir)
    #os.makedirs(file_name_dir)
    if os.path.exists(file_name_dir): os.remove(file_name_dir)
    with open(file_name_dir, 'wb') as f:
        json.dump(cluster_freq_dict, codecs.getwriter('utf-8')(f), ensure_ascii=False)
  
        
def load_cluster_freq_dict(file_name_dir):
    
    with open(file_name_dir) as json_data:
        d = json.load(json_data)
        return d
    
    """
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
    """

def write_report(filename, tot_col, k_kmeans, arguments_col, accuracyDictList, accuracyMeanList, time_duration_split, time_duration_pca, time_duration_kmean, time_duration_test,  k_pca="-", k_pca_perc="-", split="-", split_col="-"):
    file = open(filename, 'w')
    file.write('filename: ' + str(filename)+'\n')
    file.write('k_pca: ' + str(k_pca_perc) + "% " + str(k_pca)+ ' / '+str(tot_col)+'\n')
    file.write('k_kmeans: ' + str(k_kmeans)+'\n')
    
    file.write('time split: ' + str(datetime.timedelta(seconds=time_duration_split.total_seconds()))+'  \n')
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

def get_stringindexer_model_dict(string_argument_col, df):
    indexer_dict = {}
    for e in string_argument_col:
        inputCol = e[0]
        outputCol = "INDEX_" + e[1]
        stringindexer = StringIndexer(inputCol=inputCol, outputCol=outputCol) 
        stringindexer_model = stringindexer.fit(df)
        indexer_dict[e[1]] = stringindexer_model
    return indexer_dict

def load_stringindexer_model_dict(stringindexer_path):
    indexer_dict = {}
    dirs = [d for d in os.listdir(stringindexer_path) if os.path.isdir(os.path.join(stringindexer_path, d))]
    
    for d in dirs:
        stringindexer_model = StringIndexerModel.load(os.path.join(stringindexer_path, d))
        indexer_dict[d] = stringindexer_model
    return indexer_dict

def apply_stringindexer_model_dict(string_argument_col, df, indexer_dict):
    df_indexed = df
    for e in string_argument_col:
        outputCol = "INDEX_" + e[1]
        stringindexer_model = indexer_dict[e[1]]
        
        df_indexed = stringindexer_model.transform(df_indexed)
        #trasform float index in integer
        df_indexed = df_indexed.withColumn(e[1], df_indexed[outputCol].cast(IntegerType()))
    return df_indexed

def get_onehotencoding_model(arguments_col, df, ohe_col):
    ohe_col_pair = zip([x for x in arguments_col if not x == 'X_ETA'], ohe_col)
    encoders=[OneHotEncoder(inputCol=x[0], outputCol=x[1]) for x in ohe_col_pair]
    assemblerOHE = VectorAssembler(inputCols=['X_ETA']+ohe_col, outputCol="features")
    pipeline = Pipeline(stages=encoders+[assemblerOHE])
    ohe_model=pipeline.fit(df)
    return ohe_model

def apply_onehotencoding_model(df, ohe_model, ohe_col):
    df_ohe=ohe_model.transform(df)
    for x in ohe_col: df_ohe = df_ohe.drop(x)
    return df_ohe

def get_pca_model(k_pca, train_ds, pcaInputCol, pcaOutputCol):
    pca = PCA(k=k_pca, inputCol=pcaInputCol, outputCol=pcaOutputCol) #Argument with more than 65535 cols
    pca_model = pca.fit(train_ds)
    return pca_model

def start(base_filename = "data/light_r10.000",  split= [0.99, 0.01], k_pca_perc = 1, k_means_num = 100,stage_start="LOAD", stage_stop="TEST"):

    # stage_start, stage_stop  -> LOAD | PCA | KMEANS | DICT | TEST

    # INPUT DATA
    input_filename         = base_filename+".csv"
    
    string_indexer_path_dir = base_filename + "-indexer"
    ohe_path_dir = base_filename + "-onehotencoding"
    output_train_file_name = base_filename+"-train.parquet"
    output_test_file_name  = base_filename+"-test.parquet"
    
    pca_path_dir =  base_filename+"-pca-model"
    output_pca_train_filename = base_filename+"-pca-train.parquet"
    output_pca_test_filename  = base_filename+"-pca-test.parquet"
    
    
    cluster_freq_dict_filename = base_filename+"-dict.json"
   
    
    random_seed = 1
    arguments_col_string = [('STRING_X_PRESTAZIONE', 'X_PRESTAZIONE'), ('STRING_Y_UE', 'Y_UE')]
    arguments_col_x = [ 'X_ETA', 'X_SESSO', 'X_GRADO_URG', 'X_PRESTAZIONE']
    arguments_col_y = [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
    arguments_col = arguments_col_x + arguments_col_y
            
    
    #############
    # LOAD DATA #
    #############
    
    train_ds = None
    test_ds = None
    t1 = datetime.datetime.now()
    if(stage_start == "LOAD"):
        dfraw = load_from_csv (input_filename, get_input_schema())
        
        # QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE
        df = quantize(dfraw)
        
        # STRING INDEXER
        indexer_dict = get_stringindexer_model_dict(arguments_col_string, df)
        #stringindexer_path= "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000-indexer"
        #indexer_dict = load_stringindexer_model_dict(stringindexer_path)
        dfi = apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)
        
        # ONE HOT ENCODING
        ohe_col = ["OHE_"+x for x in arguments_col if not x == 'X_ETA']
        ohe_model = get_onehotencoding_model(arguments_col, dfi, ohe_col)
        #ohe_model_path = "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000-onehotencoding"
        #ohe_model=PipelineModel.load(ohe_model_path)
        df_ohe = apply_onehotencoding_model(dfi, ohe_model, ohe_col)
        
        # TRAINING / TEST SPLIT
        (train_ds, test_ds) = df_ohe.randomSplit(split, random_seed)
    
    time_duration_split = (datetime.datetime.now()-t1)
    print('time split: ' + str(datetime.timedelta(seconds=time_duration_split.total_seconds())))
    if( stage_stop == "LOAD"):
        train_ds.write.parquet(output_train_file_name, mode="overwrite")
        test_ds.write.parquet(output_test_file_name, mode="overwrite")
        print('STOP at : ' + str(stage_stop) + ", " + str(datetime.timedelta(seconds=time_duration_split.total_seconds())))
        print('Snapshot train-set: ' + output_train_file_name)
        print('Snapshot test-set: ' + output_test_file_name)
        
        if os.path.exists(string_indexer_path_dir): shutil.rmtree(string_indexer_path_dir)
        for k,indexer in indexer_dict.items():
            string_indexer_path = os.path.join(string_indexer_path_dir,k)
            print('Snaphot indexer: ' + string_indexer_path)
            indexer.save(string_indexer_path)
        
        if os.path.exists(ohe_path_dir): shutil.rmtree(ohe_path_dir)
        print('Snaphot one hot encoder: ' + ohe_path_dir)
        ohe_model.save(ohe_path_dir)
        return train_ds, test_ds, None, None, None
    
    
    #######
    # PCA #
    #######
    
    pcaInputCol="features"
    pcaOutputCol="pca_features"
    pca_model=None
    train_ds_pca=None 
    test_ds_pca=None
    t1 = datetime.datetime.now()
    if(stage_start == "LOAD" or stage_start == "PCA"):
        if(stage_start == "PCA"):
            train_ds = load_from_parquet (output_train_file_name)
            test_ds = load_from_parquet (output_test_file_name)
        tot_col = len(train_ds.head(1)[0]['features'])
        k_pca = int(tot_col*k_pca_perc/100)
        print("pca tot_col: " + str(tot_col) + " reduce to: " + str(k_pca) )
        pca_model = get_pca_model(k_pca, train_ds, pcaInputCol, pcaOutputCol)
        train_ds_pca = pca_model.transform(train_ds)
        test_ds_pca = pca_model.transform(test_ds)
        
    time_duration_pca = (datetime.datetime.now()-t1)
    print('time pca: ' + str(datetime.timedelta(seconds=time_duration_pca.total_seconds())))
    if( stage_stop == "PCA"):
        train_ds_pca.write.parquet(output_pca_train_filename, mode="overwrite")
        test_ds_pca.write.parquet(output_pca_test_filename, mode="overwrite")
        if os.path.exists(pca_path_dir): shutil.rmtree(pca_path_dir)
        pca_model.save(pca_path_dir)
        print('STOP at : ' + str(stage_stop) + ", " + str(datetime.timedelta(seconds=time_duration_pca.total_seconds())))
        print('Snapshot pca train-set: ' + output_pca_train_filename)
        print('Snapshot pca test-set: ' + output_pca_test_filename)
        return train_ds_pca, test_ds_pca, None, None, None
    
    
    ##########
    # KMEANS #
    ##########
    
    t1 = datetime.datetime.now()
    kmeans_train_ds = None
    kmeans_test_ds = None
    output_kmeans_train_ds_filename = base_filename+"-kmeans-train.parquet"
    output_kmeans_test_ds_filename = base_filename+"-kmeans-test.parquet"
    
    if(stage_start == "LOAD" or stage_start == "PCA" or stage_start=="KMEANS"):
        if(stage_start == 'KMEANS'):
            train_ds_pca = load_from_parquet (output_pca_train_filename)
            test_ds_pca = load_from_parquet (output_pca_test_filename)
            
        kmeans = KMeans().setK(k_means_num).setSeed(1).setFeaturesCol(pcaOutputCol)
        kmeans_model_fitted = kmeans.fit(train_ds_pca)
        file_name_dir_kmeans = base_filename+".kmeans"
        if os.path.exists(file_name_dir_kmeans): shutil.rmtree(file_name_dir_kmeans)
        kmeans_model_fitted.save(file_name_dir_kmeans)
        
        kmeans_train_ds = kmeans_model_fitted.transform(train_ds_pca)   
        kmeans_test_ds = kmeans_model_fitted.transform(test_ds_pca) 
        

    time_duration_kmean = (datetime.datetime.now()-t1)
    print('time kmean: ' + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds())))
    if( stage_stop == "KMEANS"):
        kmeans_train_ds.write.parquet(output_kmeans_train_ds_filename, mode="overwrite")
        kmeans_test_ds.write.parquet(output_kmeans_test_ds_filename, mode="overwrite")
        print('STOP at : ' + str(stage_stop) + ", " + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds())))
        return kmeans_train_ds, kmeans_test_ds, None, None, None
    
    
    ##############################
    # KMEAN FREQUENCY DICTIONARY #
    ##############################
    
    t1 = datetime.datetime.now()
    cluster_freq_dict = None
    if(stage_start == "LOAD" or stage_start == "PCA" or stage_start=="KMEANS" or stage_start=="DICT"):
        if(stage_start == 'DICT'):
            kmeans_train_ds = load_from_parquet (output_kmeans_train_ds_filename)
            kmeans_test_ds = load_from_parquet (output_kmeans_test_ds_filename)
            
        frequency_dict = get_cluster_freq_dict(kmeans_train_ds, arguments_col_y)
        cluster_freq_dict = build_cluster_freq_dict(frequency_dict)
        save_cluster_freq_dict(cluster_freq_dict, cluster_freq_dict_filename)
        
    time_duration_freq_dict = (datetime.datetime.now()-t1)
    print('time freq dict: ' + str(datetime.timedelta(seconds=time_duration_freq_dict.total_seconds())))
    if( stage_stop == "DICT"):
        print('STOP at : ' + str(stage_stop))
        return kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, None, None
    
    
    #################
    # TEST ACCURACY #
    #################
    t1 = datetime.datetime.now()
    if(stage_start == "LOAD" or stage_start == "PCA" or stage_start=="KMEANS" or stage_start=="DICT" or stage_start=="TEST"):
        if(stage_start == 'TEST'):
            cluster_freq_dict = load_cluster_freq_dict(cluster_freq_dict_filename)
            kmeans_train_ds = load_from_parquet (output_kmeans_train_ds_filename)
            kmeans_test_ds = load_from_parquet (output_kmeans_test_ds_filename)
        accuracyDictList = test_accuracy(kmeans_test_ds, arguments_col_y, cluster_freq_dict)
        accuracyMeanList = [e['mean_acc'] for e in accuracyDictList]
        mean_acc= np.mean(accuracyMeanList)
    time_duration_test = (datetime.datetime.now()-t1)
    print('time test: ' + str(datetime.timedelta(seconds=time_duration_test.total_seconds())))
    
    ###################
    # REPORT ACCURACY #
    ###################
    
    tot_rows = kmeans_train_ds.count() + kmeans_test_ds.count()
    split_col=(kmeans_train_ds.count(), kmeans_test_ds.count())
    
    report_filename = base_filename +".report.txt"
    if os.path.exists(report_filename): os.remove(report_filename)
    tot_col = len(kmeans_train_ds.head(1)[0]['features'])
    k_pca = int(tot_col*k_pca_perc/100)
    write_report(report_filename, tot_col, k_means_num, arguments_col_y, accuracyDictList, accuracyMeanList, time_duration_split, time_duration_pca, time_duration_kmean, time_duration_test, k_pca=k_pca, k_pca_perc=k_pca_perc, split=split, split_col=split_col)

    
    return kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc






