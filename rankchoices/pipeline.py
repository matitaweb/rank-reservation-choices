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
from pyspark.sql.functions import lead, col, count, last, greatest
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
import math
import pyspark.sql.types as types


"""
import rankchoices.pipeline as pipe
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000", stage_start="LOAD", stage_stop="LOAD")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000", stage_start="PCA", stage_stop="PCA")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000", stage_start="KMEANS", stage_stop="KMEANS")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000", stage_start="DICT", stage_stop="DICT")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000", stage_start="TEST", stage_stop="TEST")

kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000.000")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(stage_start="KMEANS")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(stage_start="TEST")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(stage_stop="PCA")

#10.000
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="LOAD", stage_stop="LOAD")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="PCA", stage_stop="PCA")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="KMEANS", stage_stop="KMEANS")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="DICT", stage_stop="DICT")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000", stage_start="TEST", stage_stop="TEST")

#10.000.000
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r10.000.000", stage_start="LOAD", stage_stop="PCA")

#100.000.000
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "/dati/data/light_r300.000.000", stage_start="LOAD", stage_stop="LOAD")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000.000", stage_start="PCA", stage_stop="PCA")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = pipe.start(base_filename = "data/light_r100.000.000", stage_start="KMEANS", stage_stop="KMEANS")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = onehotencoding.start(base_filename = "data/light_r100.000.000", stage_start="DICT", stage_stop="DICT")
kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc = onehotencoding.start(base_filename = "data/light_r100.000.000", stage_start="TEST", stage_stop="TEST")



"""

def get_input_schema(arguments_col_to_drop):
    filtered = []
    if(not "X_ETA" in arguments_col_to_drop):
        filtered.append(StructField("X_ETA", IntegerType()))
        
    if(not "X_SESSO" in arguments_col_to_drop):
        filtered.append(StructField("X_SESSO", IntegerType()))
        
    if(not "X_GRADO_URG" in arguments_col_to_drop):
        filtered.append(StructField("X_GRADO_URG", IntegerType()))
    
    if(not "STRING_X_PRESTAZIONE" in arguments_col_to_drop):
        filtered.append(StructField("STRING_X_PRESTAZIONE", StringType()))
    
    if(not "STRING_Y_UE" in arguments_col_to_drop):
        filtered.append(StructField("STRING_Y_UE", StringType()))
    
    if(not "Y_GIORNO_SETTIMANA" in arguments_col_to_drop):
        filtered.append(StructField("Y_GIORNO_SETTIMANA", IntegerType()))
    
    if(not "Y_MESE_ANNO" in arguments_col_to_drop):
        filtered.append(StructField("Y_MESE_ANNO", IntegerType()))
    
    if(not "Y_FASCIA_ORARIA" in arguments_col_to_drop):
        filtered.append(StructField("Y_FASCIA_ORARIA", IntegerType()))
    
    if(not "Y_GIORNI_ALLA_PRENOTAZIONE" in arguments_col_to_drop):
        filtered.append(StructField("Y_GIORNI_ALLA_PRENOTAZIONE", IntegerType()))
        
    #print(filtered)
    input_schema = StructType(filtered)
    
    return input_schema
    
# DATA LOADING
def load_from_csv (filename, input_schema):
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.csv(filename, header=True, mode="DROPMALFORMED",  schema=input_schema, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
    return result

def load_from_parquet (filename):
    
    spark = SparkSession.builder.master("local").appName("Rank").config("spark.python.profile", "true").getOrCreate()
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
        print("FREQU:" + ar)
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

def quantize(dfraw, my_udf, colname):
    colname_old= colname+"_OLD"
    df = dfraw.withColumnRenamed(colname, colname_old)
    df = df.withColumn(colname,my_udf(df[colname_old]))
    df = df.drop(colname_old)
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
        meta = df_indexed.schema[outputCol].metadata
  
        meta['ml_attr']['name']=e[1]
        df_indexed = df_indexed.withColumn(e[1], df_indexed[outputCol].cast(IntegerType()))
        df_indexed = df_indexed.withColumn(e[1], col(e[1]).alias(e[1], metadata=meta))
    return df_indexed

def get_onehotencoding_model(arguments_col, ohe_col, arguments_col_not_ohe):
    ohe_col_pair = zip([x for x in arguments_col if not x in arguments_col_not_ohe], ohe_col)
    encodersDict = {}
    for x in ohe_col_pair:
        ohe=OneHotEncoder(dropLast=False, inputCol=x[0], outputCol=x[1])
        encodersDict[x[1]]= ohe
    return encodersDict

def apply_onehotencoding_model(df, arguments_col_not_ohe, encoderDict, outputCol="features"):
    ohe_col = [k for k,x in encoderDict.items()]
    encoders = [x for k, x in encoderDict.items()]
    assemblerOHE = VectorAssembler(inputCols=arguments_col_not_ohe+ohe_col, outputCol=outputCol )
    pipeline = Pipeline(stages=encoders+[assemblerOHE])
    ohe_model=pipeline.fit(df)
    df_ohe=ohe_model.transform(df)
    #for x in ohe_col: df_ohe = df_ohe.drop(x)
    return df_ohe

def get_pca_model(k_pca, train_ds, pcaInputCol, pcaOutputCol):
    pca = PCA(k=k_pca, inputCol=pcaInputCol, outputCol=pcaOutputCol) #Argument with more than 65535 cols
    pca_model = pca.fit(train_ds)
    return pca_model
    
def get_metadata(df):
    metadataDict = {}
    for colname in df.columns:
        if('ml_attr' in df.schema[colname].metadata):
            metadataDict[colname]=df.schema[colname].metadata
            continue
        
        meta = {
            "ml_attr": {
                "vals": [str(x[0]) for x in df.select(colname).distinct().collect()],
                "type": "nominal", 
                "name": colname}
        }
        metadataDict[colname]=meta
    return metadataDict

def add_metadata(df, metadataDict):
    for colname in df.columns:
        #if('ml_attr' in df.schema[colname].metadata):
        #    continue
        
        if(colname in metadataDict):
            meta = metadataDict[colname]
            df = df.withColumn(colname, col(colname).alias(colname, metadata=meta))
    return df    
    
def save_metadata(df, file_name_dir):
    if os.path.exists(file_name_dir): shutil.rmtree(file_name_dir)
    os.makedirs(file_name_dir)
    for colname in df.columns:
        if(not 'ml_attr' in df.schema[colname].metadata):
            continue
        meta = df.schema[colname].metadata
        file_name_dir_col = os.path.join(file_name_dir,colname)
        with open(file_name_dir_col, 'wb') as f:
            json.dump(meta, codecs.getwriter('utf-8')(f), ensure_ascii=False)

def load_metadata(file_name_dir):
    metadataDict = {}
    dirs = [d for d in os.listdir(file_name_dir) if not os.path.isdir(os.path.join(file_name_dir, d))]
    for d in dirs:
        meta = json.load(open(os.path.join(file_name_dir, d)))
        metadataDict[d]=meta
    return metadataDict
        
def validate_with_metadata(rList, metadataDict, exceptList):
    resValidate = {}
    resValidate['valid'] = []
    resValidate['rejected'] = []
    
    for r in rList:
        rejectedCols = {key: value for key, value in r.items() if (not key in exceptList and not str(value) in metadataDict[key]['ml_attr']['vals']) }
            
        if(len(rejectedCols.keys()) == 0):
            resValidate['valid'].append(r);
            continue
        rejectedRow = {}
        rejectedRow['row'] = r
        rejectedRow['rejecterCols'] = rejectedCols
        resValidate['rejected'].append(rejectedRow)
    return resValidate

def validate_with_metadata_exceptList():
    exceptList = ["X_ETA", "Y_GIORNI_ALLA_PRENOTAZIONE"]
    return exceptList


# http://www.codehamster.com/2015/03/09/different-ways-to-calculate-the-euclidean-distance-in-python/
def euclidean0_0 (vector1, vector2):
    ''' calculate the euclidean distance
        input: numpy.arrays or lists
        return: 1. quard distance, 2. euclidean distance
    '''
    quar_distance = 0
    
    if(len(vector1) != len(vector2)):
        raise RuntimeWarning("The length of the two vectors are not the same!")
    zipVector = zip(vector1, vector2)

    for member in zipVector:
        quar_distance += (member[1] - member[0]) ** 2

    return quar_distance, math.sqrt(quar_distance)
 
 
def euclidean0_1(vector1, vector2):
    '''calculate the euclidean distance, no numpy
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist
  
    
def getArgumentsColToDrop():
    return [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']


def getArgumentsColString(arguments_col_to_drop):
    arguments_col_string_all = [('STRING_X_PRESTAZIONE', 'X_PRESTAZIONE'), ('STRING_Y_UE', 'Y_UE')]
    arguments_col_string = [x for x in arguments_col_string_all if x[0] not in arguments_col_to_drop and  x[1] not in arguments_col_to_drop ]
    return arguments_col_string
    

def getArgumentsColX(arguments_col_to_drop):
    arguments_col_x_all = [ 'X_ETA', 'X_SESSO', 'X_GRADO_URG', 'X_PRESTAZIONE']
    arguments_col_x = [x for x in arguments_col_x_all if x not in arguments_col_to_drop]
    return arguments_col_x

def getArgumentsColY(arguments_col_to_drop):
    arguments_col_y_all = [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
    arguments_col_y = [x for x in arguments_col_y_all if x not in arguments_col_to_drop]
    return arguments_col_y

def getArgumentsColNotOHE(arguments_col_to_drop):
    arguments_col_ohe_all = ['X_ETA']
    arguments_col_ohe = [x for x in arguments_col_ohe_all if x not in arguments_col_to_drop]
    return arguments_col_ohe

def save_model_info(model_info_filename, kmeans_centers, k_means_num, k_pca_perc, tot_col, wssse):
    
    k_pca = int(tot_col*k_pca_perc/100)
    
    model_info = {}
    model_info['num_custer'] = k_means_num
    model_info['tot_col'] = tot_col
    model_info['k_pca'] = k_pca
    model_info['k_pca_perc'] = k_pca_perc
    model_info['wssse'] = wssse
    model_info['kmeans_centers'] = kmeans_centers
    
    if os.path.exists(model_info_filename): os.remove(model_info_filename)
    with open(model_info_filename, 'wb') as f:
        json.dump(model_info, codecs.getwriter('utf-8')(f), ensure_ascii=False)


def start(base_filename = "data/light_r10.000",  split= [0.99, 0.01], k_pca_perc = 1, k_means_num = 100, stage_start="LOAD", stage_stop="TEST"):

    # stage_start, stage_stop  -> LOAD | PCA | KMEANS | DICT | TEST

    # INPUT DATA
    input_filename          = base_filename+".csv"
    
    string_indexer_path_dir = base_filename + "-indexer"
    df_indexed_file_name    = base_filename+"-df-indexed.parquet"
    output_train_file_name  = base_filename+"-train.parquet"
    output_test_file_name   = base_filename+"-test.parquet"
    
    pca_path_dir =  base_filename+"-pca-model"
    output_pca_train_filename = base_filename+"-pca-train.parquet"
    output_pca_test_filename  = base_filename+"-pca-test.parquet"
    
    cluster_freq_dict_filename = base_filename+"-dict.json"
    
    model_info_filename = base_filename+"-model-info.json"
   
    
    random_seed = 1
    
    # COLS to ESCLUDE TO SIMPLER MODEL
    arguments_col_to_drop = getArgumentsColToDrop()
    
    # COLS TO TRANSFORM FROM STRING TO INDEX
    arguments_col_string = getArgumentsColString([])
    
    # COLS THAT DEFINE FREQUENCY
    arguments_col_y = getArgumentsColY([])
    
    # COL TO EXCLUDE FROM ONE HOT ENCODING
    arguments_col_not_ohe = getArgumentsColNotOHE(arguments_col_to_drop)
    
    # COLUMNS TO USE IN CLUSTERING
    arguments_col = getArgumentsColX(arguments_col_to_drop) + getArgumentsColY(arguments_col_to_drop)
    print("COLUMNS TO USE IN CLUSTERING: " + str(arguments_col))

    #############
    # LOAD DATA #
    #############
    
    train_ds = None
    test_ds = None
    t1 = datetime.datetime.now()
    if(stage_start == "LOAD"):
        dfraw = load_from_csv (input_filename, get_input_schema([]))
        
        # remove column to exclude

        # QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE (ONLY ONE)
        colname_to_quantize = "Y_GIORNI_ALLA_PRENOTAZIONE"
        my_udf = functions.UserDefinedFunction(convert_y_giorni_alla_prenotazione, types.IntegerType())
        dfq = quantize(dfraw, my_udf, colname_to_quantize)
        
        
        # METADATA FOR COLUMN RELOAD
        metadataDict = get_metadata(dfq)
        df = add_metadata(dfq, metadataDict)
        metadata_file_name_dir = base_filename + "-metadata"
        save_metadata(df, metadata_file_name_dir)
        #print(df.schema['X_SESSO'].metadata)
        
        # STRING INDEXER
        indexer_dict = get_stringindexer_model_dict(arguments_col_string, df)
        if os.path.exists(string_indexer_path_dir): shutil.rmtree(string_indexer_path_dir)
        for k,indexer in indexer_dict.items():
            string_indexer_path = os.path.join(string_indexer_path_dir,k)
            print('Snaphot indexer: ' + string_indexer_path)
            indexer.save(string_indexer_path)
            
        dfi = apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)
        
        # ONE HOT ENCODING
        ohe_col = ["OHE_"+x for x in arguments_col if not x in arguments_col_not_ohe]
        featureOutputCol="features"
        encodersDict= get_onehotencoding_model(arguments_col, ohe_col, arguments_col_not_ohe)
        df_ohe = apply_onehotencoding_model(dfi, arguments_col_not_ohe, encodersDict, featureOutputCol)
        
        # TRAINING / TEST SPLIT
        (train_ds, test_ds) = df_ohe.randomSplit(split, random_seed)
    
    time_duration_split = (datetime.datetime.now()-t1)
    print('time split: ' + str(datetime.timedelta(seconds=time_duration_split.total_seconds())))
    
    
    if( stage_stop == "LOAD"):
        t1 = datetime.datetime.now()
        
        train_ds.write.parquet(output_train_file_name, mode="overwrite")
        test_ds.write.parquet(output_test_file_name, mode="overwrite")
        dfi.write.parquet(df_indexed_file_name, mode="overwrite")
        print('Snapshot train-set: ' + output_train_file_name)
        print('Snapshot test-set: ' + output_test_file_name)
        time_duration_split_save = (datetime.datetime.now()-t1)
        print('STOP at : ' + str(stage_stop) + ", save in: " + str(datetime.timedelta(seconds=time_duration_split_save.total_seconds())))
        return train_ds, test_ds, df_ohe, dfi, None
    
    
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
        if os.path.exists(pca_path_dir): shutil.rmtree(pca_path_dir)
        pca_model.save(pca_path_dir)
        
        train_ds_pca = pca_model.transform(train_ds)
        test_ds_pca = pca_model.transform(test_ds)
        
    time_duration_pca = (datetime.datetime.now()-t1)
    print('time pca: ' + str(datetime.timedelta(seconds=time_duration_pca.total_seconds())))
    if( stage_stop == "PCA"):
        t1 = datetime.datetime.now()
        
        train_ds_pca.write.parquet(output_pca_train_filename, mode="overwrite")
        test_ds_pca.write.parquet(output_pca_test_filename, mode="overwrite")
        
        time_duration_pca_save = (datetime.datetime.now()-t1)
        print('STOP at : ' + str(stage_stop) + ", save in: " + str(datetime.timedelta(seconds=time_duration_pca_save.total_seconds())))
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
            print ("TRAIN SIZE: " + train_ds_pca.count())
            
        kmeans = KMeans().setK(k_means_num).setSeed(random_seed).setFeaturesCol(pcaOutputCol)
        kmeans_model_fitted = kmeans.fit(train_ds_pca)
        
        file_name_dir_kmeans = base_filename+".kmeans"
        if os.path.exists(file_name_dir_kmeans): shutil.rmtree(file_name_dir_kmeans)
        kmeans_model_fitted.save(file_name_dir_kmeans)
        
        kmeans_train_ds = kmeans_model_fitted.transform(train_ds_pca)   
        kmeans_test_ds = kmeans_model_fitted.transform(test_ds_pca) 
        

    time_duration_kmean = (datetime.datetime.now()-t1)
    print('time kmean: ' + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds())))
    if( stage_stop == "KMEANS"):
        t1 = datetime.datetime.now()
        kmeans_train_ds.write.parquet(output_kmeans_train_ds_filename, mode="overwrite")
        kmeans_test_ds.write.parquet(output_kmeans_test_ds_filename, mode="overwrite")
        time_duration_kmean_save = (datetime.datetime.now()-t1)
        print('Snapshot kmeans train-set: ' + output_kmeans_train_ds_filename)
        print('Snapshot kmeans test-set: ' + output_kmeans_test_ds_filename)
        print('STOP at : ' + str(stage_stop) + ", save in: " + str(datetime.timedelta(seconds=time_duration_kmean_save.total_seconds())))
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
        print('Snapshot dict test-set: ' + cluster_freq_dict_filename)
        
    time_duration_freq_dict = (datetime.datetime.now()-t1)
    print('time freq dict: ' + str(datetime.timedelta(seconds=time_duration_freq_dict.total_seconds())))
    if(stage_stop == "DICT"):
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
            file_name_dir_kmeans = base_filename+".kmeans"
            kmeans_model_fitted = KMeansModel.load(file_name_dir_kmeans)  # load from file system 
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
    
    # Evaluate clustering by computing Within Set Sum of Squared Errors.
    wssse = kmeans_model_fitted.computeCost(kmeans_train_ds)
    print("Within Set Sum of Squared Errors = " + str(wssse))
    
    # save json model info
    kmeans_centers = [str(center) for center in kmeans_model_fitted.clusterCenters()]
    save_model_info(model_info_filename, kmeans_centers, k_means_num, k_pca_perc, tot_col, wssse)
    
    write_report(report_filename, tot_col, k_means_num, arguments_col_y, accuracyDictList, accuracyMeanList, time_duration_split, time_duration_pca, time_duration_kmean, time_duration_test, k_pca=k_pca, k_pca_perc=k_pca_perc, split=split, split_col=split_col)
    print('Snapshot report test-set: ' + report_filename)
    
    return kmeans_train_ds, kmeans_test_ds, cluster_freq_dict, accuracyDictList, mean_acc






