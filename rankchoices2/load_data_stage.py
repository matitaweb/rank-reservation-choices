import rank_utils
import os
import shutil
from pyspark.sql.functions import col
import json
import codecs
import datetime
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline

class DataLoaderService:

    def __init__(self):
        pass
    
    def load_data(self, spark, rankConfig, inputPipeline, pipelineSession):
        t1 = datetime.datetime.now()
        time_duration_load_data = (datetime.datetime.now()-t1)
        return time_duration_load_data
        
    def start_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        
        #load csv
        dfraw = rank_utils.load_from_csv (spark, inputPipeline.input_filename, rankConfig.get_input_schema([]))
        
        # filter row with cell null
        for x in rankConfig.get_input_schema([]):
            dfraw = dfraw.where(col(x.name).isNotNull())
        
        # QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE (ONLY ONE)
        dfq = quantize_all_cols(dfraw, rankConfig.getArgumentsColToQuantize())
        
        # METADATA FOR COLUMN RELOAD
        metadataDict = get_metadata(dfq)
        df = add_metadata(dfq, metadataDict)
        save_metadata(df, inputPipeline.metadata_file_name_dir)
        
       
        # STRING INDEXER
        arguments_col_string = rankConfig.getArgumentsColString([])
        indexer_dict = get_stringindexer_model_dict(arguments_col_string, df)
        save_indexer(inputPipeline.string_indexer_path_dir, indexer_dict)
            
        dfi = apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)
        
        
        # ONE HOT ENCODING
        arguments_col_to_drop = rankConfig.getArgumentsColToDrop()
        arguments_col_not_ohe = rankConfig.getArgumentsColNotOHE(arguments_col_to_drop)
        arguments_col = rankConfig.getArgumentsColX(arguments_col_to_drop) + rankConfig.getArgumentsColY(arguments_col_to_drop)
        ohe_col = rankConfig.getOheCol(arguments_col, arguments_col_not_ohe)
        print('OHE_COLS: ' + str(ohe_col))
        encodersDict= get_onehotencoding_model(arguments_col, ohe_col, arguments_col_not_ohe)
        df_ohe = apply_onehotencoding_model(dfi, arguments_col_not_ohe, encodersDict, rankConfig.getOheFeatureOutputColName())
        
        # TRAINING / TEST SPLIT
        (train_ds, test_ds) = df_ohe.randomSplit(inputPipeline.split, inputPipeline.random_seed)
    
        # PUT DATA IN PIPELINE SESSION
        pipelineSession.load_data_stage_train_ds = train_ds
        pipelineSession.load_data_stage_test_ds = test_ds
        
        time_duration_start_stage = (datetime.datetime.now()-t1)
        
        return train_ds, test_ds, time_duration_start_stage
        
        
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        if(pipelineSession.load_data_stage_train_ds != None):
            pipelineSession.load_data_stage_train_ds.write.parquet(inputPipeline.output_train_file_name, mode="overwrite")
        
        if(pipelineSession.load_data_stage_test_ds != None):
            pipelineSession.load_data_stage_test_ds.write.parquet(inputPipeline.output_test_file_name, mode="overwrite")
        
        time_duration_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.load_data_stage_train_ds, pipelineSession.load_data_stage_test_ds, time_duration_snapshot_stage
        
        

    
def quantize_all_cols(dfraw, arguments_col_to_quantize):
    df = dfraw
    for (colname, my_udf) in arguments_col_to_quantize:
        df = quantize(dfraw, my_udf, colname)
    return df
    

def quantize(dfraw, my_udf, colname):
    colname_old = colname+"_OLD"
    df = dfraw.withColumnRenamed(colname, colname_old)
    df = df.withColumn(colname,my_udf(df[colname_old]))
    df = df.drop(colname_old)
    return df

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
    pip = Pipeline(stages=encoders+[assemblerOHE])
    ohe_model=pip.fit(df)
    df_ohe=ohe_model.transform(df)
    #for x in ohe_col: df_ohe = df_ohe.drop(x)
    return df_ohe

def save_indexer(string_indexer_path_dir, indexer_dict):
    if os.path.exists(string_indexer_path_dir): 
        shutil.rmtree(string_indexer_path_dir)
        
    for k,indexer in indexer_dict.items():
        string_indexer_path = os.path.join(string_indexer_path_dir,k)
        print('Snaphot indexer: ' + string_indexer_path)
        indexer.save(string_indexer_path)