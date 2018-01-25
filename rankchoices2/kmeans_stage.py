import rank_utils
from pyspark.ml.clustering import KMeans
import os
import shutil
import datetime
import json
import codecs

class KmeansService:
    def __init__(self):
        pass
        
        
    def load_data(self, spark, rankConfig, inputPipeline, pipelineSession):
        t1 = datetime.datetime.now()
        
        if(pipelineSession.pca_stage_train_ds == None):
            pipelineSession.pca_stage_train_ds = rank_utils.load_from_parquet (spark, inputPipeline.output_pca_train_filename)
            
        if(pipelineSession.pca_stage_test_ds == None):
            pipelineSession.pca_stage_test_ds = rank_utils.load_from_parquet (spark, inputPipeline.output_pca_test_filename)
            
        time_duration_load_data = (datetime.datetime.now()-t1)
        return time_duration_load_data
        
    
    def start_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        
        if(pipelineSession.pca_stage_train_ds == None):
            raise ValueError('ERROR: no value for pca_stage_train_ds')
            
        if(pipelineSession.pca_stage_test_ds == None):
            raise ValueError('ERROR: no value for pca_stage_test_ds')
            

        kmeans = KMeans().setK(inputPipeline.k_means_num).setSeed(inputPipeline.random_seed).setFeaturesCol(rankConfig.getPcaFeatureOutputCol())
        pipelineSession.kmeans_model_fitted = kmeans.fit(pipelineSession.pca_stage_train_ds)

        if os.path.exists(inputPipeline.file_name_dir_kmeans): 
            shutil.rmtree(inputPipeline.file_name_dir_kmeans)
        pipelineSession.kmeans_model_fitted.save(inputPipeline.file_name_dir_kmeans)
        
        
        pipelineSession.kmeans_stage_train_ds = pipelineSession.kmeans_model_fitted.transform(pipelineSession.pca_stage_train_ds)   
        pipelineSession.kmeans_stage_test_ds = pipelineSession.kmeans_model_fitted.transform(pipelineSession.pca_stage_test_ds) 
        
        save_model_info(pipelineSession, rankConfig, inputPipeline)
        
        time_duration_start_stage = (datetime.datetime.now()-t1)
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_start_stage
        
    
    
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        if(pipelineSession.kmeans_stage_train_ds != None):
            pipelineSession.kmeans_stage_train_ds.write.parquet(inputPipeline.output_kmeans_train_ds_filename, mode="overwrite")
        
        if(pipelineSession.kmeans_stage_test_ds != None):
            pipelineSession.kmeans_stage_test_ds.write.parquet(inputPipeline.output_kmeans_test_ds_filename, mode="overwrite")
        
        time_duration_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_snapshot_stage
    
def save_model_info(pipelineSession, rankConfig, inputPipeline):
    
    
    tot_col = len(pipelineSession.pca_stage_train_ds.head(1)[0][rankConfig.getOheFeatureOutputColName()])
    
    wssse = pipelineSession.kmeans_model_fitted.computeCost(pipelineSession.kmeans_stage_train_ds)

    # save json model info
    kmeans_centers = [str(center) for center in pipelineSession.kmeans_model_fitted.clusterCenters()]

    
    k_pca = int(tot_col*inputPipeline.pca_perc/100)
    
    model_info = {}
    model_info['num_custer'] = inputPipeline.k_means_num
    model_info['tot_col'] = tot_col
    model_info['k_pca'] = k_pca
    model_info['k_pca_perc'] = inputPipeline.pca_perc
    model_info['wssse'] = wssse
    model_info['kmeans_centers'] = kmeans_centers
    
    
    model_info['arguments_col_to_drop'] = rankConfig.getArgumentsColToDrop()
    
    # COLS TO TRANSFORM FROM STRING TO INDEX
    model_info['arguments_col_string'] = rankConfig.getArgumentsColString([])
    
    # COLS THAT DEFINE FREQUENCY
    model_info['arguments_col_y'] = rankConfig.getArgumentsColY([])
    
    # COL TO EXCLUDE FROM ONE HOT ENCODING
    model_info['arguments_col_not_ohe'] = rankConfig.getArgumentsColNotOHE(rankConfig.getArgumentsColToDrop())
    
    # COLUMNS TO USE IN CLUSTERING
    model_info['arguments_col'] = rankConfig.getArgumentsColX(rankConfig.getArgumentsColToDrop()) + rankConfig.getArgumentsColY(rankConfig.getArgumentsColToDrop())
    
    if os.path.exists(inputPipeline.model_info_filename): 
        os.remove(inputPipeline.model_info_filename)
        
    with open(inputPipeline.model_info_filename, 'wb') as f:
        json.dump(model_info, codecs.getwriter('utf-8')(f), ensure_ascii=False)
