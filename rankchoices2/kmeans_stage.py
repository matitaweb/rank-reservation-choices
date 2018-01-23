import rank_utils
from pyspark.ml.clustering import KMeans
import os
import shutil
import datetime

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
        kmeans_model_fitted = kmeans.fit(pipelineSession.pca_stage_train_ds)

        if os.path.exists(inputPipeline.file_name_dir_kmeans): 
            shutil.rmtree(inputPipeline.file_name_dir_kmeans)
        kmeans_model_fitted.save(inputPipeline.file_name_dir_kmeans)
        
        pipelineSession.kmeans_stage_train_ds = kmeans_model_fitted.transform(pipelineSession.pca_stage_train_ds)   
        pipelineSession.kmeans_stage_test_ds = kmeans_model_fitted.transform(pipelineSession.pca_stage_test_ds) 
        
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
    
