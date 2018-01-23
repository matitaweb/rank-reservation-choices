import rank_utils
from pyspark.ml.feature import PCA, PCAModel
import os
import shutil
import datetime

class PcaReductionService:
    def __init__(self):
        pass
        
        
    def load_data(self, spark, rankConfig, inputPipeline, pipelineSession):
        t1 = datetime.datetime.now()
        
        if(pipelineSession.load_data_stage_train_ds == None):
            pipelineSession.load_data_stage_train_ds = rank_utils.load_from_parquet (spark, inputPipeline.output_train_file_name)
            
        if(pipelineSession.load_data_stage_test_ds == None):
            pipelineSession.load_data_stage_test_ds = rank_utils.load_from_parquet (spark, inputPipeline.output_test_file_name)
            
        time_duration_load_data = (datetime.datetime.now()-t1)
        return time_duration_load_data
        
    
    def start_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        
        if(pipelineSession.load_data_stage_train_ds == None):
            raise ValueError('ERROR: no value for load_data_stage_train_ds')
            
        if(pipelineSession.load_data_stage_test_ds == None):
            raise ValueError('ERROR: no value for load_data_stage_test_ds')
            
        tot_col = len(pipelineSession.load_data_stage_train_ds.head(1)[0][rankConfig.getOheFeatureOutputColName()])
        k_pca = int(tot_col*inputPipeline.pca_perc/100)
        print("pca tot_col: " + str(tot_col) + " reduce to: " + str(k_pca) )
        
        pca = PCA(k=k_pca, inputCol=rankConfig.getPcaFeatureInputCol(), outputCol=rankConfig.getPcaFeatureOutputCol()) #Argument with more than 65535 cols
        pca_model = pca.fit(pipelineSession.load_data_stage_train_ds)
        
        if os.path.exists(inputPipeline.pca_path_dir): 
            shutil.rmtree(inputPipeline.pca_path_dir)
        pca_model.save(inputPipeline.pca_path_dir)
        
        pipelineSession.pca_stage_train_ds = pca_model.transform(pipelineSession.load_data_stage_train_ds)
        pipelineSession.pca_stage_test_ds = pca_model.transform(pipelineSession.load_data_stage_test_ds)
        
        time_duration_start_stage = (datetime.datetime.now()-t1)
        return pipelineSession.pca_stage_train_ds, pipelineSession.pca_stage_test_ds, time_duration_start_stage
        
    
    
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        if(pipelineSession.pca_stage_train_ds != None):
            pipelineSession.pca_stage_train_ds.write.parquet(inputPipeline.output_pca_train_filename, mode="overwrite")
        
        if(pipelineSession.pca_stage_test_ds != None):
            pipelineSession.pca_stage_test_ds.write.parquet(inputPipeline.output_pca_test_filename, mode="overwrite")
        
        time_duration_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.pca_stage_train_ds, pipelineSession.pca_stage_test_ds, time_duration_snapshot_stage
    
