import rank_utils
from pyspark.ml.clustering import KMeans
import os
import shutil
import datetime

class DictService:
    def __init__(self):
        pass
        
        
    def load_data(self, spark, rankConfig, inputPipeline, pipelineSession):
        t1 = datetime.datetime.now()
        
        if(pipelineSession.kmeans_stage_train_ds == None):
            pipelineSession.kmeans_stage_train_ds = rank_utils.load_from_parquet (spark, inputPipeline.output_kmeans_train_ds_filename)
            
        if(pipelineSession.kmeans_stage_test_ds == None):
            pipelineSession.kmeans_stage_test_ds = rank_utils.load_from_parquet (spark, inputPipeline.output_kmeans_test_ds_filename)
            
        if(pipelineSession.kmeans_model_fitted == None):
            pipelineSession.kmeans_model_fitted = rank_utils.load_from_parquet (spark, inputPipeline.file_name_dir_kmeans)
        
            
        time_duration_load_data = (datetime.datetime.now()-t1)
        return time_duration_load_data
        
    
    def start_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        
        if(pipelineSession.kmeans_stage_train_ds == None):
            raise ValueError('ERROR: no value for kmeans_stage_train_ds')
            
        if(pipelineSession.kmeans_stage_test_ds == None):
            raise ValueError('ERROR: no value for kmeans_stage_test_ds')
        
        if(pipelineSession.kmeans_model_fitted == None):
            raise ValueError('ERROR: no value for kmeans_model_fitted')

        arguments_col_y = rankConfig.getArgumentsColY([])
        frequency_dict = get_cluster_freq_dict(pipelineSession.kmeans_stage_train_ds, arguments_col_y)
        cluster_freq_dict = build_cluster_freq_dict(frequency_dict)
        save_cluster_freq_dict(cluster_freq_dict, cluster_freq_dict_filename)
        print('Snapshot dict test-set: ' + cluster_freq_dict_filename)
        
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
    
def build_cluster_freq_dict(frequency_dict):
    
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
    
def save_cluster_freq_dict(cluster_freq_dict, file_name_dir):
    #if os.path.exists(file_name_dir): shutil.rmtree(file_name_dir)
    #os.makedirs(file_name_dir)
    if os.path.exists(file_name_dir): os.remove(file_name_dir)
    with open(file_name_dir, 'wb') as f:
        json.dump(cluster_freq_dict, codecs.getwriter('utf-8')(f), ensure_ascii=False)