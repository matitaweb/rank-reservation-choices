import rank_utils
from pyspark.ml.clustering import KMeansModel
from pyspark.sql.functions import count, lag, desc
from pyspark.sql.window import Window
import os
import shutil
import datetime
import json
import codecs
from multiprocessing import Process

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
            pipelineSession.kmeans_model_fitted = KMeansModel.load(inputPipeline.file_name_dir_kmeans)
        
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
        
        
        t1 = datetime.datetime.now()
        pipelineSession.cluster_freq_dict = get_cluster_freq_dict(pipelineSession.kmeans_stage_train_ds, arguments_col_y)
        save_cluster_freq_dict(pipelineSession.cluster_freq_dict, inputPipeline.cluster_freq_dict_filename)
        time_duration_start_stage = (datetime.datetime.now()-t1)
        print('SAVE DICT save in: ' + str(datetime.timedelta(seconds=time_duration_start_stage.total_seconds())))
        
        t1 = datetime.datetime.now()
        pipelineSession.cluster_freq_dict = get_cluster_freq_dict_df(pipelineSession.kmeans_stage_train_ds, arguments_col_y, inputPipeline.k_means_num)
        save_cluster_freq_dict(pipelineSession.cluster_freq_dict, inputPipeline.cluster_freq_dict_filename + ".new")
        time_duration_start_stage = (datetime.datetime.now()-t1)
        print('SAVE DICT save in: ' + str(datetime.timedelta(seconds=time_duration_start_stage.total_seconds())))
        
        #print('SAVE dict : ' + inputPipeline.cluster_freq_dict_filename)
        
        time_duration_start_stage = (datetime.datetime.now()-t1)
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_start_stage
        
    
    
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()

        time_duration_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_snapshot_stage

def get_cluster_freq_dict_df(kmeans_stage_train_ds, arguments_col_y, k_means_num):
    
    cluster_freq_dict = {}
    for c in range(k_means_num):
        cluster_freq_dict[c]={}
        for ar in arguments_col_y:
            cluster_freq_dict[c][ar]={}
            cluster_freq_dict[c][ar]['last_count'] = None
            cluster_freq_dict[c][ar]['curr_pos'] = 1
            cluster_freq_dict[c][ar]['last'] = None
            cluster_freq_dict[c][ar]['tot_distinct'] = 0
            cluster_freq_dict[c][ar]['tot_values'] = 0
    
    processes = []
    for ar in arguments_col_y:
        
        ar_group_df = kmeans_stage_train_ds.groupBy("prediction", ar).agg(count("*").alias("count"))
        ar_group_df = ar_group_df.sort([ar_group_df['prediction'], ar_group_df['count']], ascending=[True, False])
        ar_group_list = ar_group_df.collect()
        
        p = Process(target=update_cluster_freq_dict, args=(ar, ar_group_list, cluster_freq_dict))
        processes.append(p)
        
        #update_cluster_freq_dict(cluster_freq_dict, kmeans_stage_train_ds)
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
    
    return cluster_freq_dict    
        
def update_cluster_freq_dict(ar, ar_group_list, cluster_freq_dict):
    
    t1 = datetime.datetime.now()
    
    for r in  ar_group_list:
        
        c = r['prediction']
        curr_count = r['count']
        
        if(cluster_freq_dict[c][ar]['last_count'] != None and cluster_freq_dict[c][ar]['last_count'] != curr_count):
            cluster_freq_dict[c][ar]['curr_pos']=cluster_freq_dict[c][ar]['curr_pos']+1
            
        cluster_freq_dict[c][ar][str(r[ar])]= {'IDX': r[ar], 'OCCUR': curr_count, 'POS': cluster_freq_dict[c][ar]['curr_pos']}
        cluster_freq_dict[c][ar]['last_count'] = curr_count
        cluster_freq_dict[c][ar]['last'] = cluster_freq_dict[c][ar]['curr_pos']
        cluster_freq_dict[c][ar]['tot_values'] = cluster_freq_dict[c][ar]['tot_values']+curr_count
        cluster_freq_dict[c][ar]['tot_distinct'] = cluster_freq_dict[c][ar]['tot_distinct'] + 1
    time_duration = (datetime.datetime.now()-t1)
    print("FREQU: [" + ar  + "] -> " + str(datetime.timedelta(seconds=time_duration.total_seconds())))


"""
OLD METHOD
"""
def get_cluster_freq_dict(kmeans_train_ds, arguments_col_y):

    frequency_dict={}

    for ar in arguments_col_y:
        t1 = datetime.datetime.now()
        ar_group = kmeans_train_ds.groupBy("prediction",ar).agg(count("*").alias("count")).collect()
        time_duration = (datetime.datetime.now()-t1)
        print("FREQU: [" + ar  + "] -> " + str(datetime.timedelta(seconds=time_duration.total_seconds())))
        
        # make dictionary
        for r in ar_group :
            c = r['prediction']
            if not c in frequency_dict: frequency_dict[c] = {}
            if not ar in frequency_dict[c]:frequency_dict[c][ar] = []
            frequency_dict[c][ar].append({'IDX': r[ar], 'OCCUR': r['count'], 'POS': -1})

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
            cluster_freq_dict[c][ar]['tot'] = len(val_sorted)
    
    return cluster_freq_dict
    

def save_cluster_freq_dict(cluster_freq_dict, file_name_dir):
    #if os.path.exists(file_name_dir): shutil.rmtree(file_name_dir)
    #os.makedirs(file_name_dir)
    if os.path.exists(file_name_dir): 
        os.remove(file_name_dir)
    with open(file_name_dir, 'wb') as f:
        json.dump(cluster_freq_dict, codecs.getwriter('utf-8')(f), ensure_ascii=False)