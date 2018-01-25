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
import numpy as np


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
        
        pipelineSession.time_duration_dict_load_data = (datetime.datetime.now()-t1)
        return pipelineSession.time_duration_dict_load_data
        
    
    def start_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()
        
        if(pipelineSession.kmeans_stage_train_ds == None):
            raise ValueError('ERROR: no value for kmeans_stage_train_ds')
            
        if(pipelineSession.kmeans_stage_test_ds == None):
            raise ValueError('ERROR: no value for kmeans_stage_test_ds')
        
        if(pipelineSession.kmeans_model_fitted == None):
            raise ValueError('ERROR: no value for kmeans_model_fitted')

        arguments_col_y = rankConfig.getArgumentsColY([])
        
        """
        OLD
        t1 = datetime.datetime.now()
        pipelineSession.cluster_freq_dict = get_cluster_freq_dict(pipelineSession.kmeans_stage_train_ds, arguments_col_y)
        save_cluster_freq_dict(pipelineSession.cluster_freq_dict, inputPipeline.cluster_freq_dict_filename)
        time_duration_start_stage = (datetime.datetime.now()-t1)
        print('SAVE DICT save in: ' + str(datetime.timedelta(seconds=time_duration_start_stage.total_seconds())))
        """
        
        t1 = datetime.datetime.now()
        pipelineSession.cluster_freq_dict = get_cluster_freq_dict_df(pipelineSession.kmeans_stage_train_ds, arguments_col_y, inputPipeline.k_means_num)
        save_cluster_freq_dict(pipelineSession.cluster_freq_dict, inputPipeline.cluster_freq_dict_filename)
        print('SAVE DICT save in: ' + str(datetime.timedelta(seconds=(datetime.datetime.now()-t1).total_seconds())))
        
        pipelineSession.time_duration_dict_start_stage = (datetime.datetime.now()-t1)
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, pipelineSession.time_duration_dict_start_stage
        
    
    
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()

        pipelineSession.time_duration_dict_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, pipelineSession.time_duration_dict_snapshot_stage
        
        
        
    def write_report(self, rankConfig, inputPipeline, pipelineSession):
        
        tot_col = len(pipelineSession.kmeans_stage_train_ds.head(1)[0][rankConfig.getOheFeatureOutputColName()])
        k_pca = int(tot_col*inputPipeline.pca_perc/100)
        split_col=('{0:,}'.format(pipelineSession.kmeans_stage_train_ds.count()), '{0:,}'.format(pipelineSession.kmeans_stage_test_ds.count()))
    
        file = open(inputPipeline.report_dict_stage_filename, 'w')
        file.write('CONF: \n')
        file.write('------------------------------------'+' \n')
        file.write('input train: ' + str(inputPipeline.output_kmeans_train_ds_filename)+' \n')
        file.write('input test:  ' + str(inputPipeline.output_kmeans_test_ds_filename)+' \n')
        file.write('input kmeans model: ' + str(inputPipeline.file_name_dir_kmeans)+' \n\n')
        
        file.write('k_pca: ' + str(inputPipeline.pca_perc) + '% ' + str(k_pca)+ ' / '+str(tot_col)+' \n')
        file.write('k_kmeans: ' + str(inputPipeline.k_means_num)+'\n')
        file.write('train, test: ' + str(inputPipeline.split)+ ' -> ' + str(split_col) + ' \n')

        
        time_load = str(datetime.timedelta(seconds=pipelineSession.time_duration_dict_load_data.total_seconds())) if(pipelineSession.time_duration_dict_load_data != None) else "-"
        time_stage = str(datetime.timedelta(seconds=pipelineSession.time_duration_dict_start_stage.total_seconds())) if(pipelineSession.time_duration_dict_start_stage != None) else "-"
        time_snapshoot = str(datetime.timedelta(seconds=pipelineSession.time_duration_dict_snapshot_stage.total_seconds())) if(pipelineSession.time_duration_dict_snapshot_stage != None) else "-"
        
        
        file.write('\nDICTIONARY STAGE: \n')
        file.write('------------------------'+'  \n')
        file.write('time load: ' + time_load +'  \n')
        file.write('time stage: ' + time_stage +'  \n')
        file.write('time snapshoot: ' + time_snapshoot +'  \n\n')
        
        
        arguments_col_y = rankConfig.getArgumentsColY([])

        # count threshold per ogni argomento
        tot_values_cluster = [ value['Y_STER']['tot_values'] for _, value in pipelineSession.cluster_freq_dict.items()]
        mean_tot_values_cluster = np.mean(tot_values_cluster)
        min_tot_values_cluster= np.min(tot_values_cluster)
        max_tot_values_cluster = np.max(tot_values_cluster)
        median_tot_values_cluster = np.median(tot_values_cluster)
        
        file.write('CLUSTER INFO ( n. '+str(len(tot_values_cluster))+' )\n')
        file.write('------------------------------------'+'\n')
        file.write("media elementi per cluster: " + str(mean_tot_values_cluster)+'\n')
        file.write("minimo numero di elementi per cluster: " + str(min_tot_values_cluster)+'\n')
        file.write("massimo numero di elementi per cluster: " + str(max_tot_values_cluster)+'\n')
        file.write("mediano numero di elementi per cluster: " + str(median_tot_values_cluster)+'\n\n')
        
        for ar in arguments_col_y:
            tot_distinct_cluster = [ value[ar]['tot_distinct'] for _, value in pipelineSession.cluster_freq_dict.items()]
            mean_tot_distinct_cluster = np.mean(tot_distinct_cluster)
            min_tot_distinct_cluster= np.min(tot_distinct_cluster)
            max_tot_distinct_cluster = np.max(tot_distinct_cluster)
            median_tot_distinct_cluster = np.median(tot_distinct_cluster)
            file.write('CLUSTER  per  ' + str(ar) +' \n')
            file.write('------------------------------------'+'\n')
            file.write("media numero di elementi distinti per cluster: " + str(mean_tot_distinct_cluster)+'\n')
            file.write("minimo numero di elementi distinti cluster: " + str(min_tot_distinct_cluster)+'\n')
            file.write("massimo numero di elementi distinti cluster: " + str(max_tot_distinct_cluster)+'\n')
            file.write("mediana numero di elementi distinti cluster: " + str(median_tot_distinct_cluster)+'\n\n')
        
        file.close()

def get_cluster_freq_dict_df(kmeans_stage_train_ds, arguments_col_y, k_means_num):
    
    cluster_freq_dict = {}
    for c in range(k_means_num):
        c_str = str(c)
        cluster_freq_dict[c_str]={}
        for ar in arguments_col_y:
            cluster_freq_dict[c_str][ar]={}
            cluster_freq_dict[c_str][ar]['last_count'] = None
            cluster_freq_dict[c_str][ar]['curr_pos'] = 1
            cluster_freq_dict[c_str][ar]['last'] = None
            cluster_freq_dict[c_str][ar]['tot_distinct'] = 0
            cluster_freq_dict[c_str][ar]['tot_values'] = 0
    
    #processes = []
    for ar in arguments_col_y:
        
        ar_group_df = kmeans_stage_train_ds.groupBy("prediction", ar).agg(count("*").alias("count"))
        ar_group_df = ar_group_df.sort([ar_group_df['prediction'], ar_group_df['count']], ascending=[True, False])
        ar_group_list = ar_group_df.collect()
        update_cluster_freq_dict(ar, ar_group_list, cluster_freq_dict)
        #p = Process(target=update_cluster_freq_dict, args=(ar, ar_group_list, cluster_freq_dict))
        #p.start()
        #processes.append(p)
        
        #
    #for p in processes:
    #    p.start()
        
    #for p in processes:
    #    p.join()
    
    return cluster_freq_dict    
        
def update_cluster_freq_dict(ar, ar_group_list, cluster_freq_dict):
    
    t1 = datetime.datetime.now()
    
    for r in  ar_group_list:
        
        c = r['prediction']
        c_str = str(c)
        curr_count = r['count']
        
        if(cluster_freq_dict[c_str][ar]['last_count'] != None and cluster_freq_dict[c_str][ar]['last_count'] != curr_count):
            cluster_freq_dict[c_str][ar]['curr_pos']=cluster_freq_dict[c_str][ar]['curr_pos']+1
            
        cluster_freq_dict[c_str][ar][str(r[ar])]= {'IDX': r[ar], 'OCCUR': curr_count, 'POS': cluster_freq_dict[c_str][ar]['curr_pos']}
        cluster_freq_dict[c_str][ar]['last_count'] = curr_count
        cluster_freq_dict[c_str][ar]['last'] = cluster_freq_dict[c_str][ar]['curr_pos']
        cluster_freq_dict[c_str][ar]['tot_values'] = cluster_freq_dict[c_str][ar]['tot_values']+curr_count
        cluster_freq_dict[c_str][ar]['tot_distinct'] = cluster_freq_dict[c_str][ar]['tot_distinct'] + 1
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