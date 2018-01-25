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

class ReportService:
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
        
        if(pipelineSession.cluster_freq_dict == None):
            pipelineSession.cluster_freq_dict = load_cluster_freq_dict(inputPipeline.cluster_freq_dict_filename)
       
        
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

        if(pipelineSession.cluster_freq_dict == None):
            raise ValueError('ERROR: no value for cluster_freq_dict')


        
        
        
        
        

        time_duration_start_stage = (datetime.datetime.now()-t1)
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_start_stage
        
    
    
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()

        time_duration_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_snapshot_stage
        




    
def load_cluster_freq_dict(file_name_dir):
    
    with open(file_name_dir) as json_data:
        d = json.load(json_data)
        return d






def write_report(rankConfig, inputPipeline, pipelineSession):

    tot_col = len(pipelineSession.pca_stage_train_ds.head(1)[0][rankConfig.getOheFeatureOutputColName()])
    wssse = pipelineSession.kmeans_model_fitted.computeCost(pipelineSession.kmeans_stage_train_ds)
    kmeans_centers = [str(center) for center in pipelineSession.kmeans_model_fitted.clusterCenters()]
    
    arguments_col_y = rankConfig.getArgumentsColY([])
    accuracyDictList = pipelineSession.accuracyDictList 
    accuracyMeanList = [e['mean_acc'] for e in accuracyDictList]
    mean_acc = np.mean(accuracyMeanList)
    
    time_duration_split, 
    time_duration_pca, 
    time_duration_kmean, 
    time_duration_test, 
    position_threshold,  
    k_pca = int(tot_col*inputPipeline.pca_perc/100)
    pca_perc= inputPipeline.pca_perc
    split="-", 
    split_col="-"
    
    file = open(inputPipeline.report_filename, 'w')
    file.write('CONFIGURAZIONI: \n')
    file.write('filename: ' + str(inputPipeline.report_filename)+'\n')
    file.write('k_pca: ' + str(pca_perc) + "% " + str(k_pca)+ ' / '+str(tot_col)+'\n')
    file.write('k_kmeans: ' + str(rankConfig.k_kmeans)+'\n')
    
    file.write('\nLOADINGSTAGE: \n')
    file.write('------------------------')
    file.write('time split: ' + str(datetime.timedelta(seconds=time_duration_split.total_seconds()))+'  \n')
    
    
    file.write('\PCA STAGE: \n')
    file.write('------------------------')
    
    file.write('time pca: ' + str(datetime.timedelta(seconds=time_duration_pca.total_seconds()))+'  \n')
    file.write('time kmean: ' + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds()))+'  \n')
    file.write('time test: ' + str(datetime.timedelta(seconds=time_duration_test.total_seconds()))+'  \n')
    file.write('train, test: ' + str(split)+'\n')
    file.write('train, test: ' + str(split_col)+'\n\n\n')
    
    file.write('position_threshold: ' + str(position_threshold)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    # count threshold per ogni argomento
    for c in arguments_col_y:
        positionList = [e[c]["LAST_POS"] if e[c]["POS"] is None  else e[c]["POS"]  for e in accuracyDictList]
        last_pos = accuracyDictList[0][c]["LAST_POS"]
        #print(len(positionList))
        #continue
        mean_pos = np.mean(positionList)
        min_pos = np.min(positionList)
        max_pos = np.max(positionList)
        count_position = len(positionList)
        count_position_threshold = count_position
        
        
        file.write(str(c)+'\n')
        file.write('------------------------------------'+'\n')
        file.write("mean_pos: " + str(mean_pos)+'\n')
        file.write("min_pos: " + str(min_pos)+'\n')
        file.write("max_pos: " + str(max_pos)+'\n')
        file.write("last_pos: " + str(last_pos)+'\n')
        file.write("count_position: " + str(count_position)+'\n\n')
        file.write('CMC: '+'\n')
        if not (position_threshold is None):
            for i in range(position_threshold):
                positionListThreshold = [e for e in positionList if e <= i]
                count_position_threshold = len(positionListThreshold)
                count_position_threshold_perc = float(count_position_threshold)/float(count_position)
                file.write("ENTRO LA POS: " + str(i) + " -> " + str(count_position_threshold_perc) +" = " + str(count_position_threshold)  +"/" + str(count_position)+'\n')
        file.write('------------------------------------'+'\n\n')
    
    mean = np.mean(accuracyMeanList)
    file.write('mean acc.: ' + str(mean)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    
    for c in arguments_col_y:
        m = np.mean([e[c]['ACC'] for e in accuracyDictList])
        file.write("MEAN: " + str(c) + ' -> ' + str(m)+'\n')
        
    file.write('\n\n\n')
    
    maxV = np.max(accuracyMeanList)
    file.write('max acc.: ' + str(maxV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col_y:
        m = np.max([e[c]['ACC'] for e in accuracyDictList])
        file.write("MAX: " + str(c) + ' -> ' + str(m)+'\n')
    
    file.write('\n\n\n')

    minV = np.min(accuracyMeanList)
    file.write('min acc.: ' + str(minV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col_y:
        m = np.min([e[c]['ACC'] for e in accuracyDictList])
        file.write("MIN: " + str(c) + ' -> ' + str(m)+'\n')        
    
    file.write('------------------------------------'+'\n')
    
    file.write('\n\n\n')
    
    file.write('------------------------------------'+'\n')
    
    file.close()