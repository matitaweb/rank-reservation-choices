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

class AccuracyService:
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


        arguments_col_y = rankConfig.getArgumentsColY([])
        
        accuracyDictList = test_accuracy(pipelineSession.kmeans_stage_test_ds, arguments_col_y, pipelineSession.cluster_freq_dict)
        
        
        ###################
        # REPORT ACCURACY #
        ###################
        tot_rows = pipelineSession.kmeans_stage_train_ds.count() + pipelineSession.kmeans_stage_test_ds.count()
        split_col=(pipelineSession.kmeans_stage_train_ds.count(), pipelineSession.kmeans_stage_test_ds.count())
        
        
        
            
        tot_col = len(kmeans_train_ds.head(1)[0][rankConfig.getOheFeatureOutputColName()])
        k_pca = int(tot_col*k_pca_perc/100)
        
        # Evaluate clustering by computing Within Set Sum of Squared Errors.
        wssse = pipelineSession.kmeans_model_fitted.computeCost(pipelineSession.kmeans_stage_train_ds)
        
        # save json model info
        kmeans_centers = [str(center) for center in pipelineSession.kmeans_model_fitted.clusterCenters()]

    
        k_pca = int(tot_col*inputPipeline.pca_perc/100)

        if os.path.exists(inputPipeline.report_filename): 
            os.remove(inputPipeline.report_filename)
        write_report(inputPipeline.report_filename, tot_col, k_means_num, arguments_col_y, accuracyDictList, position_threshold, k_pca=k_pca, k_pca_perc=k_pca_perc, split=split, split_col=split_col)
        
        time_duration_start_stage = (datetime.datetime.now()-t1)
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_start_stage
        
    
    
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()

        time_duration_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, time_duration_snapshot_stage
        

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
    
# https://ragrawal.wordpress.com/2017/06/17/reusable-spark-custom-udf/



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
    
def load_cluster_freq_dict(file_name_dir):
    
    with open(file_name_dir) as json_data:
        d = json.load(json_data)
        return d

