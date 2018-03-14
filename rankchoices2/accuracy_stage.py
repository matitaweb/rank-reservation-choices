import rank_utils
from pyspark.ml.clustering import KMeansModel
from pyspark.sql.functions import count, lag, desc
from pyspark.sql.window import Window
import os
import shutil
import datetime
import json
import codecs
import math
from multiprocessing import Process
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType

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
       
        pipelineSession.time_duration_accuracy_load_data = (datetime.datetime.now()-t1)
        return pipelineSession.time_duration_accuracy_load_data
        
    
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
        
        t2 = datetime.datetime.now()
        pipelineSession.accuracyDictList = test_accuracy(pipelineSession.kmeans_stage_test_ds, arguments_col_y, pipelineSession.cluster_freq_dict)
        pipelineSession.time_duration_accuracy_start_stage_t1 = (datetime.datetime.now()-t2)
        
        """
        t2 = datetime.datetime.now()
        kmeans_stage_test_ds_acc = test_accuracy_df(pipelineSession.kmeans_stage_test_ds, arguments_col_y, pipelineSession.cluster_freq_dict)
        pipelineSession.accuracyDictListNew = test_accuracy_df_to_list(kmeans_stage_test_ds_acc, arguments_col_y)
        pipelineSession.time_duration_accuracy_start_stage_t2 = (datetime.datetime.now()-t2)
        """
        pipelineSession.wssse_train = pipelineSession.kmeans_model_fitted.computeCost(pipelineSession.kmeans_stage_train_ds)
        pipelineSession.wssse_test = pipelineSession.kmeans_model_fitted.computeCost(pipelineSession.kmeans_stage_test_ds)

        pipelineSession.time_duration_accuracy_start_stage = (datetime.datetime.now()-t1)
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, pipelineSession.time_duration_accuracy_start_stage
        
    
    
    def snapshot_stage(self, spark, rankConfig, inputPipeline, pipelineSession):
        
        t1 = datetime.datetime.now()

        pipelineSession.time_duration_accuracy_snapshot_stage = (datetime.datetime.now()-t1)
        
        return pipelineSession.kmeans_stage_train_ds, pipelineSession.kmeans_stage_test_ds, pipelineSession.time_duration_accuracy_snapshot_stage
        
        
    def write_report(self, rankConfig, inputPipeline, pipelineSession):
        
        tot_col = len(pipelineSession.kmeans_stage_train_ds.head(1)[0][rankConfig.getOheFeatureOutputColName()])
        k_pca = int(tot_col*inputPipeline.pca_perc/100)
        split_col=('{0:,}'.format(pipelineSession.kmeans_stage_train_ds.count()), '{0:,}'.format(pipelineSession.kmeans_stage_test_ds.count()))
    
        file = open(inputPipeline.report_accuracy_stage_filename, 'w')
        file.write('CONF: \n')
        file.write('------------------------------------'+' \n')
        file.write('input train: ' + str(inputPipeline.output_kmeans_train_ds_filename)+' \n')
        file.write('input test:  ' + str(inputPipeline.output_kmeans_test_ds_filename)+' \n')
        file.write('input kmeans model: ' + str(inputPipeline.file_name_dir_kmeans)+' \n')
        file.write('input dict: ' + str(inputPipeline.cluster_freq_dict_filename)+ ' \n\n')
        
        file.write('k_pca: ' + str(inputPipeline.pca_perc) + '% ' + str(k_pca)+ ' / '+str(tot_col)+' \n')
        file.write('k_kmeans: ' + str(inputPipeline.k_means_num)+'\n')
        file.write('train, test: ' + str(inputPipeline.split)+ ' -> ' + str(split_col) + ' \n')
        file.write('WSSSE TRAIN: ' + str(pipelineSession.wssse_train) +'  \n')
        file.write('WSSSE TEST: ' + str(pipelineSession.wssse_test) +'  \n')
        file.write('\nAccuracy position threshold: ' + str(inputPipeline.position_test_threshold)+'\n')
        
        time_load = str(datetime.timedelta(seconds=pipelineSession.time_duration_accuracy_load_data.total_seconds())) if(pipelineSession.time_duration_accuracy_load_data != None) else "-"
        time_stage = str(datetime.timedelta(seconds=pipelineSession.time_duration_accuracy_start_stage.total_seconds())) if(pipelineSession.time_duration_accuracy_start_stage != None) else "-"
        time_snapshoot = str(datetime.timedelta(seconds=pipelineSession.time_duration_accuracy_snapshot_stage.total_seconds())) if(pipelineSession.time_duration_accuracy_snapshot_stage != None) else "-"
        
        
        file.write('\nACCURACY STAGE: \n')
        file.write('------------------------'+'  \n')
        file.write('time load: ' + time_load +'  \n')
        file.write('time stage: ' + time_stage +'  \n')
        file.write('time snapshoot: ' + time_snapshoot +'  \n\n')
        
        
        arguments_col_y = rankConfig.getArgumentsColY([])

        # count threshold per ogni argomento
        accuracyDictList = pipelineSession.accuracyDictList
        accuracyMeanList = [e['mean_acc'] for e in accuracyDictList]

        tot_values_cluster = [ value[arguments_col_y[0]]['tot_values'] for _, value in pipelineSession.cluster_freq_dict.items()]
        mean_tot_values_cluster = np.mean(tot_values_cluster)
        min_tot_values_cluster= np.min(tot_values_cluster)
        max_tot_values_cluster = np.max(tot_values_cluster)
        median_tot_values_cluster = np.median(tot_values_cluster)
        
        file.write('CLUSTER INFO ( n. '+str(len(tot_values_cluster))+' )\n')
        file.write('------------------------------------'+'\n')
        file.write("media per cluster: " + str(mean_tot_values_cluster)+'\n')
        file.write("minima dimensione cluster: " + str(min_tot_values_cluster)+'\n')
        file.write("massima dimensione cluster: " + str(max_tot_values_cluster)+'\n')
        file.write("mediana dimensione cluster: " + str(median_tot_values_cluster)+'\n\n')
        
        for c in arguments_col_y:
            
            positionList = [e[c]["LAST_POS"] if e[c]["POS"] is None  else e[c]["POS"]  for e in accuracyDictList]
            if(positionList == None):
                print("Non presente: " + str(c))
                file.write(str(c) + " non presente.. "+'\n')
                
            positionList = [x for x in positionList if x is not None]
            mean_pos = 'None' if positionList == None or len(positionList) == 0 else np.mean(positionList)
            min_pos = 'None' if positionList == None or len(positionList) == 0 else np.min(positionList)
            max_pos = 'None' if positionList == None or len(positionList) == 0 else np.max(positionList)
            median_pos = 'None' if positionList == None or len(positionList) == 0 else np.median(positionList)
            count_position = 'None' if positionList == None or len(positionList) == 0 else len(positionList)
            count_position_threshold ='None' if positionList == None or len(positionList) == 0 else count_position
            
            file.write('\n')
            file.write('------------------------------------'+'\n')
            file.write('> '+str(c)+'\n')
            file.write('------------------------------------'+'\n')
            file.write("mean_pos: " + str(mean_pos)+'\n')
            file.write("min_pos: " + str(min_pos)+'\n')
            file.write("max_pos: " + str(max_pos)+'\n')
            file.write("median_pos: " + str(median_pos)+'\n')
            file.write("count_position: " + str(count_position)+'\n\n')
            
            file.write('CMC: '+'\n')
            if not (inputPipeline.position_test_threshold is None):
                for i in range(inputPipeline.position_test_threshold):
                    positionListThreshold = [e for e in positionList if e <= i]
                    count_position_threshold = len(positionListThreshold)
                    count_position_threshold_perc = float(count_position_threshold)/float(count_position)
                    file.write("ENTRO LA POS: " + str(i) + " -> " + str(count_position_threshold_perc) +" = " + str(count_position_threshold)  +"/" + str(count_position)+'\n')
            file.write('\n\n')
        
            mean = 'None' if accuracyMeanList == None else np.mean(accuracyMeanList)
            file.write('mean acc.: ' + str(mean)+'\n')
            file.write('------------------------------------'+'\n\n')
            
            
            for c in arguments_col_y:
                accList = [e[c]['ACC'] for e in accuracyDictList]
                m = 'None' if accList == None or len(accList) == 0 else np.mean(accList)
                file.write("MEAN: " + str(c) + ' -> ' + str(m)+'\n')
                
            file.write('\n\n\n')
            
            maxV = np.max(accuracyMeanList)
            file.write('max acc.: ' + str(maxV)+'\n')
            file.write('------------------------------------'+'\n\n')
            
            for c in arguments_col_y:
                accList = [e[c]['ACC'] for e in accuracyDictList]
                m = 'None' if accList == None or len(accList) == 0 else np.max(accList)
                file.write("MAX: " + str(c) + ' -> ' + str(m)+'\n')
            
            file.write('\n\n\n')
            
            minV = np.min(accuracyMeanList)
            file.write('min acc.: ' + str(minV)+'\n')
            file.write('------------------------------------'+'\n\n')
            
            for c in arguments_col_y:
                accList = [e[c]['ACC'] for e in accuracyDictList]
                m = 'None' if accList == None or len(accList) == 0 else np.min(accList)
                file.write("MIN: " + str(c) + ' -> ' + str(m)+'\n')        
            
            file.write('------------------------------------'+'\n\n')
            
        file.close()


    
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
        acc= 0.0
        accuracyDict[ar] = {}
        if ar in cluster_freq_dict[str(c)]:
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


def calculate_accuracy(cluster_freq_dict, ar):

    def _calculate_accuracy(c, val):
        acc = float(0)
        c_str = str(c)
        if ar in cluster_freq_dict[str(c_str)]:
            last_position = cluster_freq_dict[c_str][ar]['last']
            val_str = str(val)
            if val_str in cluster_freq_dict[c_str][ar]:
                position = cluster_freq_dict[c_str][ar][val_str]['POS'] - 1
                acc = (float(last_position)-float(position))/float(last_position)

        return acc
 
    return _calculate_accuracy

def calculate_last_pos(cluster_freq_dict, ar):

    def _calculate_last_pos(c, val):
        last_position = None
        c_str = str(c)
        if ar in cluster_freq_dict[str(c_str)]:
            last_position = cluster_freq_dict[c_str][ar]['last']
        return last_position
 
    return _calculate_last_pos

def calculate_pos(cluster_freq_dict, ar):

    def _calculate_pos(c, val):
        position = None
        c_str = str(c)
        if ar in cluster_freq_dict[str(c_str)]:
            val_str = str(val)
            if val_str in cluster_freq_dict[c_str][ar]:
                position = cluster_freq_dict[c_str][ar][val_str]['POS'] - 1
                
        return position
 
    return _calculate_pos


# https://ragrawal.wordpress.com/2017/06/17/reusable-spark-custom-udf/ 
# from accuracy_stage import test_accuracy_df
def test_accuracy_df(kmeans_stage_test_ds, arguments_col_y, cluster_freq_dict):
    
    kmeans_stage_test_ds_acc = kmeans_stage_test_ds
    for ar in arguments_col_y:
        udf_calculate_accuracy = udf(calculate_accuracy(cluster_freq_dict, ar), FloatType())
        kmeans_stage_test_ds_acc = kmeans_stage_test_ds_acc.withColumn("ACC_"+ar, udf_calculate_accuracy(kmeans_stage_test_ds['prediction'],kmeans_stage_test_ds[ar] ))
        
        udf_calculate_last_pos = udf(calculate_last_pos(cluster_freq_dict, ar), IntegerType())
        kmeans_stage_test_ds_acc = kmeans_stage_test_ds_acc.withColumn("LAST_POS_"+ar, udf_calculate_last_pos(kmeans_stage_test_ds['prediction'],kmeans_stage_test_ds[ar] ))
        
        udf_calculate_pos = udf(calculate_pos(cluster_freq_dict, ar), IntegerType())
        kmeans_stage_test_ds_acc = kmeans_stage_test_ds_acc.withColumn("POS_"+ar, udf_calculate_pos(kmeans_stage_test_ds['prediction'],kmeans_stage_test_ds[ar] ))
        
    return kmeans_stage_test_ds_acc
  
def test_accuracy_df_to_list(kmeans_stage_test_ds_acc, arguments_col_y):
    accuracyDictList = []
    for r in kmeans_stage_test_ds_acc.collect():
        accuracyDict = {}
        accuracyDict['prediction'] = str(r['prediction'])
        tot_acc=0.0
        for ar in arguments_col_y:
            accuracyDict[ar] = {}
            val= str(r[ar])
            accuracyDict[ar]["VAL"]=val
            accuracyDict[ar]["POS"]=r["POS_"+ar]
            accuracyDict[ar]["LAST_POS"]=r["LAST_POS_"+ar]
                
            accuracyDict[ar]['ACC'] = r["ACC_"+ar]
            tot_acc = tot_acc+r["ACC_"+ar]
            
        accuracyDict['tot_acc'] = tot_acc
        accuracyDict['mean_acc'] = tot_acc/len(arguments_col_y)
        accuracyDictList.append(accuracyDict)
        
    return accuracyDictList

    
    """
from accuracy_stage import calculate_accuracy
from pyspark.sql.functions import udf
from pyspark.sql.types import DecimalType
from pyspark.sql.types import FloatType

kmeans_stage_test_ds_acc
    """