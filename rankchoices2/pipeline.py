from rank_utils import RankConfig
from input_utils import InputPipeline
from pipeline_session import PipelineSession

from load_data_stage import DataLoaderService
from pca_stage import PcaReductionService
from kmeans_stage import KmeansService
from dict_stage import DictService
from accuracy_stage import AccuracyService

from pyspark.sql import SparkSession
import datetime




"""
import pipeline as p
p.wssse_pipeline()
p.execute(stage_start="LOAD", stage_stop="LOAD")
p.execute(base_filename = "/dati/bo_20171101-20180119_annullato_no-strt_e_prst_valide/bo_20171101-20180119_annullato_no-strt_e_prst_valide", pca_perc=0.1, k_means_num=200000, position_test_threshold=10, stage_start="LOAD", stage_stop="TEST", stage_snapshots=['PCA', 'DICT', 'TEST'], random_seed=1)
p.execute(pca_perc=0.5, k_means_num=1000, position_test_threshold=10, stage_start="LOAD", stage_stop="TEST", stage_snapshots=['LOAD', 'PCA', 'KMEANS', 'DICT', 'TEST'], random_seed=1)

"""

def execute(base_filename = "../data/bo_since19-01-2018_annullato_no-strt_e_prst_valide", split=[0.09, 0.01], pca_perc=1, k_means_num=1000, position_test_threshold=10, stage_start="LOAD", stage_stop="TEST", stage_snapshots=['LOAD', 'PCA', 'KMEANS', 'DICT', 'TEST'], random_seed=1):    
    
    # DTO with configuration
    rankConfig = RankConfig()
    
    # DTO with input params for cluster model creation
    inputPipeline = InputPipeline(base_filename = base_filename, split=split, pca_perc=pca_perc, k_means_num=k_means_num, position_test_threshold=position_test_threshold, stage_start=stage_start, stage_stop=stage_stop, stage_snapshots=stage_snapshots, random_seed=random_seed)


    # data passing in the pipeline session 
    pipelineSession = PipelineSession()
    
    
    # services
    dataLoaderService = DataLoaderService()
    pcaReductionService = PcaReductionService()
    kmeansService = KmeansService()
    dictService = DictService()
    accuracyService = AccuracyService()
    
    # spark 
    spark = SparkSession.builder.master("local[*]").appName("Rank").config("spark.python.profile", "true").getOrCreate()
    
    # START PIPELINE
    
    # LOAD
    if(stage_start == "LOAD"):
        print ("START AT " + stage_start)
        time_duration_loading_load_data = dataLoaderService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
        _, _, time_duration_loading_stage = dataLoaderService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
        if("LOAD" in  inputPipeline.stage_snapshots):
            print ("SNAPSHOT AT " + "LOAD")
            _, _, time_duration_loading_snapshot =  dataLoaderService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession)
        dataLoaderService.write_report(rankConfig, inputPipeline, pipelineSession)
    if( stage_stop == "LOAD"):
        print ("STOP AT THE END OF " + stage_stop)
        return rankConfig, inputPipeline, pipelineSession
    
    # PCA
    if(stage_start == "LOAD" or stage_start == "PCA"):
        if(stage_start == "PCA"):
            print ("START AT " + stage_start)
            time_duration_pca_load_data = pcaReductionService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
        _, _, time_duration_pca_stage = pcaReductionService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
        if("PCA" in  inputPipeline.stage_snapshots):
            print ("SNAPSHOT AT " + "PCA")
            _, _, time_duration_pca_snapshot =  pcaReductionService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession)
        pcaReductionService.write_report(rankConfig, inputPipeline, pipelineSession)
    if( stage_stop == "PCA"):
        print ("STOP AT THE END OF " + stage_stop)
        return rankConfig, inputPipeline, pipelineSession
    
    #KMEANS
    if(stage_start == "LOAD" or stage_start == "PCA" or stage_start=="KMEANS"):
        if(stage_start == 'KMEANS'):
            print ("START AT " + stage_start)
            time_duration_kmeans_load_data = kmeansService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
        _, _, time_duration_kmeans_stage = kmeansService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
        if("KMEANS" in  inputPipeline.stage_snapshots):
            print ("SNAPSHOT AT " + "KMEANS")
            _, _, time_duration_kmeans_snapshot =  kmeansService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession)
        kmeansService.write_report(rankConfig, inputPipeline, pipelineSession)
    if( stage_stop == "KMEANS"):
        print ("STOP AT THE END OF " + stage_stop)
        return rankConfig, inputPipeline, pipelineSession
    
    #DICT
    if(stage_start == "LOAD" or stage_start == "PCA" or stage_start=="KMEANS" or stage_start=="DICT"):
        if(stage_start == 'DICT'):
            print ("START AT " + stage_start)
            time_duration_dict_load_data = dictService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
        _, _, time_duration_dict_stage = dictService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
        if("DICT" in  inputPipeline.stage_snapshots):
            print ("SNAPSHOT AT " + "DICT")
            _, _, time_duration_dict_snapshot =  dictService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession)
        dictService.write_report(rankConfig, inputPipeline, pipelineSession)
    if( stage_stop == "DICT"):
        print ("STOP AT THE END OF " + stage_stop)
        return rankConfig, inputPipeline, pipelineSession
        
    #TEST
    if(stage_start == "LOAD" or stage_start == "PCA" or stage_start=="KMEANS" or stage_start=="DICT" or stage_start=="TEST"):
        if(stage_start == 'TEST'):
            print ("START AT " + stage_start)
            time_duration_dict_load_data = accuracyService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
        _, _, time_duration_dict_stage = accuracyService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
        if("TEST" in  inputPipeline.stage_snapshots):
            print ("SNAPSHOT AT " + "TEST")
            _, _, time_duration_dict_snapshot =  accuracyService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession)
        accuracyService.write_report(rankConfig, inputPipeline, pipelineSession)
    if( stage_stop == "TEST"):
        print ("STOP AT THE END OF " + stage_stop)
        return rankConfig, inputPipeline, pipelineSession



if __name__ == '__main__':
    wssse_pipeline()
    
    
def wssse_pipeline(base_filename = "../data/bo_since19-01-2018_annullato_no-strt_e_prst_valide"):    
    file = open( base_filename + ".wssse.csv", 'w')
    file.write('KMEAN,WSSSE TRAIN, WSSSE TEST\n')
    
    a = []
    k_means_num = 10
    rankConfig, inputPipeline, pipelineSession = execute(base_filename=base_filename, pca_perc=1, k_means_num=k_means_num, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=['LOAD', 'PCA', 'KMEANS', 'DICT', 'TEST'], random_seed=1)
    file.write(str(k_means_num)  +","+ str(pipelineSession.wssse_train)  +","+ str(pipelineSession.wssse_test) +'\n')
    
    k_means_num = 100
    rankConfig, inputPipeline, pipelineSession = execute(base_filename=base_filename, pca_perc=1, k_means_num=k_means_num, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=[], random_seed=1)
    file.write(str(k_means_num)  +","+ str(pipelineSession.wssse_train)  +","+ str(pipelineSession.wssse_test) +'\n')
    
    k_means_num = 500
    rankConfig, inputPipeline, pipelineSession = execute(base_filename=base_filename, pca_perc=1, k_means_num=k_means_num, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=[], random_seed=1)
    file.write(str(k_means_num)  +","+ str(pipelineSession.wssse_train)  +","+ str(pipelineSession.wssse_test) +'\n')
    
    k_means_num = 1000
    rankConfig, inputPipeline, pipelineSession = execute(base_filename=base_filename, pca_perc=1, k_means_num=k_means_num, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=[], random_seed=1)
    file.write(str(k_means_num)  +","+ str(pipelineSession.wssse_train)  +","+ str(pipelineSession.wssse_test) +'\n')
    
    k_means_num = 10000
    rankConfig, inputPipeline, pipelineSession = execute(base_filename=base_filename, pca_perc=1, k_means_num=k_means_num, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=[], random_seed=1)
    file.write(str(k_means_num)  +","+ str(pipelineSession.wssse_train)  +","+ str(pipelineSession.wssse_test) +'\n')
    
    k_means_num = 50000
    rankConfig, inputPipeline, pipelineSession = execute(base_filename=base_filename, pca_perc=1, k_means_num=k_means_num, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=[], random_seed=1)
    file.write(str(k_means_num)  +","+ str(pipelineSession.wssse_train)  +","+ str(pipelineSession.wssse_test) +'\n')
    
    k_means_num = 100000
    rankConfig, inputPipeline, pipelineSession = execute(base_filename=base_filename, pca_perc=1, k_means_num=k_means_num, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=['TEST'], random_seed=1)
    file.write(str(k_means_num)  +","+ str(pipelineSession.wssse_train)  +","+ str(pipelineSession.wssse_test) +'\n')
    
    """
    for i in range(100, 900, 100):
        rankConfig, inputPipeline, pipelineSession = execute(pca_perc=1, k_means_num=i, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=[], random_seed=1)
        a.append((i, pipelineSession.wssse_train, pipelineSession.wssse_test))
    
    for i in range(1000, 30000, 1000):
        rankConfig, inputPipeline, pipelineSession = execute(pca_perc=i, k_means_num=i, position_test_threshold=10, stage_start="KMEANS", stage_stop="TEST", stage_snapshots=[], random_seed=1)        
        a.append((i, pipelineSession.wssse_train, pipelineSession.wssse_test))
    
    for el in a:
        file.write(str(el[0]) +","+ str(el[1]) +","+ str(el[2]) +'\n')
    """
    
    file.close()
    
# import pipeline as p
# p.start()    