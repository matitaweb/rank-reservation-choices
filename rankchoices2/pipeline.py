from rank_utils import RankConfig
from input_utils import InputPipeline
from load_data_stage import DataLoaderService
from pca_stage import PcaReductionService
from kmeans_stage import KmeansService
from dict_stage import DictService
from pipeline_session import PipelineSession
from pyspark.sql import SparkSession
import datetime


rankConfig = RankConfig();
inputPipeline = InputPipeline()
dataLoaderService = DataLoaderService()
pipelineSession = PipelineSession();

spark = SparkSession.builder.master("local[*]").appName("Rank").config("spark.python.profile", "true").getOrCreate()


# LOAD
time_duration_loading_load_data = dataLoaderService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_loading_stage = dataLoaderService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_loading_snapshot =  dataLoaderService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession);
print('Snapshot train-set: ' + inputPipeline.output_train_file_name)
print('Snapshot test-set: ' + inputPipeline.output_test_file_name)
print('SNAPSHOT LOAD  save in: ' + str(datetime.timedelta(seconds=time_duration_loading_snapshot.total_seconds())))
        

# PCA
pcaReductionService = PcaReductionService()
time_duration_pca_load_data = pcaReductionService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_pca_stage = pcaReductionService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_pca_snapshot =  pcaReductionService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession);

#KMEANS
kmeansService = KmeansService()
time_duration_kmeans_load_data = kmeansService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_kmeans_stage = kmeansService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_kmeans_snapshot =  kmeansService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession);


#DICT
dictService = DictService()
time_duration_dict_load_data = dictService.load_data(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_dict_stage = dictService.start_stage(spark, rankConfig, inputPipeline, pipelineSession)
_, _, time_duration_dict_snapshot =  dictService.snapshot_stage(spark, rankConfig, inputPipeline, pipelineSession);

#TEST




"""
df = spark.read.csv("/dati/bo_20140101-20180119_annullato_no-strt_e_prst_valide/bo_20140101-20180119_annullato_no-strt_e_prst_valide.csv", header=True, schema=rankConfig.get_input_schema([]), ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
df = spark.read.csv("../data/bo_since19-01-2018_annullato_no-strt_e_prst_valide.csv", header=True, mode="DROPMALFORMED",  schema=rankConfig.get_input_schema([]), ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
from pyspark.sql.functions import col

for x in rankConfig.get_input_schema([]):
    dfa = dfa.where(col(x.name).isNotNull())
    print (x.name + " -> " + str(dfa.count()))
    
dfa = df.where(col("STRING_X_USL").isNull())

df = dataLoaderService.loading_stage(spark)
"""

def start():    
    print (rankConfig)



if __name__ == '__main__':
    start()
    
    
# import pipeline as p
# p.start()    