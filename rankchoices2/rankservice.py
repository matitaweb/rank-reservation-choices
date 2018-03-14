from flask import jsonify, request, Flask
import os
import sys
import json
import datetime
import findspark
from numpy import array
from math import sqrt
import argparse







# to calculate euclidean distance
#from scipy.spatial import distance



"""
TODO:
http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=kmeans#pyspark.ml.clustering.BisectingKMeans
KMeansModel.computeCost
da usare per calcolare la distanza dal centro del cluster e capire per poi usarlo nella 
normalizzazione del calcolo del ranking

python rankservice.py -b "/home/ubuntu/workspace/rank-reservation-choices/data/bo_since19-01-2018_annullato_no-strt_e_prst_valide"


python rankservice.py -b "/dati/bo_20140101-20180119_annullato_no-strt_e_prst_valide_kmean_10k/bo_20140101-20180119_annullato_no-strt_e_prst_valide/bo_20140101-20180119_annullato_no-strt_e_prst_valide" -s "/root/spark-2.2.1-bin-hadoop2.7"
python rankservice.py -b "/home/ubuntu/workspace/rank-reservation-choices/data/bo_since19-01-2018_annullato_no-strt_e_prst_valide"
"""



if __name__ == '__main__':
    
    # parsing parameters
    parser = argparse.ArgumentParser(description='Process rank requests.')
    parser.add_argument('-b', '--base_dir_path', type=str, help='-b base dir path ')
    parser.add_argument('-s', '--spark_home_path', type=str, help='-s spark_home path ')
    args = parser.parse_args()
    if args.base_dir_path:
        print("ARG: " + args.base_dir_path)
        base_dir = args.base_dir_path
    else:
        base_dir = "/home/ubuntu/workspace/rank-reservation-choices/data/bo_since19-01-2018_annullato_no-strt_e_prst_valide"
    
    if args.spark_home_path:
        print("ARG: " + args.spark_home_path)
        spark_home = args.spark_home_path
    else:
        spark_home = "/home/ubuntu/workspace/spark-2.3.0-bin-hadoop2.7"
        # spark_home = "/home/ubuntu/workspace/spark-2.2.1-bin-hadoop2.7"
    
    
    if not os.path.exists(spark_home): 
        print("ERROR NOT EXIST spark_home: " + spark_home)
        exit()
    
    t1 = datetime.datetime.now()
    findspark.init(spark_home)
    
    
    from rank_utils import euclidean0_1
    from rank_utils import validate_with_metadata
    from load_data_stage import load_metadata
    from load_data_stage import load_stringindexer_model_dict
    from load_data_stage import get_onehotencoding_model
    from load_data_stage import quantize_all_cols
    from load_data_stage import add_metadata
    from load_data_stage import apply_stringindexer_model_dict
    from load_data_stage import apply_onehotencoding_model
    
    from pyspark.sql import SparkSession
    from pyspark.sql import SQLContext
    from pyspark import SparkContext
    SparkContext.setSystemProperty('spark.ui.enabled', 'false') ## disable ui interface
    from pyspark.ml.feature import PCAModel
    from pyspark.ml.clustering import KMeansModel
    from pyspark.sql.functions import col
    
    from rank_utils import RankConfig
    from input_utils import InputPipeline
    from pipeline_session import PipelineSession
    
    
    from load_data_stage import DataLoaderService
    from pca_stage import PcaReductionService
    from kmeans_stage import KmeansService
    from dict_stage import DictService
    from accuracy_stage import AccuracyService
    from accuracy_stage import get_accuracy
    from accuracy_stage import test_accuracy_df
    from accuracy_stage import test_accuracy_df_to_list
    
    
    
    # dto's
    rankConfig = RankConfig();
    inputPipeline = InputPipeline(base_filename=base_dir)

    
    # LOAD SPARK ENV
    spark = SparkSession.builder.master("local[*]").appName("Rank").getOrCreate()
    sqlContext = SQLContext(spark)
    
    # LOAD METADATA COLUMNS
    metadataDict = load_metadata(inputPipeline.metadata_file_name_dir)
    
    # LOAD STRING INDEXER 
    indexer_dict = load_stringindexer_model_dict(inputPipeline.string_indexer_path_dir)
    
    
    # OHE 
    arguments_col_to_drop = rankConfig.getArgumentsColToDrop()
    arguments_col_not_ohe = rankConfig.getArgumentsColNotOHE(arguments_col_to_drop)
    #arguments_col = rankConfig.getArgumentsColX(arguments_col_to_drop) + rankConfig.getArgumentsColY(arguments_col_to_drop)
    #ohe_col = rankConfig.getOheCol(arguments_col, arguments_col_not_ohe)
    encodersDict= get_onehotencoding_model(rankConfig)
    
    
    # LOAD PCA MODEL
    pca_model = PCAModel.load(inputPipeline.pca_path_dir)

    
    # LOAD KMEANS MODEL
    kmeans_model = KMeansModel.load(inputPipeline.file_name_dir_kmeans)  # load from file system 
    
    with open(inputPipeline.model_info_filename) as model_info_file:    
        kmeans_model_info = json.load(model_info_file)
     
    
    # LOAD FREQUECY DICT
    with open(inputPipeline.cluster_freq_dict_filename) as dict_data_file:    
        cluster_freq_dict = json.load(dict_data_file)
    
    # LOAD MODEL INFO
    with open(inputPipeline.model_info_filename) as model_info_file:    
        model_info = json.load(model_info_file)
    
    
    # LOAD KMEANS MODEL CENTERS
    centers = kmeans_model.clusterCenters()
    print("MODEL ACTIVE Cluster Centers: " + str(len(centers)))
    # print(model_info)
    
    
    time_duration_loading= (datetime.datetime.now()-t1)
    print ("Successfully imported Spark Modules " + str(datetime.timedelta(seconds=time_duration_loading.total_seconds())))

    # start API REST
    app = Flask(__name__)
    
    @app.route('/', methods=['GET'])
    def default_get():
        rlist = [
            {   "X_ETA":56, "X_SESSO":2, "STRING_X_CAP": "40068", "STRING_X_USL":"167313", 
                "STRING_X_FASCIA":"1514", "X_GRADO_URG":5, "STRING_X_QDGN":"0", "STRING_X_INVIANTE":"0", "STRING_X_ESENZIONE":"0", "STRING_X_PRESCRITTORE":"807", "STRING_X_PRESTAZIONE": "2216", "STRING_X_BRANCA_SPECIALISTICA":"17", "STRING_X_GRUPPO_EROGABILE":"3641",
                "STRING_Y_STER":"17916", "STRING_Y_UE":"38977", 
                "Y_GIORNO_SETTIMANA" :2, "Y_MESE_ANNO":1, "Y_FASCIA_ORARIA":1, "Y_GIORNI_ALLA_PRENOTAZIONE":4},
            {   "X_SESSO":2, "STRING_X_CAP": "40068", "STRING_X_USL":"167313", 
                "STRING_X_FASCIA":"1514", "X_GRADO_URG":5, "STRING_X_QDGN":"0", "STRING_X_INVIANTE":"0", "STRING_X_ESENZIONE":"0", "STRING_X_PRESCRITTORE":"807", "STRING_X_PRESTAZIONE": "2216", "STRING_X_BRANCA_SPECIALISTICA":"17", "STRING_X_GRUPPO_EROGABILE":"3641",
                "STRING_Y_STER":"17916", "STRING_Y_UE":"38977", "X_ETA":57,
                "Y_GIORNO_SETTIMANA" :2, "Y_MESE_ANNO":1, "Y_FASCIA_ORARIA":1, "Y_GIORNI_ALLA_PRENOTAZIONE":4}
            ]
        #rlist = _filterCols(rlist)
        
        accuracyDictList = _predict(rlist)
        return jsonify(accuracyDictList)
    
    @app.route('/predict', methods=['POST'])
    def post_predict():
        rlist = json.loads(request.data)
        accuracyDictList = _predict(rlist)
        return jsonify(accuracyDictList)
        
    def _filterCols(rlist):
         # COLS to ESCLUDE TO SIMPLER MODEL
        resultlist = []
        for r in rlist:
            filtered_cols = {key: value for key, value in r.items() if not key in rankConfig.getArgumentsColToDrop() }
            resultlist.append(filtered_cols)
        return resultlist
    
        
    def _predict(rlistPar):
        
        t1 = datetime.datetime.now()
        validationResult = validate_with_metadata(rlistPar, metadataDict, rankConfig.validate_with_metadata_exceptList())
        print(validationResult)
        
        rlist = validationResult['valid']
    
        # TRANSFORM JSON QUERY TO DATAFRAME
        dfraw = sqlContext.createDataFrame(rlist, schema=rankConfig.get_input_schema([]))
        #print(dfraw.show())
        
        # REMOVE NULLS
        for x in rankConfig.get_input_schema([]):
            dfraw = dfraw.where(col(x.name).isNotNull())
        
        # QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE (ONLY ONE)
        dfq = quantize_all_cols(dfraw, rankConfig.getArgumentsColToQuantize())
        
        # ADD DATAFRAME METADATA
        df = add_metadata(dfq, metadataDict)
        
        
        # APPLY INDEX COLUMNS 
        arguments_col_string = rankConfig.getArgumentsColString([])
        dfi = apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)
        
        # APPLY ONE HOT ENCODING
        df_ohe = apply_onehotencoding_model(dfi, arguments_col_not_ohe, encodersDict, rankConfig.getOheFeatureOutputColName())
        
        #APPLY PCA
        df_pca = pca_model.transform(df_ohe)
        
        #PREDICT KMEANS CLUSTER
        kmeans_df_pca = kmeans_model.transform(df_pca) 
        
        # COLS THAT DEFINE FREQUENCY
        arguments_col_y = rankConfig.getArgumentsColY([])
        
        
        #print(kmeans_df_pca.head(1))
        #wssse = kmeans_model.computeCost(df_pca)
        #print("Within Set Sum of Squared Errors = " + str(wssse))
        
        time_duration_prepare = (datetime.datetime.now()-t1)
        print("PREDICTION: " + str(datetime.timedelta(seconds=time_duration_prepare.total_seconds())))
        
        
        result = {}
        result['version'] = "1.0.0"
        # result['model']= kmeans_model_info
        # result['wssse'] = wssse
        
        
        t1 = datetime.datetime.now()
        accuracyDictList = []
        
        predList = kmeans_df_pca.collect()
        for r in predList:
            cluster_center = centers[r['prediction']]
            centerdistance = euclidean0_1(r['pca_features'], cluster_center)
            #print(kmeans_df_pca.columns)
            c= str(r['prediction'])
            accuracyDict = get_accuracy(r, cluster_freq_dict, c, arguments_col_y)
            accuracyDict['centerdistance'] = centerdistance
            accuracyDict['request']={}
            for colname in dfraw.columns:
                accuracyDict['request'][colname]=r[colname]
            accuracyDictList.append(accuracyDict)
            
        tot_centerdistance= sum(accuracyDict['centerdistance'] for accuracyDict in accuracyDictList)  
        #print(tot_centerdistance)
        for accuracyDict in accuracyDictList:
            centerdistance = accuracyDict['centerdistance']
            mean_acc  = accuracyDict['mean_acc']
            accuracyDict['mean_acc_norm'] = mean_acc * (tot_centerdistance-centerdistance)/tot_centerdistance
            for ar in arguments_col_y:
                acc = accuracyDict[ar]['ACC']
                mean_acc_ar = acc * (tot_centerdistance-centerdistance)/tot_centerdistance
                accuracyDict[ar]['ACC_NORM'] = mean_acc_ar
        
        print("ACCURACY OLD: " + str(datetime.timedelta(seconds=(datetime.datetime.now()-t1).total_seconds())))
    
        """
        t1 = datetime.datetime.now()
        kmeans_stage_test_ds_acc = test_accuracy_df(kmeans_df_pca, arguments_col_y, cluster_freq_dict)
        accuracyDictList2 = test_accuracy_df_to_list(kmeans_stage_test_ds_acc, arguments_col_y)
        result['accuracyDictList2']= accuracyDictList2
        print("ACCURACY NEW: " + str(datetime.timedelta(seconds=(datetime.datetime.now()-t1).total_seconds())))
        """
    
        result['accuracyDictList']= accuracyDictList
        result['status'] ="OK"
        result['validationResult'] = validationResult
        return result
    
    
    
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    app.run(host=host, port=port, debug=True)
    app.logger.info("Starting flask app on %s:%s", host, port)
    
