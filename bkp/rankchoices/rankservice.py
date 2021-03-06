from flask import jsonify, request, Flask
import os
import sys
import json
import datetime
import findspark
from numpy import array
from math import sqrt
import argparse
from rank_utils import RankConfig
from rank_utils import validate_with_metadata





# to calculate euclidean distance
#from scipy.spatial import distance



"""
TODO:
http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=kmeans#pyspark.ml.clustering.BisectingKMeans
KMeansModel.computeCost
da usare per calcolare la distanza dal centro del cluster e capire per poi usarlo nella 
normalizzazione del calcolo del ranking


python rankservice.py -b "/dati/data/light_r300.000.000" -s "/root/spark-2.2.1-bin-hadoop2.7"
python rankservice.py -b "/home/ubuntu/workspace/rank-reservation-choices/data/bo_since19-01-2018_annullato_no-strt_e_prst_valide"
"""

# CONFIGURATION
rankConfig = RankConfig();


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
    position_threshold = None
    accuracyDictList = _predict(rlist, position_threshold)
    return jsonify(accuracyDictList)

@app.route('/predict', methods=['POST'])
def post_predict():
    rlist = json.loads(request.data)
    position_threshold = None
    accuracyDictList = _predict(rlist, position_threshold)
    return jsonify(accuracyDictList)
    
def _filterCols(rlist):
     # COLS to ESCLUDE TO SIMPLER MODEL
    arguments_col_to_drop = rankConfig.getArgumentsColToDrop()
    resultlist = []
    for r in rlist:
        filtered_cols = {key: value for key, value in r.items() if not key in arguments_col_to_drop }
        resultlist.append(filtered_cols)
    return resultlist

    
def _predict(rlistPar, position_threshold):
    t1 = datetime.datetime.now()
    validationResult = validate_with_metadata(rlistPar, metadataDict, rankConfig.validate_with_metadata_exceptList())
    print(validationResult)
    
    rlist = validationResult['valid']
    
    # TRANSFORM JSON QUERY TO DATAFRAME
    dfraw = sqlContext.createDataFrame(rlist, schema=rankConfig.get_input_schema([]))
    print(dfraw.show())
    request_col_names = dfraw.columns
    
    # QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE (ONLY ONE)
    colname_to_quantize = "Y_GIORNI_ALLA_PRENOTAZIONE"
    my_udf = pyspark.sql.functions.UserDefinedFunction(pipe.convert_y_giorni_alla_prenotazione, pyspark.sql.types.IntegerType())
    dfq = pipe.quantize(dfraw, my_udf, colname_to_quantize)
    
    # ADD DATAFRAME METADATA
    df = pipe.add_metadata(dfq, metadataDict)
    
    # INDEX (X_PRESTAZIONE, Y_UE) AS THE PCA MODEL
    dfi = pipe.apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)
    
    # APPLY ONE HOT ENCODING
    featureOutputCol = "features"
    arguments_col_to_drop = rankConfig.getArgumentsColToDrop()
    arguments_col_not_ohe = rankConfig.getArgumentsColNotOHE(arguments_col_to_drop)
    df_ohe = pipe.apply_onehotencoding_model(dfi, arguments_col_not_ohe, encodersDict, featureOutputCol)
    
    #APPLY PCA
    df_pca = pca_model.transform(df_ohe)
    
    #PREDICT KMEANS
    kmeans_df_pca = kmeans_model.transform(df_pca) 
    
    
    predList = kmeans_df_pca.collect()
    #print(kmeans_df_pca.head(1))
    wssse = kmeans_model.computeCost(df_pca)
    #print("Within Set Sum of Squared Errors = " + str(wssse))
    
    time_duration_prepare = (datetime.datetime.now()-t1)
    print("PREDICTION: " + str(datetime.timedelta(seconds=time_duration_prepare.total_seconds())))
    
    t1 = datetime.datetime.now()
    result = {}
    result['version'] = "1.0.0"
    result['model']="KMEAN:100, PCA:0.1"
    result['wssse'] = wssse
    accuracyDictList = []
    
    
    for r in predList:
        cluster_center = centers[r['prediction']]
        centerdistance = pipe.euclidean0_1(r['pca_features'], cluster_center)
        #print(kmeans_df_pca.columns)
        c= str(r['prediction'])
        accuracyDict = pipe.get_accuracy(r, cluster_freq_dict, c, arguments_col_y, position_threshold)
        accuracyDict['centerdistance'] = centerdistance
        accuracyDict['request']={}
        for colname in request_col_names:
            accuracyDict['request'][colname]=r[colname]
        accuracyDictList.append(accuracyDict)
        
    tot_centerdistance= sum(accuracyDict['centerdistance'] for accuracyDict in accuracyDictList)  
    #print(tot_centerdistance)
    for accuracyDict in accuracyDictList:
        centerdistance = accuracyDict['centerdistance']
        mean_acc  = accuracyDict['mean_acc']
        accuracyDict['mean_acc_norm'] = mean_acc * (tot_centerdistance-centerdistance)/tot_centerdistance
    
    time_duration_prediction = (datetime.datetime.now()-t1)
    print("ACCURACY: " + str(datetime.timedelta(seconds=time_duration_prediction.total_seconds())))

    result['accuracyDictList']= accuracyDictList
    result['status'] ="OK"
    result['validationResult'] = validationResult
    return result



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
        base_dir = "/home/ubuntu/workspace/rank-reservation-choices/data/10k"
    
    if args.spark_home_path:
        print("ARG: " + args.spark_home_path)
        spark_home = args.spark_home_path
    else:
        spark_home = "/home/ubuntu/workspace/spark-2.2.1-bin-hadoop2.7"
    
    
    if not os.path.exists(spark_home): 
        print("ERROR NOT EXIST spark_home: " + spark_home)
        exit()
    
    t1 = datetime.datetime.now()
    findspark.init(spark_home)
    
    import pyspark
    from pyspark import SparkConf
    from pyspark import SparkContext
    SparkContext.setSystemProperty('spark.ui.enabled', 'false')
    from pyspark.sql import SQLContext
    from pyspark.sql import SparkSession
    #from pyspark.ml import PipelineModel
    from pyspark.ml.feature import PCA, PCAModel
    from pyspark.ml.clustering import KMeans, KMeansModel
    import pipeline as pipe

    
    
    
    # COLS to ESCLUDE TO SIMPLER MODEL
    arguments_col_to_drop = rankConfig.getArgumentsColToDrop()
    
    # COLS TO TRANSFORM FROM STRING TO INDEX
    arguments_col_string = rankConfig.getArgumentsColString([])
    
    # COLS THAT DEFINE FREQUENCY
    arguments_col_y = rankConfig.getArgumentsColY([])
    
    # COL TO EXCLUDE FROM ONE HOT ENCODING
    arguments_col_not_ohe = rankConfig.getArgumentsColNotOHE(arguments_col_to_drop)
    
    # COLUMNS TO USE IN CLUSTERING
    arguments_col = rankConfig.getArgumentsColX(arguments_col_to_drop) + rankConfig.getArgumentsColY(arguments_col_to_drop)
    print("COLUMNS TO USED IN CLUSTERING: " + str(arguments_col))
    
    # LOAD SPARK
    spark = SparkSession.builder.master("local[*]").appName("Rank").config("spark.python.profile", "true").getOrCreate()
    sqlContext = SQLContext(spark)
    
    
    # LOAD METADATA COLUMNS
    metadata_file_name_dir= base_dir+"-metadata"
    metadataDict = pipe.load_metadata(metadata_file_name_dir)
    
    
    # LOAD STRING INDEXER (X_PRESTAZIONE, Y_UE)
    stringindexer_path= base_dir+"-indexer"
    indexer_dict = pipe.load_stringindexer_model_dict(stringindexer_path)
    
    
    # LOAD ONE HOT ENCODING
    ohe_col = ["OHE_"+x for x in arguments_col if not x in arguments_col_not_ohe]
    encodersDict= pipe.get_onehotencoding_model(arguments_col, ohe_col, arguments_col_not_ohe)
    
    
    # LOAD PCA MODEL
    pca_path_dir =  base_dir+"-pca-model"
    pca_model = PCAModel.load(pca_path_dir)

    
    # LOAD KMEANS MODEL
    model_file_path = base_dir+".kmeans"
    kmeans_model = KMeansModel.load(model_file_path)  # load from file system 
    
    
    # LOAD FREQUECY DICT
    dict_file_path = base_dir+"-dict.json"
    with open(dict_file_path) as dict_data_file:    
        cluster_freq_dict = json.load(dict_data_file)
    
    
    # LOAD KMEANS MODEL CENTERS
    centers = kmeans_model.clusterCenters()
    print("Cluster Centers: " + str(len(centers)))
    #print(str(len(cluster_freq_dict)))
    
    
    time_duration_loading= (datetime.datetime.now()-t1)
    print ("Successfully imported Spark Modules " + str(datetime.timedelta(seconds=time_duration_loading.total_seconds())))
    
    
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    app.run(host=host, port=port, debug=True)
    app.logger.info("Starting flask app on %s:%s", host, port)
    
