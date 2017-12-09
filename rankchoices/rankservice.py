from flask import jsonify, request, Flask
import os
import sys
import json
import datetime
import findspark
from numpy import array
from math import sqrt

# to calculate euclidean distance
#from scipy.spatial import distance



"""
TODO:
http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=kmeans#pyspark.ml.clustering.BisectingKMeans
KMeansModel.computeCost
da usare per calcolare la distanza dal centro del cluster e capire per poi usarlo nella 
normalizzazione del calcolo del ranking

"""

spark_home = "/home/ubuntu/workspace/spark-2.2.0-bin-hadoop2.7"
base_dir = "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000"


t1 = datetime.datetime.now()

findspark.init(spark_home)
import pyspark
from pyspark import SparkConf


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import PCA, PCAModel

from pyspark.ml.clustering import KMeans, KMeansModel

import pipeline as pipe

sc = pyspark.SparkContext(appName="rank")



arguments_col_string = [('STRING_X_PRESTAZIONE', 'X_PRESTAZIONE'), ('STRING_Y_UE', 'Y_UE')]
arguments_col_x = [ 'X_ETA', 'X_SESSO', 'X_GRADO_URG', 'X_PRESTAZIONE']
arguments_col_y = [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
arguments_col = arguments_col_x + arguments_col_y

# LOAD SPARK
spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
sqlContext = SQLContext(spark)


# LOAD METADATA COLUMNS
metadata_file_name_dir= base_dir+"-metadata"
metadataDict = pipe.load_metadata(metadata_file_name_dir)


# LOAD STRING INDEXER (X_PRESTAZIONE, Y_UE)
stringindexer_path= base_dir+"-indexer"
indexer_dict = pipe.load_stringindexer_model_dict(stringindexer_path)


# LOAD ONE HOT ENCODING
ohe_col = ["OHE_"+x for x in arguments_col if not x == 'X_ETA']
encodersDict= pipe.get_onehotencoding_model(arguments_col, ohe_col)

# LOAD PCA MODEL
pca_path_dir =  base_dir+"-pca-model"
pca_model = PCAModel.load(pca_path_dir)
#print(df_ohe.head(1))
#print(df_ohe.schema['OHE_Y_FASCIA_ORARIA'].metadata)


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

app = Flask(__name__)

@app.route('/', methods=['GET'])
def default_get():
    rlist = [{ "X_ETA":77, "X_SESSO":1, "X_GRADO_URG":5, "STRING_X_PRESTAZIONE": "2000", "STRING_Y_UE":"18299", "Y_GIORNO_SETTIMANA" :2, "Y_MESE_ANNO":10, "Y_FASCIA_ORARIA":0, "Y_GIORNI_ALLA_PRENOTAZIONE":11},{"X_ETA":43, "X_SESSO":2, "X_GRADO_URG":0, "STRING_X_PRESTAZIONE":"3413", "STRING_Y_UE":"17842", "Y_GIORNO_SETTIMANA":6, "Y_MESE_ANNO":3, "Y_FASCIA_ORARIA":0, "Y_GIORNI_ALLA_PRENOTAZIONE":35}]
    accuracyDictList = predict(rlist)
    return jsonify(accuracyDictList)

@app.route('/predict', methods=['POST'])
def post_predict():
    rlist = json.loads(request.data)
    accuracyDictList = predict(rlist)
    return jsonify(accuracyDictList)
    
def predict(rlist):
    t1 = datetime.datetime.now()
    
    # TRANSFORM JSON QUERY TO DATAFRAME
    dfraw = sqlContext.createDataFrame(rlist, schema=pipe.get_input_schema())
    request_col_names = dfraw.columns
    
    # QUANTIZE Y_GIORNI_ALLA_PRENOTAZIONE 
    dfq = pipe.quantize(dfraw)
    
    # ADD DATAFRAME METADATA
    df = pipe.add_metadata(dfq, metadataDict)
    
    # INDEX (X_PRESTAZIONE, Y_UE) AS THE PCA MODEL
    dfi = pipe.apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)
    
    # APPLY ONE HOT ENCODING
    df_ohe = pipe.apply_onehotencoding_model(dfi, encodersDict)
    
    #APPLY PCA
    df_pca = pca_model.transform(df_ohe)
    
    #PREDICT KMEANS
    kmeans_df_pca = kmeans_model.transform(df_pca) 
    
    
    predList = kmeans_df_pca.collect()
    #print(kmeans_df_pca.head(1))
    #wssse = kmeans_model.computeCost(df_pca)
    #print("Within Set Sum of Squared Errors = " + str(wssse))
    
    time_duration_prepare = (datetime.datetime.now()-t1)
    print("PREDICTION: " + str(datetime.timedelta(seconds=time_duration_prepare.total_seconds())))
    
    t1 = datetime.datetime.now()
    accuracyDictList = []
    
    
    for r in predList:
        cluster_center = centers[r['prediction']]
        centerdistance = pipe.euclidean0_1(r['pca_features'], cluster_center)
        #print(kmeans_df_pca.columns)
        c= str(r['prediction'])
        accuracyDict = pipe.get_accuracy(r, cluster_freq_dict, c, arguments_col_y)
        accuracyDict['centerdistance'] = centerdistance
        accuracyDict['request']={}
        for colname in request_col_names:
            accuracyDict['request'][colname]=r[colname]
        accuracyDictList.append(accuracyDict)
        
    tot_centerdistance= sum(accuracyDict['centerdistance'] for accuracyDict in accuracyDictList)  
    print(tot_centerdistance)
    for accuracyDict in accuracyDictList:
        centerdistance = accuracyDict['centerdistance']
        mean_acc  = accuracyDict['mean_acc']
        accuracyDict['mean_acc_norm'] = mean_acc * (tot_centerdistance-centerdistance)/tot_centerdistance
    
    time_duration_prediction = (datetime.datetime.now()-t1)
    print("ACCURACY: " + str(datetime.timedelta(seconds=time_duration_prediction.total_seconds())))

    return accuracyDictList



if __name__ == '__main__':
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    app.run(host=host, port=port, debug=True)
    app.logger.info("Starting flask app on %s:%s", host, port)
    
    """
    rlist = [{
        "X_ETA":77,
        "X_SESSO":1,
        "X_GRADO_URG":5,
        "STRING_X_PRESTAZIONE": "2000",
        'STRING_Y_UE':"18299", 
        'Y_GIORNO_SETTIMANA' :2, 
        'Y_MESE_ANNO':10, 
        'Y_FASCIA_ORARIA':0, 
        'Y_GIORNI_ALLA_PRENOTAZIONE':11
    },
    {
        "X_ETA":43,
        "X_SESSO":2,
        "X_GRADO_URG":0,
        "STRING_X_PRESTAZIONE":"3413",
        'STRING_Y_UE':"17842", 
        'Y_GIORNO_SETTIMANA':6, 
        'Y_MESE_ANNO':3, 
        'Y_FASCIA_ORARIA':0, 
        'Y_GIORNI_ALLA_PRENOTAZIONE':35
    }]
    #print(app.post('/predict', json=rlist))

    with app.test_request_context('/?name=Peter'):
        assert request.path == '/'
        assert request.args['name'] == 'Peter'
    """