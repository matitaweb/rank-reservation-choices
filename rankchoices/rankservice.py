from flask import jsonify, request, Flask
import os
import sys
import json
import datetime




import findspark
findspark.init("/home/ubuntu/workspace/spark-2.2.0-bin-hadoop2.7")

import pyspark

from numpy import array
from math import sqrt
from pyspark import SparkConf
# $example off$

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



    
spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
sqlContext = SQLContext(spark)


metadata_file_name_dir= "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000-metadata"
metadataDict = pipe.load_metadata(metadata_file_name_dir)


# STRING INDEXER
stringindexer_path= "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000-indexer"
indexer_dict = pipe.load_stringindexer_model_dict(stringindexer_path)


# ONE HOT ENCODING
ohe_col = ["OHE_"+x for x in arguments_col if not x == 'X_ETA']
encodersDict= pipe.get_onehotencoding_model(arguments_col, ohe_col)


pca_path_dir =  "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000-pca-model"
pca_model = PCAModel.load(pca_path_dir)
#print(df_ohe.head(1))
#print(df_ohe.schema['OHE_Y_FASCIA_ORARIA'].metadata)


model_file_path = "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000.kmeans"
kmeans_model = KMeansModel.load(model_file_path)  # load from file system 


dict_file_path = "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000-dict.json"
with open(dict_file_path) as dict_data_file:    
    cluster_freq_dict = json.load(dict_data_file)

#print(str(len(cluster_freq_dict)))





print ("Successfully imported Spark Modules")

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
    
    
    dfraw = sqlContext.createDataFrame(rlist, schema=pipe.get_input_schema())
    dfq = pipe.quantize(dfraw)
    df = pipe.add_metadata(dfq, metadataDict)
    dfi = pipe.apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)
    df_ohe = pipe.apply_onehotencoding_model(dfi, encodersDict)
    df_pca = pca_model.transform(df_ohe)
    kmeans_df_pca = kmeans_model.transform(df_pca) 
    predList = kmeans_df_pca.collect()
    #print(kmeans_df_pca.head(1))
    
    time_duration_prepare = (datetime.datetime.now()-t1)
    print("PREPARATION: " + str(datetime.timedelta(seconds=time_duration_prepare.total_seconds())))
    
    t1 = datetime.datetime.now()
    accuracyDictList = []
    for r in predList:
        c= str(r['prediction'])
        accuracyDict = pipe.get_accuracy(r, cluster_freq_dict, c, arguments_col_y)
        accuracyDictList.append(accuracyDict)
    time_duration_prediction = (datetime.datetime.now()-t1)
    print("PREDICTION: " + str(datetime.timedelta(seconds=time_duration_prediction.total_seconds())))

    return accuracyDictList

if __name__ == '__main__':
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    app.run(host=host, port=port, debug=True)
    app.logger.info("Starting flask app on %s:%s", host, port)
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
    """
    with app.test_request_context('/?name=Peter'):
        assert request.path == '/'
        assert request.args['name'] == 'Peter'
    """