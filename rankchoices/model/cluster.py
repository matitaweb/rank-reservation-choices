import rankchoices.commons.load_stage as load_stage
import rankchoices.model.kmeans as kms
import rankchoices.metrics.accuracy as accuracy
import rankchoices.metrics.report as report

import datetime
import logging
import numpy as np
import os
import shutil

from pyspark.ml.clustering import KMeansModel


# to execute
#------------
"""
import rankchoices.model.cluster as cluster


kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict, kmeans_test_ds_pca, accuracyMeanList, accuracyDictList, time_duration_kmean, time_duration_test, tot_col, k_pca, k_pca_perc, split, split_col = cluster.test_cluster()


"""


def train_kmean_model(input_filename_train, k_kmeans, arguments_col, feature_col = "pca_features"):
    
    # TRAINING
    train_ds_pca = load_stage.load_from_parquet (input_filename_train);
    
    # KMEANS
    kmeans_model_fitted = kms.build_kmean_fitted(dataset = train_ds_pca, k_kmeans = k_kmeans, feature_col = feature_col)
    kmeans_train_ds_pca = kmeans_model_fitted.transform(train_ds_pca)   
    

    # EXTRACT DICTIONARY
    cluster_freq_dict = kms.build_frequency_dict(arguments_col, kmeans_train_ds_pca)
    
    return kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict


def build_kmean_model(
    input_filename_train = "data/light_r10.000-pca-10-train.parquet",
    base_filename        = "data/light_r10.000-pca-10-kmeans-1000",
    k_kmeans = 1000,
    arguments_col = [ 'Y_UE_', 'Y_GIO_', 'Y_MES_', 'Y_FASCI_', 'Y_GIORNI_ALLA_PRENOTAZIONE_'], 
    feature_col = "pca_features"):

    # KMEAN MODEL
    kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict = train_kmean_model(input_filename_train, k_kmeans, arguments_col, feature_col = feature_col)

    # SAVE KMEANS
    file_name_dir_kmeans = base_filename+".kmeans"
    if os.path.exists(file_name_dir_kmeans): shutil.rmtree(file_name_dir_kmeans)
    kmeans_model_fitted.save(file_name_dir_kmeans)
    #kmeans_model_fitted_loaded = KMeansModel.load(filename+".kmeans")
    #kmeans_model_fitted.hasSummary
    
    dict_file_name_dir = base_filename+".dict"
    if os.path.exists(dict_file_name_dir): shutil.rmtree(dict_file_name_dir)
    kms.save_frequency_dict(cluster_freq_dict, dict_file_name_dir)
    
    return kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict

    
def test_cluster (
    input_filename_train = "data/light_r10.000-pca-10-train.parquet",
    input_filename_test  = "data/light_r10.000-pca-10-test.parquet",
    base_filename      = "data/light_r10.000-pca-10-kmeans-1000",
    k_kmeans = 1000,
    arguments_col = [ 'Y_UE_', 'Y_GIO_', 'Y_MES_', 'Y_FASCI_', 'Y_GIORNI_ALLA_PRENOTAZIONE_'], 
    feature_col = "pca_features"
    ):    
    
    
    # BUILD MODEL
    t1 = datetime.datetime.now()
    kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict = build_kmean_model(input_filename_train, base_filename, k_kmeans, arguments_col, feature_col)
    time_duration_kmean = (datetime.datetime.now()-t1)
    
    # TEST MODEL
    t1 = datetime.datetime.now()
    kmeans_test_ds_pca, accuracyMeanList, accuracyDictList = accuracy.accuracy_kmean_model(kmeans_model_fitted, input_filename_test, cluster_freq_dict, arguments_col)
    time_duration_test = (datetime.datetime.now()-t1)
    
    # REPORT ACCURACY
    tot_col = len(kmeans_train_ds_pca.columns)
    k_pca=len(kmeans_train_ds_pca.head(1)[0][feature_col])
    k_pca_perc=k_pca*100/(tot_col-1) 
    tot_rows = kmeans_train_ds_pca.count() + kmeans_test_ds_pca.count()
    split=(kmeans_train_ds_pca.count()*100/tot_rows, kmeans_test_ds_pca.count()*100/tot_rows)
    split_col=(kmeans_train_ds_pca.count(), kmeans_test_ds_pca.count())
    
    report_filename = base_filename +".report.txt"
    if os.path.exists(report_filename): os.remove(report_filename)
    report.write_report(report_filename, tot_col, k_kmeans, arguments_col, accuracyDictList, accuracyMeanList, time_duration_kmean, time_duration_test, k_pca=k_pca, k_pca_perc=k_pca_perc, split=split, split_col=split_col)

    return kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict, kmeans_test_ds_pca, accuracyMeanList, accuracyDictList, time_duration_kmean, time_duration_test, tot_col, k_pca, k_pca_perc, split, split_col

