

"""

import rankchoices.pipeline as pipeline
pipeline.execute()

"""

def execute(mode="ALL"):
    import datetime
    import sys
    import os
    import rankchoices.datasplit.split as split
    import rankchoices.reducefeatures.pca as pca
    import rankchoices.model.cluster as cluster
    
    
    base_dir_path   = 'data/light_r10.000-'

    split_input_filename  = base_dir_path + "data_clean.csv"
    split_train_file_name = base_dir_path + "split-train.parquet"
    split_test_file_name  = base_dir_path +"split-test.parquet"
    spl = [0.99, 0.01]
    
    # TRAIN TEST SPLIT
    if (mode == "ALL" or "SPLIT" in mode):
        
        train_ds, test_ds, time_duration_split = split.apply_split(
            input_filename         = split_input_filename,
            output_train_file_name = split_train_file_name,
            output_test_file_name  = split_test_file_name,
            random_seed = 1,
            split = spl)
        print("<> END SPLIT TRAIN/TEST STAGE " + str(datetime.timedelta(seconds=time_duration_split.total_seconds())))
    
    
    # PCA    
    k_pca_perc = 5
    pca_train_file_name= base_dir_path + "pca-" + str(k_pca_perc) + "-train.parquet"
    pca_test_file_name= base_dir_path + "pca-" + str(k_pca_perc) + "-test.parquet"
    
    if (mode == "ALL" or "PCA" in mode):
        dataset_pca_train, dataset_pca_test, k_pca, time_duration_pca = pca.apply_pca(
            input_filename_train = split_train_file_name,
            input_filename_test  = split_test_file_name,
            output_filename_train = pca_train_file_name,
            output_filename_test  = pca_test_file_name,
            k_pca_perc = k_pca_perc)
        print("<> END PCA STAGE " + str(datetime.timedelta(seconds=time_duration_pca.total_seconds())))
    
    # CLUSTER
    k_kmeans = 1000
    base_model_filename      = base_dir_path + "pca-" + str(k_pca_perc) +"-kmeans-" + str(k_kmeans)
    if (mode == "ALL" or "CLUSTER" in mode):
        kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict, kmeans_test_ds_pca, accuracyMeanList, accuracyDictList, time_duration_kmean, time_duration_test, tot_col, k_pca, k_pca_perc, split, split_col = cluster.test_cluster(
            input_filename_train = pca_train_file_name,
            input_filename_test  = pca_test_file_name,
            base_filename      = base_model_filename, #"data/light_r10.000-pca-10-kmeans-1000",
            k_kmeans = k_kmeans,
            arguments_col = [ 'Y_UE_', 'Y_GIO_', 'Y_MES_', 'Y_FASCI_', 'Y_GIORNI_ALLA_PRENOTAZIONE_']
            )
        print("<> END TRAINING STAGE " + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds())))
        print("<> END ACCURACY STAGE " + str(datetime.timedelta(seconds=time_duration_test.total_seconds())))


#execute(mode="SPLIT PCA")
execute(mode="CLUSTER")