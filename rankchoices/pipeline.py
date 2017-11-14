

"""

import rankchoices.pipeline as pipeline
pipeline.execute()

"""

def execute():
    import datetime
    import sys
    import rankchoices.datasplit.split as split
    import rankchoices.reducefeatures.pca as pca
    import rankchoices.model.cluster as cluster
    
    # TRAIN TEST SPLIT
    train_ds, test_ds, time_duration_split = split.apply_split()
    print("<> END SPLIT TRAIN/TEST STAGE " + str(datetime.timedelta(seconds=time_duration_split.total_seconds())))
    
    # PCA    
    dataset_pca_train, dataset_pca_test, k_pca, time_duration_pca = pca.apply_pca()
    print("<> END PCA STAGE " + str(datetime.timedelta(seconds=time_duration_pca.total_seconds())))
    
    # CLUSTER
    kmeans_model_fitted, kmeans_train_ds_pca, cluster_freq_dict, kmeans_test_ds_pca, accuracyMeanList, accuracyDictList, time_duration_kmean, time_duration_test, tot_col, k_pca, k_pca_perc, split, split_col = cluster.test_cluster()
    print("<> END TRAINING STAGE " + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds())))
    print("<> END ACCURACY STAGE " + str(datetime.timedelta(seconds=time_duration_test.total_seconds())))
    
