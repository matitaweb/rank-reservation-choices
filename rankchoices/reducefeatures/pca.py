from pyspark.ml.feature import PCA, VectorAssembler
import rankchoices.commons.load_stage as load_stage

import datetime
import logging
import numpy as np



# to execute
#------------
"""
from rankchoices.reducefeatures.pca as pca
dataset_pca_train, dataset_pca_test, k_pca, time_duration_pca = pca.apply_default_pca()

"""

def add_feature_col(loaded_dataset, outputCol="features"):
    vecAssembler = VectorAssembler(inputCols=loaded_dataset.columns, outputCol=outputCol)
    output = vecAssembler.transform(loaded_dataset)
    return output

def get_pca_model(k_pca, loaded_dataset, inputCol="features", outputCol="pca_features"):
    
    pca = PCA(k=k_pca, inputCol=inputCol, outputCol=outputCol)
    pca_model = pca.fit(loaded_dataset)
    return  pca_model
    

def apply_pca (
    input_filename_train = "data/light_r10.000-split-train.parquet",
    input_filename_test  = "data/light_r10.000-split-test.parquet",
    
    output_filename_train = "data/light_r10.000-pca-10-train.parquet",
    output_filename_test  = "data/light_r10.000-pca-10-test.parquet",
    
    k_pca_perc = 10, 
    pcaInputCol="features", 
    pcaOutputCol="pca_features"
    ):
    
    t1 = datetime.datetime.now()
    
    # TRAINING SET
    loaded_dataset_train = load_stage.load_from_parquet (input_filename_train);
    
    
    tot_col=len(loaded_dataset_train.columns)
    k_pca = int(tot_col*k_pca_perc/100)
    
    loaded_dataset_train_f = add_feature_col(loaded_dataset_train, outputCol=pcaInputCol)
    pca_model = get_pca_model(k_pca, loaded_dataset_train_f, inputCol=pcaInputCol, outputCol=pcaOutputCol)
    
    loaded_dataset_train_pca_f = pca_model.transform(loaded_dataset_train_f)
    loaded_dataset_train_pca = loaded_dataset_train_pca_f.drop(pcaInputCol)
    loaded_dataset_train_pca.write.parquet(output_filename_train, mode="overwrite")
    
    # TEST SET
    loaded_dataset_test     = load_stage.load_from_parquet (input_filename_test);
    loaded_dataset_test_f   = add_feature_col(loaded_dataset_test, outputCol=pcaInputCol)
    loaded_dataset_test_pca_f = pca_model.transform(loaded_dataset_test_f)
    loaded_dataset_test_pca = loaded_dataset_test_pca_f.drop(pcaInputCol)
    loaded_dataset_test_pca.write.parquet(output_filename_test, mode="overwrite")
    
    time_duration_pca = (datetime.datetime.now()-t1)
    
    return loaded_dataset_train_pca, loaded_dataset_test_pca, k_pca, time_duration_pca
