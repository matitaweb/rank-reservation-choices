
import rankchoices.commons.load_stage as load_stage
import datetime
import logging
import numpy as np
from pyspark.sql.functions import monotonically_increasing_id


# to execute
#------------
"""

from rankchoices.datasplit.split as split
train_ds, test_ds = split.test_split()

"""

def apply_split (
    input_filename         = "data/light_r10.000-data_clean.csv",
    output_train_file_name = "data/light_r10.000-split-train.parquet",
    output_test_file_name  = "data/light_r10.000-split-test.parquet",
    random_seed = 1,
    split = [0.95, 0.05]
    ):
        
    t1 = datetime.datetime.now()
    
    # DATA LOADING
    loaded_dataset_dup = load_stage.load_from_csv (input_filename)
    
    #add unique id
    ID_VAR= 'unique_id'
    loaded_dataset_id_var = loaded_dataset_dup.withColumn(ID_VAR, monotonically_increasing_id())

    # split in test/training set on a single col
    ds_id_var = loaded_dataset_id_var.select(ID_VAR)
    (train_ds_id_var, test_ds_id_var) = ds_id_var.randomSplit(split, random_seed)

    train = loaded_dataset_id_var.join(train_ds_id_var,ID_VAR,'inner')
    test = loaded_dataset_id_var.join(test_ds_id_var, ID_VAR,'inner')
    
    train_ds = train.drop(ID_VAR)
    test_ds = test.drop(ID_VAR)

    train_ds.write.parquet(output_train_file_name, mode="overwrite")
    test_ds.write.parquet(output_test_file_name, mode="overwrite")
    time_duration_split = (datetime.datetime.now()-t1)
    
    return train_ds, test_ds, time_duration_split
