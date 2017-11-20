
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import lead, col, sum, count
import pyspark.sql.functions as functions
from pyspark.ml import Pipeline
from pyspark.sql.functions import lead, col, sum
import pandas as pd
import os
import shutil
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession



def build_kmean_fitted(dataset, k_kmeans, feature_col = "pca_features"):
    kmeans = KMeans().setK(k_kmeans).setSeed(1).setFeaturesCol(feature_col)
    kmodel_fitted = kmeans.fit(dataset)
    return kmodel_fitted

def build_frequency_dict(arguments_col, dataset):

    clusterDict = {}

    for col_group_name in arguments_col: 
        
        # extract filter colum by argument
        filter_cols = [x for x in dataset.columns if x.startswith( col_group_name )]    
        exprs = [sum(x).alias(x) for x in filter_cols]
        
        #extract columns sum only for cols defined in arguments
        dataset_group_by = dataset.groupBy("prediction").agg(*exprs).collect()
        
        # traspone col sum in key value dictionary for every cluster 
        for r in dataset_group_by:
            c = r['prediction']
            if not c in clusterDict:
                clusterDict[c] = {}
            
            v ={'IDX':[], 'OCCUR':[], 'POS': []}
            for x in filter_cols: 
                v['IDX'].append(x)
                v['OCCUR'].append(r[x])
                v['POS'].append(-1)
            df=pd.DataFrame(v)
            
            # sorting upper the most frequent
            df.sort_values(['OCCUR'], ascending=False, inplace=True)
            vals = df.values
            
            # ADD CORRECT VALUES TO POSITION
            curr_pos = -1
            curr_val = -1
            for z in vals :
                if(z[1] != curr_val):
                    curr_val = z[1]
                    curr_pos=curr_pos+1
                z[2]= curr_pos
            df_pos = pd.DataFrame(vals, columns=['IDX', 'OCCUR', 'POS'])
            
            
            clusterDict[c][col_group_name]=df_pos

    return clusterDict







def save_frequency_dict(cluster_freq_dict, file_name_dir):
    if os.path.exists(file_name_dir): shutil.rmtree(file_name_dir)
    os.makedirs(file_name_dir)
    
    
    for c, cgroup in cluster_freq_dict.items():
        for g, df in cgroup.items():
            fname = file_name_dir+"/"+ str(c) + "." +str(g)+".parquet"
            df.to_parquet(fname)
        
def load_frequency_dict(file_name_dir):
    cluster_freq_dict =  {}
    for f in os.listdir(file_name_dir) :
        if os.path.isfile(os.path.join(file_name_dir, f)):
            df = pd.read_parquet(os.path.join(file_name_dir,f))
            fname_split = f.split(".")
            c= int(fname_split[0])
            col_group_name = str(fname_split[1])
            
            if not c in cluster_freq_dict:
                cluster_freq_dict[c]={}
                
            cluster_freq_dict[c][col_group_name]=df
            
    return cluster_freq_dict
        