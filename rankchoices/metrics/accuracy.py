import numpy as np
import rankchoices.commons.load_stage as load_stage

def test_accuracy(kmeans_test_ds, arguments_col, cluster_freq_dict):

    accuracyDictList = []

    for r in kmeans_test_ds.collect():
        accuracyDict = {}
        accuracyDict['prediction'] = r['prediction']
    
        tot_acc=0
        for col_group_name in arguments_col:
    
            # extract filter colum by argument
            filter_cols_list = [x for x in kmeans_test_ds.columns if x.startswith( col_group_name )]
    
            #cluster_freq_dict[r['prediction']][col_group_name].sort_values(['OCCUR'], ascending=False, inplace=True)
            cfd = cluster_freq_dict[r['prediction']][col_group_name]
            last_position = cfd.tail(1)['POS'].values[0]
            position = cfd.tail(1)['POS'].values[0]
            accuracyDict[col_group_name+"VAL"]='NONE'
            accuracyDict[col_group_name+"POS"]=position
            accuracyDict[col_group_name+"POS_TOT"]=last_position
            
            #extract col name with value = 1
            y_cols_name_active = [filter_col_name for filter_col_name in filter_cols_list if r[filter_col_name] == 1]
            if(len(y_cols_name_active)  > 0) :
                accuracyDict[col_group_name+"VAL"]=y_cols_name_active[0]
                position_list =[r_cdf['POS'] for idx, r_cdf in cfd.iterrows() if r_cdf['IDX'] == y_cols_name_active[0]]
                if(len(position_list) > 0):
                    position = position_list[0]
                    accuracyDict[col_group_name+"POS"]=position
                
            #calcolo l'accuracy in base alla posizione trovata piu' vicina ai primi posti piu l'accuratezza e' alta
            acc = (float(last_position)-float(position))/float(last_position)
            accuracyDict[col_group_name] = acc
            tot_acc = tot_acc+acc
            
        accuracyDict['tot_acc'] = tot_acc
        accuracyDict['mean_acc'] = tot_acc/len(arguments_col)
        accuracyDictList.append(accuracyDict)
        
    return accuracyDictList
    
def accuracy_kmean_model(kmeans_model_fitted, input_filename_test, cluster_freq_dict, arguments_col):
    
    # PCA TEST SET
    test_ds_pca = load_stage.load_from_parquet (input_filename_test)
    
    # KMEANS ON TEST SET
    kmeans_test_ds_pca = kmeans_model_fitted.transform(test_ds_pca) 
    
    # CLUSTERING: test
    #t1 = datetime.datetime.now()
    accuracyDictList = test_accuracy(kmeans_test_ds_pca, arguments_col, cluster_freq_dict)
    accuracyMeanList = [e['mean_acc'] for e in accuracyDictList]
    #tot_accuracyMeanList = np.mean(accuracyMeanList)
    #print("TEST DONE accuracy "+str(len(tot_accuracyMeanList)) +" elements, "+str((datetime.datetime.now()-t1).total_seconds())+" sec")
    
    return kmeans_test_ds_pca, accuracyMeanList, accuracyDictList    