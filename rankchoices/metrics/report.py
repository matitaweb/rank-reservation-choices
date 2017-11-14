import numpy as np
import pandas as pd
import datetime

def write_report(filename, tot_col, k_kmeans, arguments_col, accuracyDictList, accuracyMeanList, time_duration_kmean, time_duration_test,  k_pca="-", k_pca_perc="-", split="-", split_col="-"):
    file = open(filename, 'w')
    file.write('filename: ' + str(filename)+'\n')
    file.write('k_pca: ' + str(k_pca_perc) + "% " + str(k_pca)+ ' / '+str(tot_col)+'\n')
    file.write('k_kmeans: ' + str(k_kmeans)+'\n')
    
    
    file.write('time kmean: ' + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds()))+'  \n')
    file.write('time test: ' + str(datetime.timedelta(seconds=time_duration_test.total_seconds()))+'  \n')
    file.write('train, test: ' + str(split)+'\n')
    file.write('train, test: ' + str(split_col)+'\n\n\n')
    
    mean = np.mean(accuracyMeanList)
    file.write('mean acc.: ' + str(mean)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.mean([e[c] for e in accuracyDictList])
        file.write("MEAN: " + str(c) + ' -> ' + str(m)+'\n')
        
    file.write('\n\n\n')
    
    maxV = np.max(accuracyMeanList)
    file.write('max acc.: ' + str(maxV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.max([e[c] for e in accuracyDictList])
        file.write("MAX: " + str(c) + ' -> ' + str(m)+'\n')
    
    file.write('\n\n\n')

    minV = np.min(accuracyMeanList)
    file.write('min acc.: ' + str(minV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.min([e[c] for e in accuracyDictList])
        file.write("MIN: " + str(c) + ' -> ' + str(m)+'\n')        
    
    file.write('------------------------------------'+'\n')
    
    file.write('\n\n\n')
    
    p = pd.DataFrame(accuracyDictList)
    file.write(p.to_string())
    
    file.write('------------------------------------'+'\n')
    
    file.close()