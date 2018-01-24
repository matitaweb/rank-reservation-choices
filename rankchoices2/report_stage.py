


def write_report(filename, tot_col, k_kmeans, arguments_col, accuracyDictList, accuracyMeanList, time_duration_split, time_duration_pca, time_duration_kmean, time_duration_test, position_threshold,  k_pca="-", k_pca_perc="-", split="-", split_col="-"):
    file = open(filename, 'w')
    file.write('filename: ' + str(filename)+'\n')
    file.write('k_pca: ' + str(k_pca_perc) + "% " + str(k_pca)+ ' / '+str(tot_col)+'\n')
    file.write('k_kmeans: ' + str(k_kmeans)+'\n')
    
    file.write('time split: ' + str(datetime.timedelta(seconds=time_duration_split.total_seconds()))+'  \n')
    file.write('time pca: ' + str(datetime.timedelta(seconds=time_duration_pca.total_seconds()))+'  \n')
    file.write('time kmean: ' + str(datetime.timedelta(seconds=time_duration_kmean.total_seconds()))+'  \n')
    file.write('time test: ' + str(datetime.timedelta(seconds=time_duration_test.total_seconds()))+'  \n')
    file.write('train, test: ' + str(split)+'\n')
    file.write('train, test: ' + str(split_col)+'\n\n\n')
    
    file.write('position_threshold: ' + str(position_threshold)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    # count threshold per ogni argomento
    for c in arguments_col:
        positionList = [e[c]["LAST_POS"] if e[c]["POS"] is None  else e[c]["POS"]  for e in accuracyDictList]
        last_pos = accuracyDictList[0][c]["LAST_POS"]
        #print(len(positionList))
        #continue
        mean_pos = np.mean(positionList)
        min_pos = np.min(positionList)
        max_pos = np.max(positionList)
        count_position = len(positionList)
        count_position_threshold = count_position
        
        
        file.write(str(c)+'\n')
        file.write('------------------------------------'+'\n')
        file.write("mean_pos: " + str(mean_pos)+'\n')
        file.write("min_pos: " + str(min_pos)+'\n')
        file.write("max_pos: " + str(max_pos)+'\n')
        file.write("last_pos: " + str(last_pos)+'\n')
        file.write("count_position: " + str(count_position)+'\n')
        if not (position_threshold is None):
            for i in range(position_threshold):
                positionListThreshold = [e for e in positionList if e <= i]
                count_position_threshold = len(positionListThreshold)
                count_position_threshold_perc = float(count_position_threshold)/float(count_position)
                file.write("ENTRO LA POS: " + str(i) + " -> " + str(count_position_threshold_perc) +" = " + str(count_position_threshold)  +"/" + str(count_position)+'\n')
        file.write('------------------------------------'+'\n\n')
    
    mean = np.mean(accuracyMeanList)
    file.write('mean acc.: ' + str(mean)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    
    for c in arguments_col:
        m = np.mean([e[c]['ACC'] for e in accuracyDictList])
        file.write("MEAN: " + str(c) + ' -> ' + str(m)+'\n')
        
    file.write('\n\n\n')
    
    maxV = np.max(accuracyMeanList)
    file.write('max acc.: ' + str(maxV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.max([e[c]['ACC'] for e in accuracyDictList])
        file.write("MAX: " + str(c) + ' -> ' + str(m)+'\n')
    
    file.write('\n\n\n')

    minV = np.min(accuracyMeanList)
    file.write('min acc.: ' + str(minV)+'\n')
    file.write('------------------------------------'+'\n\n')
    
    for c in arguments_col:
        m = np.min([e[c]['ACC'] for e in accuracyDictList])
        file.write("MIN: " + str(c) + ' -> ' + str(m)+'\n')        
    
    file.write('------------------------------------'+'\n')
    
    file.write('\n\n\n')
    
    file.write('------------------------------------'+'\n')
    
    file.close()