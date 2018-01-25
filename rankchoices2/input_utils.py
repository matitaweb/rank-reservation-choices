
class InputPipeline:

    def __init__(self, base_filename = "../data/bo_since19-01-2018_annullato_no-strt_e_prst_valide", split=[0.09, 0.01], pca_perc=1, k_means_num=1000, position_test_threshold=10, stage_start="LOAD", stage_stop="TEST", stage_snapshots =[], random_seed=1):
        
        self.base_filename = base_filename
        self.split = split
        self.pca_perc = pca_perc
        self.k_means_num = k_means_num
        self.position_test_threshold = position_test_threshold
        self.stage_start = stage_start
        self.stage_stop = stage_stop
        self.stage_snapshots = stage_snapshots
        self.random_seed = random_seed
        self.input_filename          = base_filename+".csv"
    
        self.metadata_file_name_dir  = base_filename + "-metadata"
        self.string_indexer_path_dir = base_filename + "-indexer"
        
        self.output_train_file_name  = base_filename+"-train.parquet"
        self.output_test_file_name   = base_filename+"-test.parquet"
        
        self.pca_path_dir =  base_filename+"-pca-model"
        
        self.output_pca_train_filename = base_filename+"-pca-train.parquet"
        self.output_pca_test_filename  = base_filename+"-pca-test.parquet"
        
        self.file_name_dir_kmeans = base_filename+".kmeans"
        
        self.output_kmeans_train_ds_filename = base_filename+"-kmeans-train.parquet"
        self.output_kmeans_test_ds_filename = base_filename+"-kmeans-test.parquet"
                
        self.cluster_freq_dict_filename = base_filename+"-dict.json"
        
        self.model_info_filename = base_filename+"-model-info.json"
        
        self.report_load_data_stage_filename = base_filename +".load_data_stage.report.txt"
        self.report_pca_stage_filename = base_filename +".pca_stage.report.txt"
        self.report_kmeans_stage_filename = base_filename +".kmeans_stage.report.txt"
        self.report_dict_stage_filename = base_filename +".dict_stage.report.txt"
        self.report_accuracy_stage_filename = base_filename +".accuracy_stage.report.txt"
        
        
        
        
        
    
    