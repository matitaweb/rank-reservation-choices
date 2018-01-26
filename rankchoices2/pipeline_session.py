
class PipelineSession:
    
    
    def __init__(self):
        
        # load_data_stage
        self.load_data_stage_input_data = None
        self.load_data_stage_input_data_json = None
        self.load_data_stage_train_ds = None
        self.load_data_stage_test_ds = None
        self.indexer_dict = None
        self.time_duration_dataloader_start_stage = None
        self.time_duration_dataloader_load_data = None
        self.time_duration_dataloader_snapshot_stage = None
        self.time_duration_dataloader_quantize = None
        self.time_duration_dataloader_metadata = None
        self.time_duration_dataloader_indexer = None
        self.time_duration_dataloader_ohe = None
        
        
        self.pca_stage_train_ds = None
        self.pca_stage_test_ds = None
        self.time_duration_pca_load_data = None
        self.time_duration_pca_start_stage = None
        self.time_duration_pca_snapshot_stage = None
        
        
        self.kmeans_stage_train_ds = None
        self.kmeans_stage_test_ds = None
        self.kmeans_model_fitted = None
        self.time_duration_kmeans_load_data = None
        self.time_duration_kmeans_start_stage = None
        self.time_duration_kmeans_snapshot_stage = None
        self.wssse = None
        self.kmeans_centers = None
        
        self.cluster_freq_dict = None
        
        self.time_duration_dict_load_data = None
        self.time_duration_dict_start_stage = None
        self.time_duration_dict_snapshot_stage = None
        
        self.accuracyDictList = None
        self.time_duration_accuracy_load_data = None
        self.time_duration_accuracy_start_stage = None
        self.time_duration_accuracy_snapshot_stage = None
        
    
    