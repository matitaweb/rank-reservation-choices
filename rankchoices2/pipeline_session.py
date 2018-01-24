
class PipelineSession:
    
    
    def __init__(self):
        self.load_data_stage_train_ds = None
        self.load_data_stage_test_ds = None
        
        
        self.pca_stage_train_ds = None
        self.pca_stage_test_ds = None
        
        self.kmeans_stage_train_ds = None
        self.kmeans_stage_test_ds = None
        self.kmeans_model_fitted = None
        
        self.cluster_freq_dict = None
    
    