import pipeline as pipe

input_filename = "/home/ubuntu/workspace/rank-reservation-choices/data/light_r1.csv"
base_dir = "/home/ubuntu/workspace/rank-reservation-choices/data/light_r10.000"


arguments_col_to_drop = pipe.getArgumentsColToDrop()
arguments_col_string = pipe.getArgumentsColString(arguments_col_to_drop)
arguments_col_x = pipe.getArgumentsColX(arguments_col_to_drop)
arguments_col_y = pipe.getArgumentsColY(arguments_col_to_drop)

dfraw = pipe.load_from_csv (input_filename, pipe.get_input_schema([]))

        
# remove column to
for col_to_drop in arguments_col_to_drop:
    if(col_to_drop in dfraw.columns):
        dfraw = dfraw.drop(col_to_drop)
        
metadata_file_name_dir= base_dir+"-metadata"
metadataDict = pipe.load_metadata(metadata_file_name_dir)
df = pipe.add_metadata(dfraw, metadataDict)        


stringindexer_path= base_dir+"-indexer"
indexer_dict = pipe.load_stringindexer_model_dict(stringindexer_path)

dfi = pipe.apply_stringindexer_model_dict(arguments_col_string, df, indexer_dict)