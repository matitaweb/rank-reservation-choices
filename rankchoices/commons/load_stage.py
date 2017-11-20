from pyspark.sql import SparkSession

def load_from_csv (filename):
    
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.csv(filename, header=True, mode="DROPMALFORMED",  inferSchema=True, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
    return result
    
    
def load_from_parquet (filename):
    
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.parquet(filename)
    return result    
    
def convert_y_giorni_alla_prenotazione(x):
    if x < 2:
        return 0 #"1gg"
    if x < 4:
        return 1 #"3gg"
    if x < 6:
        return 2 #"5gg"
    if x < 8:
        return 3 #"7gg"
    if x < 16:
        return 4 #"15gg"
    if x < 21:
        return 5 #"20gg"
    if x < 31:
        return 6 #"30gg"
    if x < 61:
        return 7 #"60gg"
        
    return 8 #"oltre60gg"    