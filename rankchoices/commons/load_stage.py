from pyspark.sql import SparkSession

def load_from_csv (filename):
    
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.csv(filename, header=True, mode="DROPMALFORMED",  inferSchema=True, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
    return result
    
    
def load_from_parquet (filename):
    
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.parquet(filename)
    return result    