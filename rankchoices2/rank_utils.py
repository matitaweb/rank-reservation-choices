"""
UTILS


"""


from pyspark.sql.types import IntegerType
import pyspark.sql.functions as functions

class RankConfig:
    """
    X_ETA,X_SESSO,X_CAP,X_USL,
    X_FASCIA,X_GRADO_URG,X_QDGN,X_INVIANTE,X_ESENZIONE,X_PRESCRITTORE,
    X_PRESTAZIONE,X_BRANCA_SPECIALISTICA,X_GRUPPO_EROGABILE,
    Y_STER,Y_UE,
    Y_GIORNO_SETTIMANA,Y_MESE_ANNO,Y_FASCIA_ORARIA,Y_GIORNI_ALLA_PRENOTAZIONE

    """
    def __init__(self):
        #self.arguments_col_to_drop = [ 'X_PRESCRITTORE',                                            'Y_STER', 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
        #self.arguments_col_to_drop = [ 'X_PRESCRITTORE',                  'X_BRANCA_SPECIALISTICA', 'Y_STER', 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
        self.arguments_col_to_drop  = [ 'X_PRESCRITTORE', 'X_PRESTAZIONE', 'X_BRANCA_SPECIALISTICA', 'Y_STER', 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
        #self.arguments_col_to_drop = [                                                              'Y_STER', 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
        self.arguments_col_string_all = [
            ('STRING_X_CAP', 'X_CAP'),
            ('STRING_X_USL', 'X_USL'), 
            
            ('STRING_X_FASCIA', 'X_FASCIA'), 
            ('STRING_X_QDGN', 'X_QDGN'), 
            ('STRING_X_INVIANTE', 'X_INVIANTE'), 
            ('STRING_X_ESENZIONE', 'X_ESENZIONE'),
            ('STRING_X_PRESCRITTORE', 'X_PRESCRITTORE'),
            
            ('STRING_X_PRESTAZIONE', 'X_PRESTAZIONE'), 
            ('STRING_X_BRANCA_SPECIALISTICA', 'X_BRANCA_SPECIALISTICA'),
            ('STRING_X_GRUPPO_EROGABILE', 'X_GRUPPO_EROGABILE'), 
            
            ('STRING_Y_UE', 'Y_UE'),
            ('STRING_Y_STER', 'Y_STER')
            ]
        self.arguments_col_x_all = [ 'X_ETA','X_SESSO','X_CAP','X_USL', 'X_FASCIA','X_GRADO_URG','X_QDGN','X_INVIANTE','X_ESENZIONE','X_PRESCRITTORE', 'X_PRESTAZIONE','X_BRANCA_SPECIALISTICA', 'X_GRUPPO_EROGABILE']
        self.arguments_col_y_all = [ 'Y_STER', 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
        self.arguments_col_not_ohe_all = ['X_ETA']
        
        y_giorni_alla_prenotazione_udf = functions.UserDefinedFunction(convert_y_giorni_alla_prenotazione, IntegerType())
        self.arguments_col_to_quantize = [("Y_GIORNI_ALLA_PRENOTAZIONE", y_giorni_alla_prenotazione_udf)]
        
        self.oheFeatureOutputCol="features"
        self.pcaFeatureInputCol="features"
        self.pcaFeatureOutputCol="pca_features"
        
    def getArgumentsColToDrop(self):
        return self.arguments_col_to_drop

    def getOheCol(self, arguments_col, arguments_col_not_ohe):
        ohe_col = ["OHE_"+x for x in arguments_col if not x in arguments_col_not_ohe]
        return ohe_col
        
    def getOheFeatureOutputColName(self):
        return self.oheFeatureOutputCol
        
    def getPcaFeatureInputCol(self):
        return self.pcaFeatureInputCol
        
    def getPcaFeatureOutputCol(self):
        return self.pcaFeatureOutputCol

    def getArgumentsColString(self, arguments_col_to_drop):
        arguments_col_string = [x for x in self.arguments_col_string_all if x[0] not in arguments_col_to_drop and  x[1] not in arguments_col_to_drop ]
        return arguments_col_string
        
    
    def getArgumentsColX(self, arguments_col_to_drop):
        arguments_col_x = [x for x in self.arguments_col_x_all if x not in arguments_col_to_drop]
        return arguments_col_x
    
    def getArgumentsColY(self, arguments_col_to_drop):
        arguments_col_y = [x for x in self.arguments_col_y_all if x not in arguments_col_to_drop]
        return arguments_col_y
    
    def getArgumentsColNotOHE(self, arguments_col_to_drop):
        arguments_col_not_ohe = [x for x in self.arguments_col_not_ohe_all if x not in arguments_col_to_drop]
        return arguments_col_not_ohe
        
    def validate_with_metadata_exceptList(self):
        exceptList = ["X_ETA", "Y_GIORNO_SETTIMANA", "Y_MESE_ANNO", "Y_FASCIA_ORARIA", "Y_GIORNI_ALLA_PRENOTAZIONE"]
        return exceptList
        
    def getArgumentsColToQuantize(self):
        return self.arguments_col_to_quantize
    
    
    def get_input_schema(self, arguments_col_to_drop):

        from pyspark.sql.types import IntegerType
        from pyspark.sql.types import StringType
        from pyspark.sql.types import StructType
        from pyspark.sql.types import StructField
        
        filtered = []
        
        if(not "X_ETA" in arguments_col_to_drop):
            filtered.append(StructField("X_ETA", IntegerType(),False)) # all not nullable
        if(not "X_SESSO" in arguments_col_to_drop):
            filtered.append(StructField("X_SESSO", IntegerType(),False))
        if(not "STRING_X_CAP" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_CAP", StringType(),False))
        if(not "STRING_X_USL" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_USL", StringType(),False))
            
        if(not "STRING_X_FASCIA" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_FASCIA", StringType(),False))
        if(not "X_GRADO_URG" in arguments_col_to_drop):
            filtered.append(StructField("X_GRADO_URG", IntegerType(),False))
        if(not "STRING_X_QDGN" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_QDGN", StringType(),False))
        if(not "STRING_X_INVIANTE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_INVIANTE", StringType(),False))
        if(not "STRING_X_ESENZIONE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_ESENZIONE", StringType(),False))
        if(not "STRING_X_PRESCRITTORE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_PRESCRITTORE", StringType(),False))
        
        if(not "STRING_X_PRESTAZIONE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_PRESTAZIONE", StringType(),False))
        if(not "STRING_X_BRANCA_SPECIALISTICA" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_BRANCA_SPECIALISTICA", StringType(),False))
        if(not "STRING_X_GRUPPO_EROGABILE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_GRUPPO_EROGABILE", StringType(),False))
        
        if(not "STRING_Y_STER" in arguments_col_to_drop):
            filtered.append(StructField("STRING_Y_STER", StringType(),False))
        if(not "STRING_Y_UE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_Y_UE", StringType(),False))
        if(not "Y_GIORNO_SETTIMANA" in arguments_col_to_drop):
            filtered.append(StructField("Y_GIORNO_SETTIMANA", IntegerType(),False))
        if(not "Y_MESE_ANNO" in arguments_col_to_drop):
            filtered.append(StructField("Y_MESE_ANNO", IntegerType(),False))
        if(not "Y_FASCIA_ORARIA" in arguments_col_to_drop):
            filtered.append(StructField("Y_FASCIA_ORARIA", IntegerType(),False))
        if(not "Y_GIORNI_ALLA_PRENOTAZIONE" in arguments_col_to_drop):
            filtered.append(StructField("Y_GIORNI_ALLA_PRENOTAZIONE", IntegerType(),False))

        input_schema = StructType(filtered)
        
        return input_schema
        

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
    
    
        
def validate_with_metadata(rList, metadataDict, exceptList):
    resValidate = {}
    resValidate['valid'] = []
    resValidate['rejected'] = []
    
    for r in rList:
        rejectedCols = {key: value for key, value in r.items() if (not key in exceptList and key in metadataDict and not str(value) in metadataDict[key]['ml_attr']['vals']) }
            
        if(len(rejectedCols.keys()) == 0):
            resValidate['valid'].append(r);
            continue
        rejectedRow = {}
        rejectedRow['row'] = r
        rejectedRow['rejecterCols'] = rejectedCols
        resValidate['rejected'].append(rejectedRow)
    return resValidate
    
    
# DATA LOADING
def load_from_csv (spark, filename, input_schema):
    #spark = SparkSession.builder.master("local[*]").appName("Rank").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.csv(filename, header=True, mode="DROPMALFORMED",  schema=input_schema, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
    return result

def load_from_parquet (spark, filename):
    #spark = SparkSession.builder.master("local[*]").appName("Rank").config("spark.python.profile", "true").getOrCreate()
    result = spark.read.parquet(filename)
    return result


  