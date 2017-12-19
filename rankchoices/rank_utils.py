"""
UTILS


"""

from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField


class RankConfig:
    
    def __init__(self):
        self.arguments_col_to_drop = [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
        self.arguments_col_string_all = [
            ('STRING_X_CAP_RESIDENZA', 'X_CAP_RESIDENZA'), 
            ('STRING_X_USL_RES', 'X_USL_RES'), 
            
            ('STRING_X_FASCIA', 'X_FASCIA'), 
            ('STRING_X_QDGN', 'X_QDGN'), 
            ('STRING_X_INVIANTE', 'X_INVIANTE'), 
            ('STRING_X_ESENZIONE', 'X_ESENZIONE'),
            ('STRING_X_PRESCRITTORE', 'X_PRESCRITTORE'),
            
            ('STRING_X_PRESTAZIONE', 'X_PRESTAZIONE'), 
            ('STRING_X_BRANCA_SPECIALISTICA', 'X_BRANCA_SPECIALISTICA'), 
            
            ('STRING_Y_UE', 'Y_UE')
            ]
        self.arguments_col_x_all = [ 'X_ETA','X_SESSO','X_CAP_RESIDENZA','X_USL_RES',  'X_FASCIA','X_GRADO_URG','X_QDGN','X_INVIANTE','X_ESENZIONE','X_PRESCRITTORE',  'X_PRESTAZIONE','X_BRANCA_SPECIALISTICA']
        self.arguments_col_y_all = [ 'Y_UE', 'Y_GIORNO_SETTIMANA', 'Y_MESE_ANNO', 'Y_FASCIA_ORARIA', 'Y_GIORNI_ALLA_PRENOTAZIONE']
        self.arguments_col_not_ohe_all = ['X_ETA']
        
    def getArgumentsColToDrop(self):
        return self.arguments_col_to_drop


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
        
    def get_input_schema(self, arguments_col_to_drop):

        filtered = []
        
        if(not "X_ETA" in arguments_col_to_drop):
            filtered.append(StructField("X_ETA", IntegerType()))
        if(not "X_SESSO" in arguments_col_to_drop):
            filtered.append(StructField("X_SESSO", IntegerType()))
        if(not "STRING_X_CAP_RESIDENZA" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_CAP_RESIDENZA", StringType()))
        if(not "STRING_X_USL_RES" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_USL_RES", StringType()))
            
        if(not "STRING_X_FASCIA" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_FASCIA", StringType()))
        if(not "X_GRADO_URG" in arguments_col_to_drop):
            filtered.append(StructField("X_GRADO_URG", IntegerType()))
        if(not "STRING_X_QDGN" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_QDGN", StringType()))
        if(not "STRING_X_INVIANTE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_INVIANTE", StringType()))
        if(not "STRING_X_ESENZIONE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_ESENZIONE", StringType()))
        if(not "STRING_X_PRESCRITTORE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_PRESCRITTORE", StringType()))
        
        if(not "STRING_X_PRESTAZIONE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_PRESTAZIONE", StringType()))
        if(not "STRING_X_BRANCA_SPECIALISTICA" in arguments_col_to_drop):
            filtered.append(StructField("STRING_X_BRANCA_SPECIALISTICA", StringType()))
        
    
        if(not "STRING_Y_UE" in arguments_col_to_drop):
            filtered.append(StructField("STRING_Y_UE", StringType()))
        if(not "Y_GIORNO_SETTIMANA" in arguments_col_to_drop):
            filtered.append(StructField("Y_GIORNO_SETTIMANA", IntegerType()))
        if(not "Y_MESE_ANNO" in arguments_col_to_drop):
            filtered.append(StructField("Y_MESE_ANNO", IntegerType()))
        if(not "Y_FASCIA_ORARIA" in arguments_col_to_drop):
            filtered.append(StructField("Y_FASCIA_ORARIA", IntegerType()))
        if(not "Y_GIORNI_ALLA_PRENOTAZIONE" in arguments_col_to_drop):
            filtered.append(StructField("Y_GIORNI_ALLA_PRENOTAZIONE", IntegerType()))
            
        #print(filtered)
        input_schema = StructType(filtered)
        
        return input_schema