import sys, bz2, uuid
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

## LOAD CANCER DATASET
cancer = load_breast_cancer()
dataframe_load = pd.DataFrame(cancer.data)
dataframe_load.columns = cancer.feature_names 
data_label = cancer.target
dataframe = dataframe_load.assign(LABEL=data_label)

## CONVERT IN JSON AND THEN COMPRESSES
dataframe_json = dataframe.to_json(orient='split').encode()
compressed_data = bz2.compress(dataframe_json)

## ATTRIBUTE ID TO DATA AND THEN PUT THE IT IN MEMORY
dataframe_id = str(uuid.uuid4())
variables.put(dataframe_id, compressed_data)

## SHOW SOME INFO
print("dataframe id: ", dataframe_id)
print('dataframe size (original):   ', sys.getsizeof(dataframe_json), " bytes")
print('dataframe size (compressed): ', sys.getsizeof(compressed_data), " bytes")
print(dataframe.head())

## SPECIFY A FILE TYPE (IMAGE, CSV, ETC)
resultMetadata.put("task.dataframe_id", dataframe_id)