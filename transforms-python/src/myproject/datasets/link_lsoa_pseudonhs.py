from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.e3a02fe4-fbab-46b7-b16f-c89de4f88dfa"),
    source_df=Input("ri.foundry.main.dataset.8e0a3764-c7b6-4e43-80f0-84139cf8ff6c"),
)
def compute(source_df):

    df = source_df.select(F.col("patient_pseudo_id"), F.col("lsoa_2011_code"))
    
    return df
