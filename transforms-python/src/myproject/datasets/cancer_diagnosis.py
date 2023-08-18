from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff
from myproject.datasets import utils

@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("ri.foundry.main.dataset.57b3da18-8389-4083-bb74-499bb3208f04"),
    df_comorbidities=Input("ri.foundry.main.dataset.3f34bfba-9440-4b7b-9d54-59be8a52bb0e"),
    df_icd10_ref=Input("ri.foundry.main.dataset.3a0dfe4b-67de-4001-87eb-e43405df76d3")
)
def compute(df_comorbidities, df_icd10_ref):
    """
    Identify cancer diagnosis
    """
    # Filter to time before cut-off
    df_comorbidities = df_comorbidities.filter(F.col("date") < F.lit(date_cutoff))

    # identify cancer only, exclude non-melanoma skin cancer
    df_cancer_only = df_comorbidities.filter(F.col("diagnosis").like("C%") & ~F.col("diagnosis").like("C44%"))

    df_cancer_only = utils.create_comorbidity_features(df_cancer_only.select("patient_pseudo_id", "date", "diagnosis"), "cancer_before_cut_off")

    return df_cancer_only
