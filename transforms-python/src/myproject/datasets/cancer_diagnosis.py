from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets.config import date_cutoff
from myproject.datasets import utils

@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/cancer_diagnosis"),
    df_inpatient_comorbidities=Input("ri.foundry.main.dataset.fe042d45-d081-4221-ae22-b930587fb500"),
    df_outpatient_comorbidities=Input("ri.foundry.main.dataset.fa19d9b5-066a-4d57-aef4-59d2cda4a8ac"),
    df_icd10_ref=Input("ri.foundry.main.dataset.3a0dfe4b-67de-4001-87eb-e43405df76d3")
)
def compute(df_inpatient_comorbidities, df_outpatient_comorbidities, df_icd10_ref):
    """
    Identify cancer diagnosis
    """
    # Union comorbidities
    df_comorbidities = df_inpatient_comorbidities.unionByName(df_outpatient_comorbidities)

    # Filter to time before cut-off
    df_comorbidities = df_comorbidities.filter(F.col("date") < F.lit(date_cutoff))

    # identify cancer only, exclude non-melanoma skin cancer
    df_cancer_only = df_comorbidities.filter(F.col("diagnosis").like("C%") & ~F.col("diagnosis").like("C44%"))

    df_cancer_only = utils.create_comorbidity_features(df_cancer_only.select("patient_pseudo_id", "date", "diagnosis"), "cancer")

    return df_cancer_only
