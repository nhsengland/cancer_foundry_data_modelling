from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff
from myproject.datasets import utils


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(cancer_diagnosis_path),
    df_comorbidities=Input(
        comorbidities_path
    ),
    df_icd10_ref=Input(icd10_path),
)
def compute(df_comorbidities, df_icd10_ref):
    """
    Identify cancer diagnosis
    """
    # Filter to time before cut-off
    df_comorbidities = df_comorbidities.filter(F.col("date") < F.lit(date_cutoff))

    # identify cancer only, exclude non-melanoma skin cancer
    df_cancer_only = df_comorbidities.filter(
        F.col("diagnosis").like("C%") & ~F.col("diagnosis").like("C44%")
    )

    df_cancer_only = utils.create_comorbidity_features(
        df_cancer_only.select("patient_pseudo_id", "date", "diagnosis"),
        "cancer_before_cut_off",
    )

    return df_cancer_only
