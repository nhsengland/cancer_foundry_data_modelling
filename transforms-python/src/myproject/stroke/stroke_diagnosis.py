from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff
from myproject.datasets import utils


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(
        stroke_diagnosis_path
    ),
    df_comorbidities=Input(
        combined_comorbidities_path
    ),
    df_icd10_ref=Input(icd10_ref_path),
)
def compute(df_comorbidities, df_icd10_ref):
    """
    Identify stroke diagnosis
    """
    # Filter to time before cut-off
    df_comorbidities = df_comorbidities.filter(F.col("date") < F.lit(date_cutoff))

    # identify cancer only, exclude non-melanoma skin cancer
    df_stroke_only = df_comorbidities.filter(F.col("diagnosis").like("I6%"))

    df_stroke_only = utils.create_comorbidity_features(
        df_stroke_only.select("patient_pseudo_id", "date", "diagnosis"),
        "stroke_before_cut_off",
    )

    return df_stroke_only
