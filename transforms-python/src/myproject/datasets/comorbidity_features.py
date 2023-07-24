from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff
from myproject.datasets import utils


@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/comorbidity_features"),
    df_inpatient_comorbidities=Input("ri.foundry.main.dataset.fe042d45-d081-4221-ae22-b930587fb500"),
    df_outpatient_comorbidities=Input("ri.foundry.main.dataset.fa19d9b5-066a-4d57-aef4-59d2cda4a8ac"),
    df_icd10_ref=Input("ri.foundry.main.dataset.3a0dfe4b-67de-4001-87eb-e43405df76d3")
)
def compute(df_inpatient_comorbidities, df_outpatient_comorbidities, df_icd10_ref):
    """
    Generate features from the comorbidities
    For a given set of diagnoses, identify the first and last date when the diagnosis was made
    Create flags if the diagnosis was made in the last n weeks
    """

    # Keep latest only
    df_icd10_ref = df_icd10_ref.filter(F.col("effective_to").isNull())
    df_icd10_ref = df_icd10_ref.filter(F.col("category_3_code").isNotNull())

    # Union comorbidities
    df_comorbidities = df_inpatient_comorbidities.unionByName(df_outpatient_comorbidities)

    # Filter
    df_comorbidities = df_comorbidities.select("patient_pseudo_id", "date", "diagnosis")

    # remove null IDs
    df_comorbidities = df_comorbidities.na.drop(subset=["patient_pseudo_id"])

    # merge diagnoses with reference
    df_comorbidities = df_comorbidities.join(df_icd10_ref, df_comorbidities.diagnosis == df_icd10_ref.alt_code, "left")

    # Filter to time before cut-off
    df_comorbidities = df_comorbidities.filter(F.col("date") < F.lit(date_cutoff))

    # create binary flag, earliest and latest date for the diagnosis category
    df_category_comorbidities = utils.create_comorbidity_category_features(df_comorbidities, column_to_aggregate="category_3_code")

    # Remove those where the diagnosis has not been matched to a category from the ICD10 reference dataset
    df_category_comorbidities = df_category_comorbidities.filter(F.col("category_3_code").isNotNull())

    # Pivot to one row per patient, with columns for the diagnosis categories
    df_pivot_comorbidities = df_category_comorbidities.groupBy("patient_pseudo_id") \
        .pivot("category_3_code") \
        .agg(F.sum("category_3_code_binary").alias("binary"),
             F.first("category_3_code_date_earliest").alias("earliest_date"),
             F.first("diagnosis").alias("diagnosis"),
             F.sum("category_3_code_in_last_4_weeks").alias("last_4_weeks"),
             F.sum("category_3_code_in_last_12_weeks").alias("last_12_weeks"),
             F.sum("category_3_code_in_last_26_weeks").alias("last_26_weeks"),
             F.sum("category_3_code_in_last_52_weeks").alias("last_52_weeks"),
             F.sum("category_3_code_in_last_260_weeks").alias("last_260_weeks"))

    # drop rows which are all empty apart from the pseudo id column
    all_columns_excluding_id = df_pivot_comorbidities.columns
    all_columns_excluding_id.remove("patient_pseudo_id")
    df_pivot_comorbidities = df_pivot_comorbidities.na.drop(how='all', subset=all_columns_excluding_id)

    return df_pivot_comorbidities
