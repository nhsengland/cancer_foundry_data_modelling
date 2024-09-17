from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import diagnosis_columns_opa
from myproject.datasets import utils


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(outpatient_diagnoses_path),
    outpatient_activity_with_diagnosis=Input(
        outpatient_acitivty_diagnosis_path
    ),
)
def compute(outpatient_activity_with_diagnosis):
    """
    Create a table of outpatient diagnoses and the date from the outpatient dataset
    """
    outpatient_activity_with_diagnosis = outpatient_activity_with_diagnosis.withColumn(
        "diagnoses", F.array(diagnosis_columns_opa)
    )

    df_comorbidity_opa = utils.create_long_list_diagnoses_from_activity(
        df_activity=outpatient_activity_with_diagnosis, array_column="diagnoses"
    )

    df_comorbidity_opa = df_comorbidity_opa.withColumn("source", F.lit("OPA"))

    # remove duplicates of same diagnosis for same patient on same day
    df_comorbidity_opa = df_comorbidity_opa.drop_duplicates(
        ["patient_pseudo_id", "date", "diagnosis"]
    )

    return df_comorbidity_opa.select(
        "patient_pseudo_id", "activity_id", "date", "diagnosis", "source"
    )
