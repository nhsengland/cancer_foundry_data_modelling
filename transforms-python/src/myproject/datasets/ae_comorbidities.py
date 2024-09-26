from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets import utils


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(ae_diagnosis_path),
    df_ae_activity=Input(ae_activity_path),
    df_ecds_mapping=Input(ecds_path),
)
def compute(df_ae_activity, df_ecds_mapping):
    """
    Create a table of A&E diagnoses and the date from the emergency activity dataset
    SNOMED codes from dimention_8 are mapped to ICD10 codes to make consistent
    with the inpatient and outpatient dataset
    """
    df_ae_activity = df_ae_activity.select(
        "patient_pseudo_id", "activity_id", "attendance_date", "dimention_8"
    )

    # only including activities which had a diagnosis
    df_ae_activity = df_ae_activity.filter(F.col("dimention_8").isNotNull())

    df_ae_activity = df_ae_activity.withColumn(
        "dimention_8_array", F.split(F.col("dimention_8"), ",")
    )

    df_comorbidity_ae = utils.create_long_list_diagnoses_from_activity(
        df_activity=df_ae_activity, array_column="dimention_8_array"
    )

    df_comorbidity_ae = df_comorbidity_ae.join(
        df_ecds_mapping,
        df_comorbidity_ae.diagnosis == df_ecds_mapping.SNOMED_Code,
        "left",
    )

    df_comorbidity_ae = df_comorbidity_ae.withColumn(
        "ICD10_Mapping_remove_dots", F.regexp_replace("ICD10_Mapping", r"\.", "")
    )

    df_comorbidity_ae = df_comorbidity_ae.withColumn("source", F.lit("Emergency"))

    # Changing column names so the diagnosis column name is the ICD10 diagnosis
    df_comorbidity_ae = df_comorbidity_ae.withColumnRenamed(
        "diagnosis", "SNOMED_diagnosis"
    )
    df_comorbidity_ae = df_comorbidity_ae.withColumnRenamed(
        "ICD10_Mapping_remove_dots", "diagnosis"
    )

    # select columns
    df_comorbidity_ae = df_comorbidity_ae.select(
        "patient_pseudo_id",
        "activity_id",
        "date",
        "SNOMED_diagnosis",
        "SNOMED_Code",
        "SNOMED_UK_Preferred_Term",
        "ECDS_Group1",
        "ECDS_Group2",
        "ECDS_Group3",
        "ECDS_Description",
        "diagnosis",
        "source",
    )

    # remove duplicates of same diagnosis for same patient on same day
    df_comorbidity_ae = df_comorbidity_ae.drop_duplicates(
        ["patient_pseudo_id", "date", "diagnosis"]
    )

    return df_comorbidity_ae
