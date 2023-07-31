from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff, weeks_to_subtract
from myproject.datasets import utils


@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("ri.foundry.main.dataset.f0e0629f-e190-40fa-a7c3-a95ce8d2fe6a"),
    df_outpatient_with_diagnosis=Input("ri.foundry.main.dataset.e3a5741d-6cb5-4b64-8d3b-0be26b1b6e62"),
)
def compute(df_outpatient_with_diagnosis):
    """
    Calculate number of attendances from the outpatient dataset
    Calculate number of diagnoses based on primary diagnosis code
    Calculate % of attendances which had a diagnosis
    The output from this transformation is later merged with the patient_features table
    """

    # Determine number of OPA attendances
    # remove null IDs
    df_outpatient_with_diagnosis = df_outpatient_with_diagnosis.na.drop(subset=["patient_pseudo_id"])

    # Filter to time before cut-off
    df_outpatient_with_diagnosis = df_outpatient_with_diagnosis.filter(F.col("attendance_date") < F.lit(date_cutoff))

    # total number of attendances by ID
    dataset_opa_attend = df_outpatient_with_diagnosis.groupBy("patient_pseudo_id").count().withColumnRenamed("count", "opa_attend_total")
    dataset_opa_attend = dataset_opa_attend.cache()

    # loop through "weeks_to_subtract". Calculate the number of attendances in each period. Merge to dataset_OPA_attend
    for weeks in weeks_to_subtract:
        # get the number of outpatient attendances in the specified period
        total_number_OPA_in_period = utils.number_attendances_during_time_period(df=df_outpatient_with_diagnosis,
                                                                                 date=date_cutoff,
                                                                                 num_weeks_to_subtract=weeks,
                                                                                 variable_name_prefix="OPA_attendances_last_",
                                                                                 groupbycols=["patient_pseudo_id"]
                                                                                 )

        dataset_opa_attend = dataset_opa_attend.join(total_number_OPA_in_period,
                                                     "patient_pseudo_id",
                                                     "left")

    # Determine number of diagnoses from OPA
    df_outpatient_with_non_empty_diagnosis = df_outpatient_with_diagnosis.filter((F.col("Primary_Diagnosis_Code") != '') & (F.col("Primary_Diagnosis_Code").isNotNull()))

    dataset_opa_diagnoses = df_outpatient_with_non_empty_diagnosis.groupBy("patient_pseudo_id").count().withColumnRenamed("count", "opa_diagnoses_total")

    # Join table of count of attendances and diagnoses
    opa_features = dataset_opa_attend.join(dataset_opa_diagnoses,"patient_pseudo_id","left")

    # fill nulls with 0
    opa_features = opa_features.na.fill(0)

    # Calculate percentage of diagnoses to attendances
    opa_features = opa_features.withColumn("opa_diagnosis_pct_of_attendnaces",
                                           F.col("opa_diagnoses_total")/F.col("opa_attend_total")*100)

    return opa_features
