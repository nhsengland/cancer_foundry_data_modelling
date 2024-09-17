from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff, weeks_to_subtract
from myproject.datasets import utils


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(ae_path),
    df_emergency_activity=Input(
        emergency_activity_path
    ),
)
def compute(df_emergency_activity):
    """
    Create features related to number of attendances, and number of diagnosis codes from the A&E dataset
    The output from this transformation is later merged with the patient_features table
    """

    # remove null IDs
    df_emergency_activity = df_emergency_activity.na.drop(subset=["patient_pseudo_id"])

    # Filter to time before cut-off
    df_emergency_activity = df_emergency_activity.filter(
        F.col("attendance_date") < F.lit(date_cutoff)
    )

    # total number of emergency attendances by ID
    dataset_ae_attend = (
        df_emergency_activity.groupBy("patient_pseudo_id")
        .count()
        .withColumnRenamed("count", "ae_attend_total")
    )
    dataset_ae_attend = dataset_ae_attend.cache()

    # loop through "weeks_to_subtract". Calculate the number of attendances in each period. Merge to dataset_ae_attend
    for weeks in weeks_to_subtract:
        # get the number of emergency attendances in the specified period
        total_number_AE_in_period = utils.number_attendances_during_time_period(
            df=df_emergency_activity,
            date=date_cutoff,
            num_weeks_to_subtract=weeks,
            variable_name_prefix="ae_attendances_last_",
            groupbycols=["patient_pseudo_id"],
        )

        dataset_ae_attend = dataset_ae_attend.join(
            total_number_AE_in_period, "patient_pseudo_id", "left"
        )

    # Determine number of diagnoses (SNOMED codes in dimention_8) from AE
    df_aea_with_non_empty_diagnosis = df_emergency_activity.filter(
        (F.col("dimention_8") != "") & (F.col("dimention_8").isNotNull())
    )

    dataset_aea_diagnoses = (
        df_aea_with_non_empty_diagnosis.groupBy("patient_pseudo_id")
        .count()
        .withColumnRenamed("count", "aea_diagnoses_total")
    )

    # join AEA attendances with diagnoses

    dataset_ae_features = dataset_ae_attend.join(
        dataset_aea_diagnoses, "patient_pseudo_id", "left"
    )

    # fill nulls with 0
    dataset_ae_features = dataset_ae_features.na.fill(0)

    # calculate percentage of A&E attendances that had a diagnosis
    dataset_ae_features = dataset_ae_features.withColumn(
        "ae_diagnosis_pct_of_attendnaces",
        F.col("aea_diagnoses_total") / F.col("ae_attend_total") * 100,
    )

    return dataset_ae_features
