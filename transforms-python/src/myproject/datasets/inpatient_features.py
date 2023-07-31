from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff, weeks_to_subtract
from myproject.datasets import utils


@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/interim-datasets/IP_features"),
    inpatient_activity=Input("ri.foundry.main.dataset.793b42d6-f24d-4cca-b500-abc319dd5162"),
)
def compute(inpatient_activity):
    """
    Calculate number of attendances from the inpatient dataset
    Calculate number of diagnoses based on dimention_8 ()
    Calculate % of attendances which had a diagnosis
    The output from this transformation is later merged with the patient_features table
    """

    # Determine number of inpatient attendances
    # remove null IDs
    inpatient_activity = inpatient_activity.na.drop(subset=["patient_pseudo_id"])

    # Filter to time before cut-off
    inpatient_activity = inpatient_activity.filter(F.col("attendance_date") < F.lit(date_cutoff))

    # total number of attendances by ID
    dataset_ip_attend = inpatient_activity.groupBy("patient_pseudo_id").count().withColumnRenamed("count", "IP_attend_total")
    dataset_ip_attend = dataset_ip_attend.cache()

    # loop through "weeks_to_subtract". Calculate the number of attendances in each period. Merge to dataset_ip_attend
    for weeks in weeks_to_subtract:
        # get the number of inpatient attendances in the specified period
        total_number_IP_in_period = utils.number_attendances_during_time_period(df=inpatient_activity,
                                                                                date=date_cutoff,
                                                                                num_weeks_to_subtract=weeks,
                                                                                variable_name_prefix="IP_attendances_last_",
                                                                                groupbycols=["patient_pseudo_id"]
                                                                                )

        dataset_ip_attend = dataset_ip_attend.join(total_number_IP_in_period,
                                                   "patient_pseudo_id",
                                                   "left")

    # Determine number of diagnoses from IP
    df_inpatient_with_non_empty_diagnosis = inpatient_activity.filter((F.col("dimention_8") != '') & (F.col("dimention_8").isNotNull()))

    dataset_IP_diagnoses = df_inpatient_with_non_empty_diagnosis.groupBy("patient_pseudo_id").count().withColumnRenamed("count", "IP_diagnoses_total")

    # Join table of count of attendances and diagnoses
    IP_features = dataset_ip_attend.join(dataset_IP_diagnoses, "patient_pseudo_id", "left")

    # fill nulls with 0
    IP_features = IP_features.na.fill(0)

    # Calculate percentage of diagnoses to attendances
    IP_features = IP_features.withColumn("IP_diagnosis_pct_of_attendnaces",
                                         F.col("IP_diagnoses_total")/F.col("IP_attend_total")*100)

    return IP_features
