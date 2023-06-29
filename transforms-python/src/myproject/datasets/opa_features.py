from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets.config import date_cutoff, weeks_to_subtract
from myproject.datasets import utils


@transform_df(
    Output("ri.foundry.main.dataset.f0e0629f-e190-40fa-a7c3-a95ce8d2fe6a"),
    df=Input("ri.foundry.main.dataset.4725c357-f053-498d-9113-127bde5edc55"),
)
def compute(df):
    """
    Create features from the outpatient dataset
    The output from this transformation is later merged with the patient_features table
    """

    # remove null IDs
    df = df.na.drop(subset=["patient_pseudo_id"])

    # Filter to time before cut-off
    df = df.filter(F.col("attendance_date") < F.lit(date_cutoff))

    # total number of attendances by ID
    dataset_opa_attend = df.groupBy("patient_pseudo_id").count().withColumnRenamed("count", "opa_attend_total")
    dataset_opa_attend = dataset_opa_attend.cache()

    # loop through "weeks_to_subtract". Calculate the number of attendances in each period. Merge to dataset_OPA_attend
    for weeks in weeks_to_subtract:
        # get the number of outpatient attendances in the specified period
        total_number_OPA_in_period = utils.number_OPA_during_time_period(df=df,
                                                                         date=date_cutoff,
                                                                         num_weeks_to_subtract=weeks,
                                                                         variable_name_prefix="OPA_attendances_last_",
                                                                         groupbycols=["patient_pseudo_id"]
                                                                         )

        dataset_opa_attend = dataset_opa_attend.join(total_number_OPA_in_period,
                                                     "patient_pseudo_id",
                                                     "left")

    # fill nulls with 0
    dataset_opa_attend = dataset_opa_attend.na.fill(0)

    return dataset_opa_attend