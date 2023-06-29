from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets.config import date_cutoff, weeks_to_subtract
from myproject.datasets import utils


@transform_df(
    Output("ri.foundry.main.dataset.3b5c090a-bad0-4568-9163-d37a41c748e1"),
    df=Input("ri.foundry.main.dataset.0cc8c784-f957-47b5-9844-a69d7eee0f6a"),
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

    # total number of emergency attendances by ID
    dataset_ae_attend = df.groupBy("patient_pseudo_id").count().withColumnRenamed("count", "ae_attend_total")
    dataset_ae_attend = dataset_ae_attend.cache()

    # loop through "weeks_to_subtract". Calculate the number of attendances in each period. Merge to dataset_ae_attend
    for weeks in weeks_to_subtract:
        # get the number of emergency attendances in the specified period
        total_number_AE_in_period = utils.number_AE_during_time_period(df=df,
                                                                       date=date_cutoff,
                                                                       num_weeks_to_subtract=weeks,
                                                                       variable_name_prefix="ae_attendances_last_",
                                                                       groupbycols=["patient_pseudo_id"]
                                                                       )

        dataset_ae_attend = dataset_ae_attend.join(total_number_AE_in_period,
                                                   "patient_pseudo_id",
                                                   "left")

    # fill nulls with 0
    dataset_ae_attend = dataset_ae_attend.na.fill(0)

    return dataset_ae_attend