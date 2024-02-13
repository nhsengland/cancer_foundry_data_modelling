from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets.config import date_cutoff, weeks_to_subtract
from myproject.datasets import utils
import re


@transform_df(
    Output("ri.foundry.main.dataset.a3dfadbc-b022-4d88-98c2-8ee3392beed1"),
    source_df=Input("ri.foundry.main.dataset.a91f6452-b404-4ce3-b48a-7cafae278b7a"),
    cancer_symptoms=Input(
        "ri.foundry.main.dataset.e6f4ebc3-9175-4bd4-9ab0-550b011abde8"
    ),
)
def compute(source_df, cancer_symptoms):
    """
    Create features from the 111 dataset
    The output from this transformation is later merged with the patient_features table
    """
    # remove 999 calls and EDS calls
    source_df = source_df.filter(F.col("site_type_id") == 2)

    # remove null IDs
    source_df = source_df.na.drop(subset=["patient_pseudo_id"])

    # Filter to time before cut-off
    source_df = source_df.filter(F.col("call_date") < F.lit(date_cutoff))

    # merge cancer symptoms
    source_df = source_df.join(cancer_symptoms, "symptom_group", "left")

    # filtering to calls which had a cancer related symptom
    source_df_cancer_calls_only = source_df.filter(F.col("cancer_related") == "Y")

    # Adding a column for symptom_group which doesn't have spaces in the values
    # it also doesn't have brackets in the column name
    # this is needed to pivot the values to columns
    source_df_cancer_calls_only = source_df_cancer_calls_only.withColumn(
        "symptom_group_no_spaces", F.regexp_replace("symptom_group", r" |,|\(|\)", "_")
    )

    # total number of calls by ID
    dataset_111_calls = (
        source_df.groupBy("patient_pseudo_id")
        .count()
        .withColumnRenamed("count", "111_calls_total")
    )
    dataset_111_calls = dataset_111_calls.cache()

    # loop through "weeks_to_subtract". Calculate the number of calls in each period. Merge to dataset_111_calls
    for weeks in weeks_to_subtract:
        # get the number of calls in the specified period
        total_number_calls_in_period = utils.number_calls_during_time_period(
            df=source_df,
            date=date_cutoff,
            num_weeks_to_subtract=weeks,
            variable_name_prefix="111_calls_last_",
            groupbycols=["patient_pseudo_id"],
        )

        # get the number of calls related to cancer symptoms in the specified period
        total_number_cancer_calls_in_period = utils.number_calls_during_time_period(
            df=source_df_cancer_calls_only,
            date=date_cutoff,
            num_weeks_to_subtract=weeks,
            variable_name_prefix="111_calls_cancer_last_",
            groupbycols=["patient_pseudo_id"],
        )

        # get the number of calls related to specific symptom per person
        total_number_symptom_calls_in_period = (
            utils.number_symptom_calls_during_time_period(
                df=source_df_cancer_calls_only,
                date=date_cutoff,
                num_weeks_to_subtract=weeks,
                variable_name="111_calls_last_",
                groupbycols=["patient_pseudo_id"],
                pivotcol="symptom_group_no_spaces",
            )
        )

        # join the total number of calls in the period
        dataset_111_calls = dataset_111_calls.join(
            total_number_calls_in_period, "patient_pseudo_id", "left"
        )

        # join the total number of cancer related calls in the period
        dataset_111_calls = dataset_111_calls.join(
            total_number_cancer_calls_in_period, "patient_pseudo_id", "left"
        )

        # join number of symptom calls in the period
        dataset_111_calls = dataset_111_calls.join(
            total_number_symptom_calls_in_period, "patient_pseudo_id", "left"
        )

    # fill nulls with 0
    dataset_111_calls = dataset_111_calls.na.fill(0)

    return dataset_111_calls
