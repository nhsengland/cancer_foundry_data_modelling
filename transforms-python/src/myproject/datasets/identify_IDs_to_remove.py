from pyspark.sql import DataFrame, functions as F
from functools import reduce
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff
from myproject.datasets import utils


@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/target-datasets/IDs_to_remove"),
    patient_features_expanded=Input("ri.foundry.main.dataset.8e0a3764-c7b6-4e43-80f0-84139cf8ff6c"),
    cancer_diagnosis=Input("ri.foundry.main.dataset.57b3da18-8389-4083-bb74-499bb3208f04"),
)
def compute(patient_features_expanded, cancer_diagnosis):
    """
    Create list of patients to be removed:
    ids_without_age,
    ids_with_age_under_40,
    ids_with_age_over_120,
    ids_with_previous_or_active_cancer_phr,
    ids_with_previous_cancer_based_on_SUS_datasets,
    ids_who_passed_away_before_cut_off,
    ids_who_had_deduction_reason_other_than_death,
    ids_who_passed_away_after_cut_off_reason_other_than_cancer

    The outcome variable is updated such that those who did not have an observed cancer diagnosis after the cut-off
    date, but have cancer as a cause of death, are included in the ouucome columns cancer_diagnosis_in_next_N_weeks
    """

    # remove IDs without an age, or with age less than 40, or over 120 years
    ids_without_age = patient_features_expanded.filter(F.col("age").isNull()).select("patient_pseudo_id")
    ids_without_age = ids_without_age.withColumn("reason", F.lit("no valid age"))

    ids_with_age_under_40 = patient_features_expanded.filter(F.col("age")<40).select("patient_pseudo_id")
    ids_with_age_under_40 = ids_with_age_under_40.withColumn("reason", F.lit("age <40"))

    ids_with_age_over_120 = patient_features_expanded.filter(F.col("age")>120).select("patient_pseudo_id")
    ids_with_age_over_120 = ids_with_age_over_120.withColumn("reason", F.lit("age > 120"))

    # remove IDS with active or previous cancer (utilising PHR flags)
    ids_with_previous_or_active_cancer_phr = patient_features_expanded.filter((F.col("phr_active_cancer") == 1) | (F.col("phr_cancer") == 1)).select("patient_pseudo_id")
    ids_with_previous_or_active_cancer_phr = ids_with_previous_or_active_cancer_phr.withColumn("reason", F.lit("PHR flag of active or previous cancer"))

    # remove IDs with previous cancer diagnosis based on OPA/IP/AEA/ECDS
    ids_with_previous_cancer_based_on_SUS_datasets = cancer_diagnosis.select("patient_pseudo_id")
    ids_with_previous_cancer_based_on_SUS_datasets = ids_with_previous_cancer_based_on_SUS_datasets.withColumn("reason", F.lit("Previous cancer flag from SUS"))

    # remove IDs who passed away before cut off
    ids_who_passed_away_before_cut_off = patient_features_expanded.filter(F.col("death_month") < date_cutoff).select("patient_pseudo_id")
    ids_who_passed_away_before_cut_off = ids_who_passed_away_before_cut_off.withColumn("reason", F.lit("passed away before cut-off"))

    # remove IDs with deduction reason other than death
    ids_who_had_deduction_reason_other_than_death = patient_features_expanded.filter(((F.col("deducted") == 1) &
                                                                                     (F.col("deduction_reason") != "DEA"))).select("patient_pseudo_id")

    ids_who_had_deduction_reason_other_than_death = ids_who_had_deduction_reason_other_than_death.withColumn("reason", F.lit("deduction reason (not death)"))

    # identify those who passed away with cancer as a cause of death
    patient_features_expanded = utils.flag_if_patient_passed_away_with_cancer(patient_features_expanded)

    # remove IDs who passed away after the cut off date with a reason other than cancer
    ids_who_passed_away_after_cut_off_reason_other_than_cancer = patient_features_expanded.filter((F.col("death_month").isNotNull()) & 
                                                                                                  (F.col("death_month") >= date_cutoff) &
                                                                                                  (F.col("passed_away_with_cancer_cod") == 0)).select("patient_pseudo_id")

    ids_who_passed_away_after_cut_off_reason_other_than_cancer = ids_who_passed_away_after_cut_off_reason_other_than_cancer.withColumn("reason", F.lit("passed away after cut-off (without cancer as COD)"))

    # make list of dataframes of IDs to be removed
    all_ids_to_be_removed = [ids_without_age,
                             ids_with_age_under_40,
                             ids_with_age_over_120,
                             ids_with_previous_or_active_cancer_phr,
                             ids_with_previous_cancer_based_on_SUS_datasets,
                             ids_who_passed_away_before_cut_off,
                             ids_who_had_deduction_reason_other_than_death,
                             ids_who_passed_away_after_cut_off_reason_other_than_cancer]

    # create unioned dataframe of ids to be removed
    df_ids_to_be_removed = reduce(DataFrame.unionAll, all_ids_to_be_removed)

    return df_ids_to_be_removed