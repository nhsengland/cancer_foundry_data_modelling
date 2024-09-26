import pyspark.sql.functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff, target_weeks
from myproject.datasets import utils


@configure(profile=["NUM_EXECUTORS_8", "DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(
        patient_features_stroke_path
    ),
    patient_features_expanded=Input(
        patient_features_expanded_path
    ),
    df_ids_to_remove=Input(
        ids_to_remove_path
    ),
)
def compute(patient_features_expanded, df_ids_to_remove):
    """
    Filter dataset to only patients to be included in modelling dataset
    The outcome variable is updated such that those who did not have an observed stroke diagnosis after the cut-off
    date, but have stroke as a cause of death, are included in the outcome columns stroke_diagnosis_in_next_N_weeks
    """

    # get unique values (some IDs may have multiple reasons for removal)
    df_ids_to_be_removed = df_ids_to_remove.dropDuplicates(["patient_pseudo_id"])

    df_patient_features_subset_patients = patient_features_expanded.join(
        df_ids_to_be_removed, "patient_pseudo_id", "leftanti"
    )

    # keep all patients who do not have a reason to be removed
    # df_patient_features_subset_patients = df_patient_features_subset_patients.filter(F.col("reason").isNull())

    return df_patient_features_subset_patients
