import pyspark.sql.functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff, target_weeks
from myproject.datasets import utils


@configure(profile=['NUM_EXECUTORS_8', 'DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/target-datasets/patient_features_subset_stroke"),
    patient_features_expanded=Input("ri.foundry.main.dataset.8e0a3764-c7b6-4e43-80f0-84139cf8ff6c"),
    df_ids_to_remove=Input("ri.foundry.main.dataset.bf97af2b-f0a0-408c-89bd-418517d879ce"),
)
def compute(patient_features_expanded, df_ids_to_remove):
    """
    Filter dataset to only patients to be included in modelling dataset
    The outcome variable is updated such that those who did not have an observed stroke diagnosis after the cut-off
    date, but have stroke as a cause of death, are included in the outcome columns stroke_diagnosis_in_next_N_weeks
    """

    # get unique values (some IDs may have multiple reasons for removal)
    df_ids_to_be_removed = df_ids_to_remove.dropDuplicates(["patient_pseudo_id"])

    df_patient_features_subset_patients = patient_features_expanded.join(df_ids_to_be_removed, "patient_pseudo_id", "leftanti")

    # keep all patients who do not have a reason to be removed
    #df_patient_features_subset_patients = df_patient_features_subset_patients.filter(F.col("reason").isNull())

    return df_patient_features_subset_patients
