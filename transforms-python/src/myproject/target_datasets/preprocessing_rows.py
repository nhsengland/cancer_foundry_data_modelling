import pyspark.sql.functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff, target_weeks
from myproject.datasets import utils


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output("ri.foundry.main.dataset.00a2c97d-1499-4b4e-8cad-3af7cba8a599"),
    patient_features_expanded=Input(
        "ri.foundry.main.dataset.8e0a3764-c7b6-4e43-80f0-84139cf8ff6c"
    ),
    df_ids_to_remove=Input(
        "ri.foundry.main.dataset.b31e0565-7803-4729-8805-961db7279f07"
    ),
)
def compute(patient_features_expanded, df_ids_to_remove):
    """
    Filter dataset to only patients to be included in modelling dataset
    The outcome variable is updated such that those who did not have an observed cancer diagnosis after the cut-off
    date, but have cancer as a cause of death, are included in the ouucome columns cancer_diagnosis_in_next_N_weeks
    """

    # get unique values (some IDs may have multiple reasons for removal)
    df_ids_to_be_removed = df_ids_to_remove.dropDuplicates(["patient_pseudo_id"])

    df_patient_features_subset_patients = patient_features_expanded.join(
        df_ids_to_be_removed, "patient_pseudo_id", "left"
    )

    # keep all patients who do not have a reason to be removed
    df_patient_features_subset_patients = df_patient_features_subset_patients.filter(
        F.col("reason").isNull()
    )

    # identify those who passed away with cancer as a cause of death
    df_patient_features_subset_patients = utils.flag_if_patient_passed_away_with_cancer(
        df_patient_features_subset_patients
    )

    # identifying those whose first diagnosis of cancer after the cut-off date is as a cause of death
    df_patient_features_subset_patients = (
        df_patient_features_subset_patients.withColumn(
            "first_cancer_diagnosis_at_death_after_cut_off",
            F.when(
                (F.col("passed_away_with_cancer_cod") == 1)
                & (F.col("all_cancer_diagnoses_after_cut_off").isNull())
                & (F.col("death_month") > date_cutoff),
                1,
            ).otherwise(0),
        )
    )

    # Add those who passed away with cancer to cancer group (i.e. these may be those who did not have a cancer diagnosis while alive)
    for weeks in target_weeks:
        # obtain the date N weeks ahead of cut-off date
        later_date = utils.add_n_weeks_from_date(date_cutoff, weeks)

        # create binary column indicating if the death with cancer diagnosis as one of causes of death occurred was within N 'weeks' of the target date, e.g. 52 weeks
        df_patient_features_subset_patients = (
            df_patient_features_subset_patients.withColumn(
                "death_with_first_cancer_diagnosis_in_next_" + str(weeks) + "_weeks",
                F.when(
                    (F.col("death_month") <= later_date)
                    & (F.col("first_cancer_diagnosis_at_death_after_cut_off") == 1),
                    1,
                ).otherwise(0),
            )
        )

        # updating the outcome variable cancer_diagnosis_in_next_N_weeks, to be 1 if was already 1 from the
        #  cancer diagnosis obtained from SUS datasets OR 1 based on the flag created above (i.e. due to cancer cause of death)
        df_patient_features_subset_patients = (
            df_patient_features_subset_patients.withColumn(
                "cancer_diagnosis_in_next_" + str(weeks) + "_weeks",
                F.when(
                    (
                        F.col(
                            "death_with_first_cancer_diagnosis_in_next_"
                            + str(weeks)
                            + "_weeks"
                        )
                        == 1
                    )
                    | (F.col("cancer_diagnosis_in_next_" + str(weeks) + "_weeks") == 1),
                    1,
                ).otherwise(0),
            )
        )

    return df_patient_features_subset_patients
