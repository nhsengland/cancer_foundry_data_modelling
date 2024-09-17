# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(patient_features_path),
    patient=Input(patient_record_path),
    outcome_variable=Input(
        outcome_variable_path
    ),
    comorbidity_features=Input(
       comorbidities_features_path
    ),
    cancer_diagnosis=Input(
        cancer_diagnosis_path
    ),
)
def compute(patient, outcome_variable, comorbidity_features, cancer_diagnosis):
    """
    Add features to the patient table from the Person Ontology

    Merge outcome variable (cancer diagnosis after cut-off period)
    Merge comorbidity features (flag based on ICD-10 codes and their timing)
    Merge cancer_diagnosis (diagnoses prior to cut-off timepoint)
    """
    # Outcome variable
    patient = patient.join(outcome_variable, "patient_pseudo_id", "left")

    # Comorbidity features
    patient = patient.join(comorbidity_features, "patient_pseudo_id", "left")

    # Cancer features (Cancer diagnosis in past)
    patient = patient.join(cancer_diagnosis, "patient_pseudo_id", "left")

    columns_to_fill_na_with_0 = list(
        set(
            outcome_variable.columns
            + comorbidity_features.columns
            + cancer_diagnosis.columns
        )
    )
    columns_to_fill_na_with_0.remove("patient_pseudo_id")

    # fill nulls with 0. This will ignore string columns
    patient = patient.na.fill(value=0, subset=columns_to_fill_na_with_0)

    return patient
