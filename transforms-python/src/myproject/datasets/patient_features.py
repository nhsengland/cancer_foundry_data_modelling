# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output("ri.foundry.main.dataset.36644003-34c3-43d0-bace-751b3e071ea3"),
    patient=Input("ri.foundry.main.dataset.b2a84252-8ae1-4f7c-9948-c7e00afe36a8"),
    outcome_variable=Input(
        "ri.foundry.main.dataset.30a5ce42-d7ca-4152-8ce3-7573fc39bfe2"
    ),
    comorbidity_features=Input(
        "ri.foundry.main.dataset.7525a5d9-610c-4927-8cc4-fa12d3a17c4c"
    ),
    cancer_diagnosis=Input(
        "ri.foundry.main.dataset.57b3da18-8389-4083-bb74-499bb3208f04"
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
