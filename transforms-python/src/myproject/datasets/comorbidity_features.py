from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets.config import date_cutoff, dict_icd10_groups
from myproject.datasets import utils

@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/comorbidity_features"),
    inpatient_comorbidities=Input("ri.foundry.main.dataset.fe042d45-d081-4221-ae22-b930587fb500"),
)
def compute(inpatient_comorbidities):
    """
    Generate features from the comorbidities
    For a given set of diagnoses, identify the first and last date when the diagnosis was made
    Create flags if the diagnosis was made in the last n weeks
    """
    # remove null IDs
    inpatient_comorbidities = inpatient_comorbidities.na.drop(subset=["patient_pseudo_id"])

    # Filter to time before cut-off
    # inpatient_comorbidities = inpatient_comorbidities.filter(F.col("date") < F.lit(date_cutoff))

    # unique IDs
    specific_comorbidity_features = inpatient_comorbidities.select("patient_pseudo_id").distinct()

    # loop through dictionary of comorbidity grouping
    for key, value in dict_icd10_groups.items():
        # making a copy of the inpatient table
        filtered_disease = inpatient_comorbidities.alias(key)

        # filtering the table to the ICD10 codes specified as a list
        for comorbidity in value:
            filtered_disease = filtered_disease.filter(F.col("diagnosis").like(comorbidity))

        filtered_disease_features = utils.create_comorbidity_features(df_specific_comorbidity=filtered_disease,
                                                                      name_of_group=key)

        specific_comorbidity_features = specific_comorbidity_features.join(filtered_disease_features,
                                                                           "patient_pseudo_id",
                                                                           "left")

    # drop rows which are all empty apart from the pseudo id column
    all_columns_excluding_id = specific_comorbidity_features.columns
    all_columns_excluding_id.remove("patient_pseudo_id")

    specific_comorbidity_features = specific_comorbidity_features.na.drop(how='all', subset=all_columns_excluding_id)

    return specific_comorbidity_features
