from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(
        "/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/interim-datasets/combined_comorbidities"
    ),
    df_inpatient_comorbidities=Input(
        "ri.foundry.main.dataset.fe042d45-d081-4221-ae22-b930587fb500"
    ),
    df_outpatient_comorbidities=Input(
        "ri.foundry.main.dataset.fa19d9b5-066a-4d57-aef4-59d2cda4a8ac"
    ),
    df_ae_comorbidities=Input(
        "ri.foundry.main.dataset.cf9cd953-d9c4-486f-9130-607c252ed8e1"
    ),
    df_icd10_ref=Input("ri.foundry.main.dataset.3a0dfe4b-67de-4001-87eb-e43405df76d3"),
)
def compute(
    df_inpatient_comorbidities,
    df_outpatient_comorbidities,
    df_ae_comorbidities,
    df_icd10_ref,
):
    """
    Generate table of combined comorbidities from each of the source datasets
    Merge this with the ICD10 reference table
    This combined comorbidities is used in creating comorbidity features, and outcome variable
    """
    # ensuring all comorbidities datasets have the same columns
    df_ae_comorbidities = df_ae_comorbidities.select(
        "patient_pseudo_id", "date", "diagnosis", "source"
    )
    # removing the diagnoses which are coded as #NIS -> this refers to referral to service
    df_ae_comorbidities = df_ae_comorbidities.filter(F.col("diagnosis") != "#NIS")

    df_inpatient_comorbidities = df_inpatient_comorbidities.select(
        "patient_pseudo_id", "date", "diagnosis", "source"
    )
    df_outpatient_comorbidities = df_outpatient_comorbidities.select(
        "patient_pseudo_id", "date", "diagnosis", "source"
    )

    # Keep latest only for ICD10 reference file
    df_icd10_ref = df_icd10_ref.filter(F.col("effective_to").isNull())
    df_icd10_ref = df_icd10_ref.filter(F.col("category_3_code").isNotNull())

    # Union inpatient and outpatient
    df_comorbidities = df_inpatient_comorbidities.unionByName(
        df_outpatient_comorbidities
    )

    # Union with A&E
    df_comorbidities = df_comorbidities.unionByName(df_ae_comorbidities)

    # remove null IDs
    df_comorbidities = df_comorbidities.na.drop(subset=["patient_pseudo_id"])

    # merge diagnoses with reference
    df_comorbidities = df_comorbidities.join(
        df_icd10_ref, df_comorbidities.diagnosis == df_icd10_ref.alt_code, "left"
    )

    return df_comorbidities
