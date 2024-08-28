from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(
        combined_comorbidities_path
    ),
    df_inpatient_comorbidities=Input(
        inpatient_comorbidites_path
    ),
    df_outpatient_comorbidities=Input(
        outpatient_comorbidities_path
    ),
    df_ae_comorbidities=Input(
        ae_comorbidities_input_path
    ),
    df_icd10_ref=Input(icd10_ref_path),
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
