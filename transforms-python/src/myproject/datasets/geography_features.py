from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets import utils
from myproject.datasets.config import geographical_variables


@transform_df(
    Output(geographical_variables_path),
    df_icb=Input(icb_path),
    df_cancer_alliance=Input(
        cancer_alliance_path ),
    patient=Input(patient_record_path),
)
def compute(df_icb, df_cancer_alliance, patient):
    """
    Generate features containing the geogrpahic variables (cancer alliance, icb and LSOA)
    Using mapping tables and the LSOA code for each patient from the patient table
    The patient is mapped to the cancer alliance and icb which operates in that region
    """

    # Joining the cancer alliance and icb datasets
    df_mapping = df_icb.join(
        df_cancer_alliance, df_cancer_alliance.LSOA11CD == df_icb.LSOA11CD, how="left"
    ).select(
        df_icb.LSOA11CD,
        df_icb.ICB22NM.alias("integrated_care_board"),
        df_cancer_alliance.CAL19NM.alias("cancer_alliance"),
    )

    # Removing non-alphanumeric characters
    for col in geographical_variables:
        df_mapping = df_mapping.withColumn(
            col, F.regexp_replace(col, "[^a-zA-Z0-9 ]", "")
        )

    # Joining the mapping datset to the patient dataset
    condition = [patient.lsoa_2011_code == df_mapping.LSOA11CD]

    patient = patient.join(df_mapping, condition, how="left").select(
        patient.patient_pseudo_id,
        df_mapping.LSOA11CD,
        df_mapping.cancer_alliance,
        df_mapping.integrated_care_board,
    )

    # creating a list to store expressions for the binary columns to be created
    all_col_expressions = []

    # Converting categorical columns to binary columns
    for cat_variable in geographical_variables:
        patient = utils.clean_column_of_strings(
            patient,
            col_name=cat_variable,
            append_to_col_name="_null_removed",
            fill_string="unknown",
        )
        expression_cols = utils.create_expressions_categorical_column(
            patient, col_name=cat_variable + "_null_removed"
        )
        all_col_expressions = all_col_expressions + expression_cols

    patient_geog = patient.select(F.col("*"), *all_col_expressions)

    patient_geog = patient_geog.drop(
        "cancer_alliance_null_removed", "integrated_care_board_null_removed"
    )

    return patient_geog
