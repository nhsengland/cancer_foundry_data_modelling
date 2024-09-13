from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.datasets.config import date_cutoff, list_cancer_types, target_weeks
from myproject.datasets import utils

@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(outcome_variable_by_type_path),
    df_comorbidities=Input(
        combined_comorbidities_path
    ),
)
def compute(df_comorbidities):
    """
    Create patient level dataset of those who did have cancer type diagnosis/diagnoses in a
    designated time period(s) after the cut off date
    Identifies
    - all cancer diagnosis ICD10 codes post cut-off
    - the earliest and latest date of cancer diagnosis post cut-off date
    """
    # columns
    df_comorbidities = df_comorbidities.select("patient_pseudo_id", "date", "diagnosis")

    # Filter to time AFTER cut-off
    df_comorbidities = df_comorbidities.filter(F.col("date") > F.lit(date_cutoff))

    df_cancer_type_all = df_comorbidities.limit(0)
    df_cancer_type_all = df_cancer_type_all.withColumn("cancer_type", F.lit(""))

    # for each cancer type identify date of diagnosis, and create flags based on periods after the cut off date
    for cancer_type in list_cancer_types:
        df_cancer_type_only = utils.filter_to_cancer_type(df_comorbidities, cancer_type)
        df_cancer_type_only = df_cancer_type_only.withColumn("cancer_type", F.lit(cancer_type))

        df_cancer_type_all = df_cancer_type_all.unionByName(df_cancer_type_only)

    df_outcome_cancer_type = utils.create_cancer_type_features(df_cancer_type_all,
                                                               column_to_aggregate="cancer_type",
                                                               target_weeks=target_weeks)

    # Pivot to one row per patient, with columns for the cancer type categories
    df_pivot_cancer_type = (
        df_outcome_cancer_type.groupBy("patient_pseudo_id")
        .pivot("cancer_type")
        .agg(
            F.first("cancer_type_date_earliest").alias("cancer_earliest_date"),
            F.first("cancer_type_date_latest").alias("cancer_latest_date"),
            F.first("diagnosis").alias("cancer_diagnosis"),
            F.first("all_cancer_diagnoses_after_cut_off").alias("all_cancer_diagnoses_after_cut_off"),
            F.sum("cancer_type_diagnosis_in_next_12_weeks").alias("cancer_diagnosis_in_next_12_weeks"),
            F.sum("cancer_type_diagnosis_in_next_26_weeks").alias("cancer_diagnosis_in_next_26_weeks"),
            F.sum("cancer_type_diagnosis_in_next_52_weeks").alias("cancer_diagnosis_in_next_52_weeks"),
            F.sum("cancer_type_diagnosis_in_next_78_weeks").alias("cancer_diagnosis_in_next_78_weeks"),
            F.sum("cancer_type_diagnosis_in_next_104_weeks").alias("cancer_diagnosis_in_next_104_weeks"),
            F.sum("cancer_type_diagnosis_in_next_260_weeks").alias("cancer_diagnosis_in_next_260_weeks"),
        )
    )

    # create new column names so the cancer_type goes to the end of the column name
    # e.g. lung_cancer_diagnosis_in_next_260_weeks goes to cancer_diagnosis_in_next_260_weeks_lung
    old_column_names = list(df_pivot_cancer_type.columns)
    new_column_names = []
    # put cancer name at end
    for col_name in old_column_names:
        if col_name == "patient_pseudo_id":
            new_column_names.append("patient_pseudo_id")
        else:
            for cancer_type in list_cancer_types:
                if cancer_type in col_name:
                    new_column_names.append(col_name.replace(cancer_type + "_", "") + "_" + cancer_type)

    # replace the column names
    for i in range(len(old_column_names)):
        df_pivot_cancer_type = df_pivot_cancer_type.withColumnRenamed(old_column_names[i], new_column_names[i])

    # drop rows which are all empty apart from the pseudo id column
    all_columns_excluding_id = df_pivot_cancer_type.columns
    all_columns_excluding_id.remove("patient_pseudo_id")
    df_pivot_cancer_type = df_pivot_cancer_type.na.drop(
        how="all", subset=all_columns_excluding_id
    )
    return df_pivot_cancer_type