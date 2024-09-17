from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.target_datasets import utils_target


@configure(profile=["NUM_EXECUTORS_8", "DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(stroke_target_path),
    df_stroke_subset=Input(
        patient_features_stroke_path
    ),
    stroke_post_cutoff=Input(
        stroke_outcome_variable_path
    ),
)
def compute(df_stroke_subset, stroke_post_cutoff):
    """
    Combining the outcome variable and stroke subset of patient features
    to create the stroke target dataset
    """
    dict_name = utils_target.rural_urban_dict
    dict_name_2 = utils_target.household_type_dict
    df_mapped_1 = utils_target.df_mapping(
        df_stroke_subset, "rural_urban_classification_null_removed", dict_name
    )
    df_mapped = utils_target.df_mapping(
        df_mapped_1, "acorn_household_type_null_removed", dict_name_2
    )
    # need to edit slightly due to lack of some cancer data
    columns_to_keep = utils_target.cols_to_keep.remove(
        "first_cancer_diagnosis_at_death_after_cut_off"
    )
    df_subset = utils_target.filter_columns(df_mapped, utils_target.cols_to_keep)
    list_of_cols = [
        "rural_urban_classification_null_removed",
        "acorn_household_type_null_removed",
    ]
    all_categories = []
    for col in list_of_cols:
        list_of_categories = utils_target.generate_categories(df_subset, col)
        for cat_variable in list_of_categories:
            df_categories = utils_target.generate_binary_variables(
                df_subset, col, cat_variable
            )
            all_categories = all_categories + df_categories
        patient = df_subset.select(F.col("*"), *all_categories)

    # STRATIFIED SAMPLING

    # add age bucket
    patient = patient.withColumn(
        "age_bucket",
        F.when((F.col("age_clean") >= 40) & (F.col("age_clean") < 50), "40-50")
        .when((F.col("age_clean") >= 50) & (F.col("age_clean") < 60), "50-60")
        .when((F.col("age_clean") >= 60) & (F.col("age_clean") < 70), "60-70")
        .when((F.col("age_clean") >= 70) & (F.col("age_clean") < 80), "70-80")
        .when((F.col("age_clean") >= 80) & (F.col("age_clean") < 90), "80-90")
        .when((F.col("age_clean") >= 90), "90+")
        .otherwise("other"),
    )

    stroke_in_52_weeks = stroke_post_cutoff.filter(
        stroke_post_cutoff.stroke_diagnosis_in_next_52_weeks == 1
    )
    stroke_in_52_weeks = stroke_in_52_weeks.select(
        ["patient_pseudo_id", "stroke_diagnosis_in_next_52_weeks"]
    )
    # patients = patients.select("patient_pseudo_id")
    patient = patient.join(stroke_in_52_weeks, "patient_pseudo_id", "left")
    patient = patient.fillna(0, subset=["stroke_diagnosis_in_next_52_weeks"])

    df_age_gender_cancer = utils_target.create_stratification_crosstable(
        patient,
        age_column="age_bucket",
        gender_column="gender",
        target_column="stroke_diagnosis_in_next_52_weeks",
        name_stratification_column="category",
    )

    patient = patient.join(
        df_age_gender_cancer,
        ["age_bucket", "gender", "stroke_diagnosis_in_next_52_weeks"],
    )

    patient = utils_target.create_train_test_validation_column(
        patient,
        train_fraction=0.6,
        test_fraction=0.2,
        validation_fraction=0.2,
        unique_col="patient_pseudo_id",
        target_col="stroke_diagnosis_in_next_52_weeks",
        seed=0,
    )
    return patient
