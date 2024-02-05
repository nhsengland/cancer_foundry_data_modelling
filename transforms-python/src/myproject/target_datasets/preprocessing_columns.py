from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.target_datasets import utils_target
from myproject.datasets.config import list_tumour_sites


@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("ri.foundry.main.dataset.56032935-c25c-4262-8bd0-a8de9091ef85"),
    df_cancer_subset=Input("ri.foundry.main.dataset.00a2c97d-1499-4b4e-8cad-3af7cba8a599"),
    df_outcome=Input("ri.foundry.main.dataset.7c022978-3bba-4a0b-8e5a-5ac4dec34ba7")
)
def compute(df_cancer_subset, df_outcome):  
    """ Maps variables to new categories, filters on required columns and creates binary columns
    for the new categories

    Add column for train test validation split
    """
    dict_name = utils_target.rural_urban_dict
    dict_name_2 = utils_target.household_type_dict
    df_mapped_1 = utils_target.df_mapping(df_cancer_subset, "rural_urban_classification_null_removed", dict_name)
    df_mapped = utils_target.df_mapping(df_mapped_1, "acorn_household_type_null_removed", dict_name_2)
    df_subset = utils_target.filter_columns(df_mapped, utils_target.cols_to_keep)
    list_of_cols = ["rural_urban_classification_null_removed", "acorn_household_type_null_removed"]
    all_categories = []
    for col in list_of_cols:
        list_of_categories = utils_target.generate_categories(df_subset, col)
        for cat_variable in list_of_categories:
            df_categories = utils_target.generate_binary_variables(df_subset, col, cat_variable)
            all_categories = all_categories + df_categories
        patient = df_subset.select(F.col("*"), *all_categories)

    # STRATIFIED SAMPLING

    # add age bucket
    patient = patient.withColumn("age_bucket",
        F.when((F.col("age_clean") >= 40) & (F.col("age_clean") <50), "40-50")
        .when((F.col("age_clean") >= 50) & (F.col("age_clean") <60), "50-60")
        .when((F.col("age_clean") >= 60) & (F.col("age_clean") <70), "60-70")
        .when((F.col("age_clean") >= 70) & (F.col("age_clean") <80), "70-80")
        .when((F.col("age_clean") >= 80) & (F.col("age_clean") <90), "80-90")
        .when((F.col("age_clean") >= 90), "90+")
        .otherwise("other"))

    # creating cross table with all the categories for stratification
    df_age_gender_cancer = utils_target.create_stratification_crosstable(patient,
                                                                         age_column="age_bucket",
                                                                         gender_column="gender",
                                                                         target_column="cancer_diagnosis_in_next_52_weeks",
                                                                         name_stratification_column="category")

    patient = patient.join(df_age_gender_cancer, ["age_bucket", "gender", "cancer_diagnosis_in_next_52_weeks"])

    # dataset column created which has train, validation, test label for each row  
    patient = utils_target.create_train_test_validation_column(patient,
                                                               train_fraction=0.6,
                                                               test_fraction=0.2,
                                                               validation_fraction=0.2,
                                                               unique_col="patient_pseudo_id",
                                                               target_col="cancer_diagnosis_in_next_52_weeks",
                                                               seed=0)

    # adding tumour site columns to the dataset  
    df_outcome = df_outcome.select(["patient_pseudo_id"]+list_tumour_sites)
    
    patient = patient.join(df_outcome, "patient_pseudo_id", "left")

    columns_to_fill_na_with_0 = list_tumour_sites

    # fill nulls with 0. This will ignore string columns
    patient = patient.na.fill(value=0, subset=columns_to_fill_na_with_0)

    return patient