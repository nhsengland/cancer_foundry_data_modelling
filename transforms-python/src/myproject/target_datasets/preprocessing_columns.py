from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from myproject.target_datasets import utils_target


@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("ri.foundry.main.dataset.56032935-c25c-4262-8bd0-a8de9091ef85"),
    df_cancer_subset=Input("ri.foundry.main.dataset.00a2c97d-1499-4b4e-8cad-3af7cba8a599"),
)
def compute(df_cancer_subset):  
    """ Maps variables to new categories, filters on required columns and creates binary columns
    for the new categories"""
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
    return patient