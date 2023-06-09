# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
import pyspark.sql.functions as F
from myproject.datasets import utils
from myproject.datasets.config import list_categorical_variables, list_boolean_variables

@transform_df(
    Output("ri.foundry.main.dataset.36644003-34c3-43d0-bace-751b3e071ea3"),
    patient=Input("ri.foundry.main.dataset.b2a84252-8ae1-4f7c-9948-c7e00afe36a8"),
    dataset_111_calls=Input("ri.foundry.main.dataset.a3dfadbc-b022-4d88-98c2-8ee3392beed1")
)
def compute(patient, dataset_111_calls):
    """
    Add features to the patient table from the Person Ontology
    Features to be added: demographic, geographic, GP practice

    Merge features from 111 calls
    """
    # left join 111 dataset
    patient = patient.join(dataset_111_calls,
                           "patient_pseudo_id",
                           "left")

    # Age column: replace ages over 113 with None
    patient = patient.withColumn("age_clean", F.when(patient.age <= 113, patient.age)\
                                 .otherwise(F.lit(None)))

    # creating a list to store expressions for the binary columns to be created
    all_col_expressions = []

    # Converting boolean columns to binary columns
    for bool_variable in list_boolean_variables:
        patient = patient.withColumn(bool_variable + "_null_removed", F.when(F.col(bool_variable) == True, "True")
                                                       .when(F.col(bool_variable) == False, "False")
                                                       .otherwise("Unknown"))

        expression_cols = utils.create_expressions_categorical_column(patient, col_name=bool_variable+"_null_removed")
        all_col_expressions = all_col_expressions + expression_cols

    # Converting categorical columns to binary columns
    for cat_variable in list_categorical_variables:
        patient = utils.clean_column_of_strings(patient, col_name=cat_variable, append_to_col_name="_null_removed", fill_string='unknown')
        expression_cols = utils.create_expressions_categorical_column(patient, col_name=cat_variable+"_null_removed")
        all_col_expressions = all_col_expressions + expression_cols

    # Select all columns and add the expressions for the binary columns to be created  
    patient = patient.select(F.col("*"), *all_col_expressions)

    return patient
