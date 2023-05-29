# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
import pyspark.sql.functions as F
from myproject.datasets import utils

@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/patient_health_record_features"),
    patient_health_record=Input("ri.foundry.main.dataset.bc82a106-d0b2-4837-9d46-1aabcb4db0d2"),
)
def compute(patient_health_record):
    """
    Add features to the patient health record table   
    """

    all_col_expressions = []

    for cat_variable in ["nhs_region", "stp_code"]:
        patient_health_record = utils.clean_column_of_strings(patient_health_record, col_name=cat_variable, append_to_col_name="_null_removed", fill_string='unknown')
        expression_cols = utils.create_expressions_categorical_column(patient_health_record, col_name=cat_variable+"_null_removed")
        all_col_expressions = all_col_expressions + expression_cols

    # Select all columns and add the expressions for the binary columns to be created  
    patient_health_record = patient_health_record.select(F.col("*"), *all_col_expressions)

    return patient_health_record
