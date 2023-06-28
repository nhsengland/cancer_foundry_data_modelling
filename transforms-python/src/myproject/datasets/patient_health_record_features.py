# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
import pyspark.sql.functions as F
from myproject.datasets import utils
from pyspark.sql.window import Window
from myproject.datasets.config import list_categorical_variables_from_phr

@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/patient_health_record_features"),
    patient_health_record=Input("ri.foundry.main.dataset.bc82a106-d0b2-4837-9d46-1aabcb4db0d2"),
)
def compute(patient_health_record):
    """
    Identify latest row from patient health record table
    Add features to the patient health record table
    """

    # identifying latest entry. This code should be adapted if the PHR at a different time point/snapshot is required
    w_desc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").desc())
    patient_health_record_latest = patient_health_record.withColumn("row", F.row_number().over(w_desc)).filter(F.col("row") == 1).drop("row")

    all_col_expressions = []

    for cat_variable in list_categorical_variables_from_phr:
        patient_health_record_latest = utils.clean_column_of_strings(patient_health_record_latest, col_name=cat_variable, append_to_col_name="_null_removed", fill_string='unknown')
        expression_cols = utils.create_expressions_categorical_column(patient_health_record_latest, col_name=cat_variable+"_null_removed")
        all_col_expressions = all_col_expressions + expression_cols

    # Select all columns and add the expressions for the binary columns to be created
    patient_health_record_latest = patient_health_record_latest.select(F.col("*"), *all_col_expressions)

    return patient_health_record_latest
