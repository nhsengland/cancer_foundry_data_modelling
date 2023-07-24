from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets.config import diagnosis_columns_opa
from myproject.datasets import utils

@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/outpatient_comorbidities"),
    outpatient_activity_with_diagnosis=Input("ri.foundry.main.dataset.e3a5741d-6cb5-4b64-8d3b-0be26b1b6e62"),
)
def compute(outpatient_activity_with_diagnosis):
    """
    Create a table of outpatient diagnoses and the date from the outpatient dataset
    """

    outpatient_activity_with_diagnosis = outpatient_activity_with_diagnosis.withColumn("diagnoses",
                                                                                       F.array(diagnosis_columns_opa))

    df_comorbidity_opa = utils.create_long_list_diagnoses_from_activity(df_activity=outpatient_activity_with_diagnosis,
                                                                        array_column="diagnoses")

    df_comorbidity_opa = df_comorbidity_opa.withColumn("source", F.lit("OPA"))

    return df_comorbidity_opa.select("patient_pseudo_id", "date", "diagnosis", "source")