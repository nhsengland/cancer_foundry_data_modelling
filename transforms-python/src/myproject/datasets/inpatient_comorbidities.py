from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets import utils

@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/inpatient_comorbidities"),
    inpatient_activity=Input("ri.foundry.main.dataset.793b42d6-f24d-4cca-b500-abc319dd5162"),
)
def compute(inpatient_activity):
    """
    Create a table of inpatient diagnoses and the date from the inpatient dataset
    """
    inpatient_activity = inpatient_activity.select("patient_pseudo_id", "attendance_date", "dimention_8")

    inpatient_activity = inpatient_activity.withColumn("dimention_8_array", F.split(F.col("dimention_8"), ","))

    df_comorbidity_inpatient = utils.create_long_list_diagnoses_from_activity(df_activity=inpatient_activity,
                                                                              array_column="dimention_8_array")

    df_comorbidity_inpatient = df_comorbidity_inpatient.withColumn("source", F.lit("Inpatient"))

    return df_comorbidity_inpatient.select("patient_pseudo_id", "date", "diagnosis", "source")
