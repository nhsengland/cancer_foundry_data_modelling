from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/inpatient_comorbidities"),
    inpatient_activity=Input("ri.foundry.main.dataset.793b42d6-f24d-4cca-b500-abc319dd5162"),
)
def compute(inpatient_activity):
    """
    Create a table of diagnoses and the date from the inpatient dataset
    """

    df_comorbidity = inpatient_activity.select("patient_pseudo_id", "attendance_date", "dimention_8")

    df_comorbidity = df_comorbidity.withColumn("dimention_8_array", F.split(F.col("dimention_8"), ","))

    df_comorbidity = df_comorbidity.withColumn("diagnosis",
                                               F.explode(F.col("dimention_8_array")))

    # remove rows with empty space
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis") != '')

    # trim whitespace
    df_comorbidity = df_comorbidity.withColumn("diagnosis", F.trim(F.col("diagnosis")))

    # replace space and vertical bar (|) with empty string
    df_comorbidity = df_comorbidity.withColumn("diagnosis", F.regexp_replace("diagnosis", " |\|", ""))

    # rename attendance date to date for consistency
    df_comorbidity = df_comorbidity.withColumnRenamed("attendance_date", "date")

    # remove null diagnosis
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis").isNotNull())

    # remove empty string
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis") != "")

    return df_comorbidity.select("patient_pseudo_id", "date", "diagnosis")