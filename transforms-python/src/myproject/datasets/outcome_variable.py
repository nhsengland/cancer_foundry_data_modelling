from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from myproject.datasets.config import date_cutoff, target_weeks
from myproject.datasets import utils
from pyspark.sql.window import Window


@transform_df(
    Output(
        "/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/interim-datasets/outcome_variable"
    ),
    df_comorbidities=Input(
        "ri.foundry.main.dataset.3f34bfba-9440-4b7b-9d54-59be8a52bb0e"
    ),
)
def compute(df_comorbidities):
    """
    Create patient level dataset of those who did have cancer diagnosis/diagnoses in a
    designated time period(s) after the cut off date
    Identifies
    - all cancer diagnosis ICD10 codes post cut-off
    - the earliest and latest date of cancer diagnosis post cut-off date
    """
    # columns
    df_comorbidities = df_comorbidities.select("patient_pseudo_id", "date", "diagnosis")

    # Filter to time AFTER cut-off
    df_comorbidities = df_comorbidities.filter(F.col("date") > F.lit(date_cutoff))

    # identify cancer only, exclude non-melanoma skin cancer
    df_cancer_only = df_comorbidities.filter(
        F.col("diagnosis").like("C%") & ~F.col("diagnosis").like("C44%")
    )

    # identify earliest date of cancer diagnosis after the cut-off date
    w_asc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").asc())
    specific_comorbidity_first = (
        df_cancer_only.withColumn("row", F.row_number().over(w_asc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed(
        "date", "cancer_after_cut_off_date_earliest"
    )
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed(
        "diagnosis", "diagnosis_earliest_after_cut_off_date"
    )
    specific_comorbidity_first = specific_comorbidity_first.drop("source")

    # identify latest date of cancer diagnosis after the cut-off date
    w_desc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").desc())
    specific_comorbidity_latest = (
        df_cancer_only.withColumn("row", F.row_number().over(w_desc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed(
        "date", "cancer_after_cut_off_date_latest"
    )
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed(
        "diagnosis", "diagnosis_latest_after_cut_off_date"
    )
    specific_comorbidity_latest = specific_comorbidity_latest.drop("source")

    # identify all of the unique cancer diagnoses
    df_unique_cancer_list = df_cancer_only.groupby("patient_pseudo_id").agg(
        F.collect_set("diagnosis").alias("all_cancer_diagnoses_after_cut_off")
    )

    # merge the tables

    df_cancer_after_cut_off = specific_comorbidity_first.join(
        specific_comorbidity_latest, "patient_pseudo_id", "left"
    )

    df_cancer_after_cut_off = df_cancer_after_cut_off.join(
        df_unique_cancer_list, "patient_pseudo_id", "left"
    )

    # for each time period (e.g. 12 weeks after cut off point), identify if cancer diagnosis occurred in that time
    for weeks in target_weeks:
        later_date = utils.add_n_weeks_from_date(date_cutoff, weeks)

        df_cancer_after_cut_off = df_cancer_after_cut_off.withColumn(
            "cancer_diagnosis_in_next_" + str(weeks) + "_weeks",
            F.when(
                (F.col("cancer_after_cut_off_date_earliest") <= later_date), 1
            ).otherwise(0),
        )

    return df_cancer_after_cut_off
