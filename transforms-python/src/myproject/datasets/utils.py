import pyspark.sql.functions as F
from pyspark.sql.window import Window
import datetime
from myproject.datasets.config import date_cutoff, weeks_to_subtract, target_weeks
from typing import List


def clean_column_of_strings(
    df,
    col_name: str,
    append_to_col_name: str = "_null_replaced",
    fill_string: str = "unknown",
):
    """
    Create a new column which duplicates a column of strings
    Replaces null and empty strings with a fill_string (e.g. unknown)
    """

    df = df.withColumn(
        col_name + append_to_col_name,
        F.when(
            F.col(col_name).isNull() | (F.col(col_name) == ""), fill_string
        ).otherwise(F.col(col_name)),
    )

    return df


def create_expressions_categorical_column(df, col_name: str):
    """
    Create dummy variables for a categorical column (col)
    Convert null and empty string values to string 'unknown'
    """
    # replace blank space in category with underscore
    categories = df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
    categories_exprs = [
        F.when(F.col(col_name) == category, 1)
        .otherwise(0)
        .alias(col_name + "_" + category.replace(" ", "_"))
        for category in categories
    ]

    return categories_exprs


def clean_categorical_column_return_expressions(
    df,
    col_name: str,
    fill_string: str = "unknown",
    append_to_col_name: str = "_null_replaced",
):
    """
    First clean the categorical column (replacing null and empty strings)
    Returns expression for the dummy variables
    """
    df = clean_column_of_strings(df, col_name, fill_string)
    categories_exprs = create_expressions_categorical_column(
        df, col_name + append_to_col_name
    )

    return categories_exprs


def create_comorbidity_features(df_specific_comorbidity, name_of_group):
    """
    Input: dataframe which is already filtered to one diagnosis or set of diagnoses
    Identify the first and last date when the diagnosis was recorded
    Create output table, one row per ID
    Create binary flag to indicate presence of the group of diagnoses
    Identify if the first diagnosis was made in the last n weeks
    """
    df_specific_comorbidity = df_specific_comorbidity.withColumnRenamed(
        "diagnosis", name_of_group
    )

    # identify earliest date
    w_asc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").asc())
    specific_comorbidity_first = (
        df_specific_comorbidity.withColumn("row", F.row_number().over(w_asc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed(
        "date", name_of_group + "_date_earliest"
    )

    # identify latest date
    w_desc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").desc())
    specific_comorbidity_latest = (
        df_specific_comorbidity.withColumn("row", F.row_number().over(w_desc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed(
        "date", name_of_group + "_date_latest"
    )

    # merge
    specific_comorbidity_feature = specific_comorbidity_first.join(
        specific_comorbidity_latest, ["patient_pseudo_id", name_of_group], "left"
    )

    # make binary column
    specific_comorbidity_feature = specific_comorbidity_feature.withColumn(
        name_of_group + "_binary", F.lit(1)
    )

    # if earliest was in last 1 month, 6 months, 12 months or more

    for weeks in weeks_to_subtract:
        subtracted_date = subtract_n_weeks_from_date(date_cutoff, weeks)

        specific_comorbidity_feature = specific_comorbidity_feature.withColumn(
            name_of_group + "_in_last_" + str(weeks) + "_weeks",
            F.when(
                (F.col(name_of_group + "_date_earliest") > subtracted_date)
                & (F.col(name_of_group + "_date_earliest") < date_cutoff),
                1,
            ).otherwise(0),
        )

    return specific_comorbidity_feature


def create_comorbidity_category_features(df_comorbidities, column_to_aggregate: str):
    """
    Input dataframe should be at granularity of patient-diagnosis
    The table is grouped by a column_to_aggregate (e.g. the category of the diagnosis)
    The earliest and latest date of the diagnosis category is identified
    A binary flag is created for the diagnosis category
    Identify if the first diagnosis of that category was made in the last n weeks
    """
    # identify earliest date of diagnosis from the group
    specific_comorbidity_first = identify_first_diagnosis_of_a_group(df_comorbidities, column_to_aggregate)

    # identify latest date of diagnosis from the group
    specific_comorbidity_latest = identify_latest_diagnosis_of_a_group(df_comorbidities, column_to_aggregate)

    # merge
    specific_comorbidity_feature = specific_comorbidity_first.join(
        specific_comorbidity_latest, ["patient_pseudo_id", column_to_aggregate], "left"
    )
    # make binary column
    specific_comorbidity_feature = specific_comorbidity_feature.withColumn(
        column_to_aggregate + "_binary", F.lit(1)
    )

    # if earliest was in last 1 month, 6 months, 12 months or more

    for weeks in weeks_to_subtract:
        subtracted_date = subtract_n_weeks_from_date(date_cutoff, weeks)

        specific_comorbidity_feature = specific_comorbidity_feature.withColumn(
            column_to_aggregate + "_in_last_" + str(weeks) + "_weeks",
            F.when(
                (F.col(column_to_aggregate + "_date_earliest") > subtracted_date)
                & (F.col(column_to_aggregate + "_date_earliest") < date_cutoff),
                1,
            ).otherwise(0),
        )

    return specific_comorbidity_feature


def cancer_type_diagnosis(df_cancer_type_only, cancer_type: str, target_weeks: List):
    """
    For each cancer type, find the earliest and latest date of the diagnosis
    dataset (df_cancer_type_only) should already be filtered to include 
    diagnoses of the particular cancer type after the cut-off date

    """
    # identify earliest date of cancer diagnosis after the cut-off date
    w_asc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").asc())
    specific_comorbidity_first = (
        df_cancer_type_only.withColumn("row", F.row_number().over(w_asc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed(
        "date", cancer_type+"_cancer_after_cut_off_date_earliest"
    )
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed(
        "diagnosis", cancer_type+"_diagnosis_earliest_after_cut_off_date"
    )
    specific_comorbidity_first = specific_comorbidity_first.drop("source")

    # identify latest date of cancer diagnosis after the cut-off date
    w_desc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").desc())
    specific_comorbidity_latest = (
        df_cancer_type_only.withColumn("row", F.row_number().over(w_desc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed(
        "date", cancer_type + "_cancer_after_cut_off_date_latest"
    )
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed(
        "diagnosis", cancer_type+"_diagnosis_latest_after_cut_off_date"
    )
    specific_comorbidity_latest = specific_comorbidity_latest.drop("source")

    # identify all of the unique cancer diagnoses
    df_unique_cancer_list = df_cancer_type_only.groupby("patient_pseudo_id").agg(
        F.collect_set("diagnosis").alias(cancer_type+"_all_cancer_diagnoses_after_cut_off")
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
        later_date = add_n_weeks_from_date(date_cutoff, weeks)

        df_cancer_after_cut_off = df_cancer_after_cut_off.withColumn(
            cancer_type+"_cancer_diagnosis_in_next_" + str(weeks) + "_weeks",
            F.when(
                (F.col(cancer_type+"_cancer_after_cut_off_date_latest") <= later_date), 1
            ).otherwise(0),
        )

    return df_cancer_after_cut_off


def number_calls_during_time_period(
    df,
    date: str,
    num_weeks_to_subtract: int,
    variable_name_prefix: str = "111_calls_last_",
    groupbycols=["patient_pseudo_id"],
):
    """
    Returns the number of 111 calls in the period between
    the given date - number_weeks_to_subtract until
    the latest data present in the dataset
    """
    date_limit = subtract_n_weeks_from_date(date, num_weeks_to_subtract)
    df_limited = df.filter(F.col("call_date") >= F.lit(date_limit))

    col_alias = variable_name_prefix + str(num_weeks_to_subtract) + "_weeks"
    total_number_calls_in_period = (
        df_limited.groupBy(groupbycols).count().withColumnRenamed("count", col_alias)
    )

    return total_number_calls_in_period


def number_symptom_calls_during_time_period(
    df,
    date: str,
    num_weeks_to_subtract: int,
    variable_name: str = "111_calls_last_",
    groupbycols: str = ["patient_pseudo_id"],
    pivotcol: str = "symptom_group_no_spaces",
):
    """
    Returns the number of 111 calls for each value in the column pivotcol
    in the period between the given date - number_weeks_to_subtract until
    the latest data present in the dataset
    """
    date_limit = subtract_n_weeks_from_date(date, num_weeks_to_subtract)
    df_limited = df.filter(F.col("call_date") >= F.lit(date_limit))
    symptom_count = df_limited.groupBy(groupbycols).pivot(pivotcol).count()

    # string to add to the end of the pivoted variable name
    col_name_addition = "_" + variable_name + str(num_weeks_to_subtract) + "_weeks"

    new_col_names = []
    for col in symptom_count.columns:
        if col == "patient_pseudo_id":
            new_col_names.append(col)
        else:
            new_col_names.append(col + col_name_addition)

    # rename the created columns
    symptom_count = symptom_count.toDF(*new_col_names)

    return symptom_count


def number_attendances_during_time_period(
    df,
    date: str,
    num_weeks_to_subtract: int,
    variable_name_prefix: str = "_attendances_last_",
    groupbycols: str = ["patient_pseudo_id"],
):
    """
    Returns the number of attendances in the period between
    the given date - number_weeks_to_subtract until
    the latest data present in the dataset
    """
    date_limit = subtract_n_weeks_from_date(date, num_weeks_to_subtract)
    df_limited = df.filter(F.col("attendance_date") >= F.lit(date_limit))

    col_alias = variable_name_prefix + str(num_weeks_to_subtract) + "_weeks"
    total_number_attendances_in_period = (
        df_limited.groupBy(groupbycols).count().withColumnRenamed("count", col_alias)
    )

    return total_number_attendances_in_period


def add_n_weeks_from_date(original_date: str, num_weeks_to_subtract: int):
    """
    Given a date string in the format YYYY-mm-dd
    subtract num_weeks_to_subtract weeks and
    return the date in the same format
    """
    original_date_dt = datetime.datetime.strptime(original_date, "%Y-%m-%d")
    later_date = original_date_dt + datetime.timedelta(weeks=num_weeks_to_subtract)
    later_date = later_date.strftime("%Y-%m-%d")

    return later_date


def subtract_n_weeks_from_date(original_date: str, num_weeks_to_subtract: int):
    """
    Given a date string in the format YYYY-mm-dd
    subtract num_weeks_to_subtract weeks and
    return the date in the same format
    """
    original_date_dt = datetime.datetime.strptime(original_date, "%Y-%m-%d")
    subtracted_date = original_date_dt - datetime.timedelta(weeks=num_weeks_to_subtract)
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")

    return subtracted_date


def create_long_list_diagnoses_from_activity(df_activity, array_column: str):
    """
    Input df: activity table (e.g. inpatient or outpatient) with each row an episode,
    and a column which has the array of all the diagnoses (array_column)

    Returns df_comorbidity which has every diagnosis and the appointment date
    for each ID on a separate row

    Function to be used for inpatient and outpatient activity tables
    """

    df_comorbidity = df_activity.withColumn("diagnosis", F.explode(F.col(array_column)))

    # remove rows with empty space
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis") != "")

    # trim whitespace
    df_comorbidity = df_comorbidity.withColumn("diagnosis", F.trim(F.col("diagnosis")))

    # replace space and vertical bar (|) with empty string
    df_comorbidity = df_comorbidity.withColumn(
        "diagnosis", F.regexp_replace("diagnosis", " |\|", "")
    )

    # replace comma and dash with empty string
    df_comorbidity = df_comorbidity.withColumn(
        "diagnosis", F.regexp_replace("diagnosis", ",|-", "")
    )

    # rename attendance date to date for consistency
    df_comorbidity = df_comorbidity.withColumnRenamed("attendance_date", "date")

    # remove null diagnosis
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis").isNotNull())

    # remove empty string
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis") != "")

    return df_comorbidity


def flag_if_patient_passed_away_with_cancer(df):
    """
    creates a binary column passed_away_with_cancer_cod which is 1 if any of the
    causes of death contain a cancer ICD10 code, and 0 if not
    """

    return df.withColumn(
        "passed_away_with_cancer_cod",
        F.when(
            (
                F.col("s_underlying_cod_icd10").like("C%")
                & ~F.col("s_underlying_cod_icd10").like("C44%")
            )
            | (F.col("s_cod_code_1").like("C%") & ~F.col("s_cod_code_1").like("C44%"))
            | (F.col("s_cod_code_2").like("C%") & ~F.col("s_cod_code_2").like("C44%"))
            | (F.col("s_cod_code_3").like("C%") & ~F.col("s_cod_code_3").like("C44%"))
            | (F.col("s_cod_code_4").like("C%") & ~F.col("s_cod_code_4").like("C44%"))
            | (F.col("s_cod_code_5").like("C%") & ~F.col("s_cod_code_5").like("C44%"))
            | (F.col("s_cod_code_6").like("C%") & ~F.col("s_cod_code_6").like("C44%"))
            | (F.col("s_cod_code_7").like("C%") & ~F.col("s_cod_code_7").like("C44%"))
            | (F.col("s_cod_code_8").like("C%") & ~F.col("s_cod_code_8").like("C44%"))
            | (F.col("s_cod_code_9").like("C%") & ~F.col("s_cod_code_9").like("C44%"))
            | (F.col("s_cod_code_10").like("C%") & ~F.col("s_cod_code_10").like("C44%"))
            | (F.col("s_cod_code_11").like("C%") & ~F.col("s_cod_code_11").like("C44%"))
            | (F.col("s_cod_code_12").like("C%") & ~F.col("s_cod_code_12").like("C44%"))
            | (F.col("s_cod_code_13").like("C%") & ~F.col("s_cod_code_13").like("C44%"))
            | (F.col("s_cod_code_14").like("C%") & ~F.col("s_cod_code_14").like("C44%"))
            | (
                F.col("s_cod_code_15").like("C%") & ~F.col("s_cod_code_15").like("C44%")
            ),
            1,
        ).otherwise(0),
    )

# The filter_to_..._cancer functions below filter a dataframe of diagnoses to those for that particular cancer type 

def filter_to_head_and_neck_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C00")
        | F.col("diagnosis").contains("C01")
        | F.col("diagnosis").contains("C02")
        | F.col("diagnosis").contains("C03")
        | F.col("diagnosis").contains("C04")
        | F.col("diagnosis").contains("C05")
        | F.col("diagnosis").contains("C06")
        | F.col("diagnosis").contains("C07")
        | F.col("diagnosis").contains("C08")
        | F.col("diagnosis").contains("C09")
        | F.col("diagnosis").contains("C10")
        | F.col("diagnosis").contains("C11")
        | F.col("diagnosis").contains("C12")
        | F.col("diagnosis").contains("C13")
        | F.col("diagnosis").contains("C14")
        | F.col("diagnosis").contains("C30")
        | F.col("diagnosis").contains("C31")
        | F.col("diagnosis").contains("C32"),   
    )


def filter_to_gi_upper_and_lower_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C15")
        | F.col("diagnosis").contains("C16")
        | F.col("diagnosis").contains("C17")
        | F.col("diagnosis").contains("C18")
        | F.col("diagnosis").contains("C19")
        | F.col("diagnosis").contains("C20")
        | F.col("diagnosis").contains("C21")
        | F.col("diagnosis").contains("C22")
        | F.col("diagnosis").contains("C23")
        | F.col("diagnosis").contains("C24")
        | F.col("diagnosis").contains("C25")
        | F.col("diagnosis").contains("C26")
    )

def filter_to_lung_cancer(df):
    return  df.filter(F.col("diagnosis").contains("C33")
            | F.col("diagnosis").contains("C34")
            | F.col("diagnosis").contains("C37")
            | F.col("diagnosis").contains("C38")
            | F.col("diagnosis").contains("C39")
            | F.col("diagnosis").contains("C45"))


def filter_to_sarcoma_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C40")
        | F.col("diagnosis").contains("C41")
        | F.col("diagnosis").contains("C46")
        | F.col("diagnosis").contains("C48")
        | F.col("diagnosis").contains("C49"),
    )

def filter_to_melanoma_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C43")
    )

def filter_to_brain_and_cns_cancer(df):
    return df.filter(
            F.col("diagnosis").contains("C47")
            | F.col("diagnosis").contains("C69")
            | F.col("diagnosis").contains("C70")
            | F.col("diagnosis").contains("C71")
            | F.col("diagnosis").contains("C72")
            )

def filter_to_breast_cancer(df):
    return df.filter(
            F.col("diagnosis").contains("C50")
            )

def filter_to_gynae_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C51")
        | F.col("diagnosis").contains("C52")
        | F.col("diagnosis").contains("C53")
        | F.col("diagnosis").contains("C54")
        | F.col("diagnosis").contains("C55")
        | F.col("diagnosis").contains("C56")
        | F.col("diagnosis").contains("C57")
        | F.col("diagnosis").contains("C58"),
            )

def filter_to_urology_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C60")
        | F.col("diagnosis").contains("C61")
        | F.col("diagnosis").contains("C62")
        | F.col("diagnosis").contains("C63")
        | F.col("diagnosis").contains("C64")
        | F.col("diagnosis").contains("C65")
        | F.col("diagnosis").contains("C66")
        | F.col("diagnosis").contains("C67")
        | F.col("diagnosis").contains("C68"),
            )

def filter_to_endocrine_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C73")
        | F.col("diagnosis").contains("C74")
        | F.col("diagnosis").contains("C75"),
            )

def filter_to_haematological_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C81")
        | F.col("diagnosis").contains("C82")
        | F.col("diagnosis").contains("C83")
        | F.col("diagnosis").contains("C84")
        | F.col("diagnosis").contains("C85")
        | F.col("diagnosis").contains("C86")
        | F.col("diagnosis").contains("C87")
        | F.col("diagnosis").contains("C88")
        | F.col("diagnosis").contains("C89")
        | F.col("diagnosis").contains("C90")
        | F.col("diagnosis").contains("C91")
        | F.col("diagnosis").contains("C92")
        | F.col("diagnosis").contains("C93")
        | F.col("diagnosis").contains("C94")
        | F.col("diagnosis").contains("C95")
        | F.col("diagnosis").contains("C96"),
            )

def filter_to_unknown_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C76")
        | F.col("diagnosis").contains("C77")
        | F.col("diagnosis").contains("C78")
        | F.col("diagnosis").contains("C79")
        | F.col("diagnosis").contains("C80")
        | F.col("diagnosis").contains("C97"),
            )

def filter_to_ovarian_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C56")
            )

def filter_to_pancreatic_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C25")
            )

def filter_to_bladder_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C67")
            )

def filter_to_myeloma_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C90")
            )

def filter_to_stomach_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C16")
            )

def filter_to_oesophageal_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C15")
            )

def filter_to_kidney_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C64")
        | F.col("diagnosis").contains("C65"),
        )

def filter_to_lymphoma_cancer(df):
    return df.filter(
        F.col("diagnosis").contains("C81")
        | F.col("diagnosis").contains("C82")
        | F.col("diagnosis").contains("C83")
        | F.col("diagnosis").contains("C84")
        | F.col("diagnosis").contains("C85")
        | F.col("diagnosis").contains("C86"),
        )


def filter_to_cancer_type(df, cancer_type):
    if cancer_type == "head_and_neck":
        return filter_to_head_and_neck_cancer(df)
    
    elif cancer_type == "gi_upper_and_lower":
        return filter_to_gi_upper_and_lower_cancer(df)

    elif cancer_type == "lung":
        return filter_to_lung_cancer(df)

    elif cancer_type == "sarcoma":
        return filter_to_sarcoma_cancer(df)

    elif cancer_type == "melanoma":
        return filter_to_melanoma_cancer(df)

    elif cancer_type == "brain_and_cns":
        return filter_to_brain_and_cns_cancer(df)

    elif cancer_type == "breast":
        return filter_to_breast_cancer(df)

    elif cancer_type == "gynae":
        return filter_to_gynae_cancer(df)

    elif cancer_type == "urology":
        return filter_to_urology_cancer(df)

    elif cancer_type == "endocrine":
        return filter_to_endocrine_cancer(df)

    elif cancer_type == "haematological":
        return filter_to_haematological_cancer(df)

    elif cancer_type == "unknown":
        return filter_to_unknown_cancer(df)

    elif cancer_type == "ovarian":
        return filter_to_ovarian_cancer(df)

    elif cancer_type == "pancreatic":
        return filter_to_pancreatic_cancer(df)

    elif cancer_type == "bladder":
        return filter_to_bladder_cancer(df)

    elif cancer_type == "myeloma":
        return filter_to_myeloma_cancer(df)

    elif cancer_type == "stomach":
        return filter_to_stomach_cancer(df)

    elif cancer_type == "oesophageal":
        return filter_to_oesophageal_cancer(df)

    elif cancer_type == "kidney":
        return filter_to_kidney_cancer(df)

    elif cancer_type == "lymphoma":
        return filter_to_lymphoma_cancer(df)

    else:
        return None


def identify_first_diagnosis_of_a_group(df_comorbidities, column_to_aggregate: str):
    """
    Input dataframe should be at granularity of patient-diagnosis
    The table is grouped by a column_to_aggregate (e.g. the category of the diagnosis, or the cancer type)
    The earliest incidence of this category per patient is identified
    """
    # identify earliest date
    w_asc = Window.partitionBy(
        [F.col("patient_pseudo_id"), F.col(column_to_aggregate)]
    ).orderBy(F.col("date").asc())
    specific_comorbidity_first = (
        df_comorbidities.select(
            "patient_pseudo_id", column_to_aggregate, "date", "diagnosis"
        )
        .withColumn("row", F.row_number().over(w_asc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed(
        "date", column_to_aggregate + "_date_earliest"
    )

    return specific_comorbidity_first


def identify_latest_diagnosis_of_a_group(df_comorbidities, column_to_aggregate: str):
    """
    Input dataframe should be at granularity of patient-diagnosis
    The table is grouped by a column_to_aggregate (e.g. the category of the diagnosis, or the cancer type)
    The latest incidence of this category per patient is identified
    """
    # identify latest date
    w_desc = Window.partitionBy(
        [F.col("patient_pseudo_id"), F.col(column_to_aggregate)]
    ).orderBy(F.col("date").desc())
    specific_comorbidity_latest = (
        df_comorbidities.select("patient_pseudo_id", column_to_aggregate, "date")
        .withColumn("row", F.row_number().over(w_desc))
        .filter(F.col("row") == 1)
        .drop("row")
    )
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed(
        "date", column_to_aggregate + "_date_latest"
    )

    return specific_comorbidity_latest


def create_cancer_type_features(df_comorbidities, column_to_aggregate: str, target_weeks=target_weeks):
    """
    Input dataframe should be at granularity of patient-diagnosis
    The table is grouped by a column_to_aggregate (e.g. the category of the diagnosis)
    The earliest and latest date of the diagnosis category is identified
    A binary flag is created for the diagnosis category
    Identify if the first diagnosis of that category was made in the last n weeks
    """

    # identify earliest cancer diagnosis from the group (e.g. cancer_type)
    specific_comorbidity_first = identify_first_diagnosis_of_a_group(df_comorbidities, column_to_aggregate)

    # identify latest cancer diagnosis from the group (e.g. cancer_type)
    specific_comorbidity_latest = identify_latest_diagnosis_of_a_group(df_comorbidities, column_to_aggregate)

    # merge
    specific_comorbidity_feature = specific_comorbidity_first.join(
        specific_comorbidity_latest, ["patient_pseudo_id", column_to_aggregate], "left"
    )

    # identify all of the unique cancer diagnoses
    df_unique_cancer_list = df_comorbidities.groupby("patient_pseudo_id").agg(
        F.collect_set("diagnosis").alias("all_cancer_diagnoses_after_cut_off")
    )

    specific_comorbidity_feature = specific_comorbidity_feature.join(
        df_unique_cancer_list, "patient_pseudo_id", "left"
    )

    for weeks in target_weeks:
        later_date = add_n_weeks_from_date(date_cutoff, weeks)

        specific_comorbidity_feature = specific_comorbidity_feature.withColumn(
            column_to_aggregate + "_diagnosis_in_next_" + str(weeks) + "_weeks",
            F.when(
                (F.col(column_to_aggregate + "_date_earliest") <= later_date), 1
            ).otherwise(0),
        )

    return specific_comorbidity_feature
