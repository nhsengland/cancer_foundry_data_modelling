import pyspark.sql.functions as F
from pyspark.sql.window import Window
import datetime
from myproject.datasets.config import date_cutoff, weeks_to_subtract
import datetime


def clean_column_of_strings(df, col_name: str, append_to_col_name: str = '_null_removed', fill_string: str = 'unknown'):
    """
    Create a new column which duplicates a column of strings
    Replaces null and empty strings with a fill_string (e.g. unknown)
    """

    df = df.withColumn(col_name + append_to_col_name, F.when(F.col(col_name).isNull() | (F.col(col_name) == ''), fill_string).otherwise(F.col(col_name)))

    return df


def create_expressions_categorical_column(df, col_name: str):
    """
    Create dummy variables for a categorical column (col)
    Convert null and empty string values to string 'unknown'
    """
    # replace blank space in category with underscore
    categories = df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
    categories_exprs = [F.when(F.col(col_name) == category, 1).otherwise(0).alias(col_name+"_"+category.replace(" ", "_")) for category in categories]

    return categories_exprs


def clean_categorical_column_return_expressions(df, col_name: str,fill_string: str = 'unknown',append_to_col_name: str = '_null_removed'):
    """
    First clean the categorical column (replacing null and emppty strings)
    Returns expression for the dummy variables
    """
    df = clean_column_of_strings(df, col_name, fill_string)
    categories_exprs = create_expressions_categorical_column(df, col_name + append_to_col_name)

    return categories_exprs


def create_comorbidity_features(df_specific_comorbidity, name_of_group):
    """
    Input: dataframe which is already filtered to one diagnosis or set of diagnoses
    Identify the first and last date when the diagnosis was recorded
    Create output table, one row per ID
    Create binary flag to indicate presence of the group of diagnoses
    Identify if the first diagnosis was made in the last n weeks
    """
    df_specific_comorbidity = df_specific_comorbidity.withColumnRenamed("diagnosis", name_of_group)

    # identify earliest date
    w_asc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").asc())
    specific_comorbidity_first = df_specific_comorbidity.withColumn("row", F.row_number().over(w_asc)).filter(F.col("row") == 1).drop("row")
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed("date", name_of_group+"_date_earliest")

    # identify latest date
    w_desc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("date").desc())
    specific_comorbidity_latest = df_specific_comorbidity.withColumn("row", F.row_number().over(w_desc)).filter(F.col("row") == 1).drop("row")
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed("date", name_of_group+"_date_latest")

    # merge
    specific_comorbidity_feature = specific_comorbidity_first.join(specific_comorbidity_latest,
                                                                   ["patient_pseudo_id", name_of_group],
                                                                   "left")


    # make binary column
    specific_comorbidity_feature = specific_comorbidity_feature.withColumn(name_of_group+"_binary", F.lit(1))

    # if earliest was in last 1 month, 6 months, 12 months or more

    for weeks in weeks_to_subtract:
        subtracted_date = subtract_n_weeks_from_date(date_cutoff, weeks)

        specific_comorbidity_feature = specific_comorbidity_feature.withColumn(name_of_group+"_in_last_"+str(weeks)+"_weeks",
                                                                               F.when( (F.col(name_of_group+"_date_earliest")>subtracted_date) &
                                                                               (F.col(name_of_group+"_date_earliest")<date_cutoff), 1)
                                                                               .otherwise(0))

    return specific_comorbidity_feature


def create_comorbidity_category_features(df_comorbidities, column_to_aggregate: str):
    """
    Input dataframe should be at granularity of patient-diagnosis
    The table is grouped by a column_to_aggregate (e.g. the category of the diagnosis)
    The earliest and latest date of the diagnosis category is identified
    A binary flag is created for the diagnosis category
    Identify if the first diagnosis of that category was made in the last n weeks
    """
    # identify earliest date
    w_asc = Window.partitionBy([F.col("patient_pseudo_id"), F.col(column_to_aggregate)]).orderBy(F.col("date").asc())
    specific_comorbidity_first = df_comorbidities.select("patient_pseudo_id", column_to_aggregate, "date", "diagnosis").withColumn("row", F.row_number().over(w_asc)).filter(F.col("row") == 1).drop("row")
    specific_comorbidity_first = specific_comorbidity_first.withColumnRenamed("date", column_to_aggregate+"_date_earliest")

    # identify latest date
    w_desc = Window.partitionBy([F.col("patient_pseudo_id"), F.col(column_to_aggregate)]).orderBy(F.col("date").desc())
    specific_comorbidity_latest = df_comorbidities.select("patient_pseudo_id", column_to_aggregate, "date").withColumn("row", F.row_number().over(w_desc)).filter(F.col("row") == 1).drop("row")
    specific_comorbidity_latest = specific_comorbidity_latest.withColumnRenamed("date", column_to_aggregate+"_date_latest")

    # merge
    specific_comorbidity_feature = specific_comorbidity_first.join(specific_comorbidity_latest,
                                                                   ["patient_pseudo_id", column_to_aggregate],
                                                                   "left")
    # make binary column
    specific_comorbidity_feature = specific_comorbidity_feature.withColumn(column_to_aggregate+"_binary", F.lit(1))

    # if earliest was in last 1 month, 6 months, 12 months or more

    for weeks in weeks_to_subtract:
        subtracted_date = subtract_n_weeks_from_date(date_cutoff, weeks)

        specific_comorbidity_feature = specific_comorbidity_feature.withColumn(column_to_aggregate+"_in_last_"+str(weeks)+"_weeks",
                                                                               F.when( (F.col(column_to_aggregate+"_date_earliest")>subtracted_date) &
                                                                               (F.col(column_to_aggregate+"_date_earliest")<date_cutoff), 1)
                                                                               .otherwise(0))

    return specific_comorbidity_feature


def number_calls_during_time_period(df, date:str, num_weeks_to_subtract:int, variable_name_prefix:str = "111_calls_last_", groupbycols = ["patient_pseudo_id"]):
    """
    Returns the number of 111 calls in the period between
    the given date - number_weeks_to_subtract until
    the latest data present in the dataset
    """
    date_limit = subtract_n_weeks_from_date(date, num_weeks_to_subtract)
    df_limited = df.filter(F.col("call_date") >= F.lit(date_limit))

    col_alias = variable_name_prefix+str(num_weeks_to_subtract)+"_weeks"
    total_number_calls_in_period = df_limited.groupBy(groupbycols).count().withColumnRenamed("count", col_alias)

    return total_number_calls_in_period


def number_symptom_calls_during_time_period(df,
                                            date: str,
                                            num_weeks_to_subtract: int,
                                            variable_name: str = "111_calls_last_",
                                            groupbycols: str = ["patient_pseudo_id"],
                                            pivotcol: str = "symptom_group_no_spaces"):
    """
    Returns the number of 111 calls for each value in the column pivotcol
    in the period between the given date - number_weeks_to_subtract until
    the latest data present in the dataset
    """
    date_limit = subtract_n_weeks_from_date(date, num_weeks_to_subtract)
    df_limited = df.filter(F.col("call_date") >= F.lit(date_limit))
    symptom_count = df_limited.groupBy(groupbycols).pivot(pivotcol).count()

    # string to add to the end of the pivoted variable name 
    col_name_addition = "_" + variable_name + str(num_weeks_to_subtract)+"_weeks"

    new_col_names = []
    for col in symptom_count.columns:
        if col == "patient_pseudo_id":
            new_col_names.append(col)
        else:
            new_col_names.append(col+col_name_addition)

    # rename the created columns
    symptom_count = symptom_count.toDF(*new_col_names)

    return symptom_count

def number_OPA_during_time_period(df,
                                  date: str,
                                  num_weeks_to_subtract: int,
                                  variable_name_prefix: str = "OPA_attendances_last_",
                                  groupbycols: str = ["patient_pseudo_id"]):
    """
    Returns the number of outpatient attendances in the period between
    the given date - number_weeks_to_subtract until
    the latest data present in the dataset
    """
    date_limit = subtract_n_weeks_from_date(date, num_weeks_to_subtract)
    df_limited = df.filter(F.col("attendance_date") >= F.lit(date_limit))

    col_alias = variable_name_prefix + str(num_weeks_to_subtract) + "_weeks"
    total_number_OPA_in_period = df_limited.groupBy(groupbycols).count().withColumnRenamed("count", col_alias)

    return total_number_OPA_in_period

def number_AE_during_time_period(df,
                                 date: str,
                                 num_weeks_to_subtract: int,
                                 variable_name_prefix: str = "AE_attendances_last_",
                                 groupbycols: str = ["patient_pseudo_id"]):
    """
    Returns the number of emergency attendances in the period between
    the given date - number_weeks_to_subtract until
    the latest data present in the dataset
    """
    date_limit = subtract_n_weeks_from_date(date, num_weeks_to_subtract)
    df_limited = df.filter(F.col("attendance_date") >= F.lit(date_limit))

    col_alias = variable_name_prefix + str(num_weeks_to_subtract) + "_weeks"
    total_number_AE_in_period = df_limited.groupBy(groupbycols).count().withColumnRenamed("count", col_alias)

    return total_number_AE_in_period

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

    df_comorbidity = df_activity.withColumn("diagnosis",
                                            F.explode(F.col(array_column)))

    # remove rows with empty space
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis") != '')

    # trim whitespace
    df_comorbidity = df_comorbidity.withColumn("diagnosis", F.trim(F.col("diagnosis")))

    # replace space and vertical bar (|) with empty string
    df_comorbidity = df_comorbidity.withColumn("diagnosis", F.regexp_replace("diagnosis", " |\|", ""))

    # replace comma and dash with empty string
    df_comorbidity = df_comorbidity.withColumn("diagnosis", F.regexp_replace("diagnosis", ",|-", ""))

    # rename attendance date to date for consistency
    df_comorbidity = df_comorbidity.withColumnRenamed("attendance_date", "date")

    # remove null diagnosis
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis").isNotNull())

    # remove empty string
    df_comorbidity = df_comorbidity.filter(F.col("diagnosis") != "")

    return df_comorbidity
