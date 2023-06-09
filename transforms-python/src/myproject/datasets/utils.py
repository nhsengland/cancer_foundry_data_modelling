import pyspark.sql.functions as F
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
