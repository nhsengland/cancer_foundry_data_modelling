import pyspark.sql.functions as F

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