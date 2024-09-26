# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output(sus_opa_combined_path),
    df_opa_diag_2008_2020=Input(
        opa_diagnosis_2008_2020_path
    ),
    df_opa_diag_2021=Input(
        opa_diagnosis_2021_path
    ),
    df_opa_diag_2022=Input(
        opa_diagnosis_2022_path
    ),
    df_opa_diag_2023=Input(
        opa_diagnosis_2023_path
    ),
)
def compute(
    df_opa_diag_2008_2020, df_opa_diag_2021, df_opa_diag_2022, df_opa_diag_2023
):
    """
    Union the raw SUS OPA from across the years
    Filter to FY 2016/2017 and after
    """

    list_opa_dfs = [
        df_opa_diag_2008_2020,
        df_opa_diag_2021,
        df_opa_diag_2022,
        df_opa_diag_2023,
    ]
    df_opa_diag_union = reduce(DataFrame.unionByName, list_opa_dfs)

    # only take data from 2016/2017 onwards
    df_opa_diag_union = df_opa_diag_union.filter(
        F.col("Der_Financial_Year") >= "2016/2017"
    )

    return df_opa_diag_union
