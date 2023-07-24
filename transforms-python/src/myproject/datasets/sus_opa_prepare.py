# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


@transform_df(
    Output("/NHS/cancer-late-presentation/cancer-late-datasets/interim-datasets/sus_opa_raw_unioned"),
    df_opa_diag_2008_2020=Input("ri.foundry.main.dataset.10e87cda-7bbc-4d0c-a8b3-c2559b832dd3"),
    df_opa_diag_2021=Input("ri.foundry.main.dataset.c12f6390-def6-4fa4-bb51-1e70e3ac1c20"),
    df_opa_diag_2022=Input("ri.foundry.main.dataset.e5ca6fce-6839-4b1d-801c-81e2e976c5bd"),
    df_opa_diag_2023=Input("ri.foundry.main.dataset.5ce08265-84ab-4d90-a38a-df31e47ecb47"),
)
def compute(df_opa_diag_2008_2020, df_opa_diag_2021, df_opa_diag_2022, df_opa_diag_2023):
    """
    Union the raw SUS OPA from across the years
    Filter to FY 2016/2017 and after
    """

    list_opa_dfs = [df_opa_diag_2008_2020, df_opa_diag_2021, df_opa_diag_2022, df_opa_diag_2023]
    df_opa_diag_union = reduce(DataFrame.unionByName, list_opa_dfs)

    # only take data from 2016/2017 onwards
    df_opa_diag_union = df_opa_diag_union.filter(F.col("Der_Financial_Year") >= "2016/2017")

    return df_opa_diag_union
