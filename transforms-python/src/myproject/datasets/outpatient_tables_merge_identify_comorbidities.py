# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
import pyspark.sql.functions as F
from pyspark.sql.types import StringType


@configure(profile=["DRIVER_MEMORY_LARGE"])
@transform_df(
    Output("ri.foundry.main.dataset.e3a5741d-6cb5-4b64-8d3b-0be26b1b6e62"),
    df_sus_opa_raw_unioned=Input(
        sus_opa_raw_unioned_path
    ),
    outpatient_activity=Input(
        outpatient_acitivity_path
    ),
)
def compute(df_sus_opa_raw_unioned, outpatient_activity):
    """
    Merge the outpatients_activity with the raw SUS OPA table
    """

    # make OPA_Ident string for matching to opa_ident_min which is also string
    df_sus_opa_raw_unioned = df_sus_opa_raw_unioned.withColumn(
        "OPA_Ident_str", F.col("OPA_Ident").cast(StringType())
    )

    # merge outpatient_activity with df_sus_opa_raw_unioned
    outpatient_activity_with_diagnosis = outpatient_activity.join(
        df_sus_opa_raw_unioned,
        outpatient_activity.opa_ident_min == df_sus_opa_raw_unioned.OPA_Ident_str,
        "left",
    )

    return outpatient_activity_with_diagnosis
