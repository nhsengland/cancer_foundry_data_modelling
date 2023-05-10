"""
# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output

from myproject.datasets import utils


@transform_df(
    Output("/NHS/Cancer Late Presentation Likelihood Modelling/TARGET_DATASET_PATH"),
    source_df=Input("/NHS/Cancer Late Presentation Likelihood Modelling/SOURCE_DATASET_PATH"),
)
def compute(source_df):
    return utils.identity(source_df)
"""
