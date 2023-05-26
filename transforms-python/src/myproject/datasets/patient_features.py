# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.36644003-34c3-43d0-bace-751b3e071ea3"),
    patient=Input("ri.foundry.main.dataset.b2a84252-8ae1-4f7c-9948-c7e00afe36a8"),

)
def compute(patient):
    """
    Add features to the patient table from the Person Ontology
    Features to be added: demographic, geographic, GP practice   
    """

    return patient
