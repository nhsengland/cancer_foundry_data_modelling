# from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output, configure
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from myproject.datasets import utils
from myproject.datasets.config import (list_categorical_variables,
                                       list_boolean_variables,
                                       list_patient_health_record_variables,
                                       list_categorical_variables_from_phr,
                                       list_columns_mortality)

@configure(profile=['DRIVER_MEMORY_LARGE'])
@transform_df(
    Output("/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/interim-datasets/patient_features_expanded"),
    patient=Input("ri.foundry.main.dataset.36644003-34c3-43d0-bace-751b3e071ea3"),
    mortality=Input("ri.foundry.main.dataset.b243b273-4b66-4326-a37b-8bc060c8bdf6"),
    dataset_111_calls=Input("ri.foundry.main.dataset.a3dfadbc-b022-4d88-98c2-8ee3392beed1"),
    patient_health_record_features=Input("ri.foundry.main.dataset.195dd683-e55b-450e-abeb-5f00de85ad78"),
    ae_features=Input("ri.foundry.main.dataset.3b5c090a-bad0-4568-9163-d37a41c748e1"),
    ip_features=Input("/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/interim-datasets/IP_features"),
    opa_features=Input("ri.foundry.main.dataset.f0e0629f-e190-40fa-a7c3-a95ce8d2fe6a")
)
def compute(patient, mortality, dataset_111_calls, ae_features, ip_features, opa_features, patient_health_record_features):
    """
    Add features to the patient table from patient_features

    Merge mortality data (date of death, cause of death etc)
    Merge features from 111 calls (number of calls, symptoms of calls)
    Merge features from patient health record features (long term condtions and some geographical variables)
    Merge number of attendances features from AE, IP, OPA
    """
    # MORTALITY
    # get single mortality per ID

    w_desc = Window.partitionBy("patient_pseudo_id").orderBy(F.col("activity_id").desc())
    mortality_latest = mortality.withColumn("row", F.row_number().over(w_desc)).filter(F.col("row") == 1).drop("row")

    # Merge mortality data
    patient = patient.join(mortality_latest.select(list_columns_mortality),
                           "patient_pseudo_id",
                           "left")

    # 111 calls
    patient = patient.join(dataset_111_calls,
                           "patient_pseudo_id",
                           "left")

    # A&E attendance features

    patient = patient.join(ae_features, "patient_pseudo_id", "left")

    # IP attendance features

    patient = patient.join(ip_features, "patient_pseudo_id", "left")

    # OP attendance features

    patient = patient.join(opa_features, "patient_pseudo_id", "left")

    # PATIENT HEALTH RECORD

    # select relevant columns from patient health record table
    all_columns_phr = patient_health_record_features.columns

    # select all the ID, columns listed in list_patient_health_record_variables, and the dummy variables created from 
    columns_to_select_from_phr = ["patient_pseudo_id"] + list_patient_health_record_variables

    # add all of the binary columns created from the categorical columns
    for col in all_columns_phr:
        for cat_var in list_categorical_variables_from_phr:
            if cat_var + "_" in col:
                columns_to_select_from_phr.append(col)

    patient_health_record_features_selected = patient_health_record_features.select(columns_to_select_from_phr)

    # rename the columns to indicate their source
    for col in patient_health_record_features_selected.columns:
        # we want to keep ID column name as it is
        if col != "patient_pseudo_id":
            patient_health_record_features_selected = patient_health_record_features_selected.withColumnRenamed(col,
                                                                                                               "phr_"+col)

    # left join patient health record features tables
    patient = patient.join(patient_health_record_features_selected,
                           "patient_pseudo_id",
                           "left")

    # PATIENT FEATURES

    # Age column: replace ages over 113 with None
    patient = patient.withColumn("age_clean", F.when(patient.age <= 113, patient.age)\
                                 .otherwise(F.lit(None)))

    # creating a list to store expressions for the binary columns to be created
    all_col_expressions = []

    # Converting boolean columns to binary columns
    for bool_variable in list_boolean_variables:
        patient = patient.withColumn(bool_variable + "_null_removed", F.when(F.col(bool_variable) == True, "True")
                                                                      .when(F.col(bool_variable) == False, "False")
                                                                      .otherwise("Unknown"))

        expression_cols = utils.create_expressions_categorical_column(patient, col_name=bool_variable+"_null_removed")
        all_col_expressions = all_col_expressions + expression_cols

    # Converting categorical columns to binary columns
    for cat_variable in list_categorical_variables:
        patient = utils.clean_column_of_strings(patient, col_name=cat_variable, append_to_col_name="_null_removed", fill_string='unknown')
        expression_cols = utils.create_expressions_categorical_column(patient, col_name=cat_variable+"_null_removed")
        all_col_expressions = all_col_expressions + expression_cols

    # Select all columns and add the expressions for the binary columns to be created 
    patient = patient.select(F.col("*"), *all_col_expressions)

    return patient
