from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output
from pyspark.sql.window import Window

@transform_df(
    Output("/NHS/Cancer Late Presentation Likelihood Modelling Internal Transforms/cancer-late-datasets/interim-datasets/ecds_snomed_icd10_mapping"),
    df_ecds_etos=Input("ri.foundry.main.dataset.2341ae73-65bc-405f-a1c7-e644da6e0255"),
    df_udal_map=Input("ri.foundry.main.dataset.2a792e6d-170e-4f42-8093-2db225060034")
)
def compute(df_ecds_etos,df_udal_map):
    """
    Combine the SNOMED - ICD10 reference tables
    Where ECDS ETOS has a deprecated code, use df_udal_map which is
    [UKHD_SNOMED].[ICD10_4thEd_5_Char_Complex_To_SNOMED_Map_SCD] on UDAL
    Where multiple mappings in df_udal_map exists a single one is picked
    based on the first Map_Block, first Map_Group and highest priority
    See reference document: UK Classification Maps in the NHS Digital SNOMED CT Browser
    """

    #Filter to where the ICD10_Mapping exists (these are the non-deprecated SNOMED codes)
    df_ecds_etos_non_deprecated = df_ecds_etos.filter(F.col("ICD10_Mapping").isNotNull())

    # identify where there is no mapping to ICD_10 (these are the deprecated SNOMED codes)
    df_ecds_etos_deprecated = df_ecds_etos.filter(F.col("ICD10_Mapping").isNull())

    # identify single ICD10 code per df_udal_map
    # This is picking the first Block, first Group and highest priority
    # See UK Classification Maps in the NHS Digital SNOMED CT Browser
    w2 = Window.partitionBy("Referenced_Component_ID").orderBy(F.col("Map_Block").asc(),
                                                               F.col("Map_Group").asc(),
                                                               F.col("Map_Priority").desc())

    df_udal_map_single_per_snomed = df_udal_map.withColumn("row", F.row_number().over(w2)).filter(F.col("row") == 1).drop("row")

    # Join the udal map to the table where the ICD10 codes were null 
    df_joined_deprecated = df_ecds_etos_deprecated.join(df_udal_map_single_per_snomed,
                                                        df_ecds_etos_deprecated.SNOMED_Code == df_udal_map.Referenced_Component_ID,
                                                        "left")

    # replace the null ICD10 mapping with the ICD10 diagnosis joined from df_udal_map
    df_joined_deprecated = df_joined_deprecated.withColumn("ICD10_Mapping",F.when(F.col("ICD10_Mapping").isNull(),
                                                           F.col("Map_Target")).otherwise(F.col("ICD10_Mapping")))

    # list of columns to carry through
    common_columns = ["ECDS_UniqueID", "ECDS_Group1", "ECDS_Group2", "ECDS_Group3", "ECDS_Description", "ECDS_SearchTerms", "SNOMED_Code", "SNOMED_UK_Preferred_Term", "SNOMED_Fully_Specified_Name", "Flag_Injury", "Flag_NotifiableDisease", "Flag_Male", "Flag_Female", "Flag_Allergy", "Flag_SDEC", "Flag_ADS", "Notes", "Valid_From", "Valid_To", "ICD10_Mapping", "ICD10_Description", "ICD11_Mapping", "ICD11_Description"]

    # ensuring common names for the two tables to be unioned
    df_ecds_etos_non_deprecated = df_ecds_etos_non_deprecated.select(common_columns)
    df_joined_deprecated = df_joined_deprecated.select(common_columns)

    # union dataframes
    df_combined = df_joined_deprecated.unionByName(df_ecds_etos_non_deprecated)

    return df_combined
