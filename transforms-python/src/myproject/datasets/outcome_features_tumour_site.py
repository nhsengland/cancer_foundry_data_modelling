from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("ri.foundry.main.dataset.7c022978-3bba-4a0b-8e5a-5ac4dec34ba7"),
    source_df=Input("ri.foundry.main.dataset.30a5ce42-d7ca-4152-8ce3-7573fc39bfe2"),
)
def compute(source_df):

    # changing column type from array to string
    df = source_df.withColumn("all_cancer_diagnoses_after_cut_off",
                              F.concat_ws(", ", "all_cancer_diagnoses_after_cut_off"))


    # creating binary columns based on PHE methodology for tumour locations
    df = df.withColumn("head_and_neck_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C00") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C01") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C02") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C03") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C04") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C05") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C06") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C07") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C08") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C09") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C10") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C11") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C12") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C13") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C14") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C30") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C31") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C32"), 1).otherwise(0))

    df = df.withColumn("gi_upper_and_lower_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C15") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C16") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C17") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C18") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C19") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C20") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C21") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C22") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C23") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C24") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C25") |
                                                     F.col("all_cancer_diagnoses_after_cut_off").contains("C26"), 1).otherwise(0))

    df = df.withColumn("lung_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C33") |
                                       F.col("all_cancer_diagnoses_after_cut_off").contains("C34") |
                                       F.col("all_cancer_diagnoses_after_cut_off").contains("C37") |
                                       F.col("all_cancer_diagnoses_after_cut_off").contains("C38") |
                                       F.col("all_cancer_diagnoses_after_cut_off").contains("C39") |
                                       F.col("all_cancer_diagnoses_after_cut_off").contains("C45"), 1).otherwise(0))

    df = df.withColumn("sarcoma_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C40") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C41") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C46") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C48") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C49"), 1).otherwise(0))

    df = df.withColumn("melanoma_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C43"), 1).otherwise(0))

    df = df.withColumn("brain_and_cns_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C47") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C69") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C70") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C71") |
                                                F.col("all_cancer_diagnoses_after_cut_off").contains("C72"), 1).otherwise(0))

    df = df.withColumn("breast_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C50"), 1).otherwise(0))

    df = df.withColumn("gynae_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C51") |
                                        F.col("all_cancer_diagnoses_after_cut_off").contains("C52") |
                                        F.col("all_cancer_diagnoses_after_cut_off").contains("C53") |
                                        F.col("all_cancer_diagnoses_after_cut_off").contains("C54") |
                                        F.col("all_cancer_diagnoses_after_cut_off").contains("C55") |
                                        F.col("all_cancer_diagnoses_after_cut_off").contains("C56") |
                                        F.col("all_cancer_diagnoses_after_cut_off").contains("C57") |
                                        F.col("all_cancer_diagnoses_after_cut_off").contains("C58"), 1).otherwise(0))

    df = df.withColumn("urology_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C60") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C61") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C62") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C63") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C64") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C65") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C66") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C67") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C68"), 1).otherwise(0))

    df = df.withColumn("endocrine_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C73") |
                                            F.col("all_cancer_diagnoses_after_cut_off").contains("C74") |
                                            F.col("all_cancer_diagnoses_after_cut_off").contains("C75"), 1).otherwise(0))

    df = df.withColumn("haematological_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C81") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C82") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C83") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C84") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C85") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C86") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C87") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C88") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C89") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C90") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C91") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C92") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C93") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C94") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C95") |
                                                 F.col("all_cancer_diagnoses_after_cut_off").contains("C96"), 1).otherwise(0))

    df = df.withColumn("unknown_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C76") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C77") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C78") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C79") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C80") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C97"), 1).otherwise(0))

    # creating binary columns based on stakeholder recomendations for tumour location
    df = df.withColumn("ovarian_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C56"), 1).otherwise(0))

    df = df.withColumn("pancreatic_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C25"), 1).otherwise(0))

    df = df.withColumn("bladder_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C67"), 1).otherwise(0))

    df = df.withColumn("myeloma_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C90"), 1).otherwise(0))

    df = df.withColumn("stomach_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C16"), 1).otherwise(0))

    df = df.withColumn("oesophageal_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C16"), 1).otherwise(0))

    df = df.withColumn("kidney_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C64") |
                                         F.col("all_cancer_diagnoses_after_cut_off").contains("C65"), 1).otherwise(0))

    df = df.withColumn("lymphoma_after_cut_off",  F.when(F.col("all_cancer_diagnoses_after_cut_off").contains("C81") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C82") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C83") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C84") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C85") |
                                          F.col("all_cancer_diagnoses_after_cut_off").contains("C86"), 1).otherwise(0))

    return df