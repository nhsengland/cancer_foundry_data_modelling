# define cut-off date
date_cutoff = "2021-08-31"

weeks_to_subtract = [4, 12, 26, 52, 260]

target_weeks = [12, 26, 52, 78, 104, 260]

# categorical variables from which to create binary variables in patient_features
list_categorical_variables = [
    "gender",
    "ethnicity_code",
    "imd_decile",
    "acorn_household_type",
    "acorn_wellbeing_type",
    "rural_urban_classification",
    "ccg_of_residence",
]

# boolean variables from which to create binary variables in patient_features
list_boolean_variables = [
    "living_alone",
    "living_with_young",
    "living_with_elderly",
    "private_outdoor_space",
]

# categorical variables from which to create binary variables in patient_health_record_features
list_categorical_variables_from_phr = [
    "nhs_region",
    "stp_code",
]

# features from Patient Health Record
list_patient_health_record_variables = [
    "date",
    "date_of_birth",
    "date_of_death",
    "healthy_well",
    "ltc",
    "disability",
    "frailty_dementia",
    "organ_failure",
    "incurable_cancer",
    "highest_acuity_segment_code",
    "highest_acuity_segment_name",
    "highest_acuity_segment_description",
    "maternal_and_infant_health",
    "alcohol_dependence",
    "asthma",
    "atrial_fibrillation",
    "bronchiectasis",
    "cancer",
    "cerebrovascular_disease",
    "chronic_kidney_disease",
    "chronic_liver_disease",
    "chronic_pain",
    "copd",
    "coronary_heart_disease",
    "cystic_fibrosis",
    "depression",
    "diabetes",
    "epilepsy",
    "heart_failure",
    "hypertension",
    "inflammatory_bowel_disease",
    "multiple_sclerosis",
    "osteoarthritis",
    "osteoporosis",
    "parkinsons_disease",
    "peripheral_vascular_disease",
    "pulmonary_heart_disease",
    "rheumatoid_arthritis",
    "sarcoidosis",
    "serious_mental_illness",
    "sickle_cell_disease",
    "autism",
    "learning_disability",
    "physical_disability",
    "dementia",
    "intermediate_frailty_risk_hfrs",
    "high_frailty_risk_hfrs",
    "end_stage_renal_failure",
    "severe_interstitial_lung_disease",
    "liver_failure",
    "neurological_organ_failure",
    "severe_copd",
    "severe_heart_failure",
    "immunocompromised",
    "spleen_problems",
    "active_cancer",
    "active_lung_cancer",
    "blood_and_bone_marrow_cancer",
    "bone_marrow_and_stem_cell_transplants",
    "organ_transplants",
    "downs_syndrome",
    "rare_diseases",
    "palliative_care",
    "obesity",
    "other_chronic_respiratory_diseases",
    "pulmonary_embolism",
    "congenital_heart_disease",
    "severe_asthma",
    "cerebral_palsy",
    "other_chronic_neurological_diseases",
]
# the ICD10 codes (using % magic character) for each grouping
# Add multiple ICD10 codes to the list
dict_icd10_groups = {
    "digestive_system_diseases": ["K%"],
    "family_history_cancer": ["Z80%"],
}

diagnosis_columns_opa = [
    "Primary_Diagnosis_Code",
    "Secondary_Diagnosis_Code_1",
    "Secondary_Diagnosis_Code_2",
    "Secondary_Diagnosis_Code_3",
    "Secondary_Diagnosis_Code_4",
    "Secondary_Diagnosis_Code_5",
    "Secondary_Diagnosis_Code_6",
    "Secondary_Diagnosis_Code_7",
    "Secondary_Diagnosis_Code_8",
    "Secondary_Diagnosis_Code_9",
    "Secondary_Diagnosis_Code_10",
    "Secondary_Diagnosis_Code_11",
    "Secondary_Diagnosis_Code_12",
    "Secondary_Diagnosis_Code_13",
    "Secondary_Diagnosis_Code_14",
    "Secondary_Diagnosis_Code_15",
    "Secondary_Diagnosis_Code_16",
    "Secondary_Diagnosis_Code_17",
    "Secondary_Diagnosis_Code_18",
    "Secondary_Diagnosis_Code_19",
    "Secondary_Diagnosis_Code_20",
    "Secondary_Diagnosis_Code_21",
    "Secondary_Diagnosis_Code_22",
    "Secondary_Diagnosis_Code_23",
]

list_columns_mortality = [
    "patient_pseudo_id",
    "reg_date_of_death",
    "s_underlying_cod_icd10",
    "s_cod_code_1",
    "s_cod_code_2",
    "s_cod_code_3",
    "s_cod_code_4",
    "s_cod_code_5",
    "s_cod_code_6",
    "s_cod_code_7",
    "s_cod_code_8",
    "s_cod_code_9",
    "s_cod_code_10",
    "s_cod_code_11",
    "s_cod_code_12",
    "s_cod_code_13",
    "s_cod_code_14",
    "s_cod_code_15",
]

geographical_variables = ["cancer_alliance", "integrated_care_board"]

list_cancer_types = [
    "head_and_neck",
    "gi_upper_and_lower",
    "lung",
    "sarcoma",
    "melanoma",
    "brain_and_cns",
    "breast",
    "gynae",
    "urology",
    "endocrine",
    "haematological",
    "unknown",
    "ovarian",
    "pancreatic",
    "bladder",
    "myeloma",
    "stomach",
    "oesophageal",
    "kidney",
    "lymphoma",
]