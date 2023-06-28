# define cut-off date
date_cutoff = "2022-01-01"

weeks_to_subtract = [4, 12, 26, 52, 260]

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
dict_icd10_groups = {"digestive_system_diseases": ["K%"],
                     "family_history_cancer": ["Z80%"],
                     }
