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

# the ICD10 codes (using % magic character) for each grouping
# Add multiple ICD10 codes to the list
dict_icd10_groups = {"digestive_system_diseases": ["K%"],
                     "family_history_cancer": ["Z80%"],
                     }
