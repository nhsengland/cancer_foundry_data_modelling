from pyspark.sql import functions as F
from itertools import chain
from pyspark.sql.functions import create_map, lit


rural_urban_dict = {
    "Rural town and fringe in a sparse setting": "rural",
    "Rural town and fringe": "rural",
    "Rural hamlets and isolated dwellings": "rural",
    "unknown": "unknown",
    "Urban city and town in a sparse setting": "urban",
    "Urban city and town": "urban",
    "Rural hamlets and isolated dwellings in a sparse setting": "rural",
    "Urban minor conurbation": "urban",
    "Rural village in a sparse setting": "rural",
    "Rural village": "rural",
    "Urban major conurbation": "urban",
}

household_type_dict = {
    "1": "1",	"2": "1",	"3": "1",	"4": "1",	"5": "1",	"6": "1",	"7": "1",	"8": "1",	"9": "1",
    "10": "1",	"11": "1",	"12": "2",	"13": "2",	"14": "2",	"15": "2",	"16": "2",	"17": "2",	"18": "2",
    "19": "2",	"20": "2",	"21": "2",	"22": "3",	"23": "3",	"24": "3",	"25": "3",	"26": "3",	"27": "3",
    "28": "3",	"29": "3",	"30": "3",	"31": "3",	"32": "3",	"33": "4",	"34": "4",	"35": "4",	"36": "4",
    "37": "4",	"38": "4",	"39": "4",	"40": "4",	"41": "4",	"42": "4",	"43": "4",	"44": "4",	"45": "4",
    "46": "4",	"47": "5",	"48": "5",	"49": "5",	"50": "5",	"51": "5",	"52": "5",	"53": "5",	"54": "5",
    "55": "5",	"56": "5",	"57": "5",	"58": "unknown",	"59": "unknown",	"60": "6",	"61": "6",	"62": "6",
    "unknown": "unknown"
}


cols_to_keep = ["patient_pseudo_id", "111_calls_total",	"111_calls_last_52_weeks",	"111_calls_cancer_last_52_weeks",	"Abdominal_Pain_111_calls_last_52_weeks",	
                "Abdominal_Pain__Pregnant__Over_20_Weeks_111_calls_last_52_weeks",	"Abdominal_Pain__Rectal_Bleeding__Pregnant_Over_20_Weeks_111_calls_last_52_weeks",	
                "Abdominal__Flank__Groin_or_Back_Pain_or_Swelling_111_calls_last_52_weeks",	"Blood_in_Urine_111_calls_last_52_weeks",	"Breast_Lump_111_calls_last_52_weeks",	"Breast_Lump__Pregnant_111_calls_last_52_weeks",	"Breathing_Problems__Breathlessness_or_Wheeze_111_calls_last_52_weeks",	"Breathing_Problems__Breathlessness_or_Wheeze__Pregnant_111_calls_last_52_weeks",	"Chest_and_Upper_Back_Pain_111_calls_last_52_weeks",	"Constipation_111_calls_last_52_weeks",	"Cough_111_calls_last_52_weeks",	"Coughing_up_Blood_111_calls_last_52_weeks",	"Diarrhoea_111_calls_last_52_weeks",	"Diarrhoea_and_Vomiting_111_calls_last_52_weeks",	"Difficulty_Passing_Urine_111_calls_last_52_weeks",	"ED_Triage_Chest_Pain_111_calls_last_52_weeks",	"Easy_or_Unexplained_Bruising_111_calls_last_52_weeks",	"Face__Neck_Pain_or_Swelling_111_calls_last_52_weeks",	"Fever_111_calls_last_52_weeks",	"Genital_Problems_111_calls_last_52_weeks",	"Itch_111_calls_last_52_weeks",	"Mouth_Ulcers_111_calls_last_52_weeks",	"Pain_and/or_Frequency_Passing_Urine_111_calls_last_52_weeks",	"Rectal_Bleeding_111_calls_last_52_weeks",	"Rectal_Pain__Swelling__Lump_or_Itch_111_calls_last_52_weeks",	"Skin_Lumps_111_calls_last_52_weeks",	"Skin_Problems_111_calls_last_52_weeks",	"Tiredness__Fatigue__111_calls_last_52_weeks",	"Urinary_Problems_111_calls_last_52_weeks",	"Vaginal_Bleeding_111_calls_last_52_weeks",	"Vaginal_Discharge_111_calls_last_52_weeks",	"Vomiting_111_calls_last_52_weeks",	"Vomiting_Blood_111_calls_last_52_weeks",	"ae_attend_total",	"ae_attendances_last_52_weeks",	"cancer_before_cut_off_binary",	"cancer_before_cut_off_in_last_4_weeks",	"cancer_before_cut_off_in_last_12_weeks",	"cancer_before_cut_off_in_last_26_weeks",	"cancer_before_cut_off_in_last_52_weeks",	"cancer_before_cut_off_in_last_260_weeks",	"cancer_before_cut_off",	"cancer_before_cut_off_date_earliest",	"cancer_before_cut_off_date_latest",	"IP_attend_total",	"IP_attendances_last_52_weeks",	"IP_diagnoses_total",	"IP_diagnosis_pct_of_attendnaces",	"opa_attend_total",	"OPA_attendances_last_52_weeks",	"opa_diagnoses_total",	"opa_diagnosis_pct_of_attendnaces",	"all_cancer_diagnoses_after_cut_off",	"cancer_diagnosis_in_next_12_weeks",	"cancer_diagnosis_in_next_26_weeks",	"cancer_diagnosis_in_next_52_weeks",	"cancer_diagnosis_in_next_78_weeks",	"cancer_diagnosis_in_next_104_weeks",	"cancer_diagnosis_in_next_260_weeks",	"cancer_after_cut_off_date_earliest",	"diagnosis_earliest_after_cut_off_date",	"cancer_after_cut_off_date_latest",	"diagnosis_latest_after_cut_off_date",	"care_home",	"registered_with_closed_practice",	"diabetes",	"immunocompromised",	"ethnicity_code",	"gender",	"imd_decile",	"acorn_type",	"living_alone_null_removed",	"living_with_young_null_removed",	"living_with_elderly_null_removed",	"gender_null_removed",	"ethnicity_code_null_removed",	"imd_decile_null_removed",	"rural_urban_classification_null_removed",	"death_month",	"living_alone_null_removed_False",	"living_alone_null_removed_Unknown",	"living_alone_null_removed_True",	"living_with_young_null_removed_False",	"living_with_young_null_removed_Unknown",	"living_with_young_null_removed_True",	"living_with_elderly_null_removed_False",	"living_with_elderly_null_removed_Unknown",	"living_with_elderly_null_removed_True",	"gender_null_removed_Female",	"gender_null_removed_Not_known",	"gender_null_removed_Not_specified",	"gender_null_removed_Male",	"ethnicity_code_null_removed_K",	"ethnicity_code_null_removed_F",	"ethnicity_code_null_removed_99",	"ethnicity_code_null_removed_E",	"ethnicity_code_null_removed_B",	"ethnicity_code_null_removed_L",	"ethnicity_code_null_removed_M",	"ethnicity_code_null_removed_D",	"ethnicity_code_null_removed_C",	"ethnicity_code_null_removed_J",	"ethnicity_code_null_removed_A",	"ethnicity_code_null_removed_N",	"ethnicity_code_null_removed_X",	"ethnicity_code_null_removed_S",	"ethnicity_code_null_removed_R",	"ethnicity_code_null_removed_G",	"ethnicity_code_null_removed_P",	"ethnicity_code_null_removed_H",	"imd_decile_null_removed_7",	"imd_decile_null_removed_3",	"imd_decile_null_removed_8",	"imd_decile_null_removed_unknown",	"imd_decile_null_removed_5",	"imd_decile_null_removed_6",	"imd_decile_null_removed_9",	"imd_decile_null_removed_1",	"imd_decile_null_removed_10",	"imd_decile_null_removed_4",	"imd_decile_null_removed_2",	"acorn_household_type_null_removed",	"age_clean",	"phr_date",	"phr_date_of_birth",	"phr_healthy_well",	"phr_ltc",	"phr_disability",	"phr_frailty_dementia",	"phr_organ_failure",	"phr_incurable_cancer",	"phr_highest_acuity_segment_code",	"phr_highest_acuity_segment_name",	"phr_highest_acuity_segment_description",	"phr_nhs_region_null_removed",	"phr_stp_code_null_removed",	"phr_nhs_region_null_removed_North_West",	"phr_nhs_region_null_removed_unknown",	"phr_nhs_region_null_removed_London",	"phr_nhs_region_null_removed_South_East",	"phr_nhs_region_null_removed_East_of_England",	"phr_nhs_region_null_removed_North_East_and_Yorkshire",	"phr_nhs_region_null_removed_South_West",	"phr_nhs_region_null_removed_Midlands",	"phr_hypertension",	"phr_depression",	"phr_diabetes",	"phr_asthma",	"phr_osteoarthritis",	"phr_coronary_heart_disease",	"phr_cancer",	"phr_date_of_death",	"phr_atrial_fibrillation",	"phr_cerebrovascular_disease",	"phr_osteoporosis",	"phr_copd",	"phr_obesity",	"phr_heart_failure",	"phr_immunocompromised",	"phr_peripheral_vascular_disease",	"phr_chronic_kidney_disease",	"phr_dementia",	"phr_chronic_pain",	"phr_intermediate_frailty_risk_hfrs",	"phr_palliative_care",	"phr_high_frailty_risk_hfrs",	"phr_serious_mental_illness",	"phr_physical_disability",	"phr_severe_heart_failure",	"phr_epilepsy",	"phr_pulmonary_heart_disease",	"phr_rheumatoid_arthritis",	"phr_severe_copd",	"phr_alcohol_dependence",	"phr_chronic_liver_disease",	"phr_inflammatory_bowel_disease",	"phr_other_chronic_respiratory_diseases",	"phr_severe_asthma",	"phr_pulmonary_embolism",	"phr_learning_disability",	"phr_congenital_heart_disease",	"phr_blood_and_bone_marrow_cancer",	"phr_bronchiectasis",	"phr_end_stage_renal_failure",	"phr_liver_failure",	"phr_parkinsons_disease",	"phr_autism",	"phr_active_cancer",	"phr_neurological_organ_failure",	"phr_spleen_problems",	"phr_severe_interstitial_lung_disease",	"phr_rare_diseases",	"phr_organ_transplants",	"phr_multiple_sclerosis",	"phr_other_chronic_neurological_diseases",	"phr_cerebral_palsy",	"phr_sarcoidosis",	"phr_active_lung_cancer",	"phr_downs_syndrome",	"phr_sickle_cell_disease",	"phr_bone_marrow_and_stem_cell_transplants",	"phr_cystic_fibrosis",	"Z80-Z99_last_52_weeks", "first_cancer_diagnosis_at_death_after_cut_off"

]


def df_mapping(df, col_name, dict_name: dict):
    """ This function is used to map values from dictionary 
        to replace values in the column"""
    mapping_expr = create_map([lit(x) for x in chain(*dict_name.items())])
    df_to_be_mapped = df.withColumn(col_name, mapping_expr[df[col_name]])
    return df_to_be_mapped


# we need to remove anything that ends with 4, 12, 26, 260 weeks
# remove household type breakdowns completely except null_removed
# same as above for rural and urban
# recreate columns for household types and rural and urban
# comorbidity tables to be removed

def filter_columns(df, cols_to_keep):
    """ This function is used to filter columns needed
    for the analysis"""
    df_subset = df.select(*cols_to_keep)
    return df_subset


def generate_categories(df, base_col):
    """ This function is to generate binary variables from categorical columns"""
    categories = df.select(base_col).distinct().rdd.flatMap(lambda x: x).collect()  # noqa
    return categories


def generate_binary_variables(df, base_col, category):
    categories_exprs = [F.when(F.col(base_col) == category, 1).otherwise(0).alias(base_col+"_"+category.replace(" ", "_"))]# for category in categories]
    return categories_exprs