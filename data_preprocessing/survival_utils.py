import re
import numpy as np
import pandas as pd


def condense_codes(df, index, label):
    # concatenate all procedures into procedure_codeP field
    # combine all diagnosis, keep unique codes, count them, and put them all in diagnosis fields 1-50
    # use the rest of the information from the very first record
    _i = index.index[0]
    cols = [f'{label}_code{i}' for i in range(1,51)]
    codes = list(set(filter(pd.notnull, df[cols].fillna('').values.flatten().tolist())))
    num_codes = len(codes)
    if num_codes > 50:
        codes = codes[:49] + [','.join(codes[:50])]
        num_codes = 50

    cols = [f'{label}_code{i}' for i in range(1,num_codes+1)]
    index.loc[_i,cols] = codes

    primary_col = f'{label}_codeP'
    primary_code = list(set(filter(pd.notnull, df[primary_col].tolist())))
    index.loc[_i,primary_col] = ','.join(primary_code)


def filter_transfers(df):
    if len(df) == 1:
        return df
    else:
        # this is a transfer
        index = df.iloc[[0]]

        condense_codes(df, index, 'procedure')
        condense_codes(df, index, 'diagnosis')
        return index


def pseudo_lambda(df, regex):
    return np.array([bool(regex.search(i)) for i in df])


def build_re_patterns():
    chad = [
        # H - hypertension
        ['hypertension', 'I10'],
        ['hypertension_complicated', 'I1[1235]'],

        # D - diabetes
        ['diabetes','E1[0-4]'],

        # S_2 - stroke/TIA/thrombo-embolism
        ['stroke', 'G45,I63,I64,I74'],
        # V - vascular disease:
        # coronary artery disease, myocardial infarction, peripheral artery disease, aortic plaque
        # myocardial infarction
        # use charlson

        # conronary artery disease - same as coronary heart disease
        ['coronary_artery', 'I2[0-2],I24,I25,Z95.1,Z95.5'],
        # peripheral arterial occlusive disease
        ['peripheral_diseases', 'I79.2,I79.8,I70.2,I73.1,I73.8'],
        # atherosclerosis of the aorta [aortic plaque]
        ['atherosclerosis', 'I70.0'],
    ]

    # HAS-BLED
    hasbled = [
        ['kidney', 'N1[7-9]'],
        ['liver', 'K7[0-7]'],
        ['previous_bleeding', 'K25.[024],K26.[024],K27.[024],K28.0'],
        #['labile', 'R79.1'],
        #['alcohol', 'K70,T52,K86.0,E52']
    ]

    # charlson
    charlson = [
        ['charlson_myocardial_infarction', 'I2[12],I25.2'],
        ['charlson_congestive_heart_failure', 'I09.9,I11.0,I13.[02],I25.5,I42.[05-9],I43,I50,P29.0'],
        ['charlson_peripheral_vascular_disease', 'I7[01],I73.[189],I77.1,I79.[02],K55.[189],Z95.[89]'],

        ['charlson_renal_disease', 'I12.0,I13.1,N03.[2-7],N1[89],N25.0,Z49.[0-2],Z94.0,Z99.2'],
        ['charlson_moderate_severe_liver_disease', 'I85.[09],I86.4,I98.2,K70.4,K71.1,K72.[19],K76.[5-7]'],
        ['charlson_mild_liver_disease', 'B18,K70.[0-39],K71.[3-57],K7[34],K76.[02-489],Z94.4'],

        #['charlson_peptic_ulcer_disease', 'K2[5-8].'],
        #['charlson_rheumatic_disease', 'M0[56].,M31.5,M3[2-4].,M35.[13],M36'],
        #['charlson_cerebrovascular_disease', 'G4[56].,H34.0,I6[0-9].'],
        #['charlson_dementia', 'F0[0-3].,F05.1,G30.,G31.1'],
        #['charlson_chronic_pulmonary_disease', 'I27.[89],J4[0-7].,J6[0-7].,J68.4,J70.[13]'],
        #['charlson_diabetes_uncomplicated', 'E10.[01689],E11.[01689],E12.[01689],E13.[01689],E14.[1689]'],
        #['charlson_diabetes_complicated', 'E10.[2-57],E11.[2-57],E12.[2-57],E13.[2-57],E14.[2-57]'],
        #['charlson_hemiplegia_paraplegia', 'G04.1,G11.4,G80.[12],G81.,G82.,G83.[0-49]'],
        #['charlson_malignancy', 'C[0-2][0-6].,C3[0-4].,C3[7-9].,C4[0135-9].,C5[0-8].,C6[0-9].,C7[0-6].,C8[1-58].,C9[0-7].'],
        #['charlson_metastatic_solid_tumour', 'C[7-9].,C80.'],
        #['charlson_aids_hiv', 'B2[0-24].']]
        ]

    procedures = [
        ['cardioversion', '13400-00'],
    ]

    other = [ ['major_bleeding','I85.0,I98.3,K2[5-8].[0-24-6],K62.5,K92.2,D62.9'],
            ['chronic_kidney_disease', 'N18'],
            ['chronic_liver_disease', 'K76.9'],
            ]
    # major bleedling I85.0, I98.3, K25-28[0-2 + 4-6], K62.5,K92.2,D62.9
    # chronic kidney disease N18
    # chronic liver disease k76.9

    feature_sets = chad + hasbled + charlson + procedures + other

    feature_extraction_re = []
    for name,codes in feature_sets:
        split_codes = [c.replace('.','\.') for c in codes.split(',')]
        regex = '|'.join([f'(?:{c})' for c in split_codes])
        r = re.compile(regex)
        feature_extraction_re.append((name,r))

    return feature_extraction_re


def get_pt_features(df, feature_re):
    features = {}

    diagnosis_cols = ['diagnosis_codeP'] + [f'diagnosis_code{i}' for i in range(1,51)]
    proc_cols = ['procedure_codeP'] + [f'procedure_code{i}' for i in range(1,51)]

    df_temp = df[diagnosis_cols + proc_cols].astype(str)
    df_flat = df_temp.values.flatten()

    for feature_name, regex in feature_re:
        features[feature_name] = pseudo_lambda(df_flat, regex).sum()

    return features


def cha2ds2vasc(patient):
    score = 0

    if patient['age_recode'] >= 65 and patient['age_recode'] <75:
        score += 1
    elif patient['age_recode'] >= 75:
        score += 2

    if patient['sex'] == 1:
        score += 1

    if patient['charlson_congestive_heart_failure']:
        score += 1

    if patient['hypertension']:
        score += 1

    if patient['stroke']:
        score += 2

    if patient['charlson_myocardial_infarction'] or patient['coronary_artery'] or patient['peripheral_diseases'] or patient['atherosclerosis']:
        score += 1

    if patient['diabetes']:
        score += 1

    return score
