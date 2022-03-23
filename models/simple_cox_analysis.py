import argparse
import yaml
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold

from model_utils import get_sla_quantile, merge_scifa_index, patient_insurance_features


def analyze(df, model, fout=None):
    cph = CoxPHFitter()
    cph.fit(df, duration_col='duration', event_col='observed', formula=model)
    cph.print_summary(decimals=3)
    if fout:
        cph.summary.to_csv(fout)

    # evaluate model fit
    scores = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(df):
        cph.fit(df.iloc[train], duration_col='duration', event_col='observed', formula=model)
        test_score = cph.score(df.iloc[test], scoring_method='concordance_index')
        scores.append(test_score)

    test_score = np.mean(scores)
    test_std = np.std(scores) 
    print(f'Test: {test_score:.2f} ({test_std:.2f})')


def experiments(df, output_file):
    base_model = 'age_recode + sex'
    analyze(df, model=base_model, fout='base_'+output_file+'.csv')

    # model 1 -> clinical factors
    clinical_adjustment = 'cardioversion + charlson_congestive_heart_failure + hypertension' 
    clinical_adjustment += '+ diabetes + stroke + charlson_myocardial_infarction + coronary_artery'
    clinical_adjustment += '+ major_bleeding'
    adjustment_cols = [i.strip() for i in clinical_adjustment.split('+')]
    df.loc[:,adjustment_cols] = df[adjustment_cols].astype(bool).astype(int)

    adjusted_model = base_model + ' + ' + clinical_adjustment
    analyze(df, model=adjusted_model, fout='clinical_'+output_file+'.csv')

    # model 2 -> sociodemographic factors
    non_clinical_adjustment = ' year + private_pt + sla_2011_quantile'
    #non_clinical_adjustment += '+ sla_2011_q2 + sla_2011_q3 + sla_2011_q4 + sla_2011_q5'
    adjusted_model = base_model + ' + ' + non_clinical_adjustment
    analyze(df, model=adjusted_model, fout='nonclinical_'+output_file+'.csv')

    # model 3 -> all - clinical and non-clinical
    adjusted_model = f'{base_model} + {clinical_adjustment} + {non_clinical_adjustment}'
    analyze(df, model=adjusted_model, fout='all_'+output_file+'.csv')


def main():
    '''
    Fits four Cox models using (1) sex and age, (2) clinical factors, 
    (3) sociodemographic factors, and (4) all variables.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, help='output file to save summary results')
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.safe_load(fin)
    args = parser.parse_args()

    np.random.seed(123)

    df = pd.read_csv(config['survival_data'], low_memory=False)
    df = df[df['year'] > 2008]

    # remove people who died on index admission
    dead_mask = (df.observed==0) & (df.duration < 0.01)
    df = df[~dead_mask].copy()

    # get socioeconomic status based on residence area
    df_scifa = pd.read_csv(config['scifa'], dtype='Int64')
    # merge with patient features
    df = merge_scifa_index(df, df_scifa)
    # stratify socioeconomic index into quantiles
    df = get_sla_quantile(df)

    df.loc[:,'cardioversion'] = df['cardioversion'].fillna(0)

    # get private insurance features
    df = patient_insurance_features(df)

    experiments(df, args.output_file)


if __name__ == '__main__':
    main()
