import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import statsmodels.api as sm

from model_utils import get_sla_quantile, merge_scifa_index, patient_insurance_features


def analyze(df, model, fout=None):
    x_cols = [col.strip() for col in model.split('+')]
    X = df[x_cols]
    Y = df[['Y']]

    log_reg = sm.Logit(Y, X).fit()
    print(log_reg.summary())
    if fout:
        with open(fout, 'w') as f:
            f.write(log_reg.summary().as_csv())

    # evaluate model fit
    train_auc_scores = []
    train_accuracy = []
    test_auc_scores = []
    test_accuracy = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(df):
        log_reg = sm.Logit(Y.iloc[train], X.iloc[train]).fit()
        proba_ = log_reg.predict(X.iloc[test])
        test_auc_scores.append(metrics.roc_auc_score(Y.iloc[test], proba_))
        predictions = proba_ > 0.5
        test_accuracy.append(metrics.accuracy_score(Y.iloc[test], predictions))

        proba_ = log_reg.predict(X.iloc[train])
        train_auc_scores.append(metrics.roc_auc_score(Y.iloc[train], proba_))
        predictions = proba_ > 0.5
        train_accuracy.append(metrics.accuracy_score(Y.iloc[train], predictions))

    print(f'Train AUC: {np.mean(train_auc_scores):.2f} ({np.std(train_auc_scores):.2f})')
    print(f'Train acc: {np.mean(train_accuracy):.2f} ({np.std(train_accuracy):.2f})')

    print(f'Test AUC: {np.mean(test_auc_scores):.2f} ({np.std(test_auc_scores):.2f})')
    print(f'Test acc: {np.mean(test_accuracy):.2f} ({np.std(test_accuracy):.2f})')


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
    adjusted_model = base_model + ' + ' + non_clinical_adjustment
    analyze(df, model=adjusted_model, fout='nonclinical_'+output_file+'.csv')

    # model 3 -> all - clinical and non-clinical
    adjusted_model = f'{base_model} + {clinical_adjustment} + {non_clinical_adjustment}'
    analyze(df, model=adjusted_model, fout='all_'+output_file+'.csv')


def main():
    '''
    Script that builds a logistic regression with early ablation as the outcome.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('survival_data', type=str, help='files with survival data')
    parser.add_argument('--output_file', type=str, help='output file to save summary results')
    args = parser.parse_args()

    np.random.seed(123)

    df = pd.read_csv(args.survival_data, low_memory=False)

    df = df[df.observed == 1].copy()
    df['Y'] = (df.duration <= 12).astype(int)
    df = df[df['year'] > 2008]

    # get socioeconomic status based on residence area
    df_scifa = pd.read_csv('sla_2011_scifa_index.csv', dtype='Int64')
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
