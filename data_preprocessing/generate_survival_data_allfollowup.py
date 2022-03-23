import argparse
import glob
import logging
import re
import yaml
import pandas as pd
import numpy as np

from survival_utils import build_re_patterns
from survival_utils import filter_transfers
from survival_utils import get_pt_features
from survival_utils import cha2ds2vasc

# for profiling
#import cProfile, pstats, io

valvular_disease_re = re.compile('(I05.[01289])|(I34.[01289])|(Q23.[1389])')
def check_valvular(df):
    df_flat = df.astype(str).values.flatten()
    return np.array([bool(valvular_disease_re.search(i)) for i in df_flat]).any()


valve_replacement_re = re.compile('(38488-0[239])|(38489-02)')
def check_valve_replacement(df):
    df_flat = df.astype(str).values.flatten()
    return np.array([bool(valve_replacement_re.search(i)) for i in df_flat]).any() 


cardioversion_re = re.compile('13400-00')
def check_cardioversion(df):
    df_flat = df.astype(str).values.flatten()
    return np.array([bool(cardioversion_re.search(i)) for i in df_flat]).any() 


any_af_re = re.compile('I48')
def check_any_af(df):
    df_flat = df.astype(str).values.flatten()
    return np.array([bool(any_af_re.search(i)) for i in df_flat]).any() 
                

ABLATION_CODES = '(38290-01)|(38287-02)'
ablation_re = re.compile(ABLATION_CODES)
def check_ablation(df):
    df_flat = df.astype(str).values.flatten()
    return np.array([bool(ablation_re.search(i)) for i in df_flat]).any() 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_data', metavar='episode_data', type=str, nargs='+',
                        help='patient chunks of episode data from APDC')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--dropna', action='store_true')
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--lookback', type=int, default=1)
    parser.add_argument('--lookahead_min', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config) as fin:
            config = yaml.full_load(fin)

    logging.basicConfig(filename=config['log'], level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)

    diagnosis_cols = ['diagnosis_codeP'] + [f'diagnosis_code{i}' for i in range(1,51)]
    procedure_cols = ['procedure_codeP'] + [f'procedure_code{i}' for i in range(1,51)]

    nrows = args.nrows
    lookback = config['lookback'] if config else args.lookback

    if args.all:
        usecols = None
    else:
        usecols = ['episode_start_date', 'ppn', 'stayseq', 
                    'hospital_type', 'sex', 'emergency_status_recode', 'payment_status_on_sep',
                    'health_insurance_on_admit',
                    'area_identifier',
                    'age_recode',
                    'block_numP',
                    'age_grouping_recode',
                    'marital_status',
                    'indigenous_status',
                    'source_of_referral_recode',
                    ]
        usecols.extend(diagnosis_cols)
        usecols.extend(procedure_cols)

    # regular expressions for diagnosis and procedures of interest
    diagnosis_procedure_re = build_re_patterns()

    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    date_lookback = pd.offsets.DateOffset(months=config['lookback_months'])
    min_date_lookahead = pd.offsets.DateOffset(months=config['min_followup_months'])
    lookahead = -1
    i = 0
    valvular_drop = 0
    any_af_drop = 0
    cardioversion_drop = 0
    ablation_counter = 0
    for fname in sorted(glob.glob(config['data'])):
        logging.info(f'Processing {fname}')
        print(fname)
        df = pd.read_csv(fname, nrows=nrows, dtype='str')
        df['age_recode'] = df['age_recode'].astype('float')
        df['sex'] = pd.to_numeric(df['sex'], downcast='integer')
        df.loc[:,'sex'] = df.sex - 1
        df['stayseq'] = df['stayseq'].astype('int')

        df['start_date'] = pd.to_datetime(df['episode_start_date'])
        df['year'] = df['start_date'].apply(lambda x: x.year)
        df['procedure_codeP'] = df['procedure_codeP'].astype('str')

        patient_group = df.groupby('ppn')
        print('Done grouping', flush=True)

        records = []
        for name,group in patient_group:

            episodes_group = group.groupby('stayseq')
            # condense transfers into a single episode (saving all diagnoses and procedures)
            episodes = episodes_group.apply(filter_transfers).reset_index(drop=True)
            episodes.sort_values('stayseq', inplace=True)

            study_group = episodes
            # look for primary diagnosis of AF
            af_mask = study_group['diagnosis_codeP'].str.contains('I48', na=False)
            af_episodes = study_group[af_mask]

            if af_episodes.empty: 
                continue

            # now safely can check first AF
            index_admission = af_episodes.iloc[0]
            # exclude those with not enough lookback or lookahead data
            if index_admission.start_date < start_date+date_lookback or \
                index_admission.start_date > end_date-min_date_lookahead\
                or index_admission.age_recode < 18:
                continue

            # medical history, include index admission
            history = study_group[(study_group.start_date <= index_admission.start_date) & (study_group.start_date >= index_admission.start_date - date_lookback)]

            # exclude patients with valvular disease
            if check_valvular(history[diagnosis_cols]) or check_valve_replacement(history[procedure_cols]):
                valvular_drop += 1
                continue

            # medical history without index admission
            history_no_index = study_group[(study_group.start_date < index_admission.start_date) & (study_group.start_date >= index_admission.start_date - date_lookback)]
            # exclude patients with any AF diagnosis
            if check_any_af(history_no_index[diagnosis_cols]):
                any_af_drop += 1
                continue

            # exclude patients with cardioversion prior to index AF
            if check_cardioversion(history_no_index[procedure_cols]):
                cardioversion_drop += 1
                continue

            # check if ablation was performed during index admission
            if check_ablation(index_admission[procedure_cols]):
                continue

            # this is the patient's followup data starting after index admission
            follow_up = study_group[study_group.start_date > index_admission.start_date]

            # this covers ablations that occurred at a later point in time after an AF index admission

            # find ablation procedures in follow-up
            ablation = follow_up[procedure_cols].apply(lambda c: c.str.contains(ABLATION_CODES, na=False)).any(axis=1)
            # ablation was never performed, right-censored
            if not ablation.any():
                observed = 0
                duration = (end_date - index_admission.start_date).days / 365 * 12
                ablation_date = None
                ablation_recnum = None
            else:
                ablation_counter += 1
                # ablation was performed during follow-up
                ablation_episode = follow_up[ablation].iloc[0]
                # this contains days of follow-up when ablation occurred
                diff = (ablation_episode.start_date - index_admission.start_date)
                # sometimes this happens
                # it seems to be that the records are out of order, so dates are not sorted, but
                # technically these are coming from strata
                if diff.days < 0:
                    print('debugging ', ablation_episode.start_date, index_admission.start_date)
                    print(follow_up['start_date'])
                    continue

                observed = 1
                duration = diff.days/365 * 12

                ablation_date = ablation_episode.start_date
                ablation_recnum = ablation_episode.recnum

            rec = index_admission.to_dict()
            rec['observed'] = observed
            rec['duration'] = duration
            rec['episodes'] = len(history)
            rec['ablation_date'] = ablation_date
            rec['ablation_recnum'] = ablation_recnum

            # collect patient history using lookback period
            history_features = get_pt_features(history, diagnosis_procedure_re)
            rec.update(history_features)
            rec['cha2ds2vasc'] = cha2ds2vasc(rec)

            records.append(rec)

        print(f'drops af {any_af_drop} valvular {valvular_drop} cardioversion {cardioversion_drop}')
        d = pd.DataFrame(records)
        print(d.describe())
        print(d['duration'][d['duration']<0])
        print(d.head())
        print(len(d))

        header = i == 0
        mode = 'a' if i > 0 else 'w'

        d.to_csv(f'survival_sets/aug_2021_ablation_vs_noablation_{start_date.year}_{end_date.year}_back{lookback}_followup{lookahead}_{nrows}.csv', 
                index=False, mode=mode, header=header)
        i += 1

        logging.info(f'drops af {any_af_drop} valvular {valvular_drop} cardioversion {cardioversion_drop}')
        logging.info(f'Saved {len(d)} rows')
        logging.info(f'Finished processing {fname}')
    print(f'ablation counter {ablation_counter}')


if __name__ == '__main__':
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('cumtime')
    #ps.print_stats()
