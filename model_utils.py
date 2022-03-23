import pandas as pd

def get_sla_quantile(df):
    def decile_to_quantile(decile):
        if decile < 1:
            return 0
        elif decile <= 2:
            return 0
        elif decile <= 4:
            return 1
        elif decile <= 6:
            return 2
        elif decile <= 8:
            return 3
        else:
            return 4
            
    sla_decile = 'sla_2011_adv_dis_decile'
    df.loc[:,sla_decile] = df[sla_decile].fillna(0)

    df['sla_2011_q1'] = ((df[sla_decile] == 1)|(df[sla_decile] == 2)).astype(int)
    df['sla_2011_q2'] = ((df[sla_decile] == 3)|(df[sla_decile] == 4)).astype(int)
    df['sla_2011_q3'] = ((df[sla_decile] == 5)|(df[sla_decile] == 6)).astype(int)
    df['sla_2011_q4'] = ((df[sla_decile] == 7)|(df[sla_decile] == 8)).astype(int)
    df['sla_2011_q5'] = ((df[sla_decile] == 9)|(df[sla_decile] == 10)).astype(int)
    df['sla_2011_quantile'] = df[sla_decile].apply(decile_to_quantile)
    return df


def merge_scifa_index(df, df_scifa):
    df.loc[:,'SLA_2011_CODE'] = df['SLA_2011_CODE'].str.replace('\..*','')
    df.loc[:,'SLA_2011_CODE'] = df['SLA_2011_CODE'].str.replace('X9.*','-1')
    df.loc[:,'SLA_2011_CODE'] = df['SLA_2011_CODE'].str.replace(' ','-1')
    df.loc[:,'SLA_2011_CODE'] = df['SLA_2011_CODE'].fillna(0).astype(int)
    df = pd.merge(df, df_scifa, left_on='SLA_2011_CODE', right_on='sla_2011_code', how='left')
    return df


def patient_insurance_features(df):
    '''
    See APDC data dictionary for details of how paytment status on separation is coded.
    '''
    df.loc[:,'public_pt'] = ((df['payment_status_on_sep']<=25)|(df['payment_status_on_sep'].isin([45,46,52,60]))).astype(int)
    private_pt = (df['payment_status_on_sep']<=39) & (df['payment_status_on_sep']>=30)
    other_pt = df['payment_status_on_sep'].isin([40, 41, 42, 50, 51, 55])
    df.loc[:,'private_pt'] = (private_pt | other_pt).astype(int)
    return df
