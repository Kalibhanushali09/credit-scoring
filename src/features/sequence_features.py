import pandas as pd
import numpy as np

def build_bureau_features(bureau_df):
    df = bureau_df.fillna(0)   
    bureau_agg = df.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['count'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE':['sum','max'],
        'AMT_CREDIT_MAX_OVERDUE':['max'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum']
        
    })
    bureau_agg.columns = ['_'.join(col) for col in bureau_agg.columns]
    bureau_agg = bureau_agg.reset_index()
    
    return bureau_agg


'''installments_payments.csv has 13 million rows — multiple rows per borrower. One row per payment made. Borrower 161674 might have 50 rows, one for each monthly payment they made on previous loans.
But your model needs one row per borrower to match application_train.csv. So you need to squash those 50 rows into a single summary row. That's what groupby().agg() does.'''



def build_installment_features(installment_df):
    df = installment_df.fillna(0)
    df['DAYS_LATE'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['PAYMENT_RATIO'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT'].replace(0, 1)

    installment_agg = df.groupby('SK_ID_CURR').agg({
        'DAYS_LATE': ['mean', 'max'],
        'PAYMENT_RATIO': ['mean', 'min'],
        'AMT_INSTALMENT': ['count'],  
    })
    
    installment_agg.columns = ['_'.join(col) for col in installment_agg.columns]

    installment_agg = installment_agg.reset_index()
    
    return installment_agg


def build_previous_app_features(prev_df):
    df = prev_df.fillna(0)

    df['TRUST_RATIO']=df['AMT_CREDIT']/df['AMT_APPLICATION'].replace(0,1)

    prev_agg=df.groupby('SK_ID_CURR').agg({
        'AMT_DOWN_PAYMENT':['max'],
        'CNT_PAYMENT':['count'],
        'TRUST_RATIO': ['mean']
    })

    refused_count = df[df['NAME_CONTRACT_STATUS'] == 'Refused'] \
        .groupby('SK_ID_CURR').size() \
        .reset_index(name='PREV_REFUSED_COUNT')

    prev_agg.columns = ['_'.join(col) for col in prev_agg.columns]

    prev_agg = prev_agg.reset_index()
    prev_agg = prev_agg.merge(refused_count, on='SK_ID_CURR', how='left')
    prev_agg['PREV_REFUSED_COUNT'] = prev_agg['PREV_REFUSED_COUNT'].fillna(0)
    
    return prev_agg


def build_credit_card_bal_app_features(credits_df):
    df = credits_df.fillna(0)

    df['CREDIT_RATIO']=df['AMT_BALANCE']/df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0,1)

    cc_agg = df.groupby('SK_ID_CURR').agg({
        'CREDIT_RATIO': ['mean'],      # derived: AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
        'SK_DPD': ['max', 'mean'],           # days past due — worst and typical
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'sum'],  # cash advances
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean'],        # payment behaviour
        'MONTHS_BALANCE': ['count'],         # length of history
    })
    cc_agg.columns = ['_'.join(col) for col in cc_agg.columns]
    cc_agg = cc_agg.reset_index()
    
    return cc_agg


def build_bureau_bal_app_features(bureau_bal_df):
    df = bureau_bal_df.fillna(0)

    df['STATUS_OVERDUE']=df['STATUS'].isin(['1','2','3','4','5']).astype(int)
    bb_agg = df.groupby('SK_ID_BUREAU').agg({       
        'MONTHS_BALANCE': ['count'],
        'STATUS_OVERDUE': ['sum', 'mean']        
    })
    bb_agg.columns = ['_'.join(col) for col in bb_agg.columns]
    
    bb_agg = bb_agg.reset_index()
    return bb_agg


def build_sequences(inst_df, max_len=50):
    df = inst_df.fillna(0)
    df['DAYS_LATE'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['PAYMENT_RATIO'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT'].replace(0, 1)

    sequences = []
    ids = []
    
    for sk_id, group in df.groupby('SK_ID_CURR'):
        group = group.sort_values('DAYS_INSTALMENT').tail(max_len)
        data=group[['DAYS_LATE','PAYMENT_RATIO','AMT_PAYMENT']].values
        seq=np.zeros((max_len,3))
        seq[-len(data):]=data
        sequences.append(seq)
        ids.append(sk_id)
    
    return np.array(sequences), ids