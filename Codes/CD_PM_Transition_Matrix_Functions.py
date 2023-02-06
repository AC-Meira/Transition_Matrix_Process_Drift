# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
from functools import reduce

import scipy.stats as ss
from scipy.stats import mannwhitneyu, wilcoxon, ranksums, power_divergence, ks_2samp, kstest, chisquare, ttest_rel
from scipy.stats import ttest_ind, spearmanr, pearsonr, cramervonmises_2samp, entropy
from scipy.stats.contingency import association
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

# from source import CD_PM_Transition_Matrix_Functions as tm_f


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    """
    chi2 = ss.chi2_contingency(confusion_matrix, correction=True, lambda_='log-likelihood')[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   Transition Matrix                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_TM_CF_representation(log_transition, threshold_anomaly=0): 
    
    ### Get transition frequencies and probabilities
    
    # MAkes any necessary preparation in the log_transition
    log_prep = log_transition
    
    # Get transition percentuals
    transition_matrix_perc = pd.crosstab(log_prep["From"], log_prep["To"], normalize='all').stack().reset_index().rename(columns={0: "Perc"})
    
    # Get transition frequencies
    transition_matrix_freq = pd.crosstab(log_prep["From"], log_prep["To"], normalize=False).stack().reset_index().rename(columns={0: "Freq"})
    
    # Get transition probabilities
    transition_matrix_prob = pd.crosstab(log_prep["From"], log_prep["To"], normalize='index').stack().reset_index().rename(columns={0: "Prob"})
    
    # Merge all transition matrices
    dfs_to_merge = [transition_matrix_perc, transition_matrix_freq, transition_matrix_prob]
    transition_matrix = reduce(lambda  left,right: pd.merge(left, right, on=['From', 'To'], how='outer'), dfs_to_merge).fillna(0)
    transition_matrix = transition_matrix.sort_values(by=['From', 'To'], ascending=[True, True]).reset_index(drop=True).set_index(['From','To'])
    
    ### Remove transitions with less percentage than threshhold (anomalies)
    if threshold_anomaly < 1 :
        transition_matrix.iloc[transition_matrix['Perc'] <= threshold_anomaly] = 0
    else:
        transition_matrix.iloc[transition_matrix['Freq'] <= threshold_anomaly] = 0
        
    ### Alpha relations
    
    # Direct succession: x>y if for some case x is directly followed by y
    transition_matrix['Direct_succession'] = np.where(transition_matrix['Perc']>0, 1, 0)
    
    # Opposite direction: if y>x
    transition_matrix_inverted = transition_matrix.reset_index()[['From','To','Direct_succession']]
    transition_matrix_inverted.columns = ['To','From', 'Opposite_direction']
    transition_matrix = pd.merge(transition_matrix,transition_matrix_inverted.set_index(['From','To']),on=['From', 'To'], how='left')

    # Causality: x→y if x>y and not y>x
    transition_matrix['Causality'] = np.where((transition_matrix['Direct_succession']==1) & (transition_matrix['Opposite_direction']==0), 1, 0)
    
    # Parallel: x||y if x>y and y>x
    transition_matrix['Parallel'] = np.where((transition_matrix['Direct_succession']==1) & (transition_matrix['Opposite_direction']==1), 1, 0)
    
    # Choice: x#y if not x>y and not y>x and not x--->y
    transition_matrix['Choice'] = np.where((transition_matrix['Direct_succession']==0) & (transition_matrix['Opposite_direction']==0), 1, 0)
    

    return transition_matrix.fillna(0)
    



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                  Measures genarator                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def run_offline_window_method(log_df, window_size=500, sliding_step=250, window_ref_mode='static'):
    
    # Check if log_df is not null
    if len(log_df) > 0:
        pass
    else:
        return []

    
    # Define measures and weight variables
    # all measures: 'Perc', 'Freq', 'Prob', 'Direct_succession', 'Opposite_direction' , 'Causality', 'Parallel', 'Choice'
    measures = ['Perc', 'Freq', 'Prob', 'Direct_succession', 'Opposite_direction' , 'Causality', 'Parallel', 'Choice']
    weight = ['Freq']

    # Create final dataset
    result = pd.DataFrame() #columns=['init', 'end']+measures


    # Setting looping to simulate window method
    if sliding_step > 0 :
        # loop = range(0, log_df['Trace_order'].nunique() - window_size + 1, sliding_step)
        loop = range(0, len(log_df), sliding_step)
    else:
        loop = range(0, len(log_df), window_size)
        # loop = range(0, log_df['Trace_order'].nunique(), window_size)
      

    ### Loop
    for i in loop:
        # print("Window from ", i, " to ", i+window_size)
        
        ### Get log_df in loop window
        log_df_window = log_df[i : i+window_size]
        # log_df_window = log_df[log_df['Trace_order'].between(i, i+window_size)]
        # print("Lenght events in window: ", len(log_df_window))
        
        ### Check if window is complete
        if len(log_df_window) == window_size :
            
            # --------------------------------------------#
            # Get Transition Matrix Representation        #
            # --------------------------------------------#
            representation_curr = get_TM_CF_representation(log_df_window, 0)[measures] #+weight
            
            
            ### Get difference from past representation (if it exists)
            if i > 0:
                
                ### window reference moving test
                if window_ref_mode=='moving' and i>=window_size:
                    # Set the currunt window as the reference window
                    log_df_window_ref = log_df[i-window_size : i]
                    representation_ref = get_TM_CF_representation(log_df_window_ref, 0)[measures]
                    # print('representation_ref - init: ', i-window_size, ', end:', i)
                               
                
                ### Get the difference between the reference window and the current moving window - Delta Matrix
                difference_df = abs(representation_ref.subtract(representation_curr, fill_value=0))#.drop(columns=weight)
                
                ### Aggregate all differences - Delta Vector
                diff_agg = (abs(difference_df)+0).sum()
                
                ### Get the proportion of difference to maximum difference possible
                max_difference = representation_ref.sum() + representation_curr.sum()
                prop_max_difference = diff_agg/max_difference
                prop_max_difference = prop_max_difference.add_suffix("_prop")
                diff_agg = pd.concat([diff_agg, prop_max_difference])
                
                ### Divide the difference by 2 to get the real value of changing, not difference. 
                # difference_df = difference_df.divide(2, fill_value=0)
                
                ### Apply a weight in the difference
                diff_agg['Prob_Freq'] =  (abs(difference_df['Prob'].multiply(difference_df['Freq'].values, axis="index", fill_value=0))+0).sum()
                diff_agg['Prob_Freq_prop'] = diff_agg['Prob_Freq']/max_difference['Freq']
                # difference_df = difference_df.drop(columns=['Prob', 'Freq'])
                # difference_df = difference_df.multiply(difference_df[weight].values, axis="index", fill_value=0)#.drop(columns=weight)
    

                ### Add some statistical hypothesis tests

                # ## Contingency table
                contingency_table_prob = pd.merge(pd.Series(representation_ref['Prob'].multiply(representation_ref['Freq'].values, axis="index", fill_value=0), name='Prob_Freq') 
                                                  , pd.Series(representation_curr['Prob'].multiply(representation_curr['Freq'].values, axis="index", fill_value=0), name='Prob_Freq') 
                                                  , left_index=True, right_index=True, how='outer', suffixes=('_ref', '_curr')).fillna(0).astype(int)
                # contingency_table_prob = pd.merge(representation_ref['Prob'], representation_curr['Prob'], left_index=True, right_index=True, how='outer', suffixes=('_ref', '_curr')).fillna(0)
                contingency_table_prob = contingency_table_prob.loc[(contingency_table_prob!=0).any(axis=1)]
                contingency_table_prob += 5
                
                # contingency_table_prob2 = pd.merge(representation_ref['Prob'], representation_curr['Prob'], left_index=True, right_index=True, how='outer', suffixes=('_ref', '_curr')).fillna(0)
                # contingency_table_prob2 = contingency_table_prob2.loc[(contingency_table_prob2!=0).any(axis=1)]
                
                diff_agg['CramersV_prob'] = cramers_corrected_stat(contingency_table_prob)
                
                ## Chi2 or G-test (add: lambda_='log-likelihood' in chi2_contingency)
                diff_agg['Chi2_prob'] = ss.chi2_contingency(contingency_table_prob, correction=True, lambda_='log-likelihood')[1]
                diff_agg['Chi2_stat_prob'] = ss.chi2_contingency(contingency_table_prob, correction=True, lambda_='log-likelihood')[0]
                diff_agg['Chi2_bool_prob'] = 1 if diff_agg['Chi2_prob'] <= 0.025 else 0
                diff_agg['Chi2_Cramer_bool_prob'] = 1 if (diff_agg['Chi2_prob'] <= 0.025 and diff_agg['CramersV_prob'] > 0.05) else 0
                # diff_agg['Wilcoxon_prob'] = wilcoxon(contingency_table_prob['Prob_ref'], contingency_table_prob['Prob_curr'])[1]
                
                ## Contingency table
                contingency_table = pd.merge(representation_ref['Freq'], representation_curr['Freq'], left_index=True, right_index=True, how='outer', suffixes=('_ref', '_curr')).fillna(0).astype(int)
                contingency_table = contingency_table.loc[(contingency_table!=0).any(axis=1)]
                contingency_table += 5
                
                
                ## Cramer's V corrected
                diff_agg['CramersV'] = cramers_corrected_stat(contingency_table)
                
                ## Chi2 or G-test (add: lambda_='log-likelihood' in chi2_contingency)
                diff_agg['Chi2'] = ss.chi2_contingency(contingency_table, correction=True, lambda_='log-likelihood')[1]
                diff_agg['Chi2_stat'] = ss.chi2_contingency(contingency_table, correction=True, lambda_='log-likelihood')[0]
                diff_agg['Chi2_bool'] = 1 if diff_agg['Chi2'] <= 0.025 else 0
                diff_agg['Chi2_Cramer_bool'] = 1 if (diff_agg['Chi2'] <= 0.025 and diff_agg['CramersV'] > 0.05) else 0
                # diff_agg['Wilcoxon'] = wilcoxon(contingency_table['Freq_ref'], contingency_table['Freq_curr'])[1]
                
                
                ### Distances measures
                # diff_agg['Cosine_dist_Prob_Freq'] = distance.cosine(contingency_table_prob['Prob_Freq_ref'], contingency_table_prob['Prob_Freq_curr'])
                
                ### Entropy
                # diff_agg['Entropy_Prob_Freq'] = entropy(contingency_table_prob['Prob_Freq_ref'], contingency_table_prob['Prob_Freq_curr'])
                # diff_agg['Entropy_bool'] = 1 if diff_agg['Entropy'] < 0.05 else 0
                
                ###
                diff_agg['Freq_prop_bool'] = 1 if diff_agg['Freq_prop'] > 0.05 else 0
                
                ### Get the initial and final indexes of the current window 
                diff_agg['init'] = i
                diff_agg['end'] = i+window_size
                # print('representation_curr - init: ', i, ', end:', i+window_size)
    
                ### Append all information in the result table
                result = result.append(diff_agg, ignore_index=True)
                
                
            else:
                # Set the currunt window as the reference window
                representation_ref = representation_curr.copy()
                # print('representation_ref - init: ', i, ', end:', i+window_size)
                     
            
    ### Smooth
    result['Chi2_test_window'] = np.where(result['Chi2_bool'].rolling(window=5).mean().fillna(0) > 0.5, 1, 0) 
    result['Chi2_Cramer_test_window'] = np.where(result['Chi2_Cramer_bool'].rolling(window=5).mean().fillna(0) > 0.5, 1, 0) 
    result['Chi2_prob_test_window'] = np.where(result['Chi2_bool_prob'].rolling(window=5).mean().fillna(0) > 0.5, 1, 0) 
    result['Chi2_Cramer_prob_test_window'] = np.where(result['Chi2_Cramer_bool_prob'].rolling(window=5).mean().fillna(0) > 0.5, 1, 0)  
    result['Freq_prop_bool_window'] = np.where(result['Freq_prop_bool'].rolling(window=5).mean().fillna(0) > 0.5, 1, 0)      
                
                
    return result.set_index(['init', 'end'])


