# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import scipy.stats as ss

def process_instance(el):
    """
        Process each 'process instance' element from the .mxml file
        and returns as dict
    """
    resp = []
    for entry in el[1:]:
        r = {
            "TraceId": el.get("id")
        }
        for item in entry:
            if item.tag == 'Data':
                r[item.tag] = item[-1].text
            else:
                r[item.tag] = item.text
        resp.append(r)
    return resp

def read_mxml(file):
    """
        Read MXML file into a Pandas DataFrame
    """
    root = ET.parse(file).getroot()
    process = root[-1]
    
    resp = []
    for p_instance in process:
        for r in process_instance(p_instance):
            resp.append(r)
    
    return pd.DataFrame.from_dict(resp)


def cumulative_counting(traces):
    """
        Cumulative counting in column
    """
    t_ant = None
    cnt = 0
    
    resp = []
    for t in traces:
        if t != t_ant:
            cnt += 1
            t_ant = t
        resp.append(cnt)
        
    return(pd.Series(resp) - 1)
    

def parse_mxml(file, aliases=None, replace_whitespaces="_"):
    """
        Runs all basic prep and return preped DataFrame
    """
    df = read_mxml(file)
    
    df["WorkflowModelElement"] = df.WorkflowModelElement.apply(lambda x: x.replace(' ', replace_whitespaces))
    
    if aliases is not None:
        df["Activity"] = df.WorkflowModelElement.replace(aliases)
    else:
        df["Activity"] = df.WorkflowModelElement

    return df

def log_to_transitions(log):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """
    
    ### Get next activity to create a from-to representation
    log_transition = log[['Activity']].rename(columns={"Activity": "From"})
    log_transition['To'] = log.groupby('Trace_order').shift(-1)['Activity']
    
    ### Get trace order identification
    log_transition['Trace_order'] = log['Trace_order']
    
    ### Clean dataset
    # log_prep["To"] = log_prep["To"].fillna('END')
    log_transition = log_transition.dropna(axis=0)
    log_transition = log_transition.reset_index().drop(columns=['index'])
    
    ### Get transition order identification 
    log_transition['Transition_order'] = list(log_transition.index)
    
    return log_transition

