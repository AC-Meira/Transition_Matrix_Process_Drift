# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import TMPD_utils
import TMPD_process_features
import TMPD_change_features
import TMPD_detection_tasks
import pm4py

from tqdm.notebook import tqdm_notebook
import time

import ruptures as rpt
from ruptures.metrics import precision_recall, meantime

from pm4py.objects.log.importer.xes import importer as xes_importer


class TMPD():

    """Transition Matrix Process Drift.
    Class for deal with Process Drift (or Concept Drift) in Process Mining using transition matrices as a unified data structure. 
    """

    def __init__(self, scenario='offline'):

        """Initialize a Pelt instance.
        Args:
            scenario (str, optional): Data Scenario. Online (Stream) or offline (Batch). 
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
        """

        self.scenario = scenario
  

    def set_transition_log(self, event_log_original, case_id, activity_key, timestamp_key, other_columns_keys=[]):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        self.case_id = case_id
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.other_columns_keys = other_columns_keys
        self.event_log = event_log_original.copy()
        

    def run_transition_log(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """
        
        event_log = self.event_log.copy()

        event_log = event_log[[self.case_id, self.activity_key, self.timestamp_key] + self.other_columns_keys]

        event_log = event_log.rename(columns={self.case_id:'case_id', self.activity_key:'activity', self.timestamp_key:'timestamp'})

        transition_log = pd.concat([event_log[['case_id']], event_log.add_suffix('_from'), event_log.groupby('case_id').shift(-1).add_suffix('_to')], axis=1).drop(columns=['case_id_from'])
        
        transition_log = transition_log.dropna(subset = ['activity_to'])

        transition_log = transition_log.reset_index(names='original_index') 

        transition_log["transition_id"] = transition_log.index
        
        transition_log['case_order'] = transition_log.groupby('case_id').cumcount()

        self.transition_log = transition_log


    def get_transition_log(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        return self.transition_log

        

    def set_windowing_strategy(self, window_size_mode = "Fixed", window_size = None, window_size_max = None, window_size_min = None, window_ref_mode = "Fixed"
        , overlap = True, sliding_step = None, continuous = True, gap_size = None):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """
        
        self.window_size_mode = window_size_mode
        self.window_size = window_size
        self.window_size_max = window_size_max
        self.window_size_min = window_size_min
        self.window_ref_mode = window_ref_mode
        self.overlap = overlap
        self.sliding_step = sliding_step
        self.continuous = continuous
        self.gap_size = gap_size


    def run_windowing_strategy(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        windows_index = pd.DataFrame()
        # Getting the windows index
        if self.window_size_mode == 'Fixed':
            
            if self.overlap == False:
                if self.continuous == True:
                    windows_index['start'] = range(0, len(self.transition_log), self.window_size)
                else:
                    windows_index['start'] = range(0, len(self.transition_log), self.window_size + self.gap_size)
            else:
                windows_index['start'] = range(0, len(self.transition_log), self.sliding_step)

            windows_index['end'] = windows_index['start'] + self.window_size

            # Get only windows completed
            windows_index = windows_index[windows_index['end'] <= len(self.transition_log)]

        # else:
        #     # TODO

        # Add window index number
        windows_index['window_index'] = windows_index.index

        self.windows_index_dict = windows_index.to_dict('index') 


    def get_windowing_strategy(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        return self.windows_index_dict
        # return 'window_size_mode: ' + self.window_size_mode + ', window_size: ' + str(self.window_size) + ', window_size_max: ' + str(self.window_size_max) + \
        #     ', window_size_min: ' + str(self.window_size_min) + ', window_ref_mode: ' + self.window_ref_mode + ', overlap: ' + self.overlap + ', sliding_step: ' + \
        #         str(self.sliding_step) + ', continuous: ' + self.continuous + ', gap_size: ' + str(self.gap_size)



    def set_process_representation(self, threshold_anomaly=0
                                    , control_flow_features=['frequency', 'probability', 'causality', 'parallel', 'choice']
                                    , time_features={}
                                    , resource_features={}
                                    , data_features={}): 

        """Instanciate a Transition Matrix (TM) Data Structure as a representation of the process using defined features list. 
        Args:
            threshold_anomaly (int, optional): Filter for anomaly transitions (few frequency). If less than 1, is considered a percentage threshold, otherwise is considered a frequency threshold.
            features (list, optional): List of features to be used. Default features list is ['frequency', 'probability']. Possible features are: 
                'frequency'
                'probability'
                'causality'
                'parallel'
                'choice'
        """
        self.threshold_anomaly = threshold_anomaly
        self.control_flow_features = control_flow_features
        self.time_features = time_features
        self.resource_features = resource_features
        self.data_features = data_features


    def run_process_representation(self, transition_log_sample_original):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        transition_log = transition_log_sample_original.copy()
        
        # Create a standard Transition Matrix (TM) process representation using frequency and percentual features for anomaly filter
        process_representation_df = pd.crosstab(transition_log["activity_from"], transition_log["activity_to"], normalize=False).stack().reset_index().rename(columns={0: "frequency"})
        process_representation_df["percentual"] = process_representation_df["frequency"]/process_representation_df["frequency"].sum()
        process_representation_df = process_representation_df.sort_values(by=['activity_from', 'activity_to'], ascending=[True, True]).set_index(['activity_from','activity_to'])

        # Remove transitions with less percentage or frequency than threshhold (anomaly filter)
        if self.threshold_anomaly < 1 :
            process_representation_df.iloc[process_representation_df['percentual'] <= self.threshold_anomaly] = 0
        else:
            process_representation_df.iloc[process_representation_df['frequency'] <= self.threshold_anomaly] = 0

        # Loop to add all other features from the arg list.
        process_features_dict = {
            "frequency": process_representation_df[["frequency"]]
            , "percentual": process_representation_df[["percentual"]]
        }

        for control_flow_feature in [other_control_flow_features for other_control_flow_features in self.control_flow_features if other_control_flow_features not in ('frequency', 'percentual')]:
            try:
                process_features_dict[control_flow_feature] = getattr(TMPD_process_features, "get_feature_" + control_flow_feature)(process_representation_df, control_flow_feature)
            except AttributeError as e:
                print("Unknown feature: ", control_flow_feature)
                print("Error: ", e)

        for time_feature in self.time_features:
            try:
                process_features_dict[time_feature] = getattr(TMPD_process_features, "get_feature_" + time_feature)(process_representation_df, transition_log, time_feature, self.time_features[time_feature])
            except AttributeError as e:
                print("Unknown feature: ", time_feature)
                print("Error: ", e)

        # Merge all features transition matrices
        process_representation_df = reduce(lambda  left,right: pd.merge(left, right, on=['activity_from', 'activity_to'], how='outer'), process_features_dict.values()).fillna(0)

        # Keep only defined features
        self.process_representation_df = process_representation_df[self.control_flow_features + list(self.time_features.keys()) + list(self.resource_features.keys()) + list(self.data_features.keys())]


    def get_process_representation(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        return self.process_representation_df


    def set_change_representation(self
        , change_features_strategy_dict = {
        'delta_matrix_strategy': 
            {
                'frequency_delta' : {'process_feature':'frequency', 'method':'aggregation', 'agg_function' : 'sum'}
                , 'probability_delta' : {'process_feature':'probability', 'method':'aggregation', 'agg_function' : 'sum'}
                , 'frequency_delta_percentage' : {'process_feature':'frequency', 'method':'percentage'}
                , 'prob_freq_delta_weight' : {'process_feature':'probability', 'method':'aggregation_weight', 'agg_function' : 'sum', 'weight_feature' : 'frequency'}
            }
        , 'statistic_test_strategy' : 
            {
                'frequency_gtest_pvalue' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
                , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
            }
        }
    ):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        self.change_features_strategy_dict = change_features_strategy_dict
        


    def run_change_representation(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        # Initiating the change representation
        change_representation_dict = self.windows_index_dict.copy()

        # for window_index, window in self.windows_index.iterrows():
        for window_index, window_info in self.windows_index_dict.items():

            # Run and get the process representation with window sample
            self.run_process_representation(self.transition_log.iloc[window_info['start'] : window_info['end']])
            process_representation_detection_window_df = self.get_process_representation()

            # Check if it's the first window
            if window_index > 0:

                # Add information about which window was used as reference
                change_representation_dict[window_index].update({'reference_window_index' : str(reference_window_index)})
                
                # Loop to call the change features methods - Delta Vector
                for change_feature_strategy, change_feature_params_dict in self.change_features_strategy_dict.items():
                    try:
                        # Call methods functions and update results in dict
                        change_representation_dict[window_index].update(getattr(TMPD_change_features, "get_" + change_feature_strategy)(process_representation_reference_window_df
                                                                                                                                            , process_representation_detection_window_df
                                                                                                                                            , change_feature_params_dict))

                    except AttributeError as e:
                        print("Error in change feature strategy: ", change_feature_strategy)   
                        print("Error: ", e)
                
                # If window reference mode is Sliding, the detection window become the reference window
                if self.window_ref_mode == "Sliding":
                    process_representation_reference_window_df = process_representation_detection_window_df.copy()
                    reference_window_index = window_index

            else:
                process_representation_reference_window_df = process_representation_detection_window_df.copy()
                reference_window_index = window_index

        # Merge all features
        self.change_representation_df = pd.DataFrame.from_dict(change_representation_dict, orient='index') #.fillna(0)


    def get_change_representation(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        return self.change_representation_df


    def set_detection_task(self
        , detection_task_strategy_dict = {
            'time_series_strategy': 
            {
                'cpd_frequency_delta' : {'change_features':['frequency_delta'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
                , 'cpd_prob_freq_delta' : {'change_features':['prob_freq_delta_weight'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
            }
            , 'threshold_strategy' : 
            {
                'gtest_frequency' : {'change_features':['frequency_gtest_pvalue'], 'method':'comparison_operator', 'operator' : 'le', 'threshold_value' : '0.025', 'smooth' : '3'}
                , 'fixed_frequency_delta_percentage' : {'change_features':['frequency_delta_percentage'], 'method':'comparison_operator', 'operator' : 'ge', 'threshold_value' : '0.05', 'smooth' : '3'}
            }
        }
        ):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        self.detection_task_strategy_dict = detection_task_strategy_dict


    def run_detection_task(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        # Initiating the detection task result
        detection_task_result_dict = {}

        # Loop to call the change features methods - Delta Vector
        for detection_task_strategy, detection_task_params_dict in self.detection_task_strategy_dict.items():
            try:
                # Call methods functions and update results in dict
                detection_task_result_dict[detection_task_strategy] = getattr(TMPD_detection_tasks, "get_" + detection_task_strategy)(self.change_representation_df, detection_task_params_dict)

            except AttributeError as e:
                print("Error in detection task strategy: ", detection_task_strategy)  
                print("Error: ", e)

        # Prepare result dict to dataframe
        detection_task_result_df = pd.DataFrame.from_dict(detection_task_result_dict, orient='index')
        detection_task_result_df = detection_task_result_df.reset_index(names='detection_strategy').melt(id_vars=['detection_strategy'], var_name='detection_feature', value_name='detection_results')

        self.detection_task_result_df = detection_task_result_df.dropna(axis=0).reset_index(drop=True)


    def get_detection_task(self):

        """... 
        Args:
            transition_log (DataFrame): Event log as Pandas Dataframe. 
        """

        return self.detection_task_result_df
