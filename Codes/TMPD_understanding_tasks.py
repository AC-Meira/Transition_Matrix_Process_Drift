# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 17:00:46 2024

@author: Antonio Carlos Meira Neto
"""

import os
import yaml
import ast
import sys
thismodule = sys.modules[__name__]
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as ss
import pm4py
from pm4py.objects.dfg.obj import DFG
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from graphviz import Digraph
from IPython.display import display, Image
from openai import OpenAI
from langchain_openai import ChatOpenAI
import google.generativeai as genai



# Helper function to identify the type of variable
def identify_statistical_test(series):
    """
    Identify the type of variable based on its values.
    - 'proportion' if values are between 0 and 1.
    - 'continuous' otherwise.
    """
    if np.all((series >= 0) & (series <= 1)):
        return 'proportion_test'
    return 'count_test'


# Calculate Cohen's h effect size for proportions.
def cohen_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


# Calculate Cramers V statistic for categorical-categorical association
def cramers_corrected_stat(data, ref_column, det_column, ref_value, det_value):
    # Calculate total counts for ref and det
    total_ref = data[ref_column].sum() - ref_value
    total_det = data[det_column].sum() - det_value

    # Constructing the contingency table
    contingency_table = np.array([[ref_value, total_ref], [det_value, total_det]])

    # Perform Chi-squared test with correction
    chi2 = ss.chi2_contingency(contingency_table, correction=True, lambda_='log-likelihood')[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# Function to perform count test (Chi-squared or Fisher's exact test)
def perform_count_test(merged_windows, row, variable_ref, variable_det):
    freq_reference = row[variable_ref]
    freq_detection = row[variable_det]
    total_freq_reference = merged_windows[variable_ref].sum() - freq_reference
    total_freq_detection = merged_windows[variable_det].sum() - freq_detection
    contingency_table = [[freq_reference, total_freq_reference], [freq_detection, total_freq_detection]]

    if 0 in contingency_table or min(min(contingency_table)) < 5:
        _, p_value = fisher_exact(contingency_table)
    else:
        _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value


# Adjusted function with pseudo-count (also known as Laplace smoothing) for handling zero frequencies
def perform_proportions_test(merged_windows, row, transition, variable_ref, variable_det, pseudo_count):
    prob_reference = row[variable_ref]
    prob_detection = row[variable_det]

    # Check for equal values, return non-significant p-value
    if prob_reference == prob_detection:
        return 1

    total_count_reference = merged_windows.loc[merged_windows['activity_from'] == transition[0], 'frequency_ref'].sum()
    total_count_detection = merged_windows.loc[merged_windows['activity_from'] == transition[0], 'frequency_det'].sum()

    # Adjust counts by adding pseudo-counts directly to the successes and total counts
    # This approach ensures that zero frequencies are adjusted to allow for statistical testing
    success_reference = prob_reference * total_count_reference + pseudo_count
    success_detection = prob_detection * total_count_detection + pseudo_count
    total_count_reference += pseudo_count
    total_count_detection += pseudo_count

    # Calculate the successes and attempts for both groups, adjusting for pseudo-counts
    nobs = [total_count_reference, total_count_detection]
    count = [success_reference, success_detection]

    # Perform the proportions Z-test
    stat, p_value = proportions_ztest(count, nobs)

    # # Add pseudo-count
    # success_reference = int(round(prob_reference * (total_count_reference + pseudo_count)))
    # success_detection = int(round(prob_detection * (total_count_detection + pseudo_count)))

    # # Calculate standard deviation
    # std = np.sqrt((prob_reference * (1 - prob_reference) / (total_count_reference + pseudo_count)) +
    #               (prob_detection * (1 - prob_detection) / (total_count_detection + pseudo_count)))

    # # Avoid divide by zero
    # if std == 0:
    #     return 1

    # # Perform the test
    # stat, p_value = proportions_ztest([success_reference, success_detection], 
    #                                   [total_count_reference + pseudo_count, total_count_detection + pseudo_count])
    
    return p_value


def changed_transitions_detection(self, merged_windows_df, features_windows):

    # Initialize a dictionary to store the results
    changed_transitions = pd.DataFrame(columns=['transition', 'feature', 'p_value', 'effect_size', 'ref_value', 'det_value', 'dif_value'])

    # Perform statistical tests for each variable and transition
    i=0
    for index, row in merged_windows_df.iterrows():
        transition = (row['activity_from'], row['activity_to'])

        for feature in features_windows:

            variable_ref = feature + '_ref'
            variable_det = feature + '_det'

            ref_value = row[variable_ref]
            det_value = row[variable_det]

            # Check if the feature is a proportion or count
            test_type = identify_statistical_test(merged_windows_df[[variable_ref, variable_det]].values.flatten())

            # Perform the appropriate statistical test
            if test_type == 'count_test':
                p_value = perform_count_test(merged_windows_df, row, variable_ref, variable_det)
                # Calculate corrected Cramér's V
                effect_size = cramers_corrected_stat(merged_windows_df, variable_ref, variable_det, ref_value, det_value)
                is_significant = p_value is not None and p_value < self.pvalue_threshold_localization and abs(effect_size) > self.effect_count_threshold_localization
            else:
                p_value = perform_proportions_test(merged_windows_df, row, transition, variable_ref, variable_det, self.pseudo_count_localization)
                # Calculate Cohen's h
                effect_size = cohen_h(ref_value, det_value)
                is_significant = p_value is not None and p_value < self.pvalue_threshold_localization and abs(effect_size) > self.effect_prop_threshold_localization
            # Record the significant result
            if is_significant:
                changed_transitions.loc[i] = [transition, feature, p_value, effect_size, ref_value, det_value, det_value - ref_value]
                i += 1
        
    return changed_transitions  


def create_dfg_from_dataset(dataset):
    # Creating a DFG from the dataset
    dfg_transitions = {(row['activity_from'], row['activity_to']): row['frequency'] for index, row in dataset.iterrows() if row['frequency'] > 0}

    # Identifying real start activities
    real_start_activities = set(to for from_, to in dfg_transitions if from_ == 'START')
    start_activities_freq = {activity: dfg_transitions[('START', activity)] for activity in real_start_activities}

    # Identifying real end activities
    real_end_activities = set(from_ for from_, to in dfg_transitions if to == 'END')
    end_activities_freq = {activity: dfg_transitions[(activity, 'END')] for activity in real_end_activities}

    # Removing transitions that involve 'START' and 'END'
    dfg_transitions = {k: v for k, v in dfg_transitions.items() if 'START' not in k and 'END' not in k}

    return create_dfg_from_transitions(dfg_transitions, start_activities_freq, end_activities_freq)


def create_dfg_from_transitions(dfg_transitions, start_activities_freq, end_activities_freq):
    dfg = DFG()

    # Adding transitions to the DFG
    for (from_act, to_act), count in dfg_transitions.items():
        dfg.graph[(from_act, to_act)] += count

    # Adding real start activities
    for act, count in start_activities_freq.items():
        dfg.start_activities[act] += count

    # Adding real end activities
    for act, count in end_activities_freq.items():
        dfg.end_activities[act] += count

    return dfg


def compare_dfgs(dfg1, dfg2):
    # Retrieve transition sets from the graphs
    dfg1_transitions = set(dfg1.graph.keys())
    dfg2_transitions = set(dfg2.graph.keys())

    # Calculate new, deleted, and altered transitions
    new_transitions = dfg2_transitions - dfg1_transitions
    deleted_transitions = dfg1_transitions - dfg2_transitions
    
    # Get activities
    dfg1_activities = set(t[0] for t in dfg1.graph.keys()) | set(t[1] for t in dfg1.graph.keys())
    dfg2_activities = set(t[0] for t in dfg2.graph.keys()) | set(t[1] for t in dfg2.graph.keys())

    # Calculate new and deleted activities
    new_activities = dfg2_activities - dfg1_activities
    deleted_activities = dfg1_activities - dfg2_activities
    
    # Get start and end activities
    dfg1_start_activities = set(dfg1.start_activities.keys())
    dfg2_start_activities = set(dfg2.start_activities.keys())
    dfg1_end_activities = set(dfg1.end_activities.keys())
    dfg2_end_activities = set(dfg2.end_activities.keys())

    # Calculate new and deleted start and end activities
    # new_start_activities = dfg2_start_activities - dfg1_start_activities
    # deleted_start_activities = dfg1_start_activities - dfg2_start_activities
    # new_end_activities = dfg2_end_activities - dfg1_end_activities
    # deleted_end_activities = dfg1_end_activities - dfg2_end_activities

    dfg_changes = {
        'Transitions_new': list(new_transitions) if new_transitions else ["None"]
        ,'Transitions_Deleted': list(deleted_transitions) if deleted_transitions else ["None"]
        ,'Activities_new': list(new_activities) if new_activities else ["None"]
        ,'Activities_Deleted': list(deleted_activities) if deleted_activities else ["None"]
        # ,'Start_Activities_new': list(new_start_activities)
        # ,'Start_Activities_Deleted': list(deleted_start_activities)
        # ,'End_Activities_new': list(new_end_activities)
        # ,'End_Activities_Deleted': list(deleted_end_activities)

        # ,'Transitions_reference': list(dfg1_transitions)
        # ,'Transitions_detection': list(dfg2_transitions)
        # ,'Activities_reference': list(dfg1_activities)
        # ,'Activities_detection': list(dfg2_activities)
        # ,'Start_Activities_reference': list(dfg1_start_activities)
        # ,'Start_Activities_detection': list(dfg2_start_activities)
        # ,'End_Activities_reference': list(dfg1_end_activities)
        # ,'End_Activities_detection': list(dfg2_end_activities)
    }

    return dfg_changes


def create_bpmn_from_dfg(dfg):
    return pm4py.discover_bpmn_inductive(dfg, noise_threshold=0)



def localization_dfg_visualization(dfg, change_informations, bgcolor="white", rankdir="LR", node_penwidth="2", edge_penwidth="2"):
    
    dfg_graph = dfg.graph
    start_activities = dfg.start_activities
    end_activities = dfg.end_activities
    new_transitions = change_informations['Transitions_new']
    deleted_transitions = change_informations['Transitions_Deleted']
    new_activities = change_informations['Activities_new']
    deleted_activities = change_informations['Activities_Deleted']

    edge_annotations = {}
    for key, transitions in {key: val for key, val in change_informations.items() if key.startswith("Changed_transition_")}.items():
        suffix = key.split('_')[-1]  # Extract suffix
        for transition in transitions:
            if transition in edge_annotations:
                edge_annotations[transition].append(suffix)
            else:
                edge_annotations[transition] = [suffix]


    dot = Digraph(engine='dot', graph_attr={'bgcolor': bgcolor, 'rankdir': rankdir})

    # Create a unique start and end node for visualization
    dot.node('START', shape='circle', label='START', width='0.8', style='filled', fillcolor='white', penwidth=node_penwidth)
    dot.node('END', shape='doublecircle', label='END', width='0.8', style='filled', fillcolor='white', penwidth=node_penwidth)

    # Add nodes and edges to the graph
    for (source, target), count in dfg_graph.items():
        # Set node shapes and labels
        source_label = f"{source} ({start_activities.get(source, count)})"
        target_label = f"{target} ({end_activities.get(target, count)})"

        # Determine node colors based on activity status
        source_color = 'blue' if source in new_activities else 'red' if source in deleted_activities else 'black'
        target_color = 'blue' if target in new_activities else 'red' if target in deleted_activities else 'black'

        # # Add nodes
        dot.node(source, label=source_label, shape='box', style='filled', fillcolor='white', color=source_color, penwidth=node_penwidth)
        dot.node(target, label=target_label, shape='box', style='filled', fillcolor='white', color=target_color, penwidth=node_penwidth)

        # Set edge colors based on transition type
        edge_color = 'black'
        if (source, target) in new_transitions:
            edge_color = 'blue'
        elif (source, target) in deleted_transitions:
            edge_color = 'red'
        elif (source, target) in edge_annotations:
            edge_color = 'orange'

        # Add edges
        if (source, target) in edge_annotations:
            dot.edge(source, target, label=str(count) + ' (Dif. in ' +', '.join(edge_annotations[(source, target)]) + ')', color=edge_color, penwidth=edge_penwidth) 
        else: 
            dot.edge(source, target, label=str(count), color=edge_color, penwidth=edge_penwidth)

    # Connect the start node to the real start activities and the real end activities to the end node
    for act in start_activities:
        if act not in end_activities:  # Avoid connecting end activities again
            count = start_activities.get(act, 0)  # Get the count for the activity

            # Set edge colors based on transition type
            edge_color = 'black'
            if ('START', act) in new_transitions:
                edge_color = 'blue'
            elif ('START', act) in deleted_transitions:
                edge_color = 'red'
            elif ('START', act) in edge_annotations:
                edge_color = 'orange'

            if ('START', act) in edge_annotations: 
                dot.edge('START', act, label=str(count) + ' (Dif. in ' +', '.join(edge_annotations[('START', act)]) + ')', color=edge_color, style='bold', penwidth=edge_penwidth)
            else:
                dot.edge('START', act, label=str(count), color=edge_color, style='bold', penwidth=edge_penwidth)

    for act in end_activities:
        if act not in start_activities:  # Avoid connecting start activities again
            count = end_activities.get(act, 0)  # Get the count for the activity

            # Set edge colors based on transition type
            edge_color = 'black'
            if (act, 'END') in new_transitions:
                edge_color = 'blue'
            elif (act, 'END') in deleted_transitions:
                edge_color = 'red'
            elif (act, 'END') in edge_annotations:
                edge_color = 'orange'

            if (act, 'END') in edge_annotations: 
                dot.edge(act, 'END', label=str(count) + ' (Dif. in ' +', '.join(edge_annotations[(act, 'END')]) + ')', color=edge_color, style='bold', penwidth=edge_penwidth)
            else:
                dot.edge(act, 'END', label=str(count), color=edge_color, style='bold', penwidth=edge_penwidth)

    # Render and display the graph inline in Jupyter Notebook
    png_data = dot.pipe(format='png')
    display(Image(png_data))


def create_process_tree_from_dfg(dfg, parameters):
    return inductive_miner.apply(dfg, parameters=parameters) 
    

def llm_instanciating(llm_company, llm_model, api_key):
        
    if llm_company == "openai":

        # insert API_KEY in the file to be read here
        if api_key:
            with open(api_key, 'r') as file:
                os.environ["OPENAI_API_KEY"] = file.read().rstrip()

        # Instanciating LLM class
        return OpenAI()
    
    elif llm_company == "google":

        # insert API_KEY in the file to be read here
        if api_key:
            with open(api_key, 'r') as file:
                os.environ["GOOGLE_CLOUD_API_KEY"] = file.read().rstrip()
                genai.configure(api_key=os.environ['GOOGLE_CLOUD_API_KEY'])

        # Instanciating LLM class
        return genai.GenerativeModel(llm_model)
    

def llm_call_response(llm_company, llm_model, llm, user_prompt):
    if llm_company == "openai":

        response = llm.chat.completions.create(
            temperature=0
            , top_p=0.000000000000001
            , seed=42
            , model=llm_model
            , messages=[
            # {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content
    
    elif llm_company == "google":

        response = llm.generate_content(user_prompt,
            generation_config=genai.types.GenerationConfig(
                # Only one candidate for now.
                candidate_count=1
                # , stop_sequences=['x']
                # , max_output_tokens=None
                , temperature=0
                , top_p=0.000000000000001
                , top_k=40
            )
        )
        
        return response.text
        

def llm_instructions(llm_instructions_path, reference_bpmn_text, detection_bpmn_text, change_informations):

    # Open the JSON file for reading
    with open(llm_instructions_path, 'r') as file:
        # Parse the yaml file into a Python dictionary
        llm_instructions = yaml.safe_load(file)

    llm_instructions["changes_informations"] += (
        " - Transitions with variations in probability: {0}. \n"
        " - Transitions with variations in in frequence: {1}. \n"
        " - New transitions added to the process: {2}. \n"
        " - Deleted transitions from the process: {3}. \n"
        " - New activities added to the process: {4}. \n"
        " - Deleted activities from the process: {5}. \n"
    ).format(change_informations["Changed_transition_probability"], change_informations["Changed_transition_frequency"], change_informations["Transitions_new"], change_informations["Transitions_Deleted"]
            , change_informations["Activities_new"], change_informations["Activities_Deleted"] )
    
    llm_instructions["bpmn_informations"] += (
        " - The BPMN before the concept drift: {0}. \n"
        " - The BPMN after the concept drift: {1}. \n"
    ).format(reference_bpmn_text, detection_bpmn_text)

    # transform controlflow_change_patterns in dict
    consolidated_dict = {}
    for instruction_dict in llm_instructions["controlflow_change_patterns"]:
        for key, value in instruction_dict.items():
            consolidated_dict[key] = value
    llm_instructions["controlflow_change_patterns"] = consolidated_dict
                            
    return llm_instructions
     

def llm_prompt_analysis(llm_instructions):

    prompt = (llm_instructions["introduction"] 
                   + llm_instructions["changes_informations"] 
                   + llm_instructions["bpmn_informations"] 
                   + llm_instructions["concept_drift_analysis"])
    return prompt


def llm_prompt_classification(llm_instructions, change_informations):

    prompt = (llm_instructions["introduction"] 
        + llm_instructions["bpmn_informations"] 
        + llm_instructions["changes_informations"] 
        + "\n### Control Flow Change Patterns ### \n"
    )

    # prompt = ("\n### Concept drift analysis of the BPMN diagrams and the detailed variations in activities and transitions ###\n" + characterization_analysis)

    # If there is at least a new or deleted activity, then suggest SRE, PRE, CRE, or RP
    if change_informations['Activities_new'] != ['None'] or change_informations['Activities_Deleted'] != ['None']:
        prompt += (llm_instructions['controlflow_change_patterns']['sre_instructions'] 
                  + llm_instructions['controlflow_change_patterns']['pre_instructions'] 
                  + llm_instructions['controlflow_change_patterns']['cre_instructions'] 
                  + llm_instructions['controlflow_change_patterns']['rp_instructions'] 
        )

    # If the changes don't involve addition or deletion of activities but rather addition or deletion of transitions between existing activities, then suggest SM, CM, PM, or SW, CF, PL, LP,CD,  CB, or CP
    elif change_informations['Transitions_new'] != ['None'] or change_informations['Transitions_Deleted'] != ['None']:
        # Movement Patterns
        prompt += (llm_instructions['controlflow_change_patterns']['sm_instructions'] 
                + llm_instructions['controlflow_change_patterns']['cm_instructions'] 
                + llm_instructions['controlflow_change_patterns']['pm_instructions'] 
                + llm_instructions['controlflow_change_patterns']['sw_instructions'] 
        )

        # Gateway Type Changes
        prompt += (llm_instructions['controlflow_change_patterns']['pl_instructions'] 
                + llm_instructions['controlflow_change_patterns']['cf_instructions'] 
        )

        # Synchronization (Parallel involved)
        prompt += (llm_instructions['controlflow_change_patterns']['cd_instructions'] 
        )

        # Bypass (XOR involved)
        prompt += (llm_instructions['controlflow_change_patterns']['cb_instructions'] 
        )

        # Loop Fragment Changes
        prompt += (llm_instructions['controlflow_change_patterns']['cp_instructions'] 
                + llm_instructions['controlflow_change_patterns']['lp_instructions'] 
        )

    # If the changes don't involve addition or deletion of activities nor addition or deletion of transitions between existing activities, but rather only changes in the transitions, then is FR
    else:
        prompt += (
            llm_instructions['controlflow_change_patterns']['fr_instructions'] 
        )

    prompt += llm_instructions["change_pattern_classification"]

    return prompt



def llm_classification_formatting(characterization_classification):

    # Finding the start and end of the dictionary string
    try:
        start_str = "result_dict = {"
        end_str = "}"
        start_index = characterization_classification.find(start_str) + len(start_str) - 1
        end_index = characterization_classification.find(end_str, start_index) + 1

        return ast.literal_eval(characterization_classification[start_index:end_index].strip())
    except:
        return "Classification not in the expected format."



def set_characterization_main_instructions_default(reference_bpmn_text, detection_bpmn_text, change_informations):

    characterization_main_instructions = {
        # "general_introduction" : ("\n### Objective ###\n" 
        #     "- Your primary task is to identify concept drift in a business process by comparing BPMN diagrams from two periods and analyzing detailed change information. Concept drift refers to changes in the process's behavior over time. " 
        #     )

        # , "bpmn_windows_informations" : ("\n### BPMN Diagram Analysis ###\n"
        #     "- **Understand BPMN Symbols**: Familiarize yourself with symbols denoting sequential operations (`->`), parallel operations (`+`), conditional operations (`X`), looping operations (`*`), and silent transitions (`tau`). Nested operations are enclosed in parentheses `()`."
        #     "- **Compare BPMN Diagrams**: Examine two BPMN diagrams representing the process before and after potential concept drift. Identify structural changes, including added or deleted activities and transitions."
        #     "- **BPMN diagrams representing the process before**: {0}. "
        #     "- **BPMN diagrams representing the process after**: {1}. "
        #     ).format(reference_bpmn_text, detection_bpmn_text)

        # , "concept_drift_informations" : ("\n### Changes Information Analysis ###\n"
        #     "- Analyze provided details on changes between the two periods, focusing on transition probability, transition frequency, new transitions, deleted transitions, new activities, and deleted activities."
        #     "- **Provided details on changes between the two periods**: {0}."
        #     ).format(change_informations)

        # , "rules_instructions" : ("\n### Classification Rules ###\n"
        #     "1. **Activity Addition or Deletion**:"
        #     "- Classify as RP if a new activity replaces a deleted one in the same location."
        #     "- Use SRE, PRE, or CRE based on whether the activity was part of a sequential, parallel, or XOR flow, respectively."

        #     "2. **Transition Changes Without Activity and Transitions Addition/Deletion**:"
        #     "- Classify as FR if there are changes in transitions without any activity or transition being added or deleted."

        #     "3. **Transition Addition or Deletion**:"
        #     "- **Movement Patterns**: Classify as SM, PM, CM, or SW if an activity changes BPMN fragments, requiring new and deleted transitions. The classification depends on the new structure (sequential, parallel, XOR) or if activities swap positions."
        #     "- **Fragment Type Changes**: Classify as CF or PL if two activities change BPMN fragment types together."
        #     "- **Synchronization**: Classify as CD if two parallel activities are now sequential within the same parallel fragment or one is immediately outside the fragment."
        #     "- **Loop Fragment Changes**: Classify as CP or LP for new or deleted BPMN LOOP fragments, based on transition changes."
        #     "- **Conditional Bypass**: Classify as CB if a new transition enables another activity to be skipped."


        "general_introduction" : ("\n### Objective ###\n" 
            # "Your primary task is to explore the concept drift within a business process, which refers to the evolution or changes in the process's behavior over time. " 
            # "You will compare BPMN (Business Process Model and Notation) diagrams and changes informations from two distinct periods to identify these changes. "
            "Your primary task is to understand the concept drift within a business process and classify which change pattern occurred. " 
            "You will take into consideration all the changes provided, compare BPMN (Business Process Model and Notation) diagrams before and after the concept drift and use the change patterns definitions to classify which occured. "
            "Think step by step.")

        , "concept_drift_informations" : ("\n### Changes Information Analysis ###\n"
            "Analyze these changes regarding activities and transitions to understand the nature of the concept drift: "
                " - Transitions with statistically significant alterations in probability: {0}. "
                " - Transitions with statistically significant alterations in frequence: {1}. "
                " - New transitions added to the process: {2}. "
                " - Deleted transitions from the process: {3}. "
                " - New activities added to the process: {4}. "
                " - Deleted activities from the process: {5}. "
            ).format(change_informations["Changed_transition_probability"], change_informations["Changed_transition_frequency"], change_informations["Transitions_new"], change_informations["Transitions_Deleted"]
                     , change_informations["Activities_new"], change_informations["Activities_Deleted"] )

        , "bpmn_windows_informations" : ("\n### BPMN Diagram Analysis ###\n"
            "Understand BPMN Symbols: Familiarize yourself with the BPMN symbols used in the diagrams. " 
            "Sequential fragments are denoted by '->', parallel fragments by '+', conditional fragments by 'X', looping fragments by '*', and silent transitions by 'tau'. Nested fragments are enclosed in parentheses '()'. "
            "Compare BPMN Diagrams: You are given two BPMN diagrams representing the process flow before and after the detection of concept drift. " 
            "Carefully compare these diagrams to identify any structural changes in the fragments such as sequences, parallelisms, conditional flows, loops, and silent transitions." 
            " - The BPMN before the concept drift: {0}. "
            " - The BPMN after the concept drift: {1}. "
            ).format(reference_bpmn_text, detection_bpmn_text)

        # , "concept_drift_informations" : ("\n### Changes Information Analysis ###\n"
        #     "You will also be provided with a detailed breakdown of the changes between the two periods, categorized as follows: "
        #         " - Changed_transition_probability: Transitions with statistically significant alterations in likelihood. "
        #         " - Changed_transition_frequency: Transitions with statistically significant alterations in occurrence. "
        #         " - Transitions_new: New transitions introduced into the process. "
        #         " - Transitions_Deleted: Transitions deleted from the process. "
        #         " - Activities_new: New activities added to the process. "
        #         " - Activities_Deleted: Activities deleted from the process. "
        #         "Analyze these changes to understand the nature and impact of the concept drift. "
        #         "Follow the changes informations: {0}. "
        #     ).format(change_informations)

        , "rules_instructions" : ("\n### Change Patterns Definitions ###\n"
        
        
        # Analyze these changes regarding activities and transitions to understand the nature of the concept drift:  
        # - Transitions with statistically significant alterations in probability: [('Appraise_property', 'Assess_eligibility'), ('Appraise_property', 'Assess_loan_risk'), ('Assess_loan_risk', 'Appraise_property'), ('Assess_loan_risk', 'Assess_eligibility'), ('Check_credit_history', 'Appraise_property'), ('Check_credit_history', 'Assess_loan_risk')].  
                                #   - Transitions with statistically significant alterations in frequence: [('Appraise_property', 'Assess_eligibility'), ('Appraise_property', 'Assess_loan_risk'), ('Assess_loan_risk', 'Appraise_property'), ('Assess_loan_risk', 'Assess_eligibility'), ('Check_credit_history', 'Appraise_property'), ('Check_credit_history', 'Assess_loan_risk')].  
                                #   - New transitions added to the process: ['None'].  
                                #   - Deleted transitions from the process: [('Appraise_property', 'Assess_eligibility'), ('Assess_loan_risk', 'Appraise_property')].  
                                #   - New activities added to the process: ['None'].  
                                #   - Deleted activities from the process: ['None']. 


            "1. Activity Addition or Deletion: there are new or deleted activities in the process. " 
                "For example: - New activities added to the process: ['A', 'B'] - Deleted activities from the process: ['C', 'D']. "
                " - If the new activity replaces the deleted activity in the exact same location, classify as RP (Replace). For example: '->(A, B, C)' to '->(A, D, C)'. "
                " - If the activity was part of a sequential flow, classify as SRE (Serial Routing Enablement). For example: '->(A, B)' to '->(A, C, B)'. "
                " - If the activity was part of a parallel flow (+), classify as PRE (Parallel Routing Enablement). For example: '->(A, B, C)' to '->(A, +(B, D), C)'. "
                " - If the activity was part of a XOR flow (X), classify as CRE (Conditional Routing Enablement). For example: '->(A, B, C)' to '->(A, X(B, D), C)'. "

            "2. Transition Addition or Deletion: no activity is added or deleted but there are new or deleted transitions in the process. "
                "For example: - New activities added to the process: ['None'] - Deleted activities from the process: ['None'] - New transitions added to the process: [('A', 'B')] - Deleted transitions from the process: [('B', 'C')]. "
                " - Movement Patterns: an activity changes BPMN fragments, requiring both new and deleted transitions, classify based on the new structure. "
                    " - Sequential (→) as SM (Serial Move). For example: '->(A, B, C, D, E)' to '->(A, C, D, B, E)'. "
                    " - Parallel (+) as PM (Parallel Move). For example: '->(A, B, C, D, E, F)' to '->(A, C, D, +(B, E), F)'. "
                    " - XOR (X) as CM (Conditional Move). For example: '->(A, B, C, D, E, F)' to '->(A, C, D, X(B, E), F)'. "
                    " - If two activities swap positions with identical transition adjustments, classify as SW (Swap). For example: '->(A, B, C, D, E, F)' to '->(A, E, C, D, B, F)'. "
                    
                " - Fragment Type Changes: two activities change BPMN fragment types together. "
                        " - From/to XOR (X) as CF (Conflict). For example: '->(A, X(B, C), D)' to '->(A, B, C, D)'. "
                        " - From/to Parallel (+) as PL (Parallel). For example: '->(A, +(B, C), D)' to '->(A, B, C, D)'. "

                " - Synchronization: two parallel activities are now sequential within the same parallel fragment, or one is immediately outside the fragment. "
                    " - Classify as CD (Control Dependancy or Synchronization). For example: '->(A, +(B, ->(C, D)), E)' to '->(A, ->(+(B, C), D), E)'. "

                " - Loop Fragment Changes: new or deleted BPMN LOOP fragments, classify based on transition changes. "
                    " - With deleted transitions in the LOOP, classify as CP (Copy). For example: '->(A, B, C, D)' to '->(A, B, C, A, D)'. "
                    " - Without deleted transitions, classify as LP (Loop). For example: '->(A, *(B, C, D), E)' to '->(A, B, C, D, E)'.  "

                " - Conditional Bypass: there are activities with new transition enables another activity to be skipped. "
                    "- Classify as CB (Conditional Branching or Skip). For example: '->(A, B, C)' to '->(A, X(B, null), C)'. "

            "3. Transition Changes only: there are ONLY transitions with statistically significant alterations in probability or frequency, with NO activity being added or deleted, and NO transition being added or deleted. "
                "For example: - New activities added to the process: ['None'] - Deleted activities from the process: ['None'] - New transitions added to the process: ['None'] - Deleted transitions from the process: ['None']."
                " - Classify as FR (Frequency). For example: '->(A, X(B[550], C[450]), D)' to '->(A, X(B[700], C[300]), D)'. "

            # "1. Activity Addition or Deletion:"
            # "    - If at least one activity is added or deleted:"
            # "        - If the new activity replaces the deleted activity in the exact same location, classify as RP."
            # "        - Otherwise, determine the pattern based on the structure:"
            # "            - If the activity was part of a sequential flow, classify as SRE."
            # "            - If the activity was part of a parallel flow (+), classify as PRE."
            # "            - If the activity was part of a XOR flow (X), classify as CRE."
            # "2. Transition Changes Without Activity Addition/Deletion:"
            # "    - If there are changes in transitions without any activity being added or deleted:"
            # "        - If transitions are modified but no activities are added or deleted, classify as FR."
            # "3. Transition Addition or Deletion with Existing Activities:"
            # "    - If there are additions or deletions of transitions between existing activities, without adding or deleting the activities themselves, classify according to the detailed rules:"
            # "        - Parallel to Sequential Change:"
            # "            - If two parallel activities are now sequential within the same parallel fragment, or one is immediately outside the fragment, classify as CD."
            # "        - Movement Patterns:"
            # "            - If an activity changes BPMN fragments, requiring new and deleted transitions, classify based on the new structure:"
            # "                - Sequential (→) as SM."
            # "                - Parallel (+) as PM."
            # "                - XOR (X) as CM."
            # "                - If two activities swap positions with identical transition adjustments, classify as SW."
            # "        - Fragment Type Changes:"
            # "            - If two activities change BPMN fragment types together, classify as:"
            # "                - From/to XOR (X) as CF."
            # "                - From/to Parallel (+) as PL."
            # "        - Loop Fragment Changes:"
            # "            - For new or deleted BPMN LOOP fragments, classify based on transition changes:"
            # "                - With deleted transitions in the LOOP, classify as CP."
            # "                - Without deleted transitions, classify as LP."
            # "        - Conditional Bypass:"
            # "            - If at least one activity with a new transition enables another activity to be skipped, classify as CB."

            # "1. If there is at least a new or deleted activity, then suggest SRE, PRE, CRE, or RP: " 

            #     " - If the new activity is in the exactly place as the deleted then is RP. " 

            #     " - Otherwise: "
            #         " - If the new or deleted activity is or was in a sequence then is SRE. "
            #         " - If the new or deleted activity is or was in a parallel (+) then is PRE. "
            #         " - If the new or deleted activity is or was in a XOR (X) then is CRE. "

            # "2. If the changes don't involve addition or deletion of activities nor addition or deletion of transitions between existing activities, but rather only changes in the transitions, then is FR. "

            # "3. If the changes don't involve addition or deletion of activities but rather addition or deletion of transitions between existing activities, then suggest SM, CM, PM, or SW, CF, PL, LP,CD,  CB, or CP: "

            #     " - If there is at least an activity that is in a different BPMN fragment, and necessarally have new and deleted transitions, then suggest a movement patterns like SM, CM, PM, or SW: "
            #         " - If the activity is now in a sequence then is SM. "
            #         " - If the activity is now in a parallel (+) then is PM. "
            #         " - If the activity is now in a XOR (X) then is CM. "
            #         " - If there are two activities that changed their positions to the exact position of the other, and they have the same transitions with the activities that the other had in the past, then is SW. "

            #     " - If there are at least two activities that were in the same BPMN fragment type such as sequence, XOR, or parallel, but now both changed to other BPMN fragment type, then suggest CF, or PL: "
            #         " - If involved a XOR (X) then is CF. "
            #         " - If involved a parallel (+) then is PL. "

            #     " - If there are at least two activities that were parallel in a parallel (+) fragment and now they are sequence in this parallel (+) fragment or one of them is outside the fragment, being the right next activity, then is CD. "

            #     " - If there is a new or deleted BPMN LOOP fragment, then suggest CP or LP: "
            #         " - If there are deleted transitions involving activities in the BPMN LOOP fragment then CP. "
            #         " - If there aren't deleted transitions involving activities in the BPMN LOOP fragment then LP. "

            #     " - If there is at least an activity with new transition that make at least another activity be skippable then is CB. "

            # transition new, deleted, changed
            # activities new and deleted
            # bpmn with fragment types

            # "1 - Identify the Nature of Change: " 
            #     " - Activity-Based Changes: Look for the addition, deletion, or duplication of activities within the process. Such changes often signify SRE, PRE, CRE, or RP patterns. "
            #     " - Transition-Based Changes: Changes that don't involve direct addition or deletion of activities but rather alterations in the flow between existing activities suggest patterns like SW, SM, CM, PM, CF, PL, CD, LP, or CP. "
            # "2 - Focus on Branching and Flow: "
            #     " - SRE, PRE, CRE"
            #     " - SM, CM, PM"
            #     " - CF, PL"
            #     " - Branching Creation: Pay special attention to new or deleted branches, that can indicate Activity-Based Changes patterns (such as SRE, PRE, CRE, or RP) or activity movement (such as SM, CM, PM). "
            #     " - Branching Modifications: Any alterations in AND (+) or XOR (X) branches indicate potential CF, PL"
            # "3 - Analyze Activity Positions and Relations: "
            #     " - Movement and Swap: Look for activities that have changed their position within the process without being added or deleted. This could indicate SM, CM, PM, or SW patterns."
            #     " - Replacement: If an activity is deleted and another appear in the exactly place, suggest RP. "
            # "4 - Consider Execution Frequency: "
            #     " - Frequency Changes: Variations in transitions without altering the control flow (i.e. no new or deleted activity and transitions), suggest the FR pattern."
            # "5 - Examine Loop Fragments: "
            #     " - Loop Adjustments: Modifications to looping behavior, including the addition or removal of loops or changes in loop execution, point to the LP pattern."
            # "6 - "
            #     " - Duplication: CP"
            #     " - Synchronization: CD"
            #     " - skippability: CB"
            #     "RP, SW"
            )

        , "question_separeted" : ("\n### Conclusion Formation ###\n"
            "Based on your analysis of the BPMN diagrams and the detailed changes information, conclude whether a concept drift occurred, whether this specific change pattern has occurred, and which activities are involved. "
            "Your conclusion must be supported by evidence from your analysis, and be aware that another change pattern may has occurred and not the one you are analyzing. "
            "Format your conclusion as follows, providing a clear and concise verdict on whether the change pattern is present: "
            "'### result_dict = {'concept_drift' : 'No', 'change_pattern' : 'None', 'activities' : []} " 
            "or '### result_dict = {'concept_drift' : 'Yes', 'change_pattern' : 'Other', 'activities' : ['A', 'B']} "
            "or '### result_dict = {'concept_drift' : 'Yes', 'change_pattern' : 'pattern_acronym', 'activities' : ['A', 'B']}. "
            # "Your analysis and conclusion are crucial in understanding the impact of concept drift on the business process. Approach this task methodically, focusing on accuracy and evidence to support your conclusions. "
            )

        , "question_unified" : ("\n### Conclusion Formation ###\n"
            "Make a list with all 14 patterns and the probability of being each pattern. Then, conclude whether concept drift occurred, the specific change pattern, and which activities are involved. Your conclusion must be evidence-based. "
            "Avoid presupposing any particular change pattern. Instead, let your analysis of the diagrams and change information guide your classification."
            "Format your conclusion as follows: "
            "'### result_dict = {'concept_drift' : 'No', 'change_pattern' : 'None', 'activities' : []} " 
            "or '### result_dict = {'concept_drift' : 'Yes', 'change_pattern' : 'Other', 'activities' : ['A', 'B']} "
            "or '### result_dict = {'concept_drift' : 'Yes', 'change_pattern' : 'pattern_acronym', 'activities' : ['A', 'B']}. ")

    }
    return characterization_main_instructions


# def get_characterization_controlflow_change_patterns_default():
#     characterization_controlflow_change_patterns = {
#         ### SRE (Serial Routing Enablement)
#         "sre_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'SRE' (Serial Routing Enablement) involves the addition or deletion of one or more activities in or from a sequential flow. " 
#             "For example, the sequence could change from '->(A, B)' to '->(A, C, B)' or vice versa, where 'A', 'B', and 'C' are activities, and '->' symbolizes sequential flow. " 
#             "This scenario implies that after the drift, activity 'A' is inserted in the process and in a sequential flow between other two activities, in this case 'A' and 'B', "
#             "or it could be deleted from the process and from a sequential flow bteween two other activities. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information shows differences in Sequences (->) fragments. "
#             # "if the involved activities are new or deleted, and if they have new and deleted transitions. "
#             "This scenario involve the addition and deletion of activities and transitions, so look if activities and transitions are inserted and deleted. ")

#         ### PRE (Parallel Routing Enablement)
#         , "pre_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'PRE' (Parallel Routing Enablement) involves the addition or deletion of one or more activities in or from a parallel branch. " 
#             "For example, the sequence could change from '->(A, B, C)' to '->(A, +(B, D), C)' or vice versa, where 'A', 'B', 'C', and 'D' are activities, '->' symbolizes sequential flow, " 
#             "and '+' represents an AND or parallel operation within the BPMN (Business Process Model and Notation). "
#             "This scenario implies that after the drift, activitiy 'D' is  inserted in the process and in a parallel branch, or 'D' could be deleted from the process and from a parallel branch. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information shows new or deleted ANDs (+) fragments. "
#             "This scenario involve the addition and deletion of activities and transitions, so look if activities and transitions are inserted and deleted. ")

#         ### CRE (Conditional Routing Enablement)
#         , "cre_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'CRE' (Conditional Routing Enablement) involves the addition or deletion of one or more activities in or from a conditional branch. "
#             "For example, the sequence could change from '->(A, B, C)' to '->(A, X(B, D), C)' or vice versa, where 'A', 'B', 'C', and 'D' are activities, '->' symbolizes sequential flow, " 
#             "and 'X' represents a XOR or conditional operation within the BPMN (Business Process Model and Notation). "
#             "This scenario implies that the drift, activitiy 'D' is  inserted in the process and in a conditional branch, or 'D' could be deleted from the process and from a conditional branch. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information shows new or deleted XORs (X) fragments. "
#             "This scenario involve the addition and deletion of activities and transitions, so look if activities and transitions are inserted and deleted. ")

#         ### CP (Copy) 
#         , "cp_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'CP' (Copy) involves duplicate an existing activity of the process to another location in the process. " 
#             "For example, the sequence could change from '->(A, B, C, D)' to '->(A, B, C, A, D)' or vice versa, where 'A', 'B', 'C', and 'D' are activities, '->' indicates a sequential operation. "
#             "This scenario implies that after the drift, activities 'A' appears in another location in the process as well. "
#             "Note: Identifying this pattern can be challenging, especially because neither the BPMN diagrams nor the transitions information may be correctly identified the duplication of an activity. "
#             "Both informations may only show that the duplicated activity has new transitions or maybe identify the duplication as a loop. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates the new transitions. " 
#             "Using the previous example, activity 'A' would have a new transition from activity 'C' ('C', 'A') and a new transition to activity 'D' ('A', 'D'). " 
#             "These two new transitions should be undestood as a new addition between 'C' and 'D', and, consequently, identified the duplication of 'A'. ")

#         ### RP (Replace)
#         , "rp_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'RP' (Replace) involves the replace of one or more activities by other new activities. " 
#             "For example, the sequence could change from '->(A, B, C)' to '->(A, D, C)', where 'A', 'B', 'C', and 'D' are activities, '->' symbolizes sequential flow. "  
#             "This scenario implies that after the drift, activity 'B' is replaced by activity 'D', which are a new activity in the process. "
#             "Tip: This scenario involve the addition and deletion of activities, so look if activities are inserted and another deleted. ")

#         ### SW (Swap)
#         , "sw_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'SW' (Swap) involves the swap of two existing activities in the process. " 
#             "For example, the sequence could change from '->(A, B, C, D, E, F)' to '->(A, E, C, D, B, F)', where 'A', 'B', 'C', 'D', 'E', and 'F' are activities, '->' symbolizes sequential flow. "  
#             "This scenario implies that after the drift, activity 'B' is swapped with activity 'E'. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information may shows the activities in different locations in the process, both occupying the exact position of the other, "
#             "and if both involved activities have new and deleted transitions. This scenario DOESN'T involve activities addition or deletion. ")

#         ### SM (Serial Move)
#         , "sm_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'SM' (Serial Move) involves move an existing activity to between two existing activities. " 
#             "For example, the sequence could change from '->(A, B, C, D, E)' to '->(A, C, D, B, E)', where 'A', 'B', 'C', 'D', and 'E' are activities, '->' symbolizes sequential flow. "  
#             "This scenario implies that after the drift, activity 'B', which was between activities 'A' and 'C', is moved to between activities 'D' and 'E'. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information may shows the activity in different locations in the process, "
#             "and if the involved activitiy have new and deleted transitions. This scenario DOESN'T involve activities addition or deletion. ")

#         ### CM (Conditional Move)
#         , "cm_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'CM' (Conditional Move) involves move an existing activity into or out of a conditional branch. " 
#             "For example, the sequence could change from '->(A, B, C, D, E, F)' to '->(A, C, D, X(B, E), F)', where 'A', 'B', 'C', 'D', 'E', and 'F' are activities, '->' symbolizes sequential flow, "
#             "and 'X' represents a XOR or conditional operation within the BPMN (Business Process Model and Notation). "  
#             "This scenario implies that after the drift, activity 'B', which was between activities 'A' and 'C', is moved to between activities 'D' and 'F' and in a conditional branch with activity 'E'. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information may shows the activity in different locations in the process, shows new or deleted XORs (X) fragments, "
#             "and if the involved activitiy have new and deleted transitions. This scenario DOESN'T involve activities addition or deletion. ")

#         ### PM (Parallel Move)
#         , "pm_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'PM' (Parallel Move) involves move an existing activity into or out of a parallel branch. " 
#             "For example, the sequence could change from '->(A, B, C, D, E, F)' to '->(A, C, D, +(B, E), F)', where 'A', 'B', 'C', 'D', 'E', and 'F' are activities, '->' symbolizes sequential flow, "
#             "and '+' represents an AND or parallel operation within the BPMN (Business Process Model and Notation). " 
#             "This scenario implies that after the drift, activity 'B', which was between activities 'A' and 'C', is moved to between activities 'D' and 'F' and in a parallel branch with activity 'E'. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information may shows the activity in different locations in the process, shows new or deleted ANDs (+) fragments, "
#             "and if the involved activitiy have new and deleted transitions. This scenario DOESN'T involve activities addition or deletion. ")

#         ### CF (Conflict)
#         , "cf_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'CF' (Conflict) involves making activities initially mutually exclusive to be sequential, or vice versa. " 
#             "For example, the sequence could change from '->(A, X(B, C), D)' to '->(A, B, C, D)' or vice versa, where 'A', 'B', 'C', and 'D' are activities, '->' indicates a sequential operation, "
#             "and 'X' represents a XOR or conditional operation within the BPMN (Business Process Model and Notation). "
#             "This indicates that before the drift, activities 'B' and 'C' were mutually exclusive, suggesting that only one could occur in any process instance, "
#             "while after the drift, they have become sequential, implying a fixed order of execution. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information shows new or deleted XORs (X) fragments, "
#             "and if the involved activities have new and deleted transitions. This scenario DOESN'T involve activities addition or deletion. ")

#         ### PL (Parallel)
#         , "pl_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'PL' (Parallel) involves making activities initially parallel to be sequential, or vice versa. " 
#             "For example, the sequence could change from '->(A, +(B, C), D)' to '->(A, B, C, D)' or vice versa, where 'A', 'B', 'C', and 'D' are activities, '->' indicates a sequential operation, "
#             "and '+' represents an AND or parallel operation within the BPMN (Business Process Model and Notation). "
#             "This indicates that before the drift, activities 'B' and 'C' were parallel, suggesting that both would occur with no defined order between them,"
#             "while after the drift, they have become sequential, implying a fixed order of execution. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information shows new or deleted ANDs (+) fragments, "
#             "and if the involved activitiy have new and deleted transitions. This scenario DOESN'T involve activities addition or deletion. ")

#         ### CD (Control Dependancy) or Synchronization
#         , "cd_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'CD' (Control Dependancy) involves synchronizing two or more activities, where the convergence of multiple branches into a single subsequent branch occurs "
#             "such that the thread of control passes to the subsequent branch only when all preceding branches have been executed. " 
#             "For example, the sequence could change from '->(A, +(B, ->(C, D)), E)' to '->(A, ->(+(B, C), D), E)' or vice versa, where 'A', 'B', 'C', and 'D' are activities, '->' indicates a sequential operation, "
#             "and '+' represents an AND or parallel operation within the BPMN (Business Process Model and Notation). "
#             "This scenario implies that before the drift, activities 'B', 'C', and 'D' could proceed in parallel, but after the drift, 'B' and 'C' need to synchronize before initiating 'D'. "
#             "Tip: In such scenarios, it's advisable to delve deeper and investigates if the BPMN information shows differences inside ANDs (+) fragments, "
#             "and if the involved activitiy have new and deleted transitions. This scenario DOESN'T involve activities addition or deletion. ")

#         ### LP (Loop)
#         , "lp_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'LP' (Loop) involves making activities initially loopable to be non-loopable, or vice versa. " 
#             "For example, the sequence could change from '->(A, *(B, C, D), E)' to '->(A, B, C, D, E)' or vice versa, where 'A', 'B', 'C', 'D', and 'E' are activities, '->' indicates a sequential operation, "
#             "and '*' represents a loop operation within the BPMN (Business Process Model and Notation). "
#             "This indicates that before the drift, activities 'B', 'C', and 'D' were in a loop, while after the drift, they were not in a loop. "
#             "Tip: In such scenarios, if the BPMN information doesn't show the loop, it's advisable to delve deeper and investigates if deleted transitions led to activities that could lead to the activity in question, " 
#             "which define a loop in the process. ")

#         ### CB (Conditional Branching) or Skip
#         , "cb_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'CB' (Conditional Branching) involves the modifications to the skippability of one or more activities. Specifically, an activity might become optional (skippable) or mandatory (non-skippable). " 
#             "For example, the sequence could change from '->(A, B, C)' to '->(A, X(B, null), C)', where 'A', 'B', and 'C' are activities, '->' indicates a sequential operation, "
#             "and 'X' represents a XOR or conditional operation within the BPMN (Business Process Model and Notation). "
#             "The term 'null' is used to indicate the absence of an activity in that branch, highlighting the conditional execution of 'B'. "
#             "Note: Identifying this pattern can be challenging, especially in BPMN diagrams where the new conditional operations or XOR may not be correctly identified because of the branch with no activitiy, "
#             "making the new skippable nature of 'B' less apparent. "
#             "Tip: In such scenarios, investigates all new or deleted transition carefully to understanding if it could represent a change in the skippablability to others activities. "
#             "Using our previous example, the new transition between 'A' to 'C' is causing B to become skippable. This scenario DOESN'T involve activities addition or deletion. ")

#         ### FR (Frequency)
#         , "fr_instructions" : (
#             # "\n### Change Pattern Instruction ###\n"
#             "The 'FR' (Frequency) involves change the branching frequency of activities. " 
#             "For example, the sequence could change from '->(A, X(B[550], C[450]), D)' to '->(A, X(B[700], C[300]), D)' or vice versa, where 'A', 'B', 'C', and 'D' are activities, '->' symbolizes sequential flow, " 
#             "'X' represents a XOR or conditional operation within the BPMN (Business Process Model and Notation), and the information inside the '[]' represent the activity frequency. "
#             "This indicates that before the drift, activity 'B' and 'C' frequency were similar, while after the drift, they were different. "
#             "In other words, before the drift, the frequency of activities 'B' and 'C' had not statistical difference, and after the drift, theirs frequencies had statiscial difference. "
#             "Note: In such scenarios, the involved activities hasn't any control-flow difference before and after the drift, only the branch frequency or branch probability have difference. "
#             "Tip: Investigate activities which are in XORs (X) fragments, and transitions which are in Changed_transition_probability or Changed_transition_frequency but is not in Transitions_new or Transitions_deleted. ")
#     }

#     return characterization_controlflow_change_patterns