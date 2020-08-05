###########
# IMPORTS #
###########

# Standard Library Imports
from collections import OrderedDict, Counter
import os

# Third Party Library Imports
import pandas as pd

# Local Library Imports


########
# CODE #
########


def get_labels(stage_num: str, team_num: str, team_path: str) -> None:
    """
    - Best, Invalid, Conflict, and Argument Labels
    - Best Labels Gets The Majority Label For A Sentence
    - Invalid Label Is For Invalid Sentences
    - Conflict Label Is For Sentences That Don't Have A Clear Majority Label
    - Argument Label Is For The 5 Main Categories: [FACT, ISSUE, RLH, ANALYSIS, CONCLUSION]
    - Non-Argument Label Is For The Other 2 Categories: [OTHER, INVALID SENTENCE]
    - Takes all the CSV files in a team's folder and gets labels and put them into their respective DataFrames
    - Convert DataFrames into a CSV file
        - "best_labels_filename" format:        "s<stage_number>_t<team_number>_best_labels.csv"
        - "conflict_labels_filename" format:    "s<stage_number>_t<team_number>_conflict_labels.csv"
        - "invalid_labels_filename" format:     "s<stage_number>_t<team_number>_invalid_labels.csv"
        - "argument_labels_filename" format:     "s<stage_number>_t<team_number>_argument_labels.csv"

    :param stage_num:   Stage Number In Dataset
    :param team_num:    Team Number In Stage
    :param team_path:   The path to where the team's annotation folder is located
    :return:            None
    """

    # Map Label IDs To Label Text
    labels = {
        1: 'Fact',
        2: 'Issue',
        3: 'Rule/Law/Holding',
        4: 'Analysis',
        5: 'Conclusion',
        6: 'Other',
        7: 'Invalid sentence'
    }

    data = OrderedDict()                                                                # Keeps Sentences (Keys) in Order of Input
    best_df, conflict_df, invalid_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()   # DataFrames for BIC Labels
    argument_df = pd.DataFrame()                                                        # DataFrame For Argument and Non-Argument Sentences
    csv_files = sorted(os.listdir(team_path))                                           # Get List of All CSV Files in a Team

    for csv_file in csv_files:
        # Skip Over "best_labels.csv" or "conflict_labels.csv" Files, If Found
        if "best_labels" in csv_file or "conflict_labels" in csv_file or "invalid_labels" in csv_file:
            continue

        csv_file_path = os.path.join(team_path, csv_file)
        temp_df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

        # Extract Sentences And Label_IDs From CSV File
        texts, label_ids = temp_df['text'].to_numpy(), temp_df['label_id'].to_numpy()

        for text, label_id in zip(texts, label_ids):
            if data.get(text, 0) != 0:
                data[text].append(label_id)     # If Sentence Is In The Dict, Append To List of Team's Labels
            else:
                data[text] = [label_id]         # Otherwise, Add New Sentence To Dict, With List of Labels

    # Find The Appropriate Label For Every Sentence
    for text, label_ids in data.items():
        counts = Counter(label_ids)
        two_most_freq = counts.most_common(2)   # Format: [(label_id, frequency_count), (...), ...]

        # Check If The Top 2 Labels Are The Same Count (This Indicates That There Is No Majority Label)
        if len(two_most_freq) > 1 and two_most_freq[0][1] == two_most_freq[1][1]:
            label_texts = ", ".join(labels[label_id] for label_id in label_ids)     # Convert List of Labels To A String
            conflict_df = conflict_df.append([[text, label_texts]], ignore_index=True)

        # Check If The Majority Label is a Invalid Sentence
        elif (best_label_id := two_most_freq[0][0]) == 7:
            invalid_df = invalid_df.append([[text, best_label_id, labels[best_label_id]]], ignore_index=True)
            argument_df = argument_df.append([[text, 1, "Non-Argument"]], ignore_index=True)

        # Append The Best Label To The Best DataFrame (Do Not Append "Other" Sentences)
        else:
            if best_label_id != 6:
                best_df = best_df.append([[text, best_label_id, labels[best_label_id]]], ignore_index=True)
                argument_df = argument_df.append([[text, 0, "Argument"]], ignore_index=True)
            else:
                argument_df = argument_df.append([[text, 1, "Non-Argument"]], ignore_index=True)

    best_df.columns = ['text', 'label_id', 'label_text']        # Set Column Names Of Best Labels DataFrame
    best_df.sort_values(by=['label_id'], inplace=True)          # Sort Sentences By Label ID, For Best DataFrame
    argument_df.columns = ['text', 'label_id', 'label_text']    # Set Column Names Of Argument Labels DataFrame
    argument_df.sort_values(by=['label_id'], inplace=True)      # Sort Sentences By Label ID, For Argument DataFrame
    invalid_df.columns = ['text', 'label_id', 'label_text']     # Set Column Names Of Invalid Labels DataFrame
    conflict_df.columns = ['text', 'label_text']                # Set Column Names Of The Conflict Labels DataFrame

    # Make Filename For BIC Label CSV Files
    best_labels_filename = "".join(['s', stage_num,
                                    '_t', team_num,
                                    '_best_labels.csv'])
    invalid_labels_filename = "".join(['s', stage_num,
                                       '_t', team_num,
                                       '_invalid_labels.csv'])
    conflict_labels_filename = "".join(['s', stage_num,
                                        '_t', team_num,
                                        '_conflict_labels.csv'])
    argument_labels_filename = "".join(['s', stage_num,
                                        '_t', team_num,
                                        '_argument_labels.csv'])

    # Get Full Path To Best Labels and Conflict Labels CSV Files
    best_labels_path = os.path.join(team_path, best_labels_filename)
    invalid_labels_path = os.path.join(team_path, invalid_labels_filename)
    conflict_labels_path = os.path.join(team_path, conflict_labels_filename)
    argument_labels_path = os.path.join(team_path, argument_labels_filename)

    # Convert Best Labels and Conflict Labels DataFrames Into CSV File
    best_df.to_csv(best_labels_path, index=False, encoding='ISO-8859-1')
    invalid_df.to_csv(invalid_labels_path, index=False, encoding='ISO-8859-1')
    conflict_df.to_csv(conflict_labels_path, index=False, encoding='ISO-8859-1')
    argument_df.to_csv(argument_labels_path, index=False, encoding='ISO-8859-1')


def clean(dataset_path: str) -> None:
    """
    - Clean Raw Docanno CSV Files for Training
    - Splits Data Into Best Labels and Conflict Labels for Each Stage and Each Team

    :return:    None
    """

    stages = sorted(os.listdir(dataset_path))      # List All Stages in Dataset Folder

    # Loop Through All Selected Stages
    for stage in stages:
        stage_path = os.path.join(dataset_path, stage)
        teams = sorted(os.listdir(stage_path))
        stage_num = stage[-1]

        # Loop Through All Teams In That Stage
        for team in teams:
            team_path = os.path.join(stage_path, team)
            team_num = team[-1]
            get_labels(stage_num, team_num, team_path)  # Compile and Consolidate Best Labels Among Team Members
