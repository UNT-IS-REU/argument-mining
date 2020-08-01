###########
# IMPORTS #
###########

# Standard Library Imports
import os
from collections import OrderedDict, Counter

# Third Party Library Imports
import pandas as pd

########
# CODE #
########


def get_best_and_conflict_labels(stage_num: str, team_num: str, team_path: str) -> None:
    """
    - Takes all the CSV files in a team's folder and gets the majority selected label for every sentence and any sentences
    that has conflicted labels (meaning there is not definitive majority label)
    - Puts best label and best label ID into a DataFrame, along with the sentence
    - Puts conflict label and conflict label text into a DataFrame, along with the sentence
    - Convert DataFrames into a CSV file
        - "best_labels_filename" format: "s<stage_number>_t<team_number>_best_labels.csv"
        - "conflict_labels_filename" format: "s<stage_number>_t<team_number>_conflict_labels.csv"

    :param stage_num:   Stage Number In Dataset
    :param team_num:    Team Number In Stage
    :param team_path:   The path to where the team's annotation folder is located
    :return:            None
    """

    # Map Label IDs To Label Text
    labels = {
        1: 'Facts',
        2: 'Issue',
        3: 'Rule/Law/Holding',
        4: 'Analysis',
        5: 'Conclusion',
        6: 'Others',
        7: 'Invalid sentences'
    }

    data = OrderedDict()                        # Keeps Sentences (Keys) in Order of Input
    best_df = pd.DataFrame()                    # DataFrame for Best Labels
    conflict_df = pd.DataFrame()                # DataFrame for Best Labels
    csv_files = sorted(os.listdir(team_path))   # Get List of All CSV Files in a Team

    for csv_file in csv_files:
        # Skip Over "best_labels.csv" or "conflict_labels.csv" Files, If Found
        if "best_labels" in csv_file or "conflict_labels" in csv_file:
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

    # Find The Best (Majority) Label For Every Sentence
    for text, label_ids in data.items():
        counts = Counter(label_ids)
        two_most_freq = counts.most_common(2)

        # Check If The Top 2 Labels Are The Same Count
        if len(two_most_freq) > 1 and two_most_freq[0][1] == two_most_freq[1][1]:
            label_texts = ", ".join([labels[label_id] for label_id in label_ids])
            conflict_df = conflict_df.append([[text, label_texts]], ignore_index=True)  # Append To Conflict DataFrame As Row
        else:
            best_label_id = two_most_freq[0][0]
            if best_label_id != 7:
                best_df = best_df.append([[text, best_label_id, labels[best_label_id]]], ignore_index=True)  # Append To Best DataFrame As Row

    best_df.columns = ['text', 'label_id', 'label_text']    # Set Column Names Of Best Labels DataFrame
    best_df.sort_values(by=['label_id'], inplace=True)      # Sort Sentences By Label ID
    conflict_df.columns = ['text', 'label_text']            # Set Column Names Of The Conflict Labels DataFrame

    # Make Filename For Best Labels and Conflict Labels CSV File
    best_labels_filename = "".join(['s', stage_num,
                                    '_t', team_num,
                                    '_best_labels.csv'])
    conflict_labels_filename = "".join(['s', stage_num,
                                        '_t', team_num,
                                        '_conflict_labels.csv'])

    # Get Full Path To Best Labels and Conflict Labels CSV Files
    best_labels_path = os.path.join(team_path, best_labels_filename)
    conflict_labels_path = os.path.join(team_path, conflict_labels_filename)

    # Convert Best Labels and Conflict Labels DataFrames Into CSV File
    best_df.to_csv(best_labels_path, index=False, encoding='ISO-8859-1')
    conflict_df.to_csv(conflict_labels_path, index=False, encoding='ISO-8859-1')


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
            get_best_and_conflict_labels(stage_num, team_num, team_path)  # Compile and Consolidate Best Labels Among Team Members
