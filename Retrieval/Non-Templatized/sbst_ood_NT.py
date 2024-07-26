import re
import torch  # the main PyTorch library
import torch.nn.functional as f
from sentence_transformers import CrossEncoder
import numpy as np
import time
import json
import pickle

# Define the folder for output
folder_2 = 'outputs'

# Define the list of databases to process
db_list = ['debit_card_specializing', 'toxicology', 'financial', 'european_football_2', 'superhero', 'student_club', 'thrombosis_prediction', 'california_schools', 'card_games', 'codebase_community', 'formula_1']

# Process each database in the list
for db_name in db_list:
    print(f'******************************************')
    print(f'*******>>>> db_name: {db_name}************\n')

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the path to the database directory
    path_to_database_directory_dev = '/path/to/databases/dev_databases'

    # Initialize the CrossEncoder model
    model = CrossEncoder("cross-encoder/stsb-roberta-base")

    # Load the generated data for the current database
    with open(f'/path/to/corrected/data/{db_name}_set1.json', 'r') as file:
        generated_data1 = json.load(file)
    with open(f'/path/to/comp_splits_generic/{db_name}_set2.json', 'r') as f:
        generated_data2 = json.load(f)

    recall = 10
    pred_evidences = {}

    # Iterate over the keys in the second set of generated data
    for oidx, k in enumerate(generated_data2.keys()):
        test = generated_data2[k]['question']
        print(oidx + 1)
        ev_scores = {}

        # Replace numbers in the question with a placeholder
        for q in test.split():
            try:
                float(q)
                test = test.replace(q, '100')
            except:
                continue

        cnt = -1
        all_evs_list = []
        similarity = []

        # Calculate similarity scores for each evidence
        for oidx, k2 in enumerate(generated_data1.keys()):
            for idx, j in enumerate(generated_data1[k2]['actual_evidence']):
                cnt += 1
                ev = j
                ev_scores[j] = model.predict([test, ev])

        # Select the top evidences based on similarity scores
        top_evs = list(dict(sorted(ev_scores.items(), key=lambda item: item[1])).keys())
        top_evs.reverse()
        top_evs = top_evs[:recall]
        ev_filtered = []

        for i in top_evs:
            ev_filtered.append([i, np.float64(ev_scores[i])])
        pred_evidences[generated_data2[k]['question']] = ev_filtered

        # Save the predicted evidences to a JSON file
        output_path = f'/path/to/outputs/{folder_2}/{db_name}_pred_evidences_ood_matrix.json'
        with open(output_path, 'w') as file:
            json.dump(pred_evidences, file)
            print(f'saved: {folder_2}/{db_name}_pred_evidences_ood_matrix.json')

