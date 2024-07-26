import re
import torch  # the main PyTorch library
import torch.nn.functional as f
from sentence_transformers import CrossEncoder
import numpy as np
import time
import json
import pickle
import os

# Define the folder for output
folder_2 = 'output'

# Check if the output directory exists, if not, create it
output_dir = f'/path/to/outputs/{folder_2}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Subdirectory '{folder_2}' created successfully.")
else:
    print(f"Subdirectory '{folder_2}' already exists.")

# Define the list of databases to process
db_list = ['thrombosis_prediction', 'california_schools', 'card_games', 'codebase_community', 'formula_1']

# Process each database in the list
for db_name in db_list:
    print(f'******************************************')
    print(f'*******>>>> db_name: {db_name}************\n')

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the path to the database directory
    path_to_database_directory_dev = f'/path/to/databases/dev_databases'

    # Initialize the CrossEncoder model
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Load the generated data for the current database
    with open(f'/path/to/corrected/data/{db_name}_set1.json', 'r') as file:
        generated_data1 = json.load(file)

    recall = 10
    pred_evidences = {}

    # Iterate over the keys in the generated data
    for oidx, k in enumerate(generated_data1.keys()):
        test = generated_data1[k]['question']
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
        pred_evidences[generated_data1[k]['question']] = ev_filtered

        # Save the predicted evidences to a JSON file
        output_path = f'{output_dir}/{db_name}_pred_evidences_iid_matrix.json'
        with open(output_path, 'w') as file:
            json.dump(pred_evidences, file)
            print(f'saved: {output_path}')
