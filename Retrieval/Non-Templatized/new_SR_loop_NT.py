import re
import torch  # the main PyTorch library
import numpy as np
import json
import pickle
import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_factorial(n):
    fact = 1
    for i in range(1, n+1):
        fact = fact * i
    return fact

# Specify the parent directory where databases are located
path_to_database_directory = '/path/to/databases/directory'

# Specify the directory where embeddings and outputs will be saved
output_directory = '/path/to/output/directory'

# List of databases to process
db_list = ['database1', 'database2', 'database3']

# Loop over each database
for db_name in db_list:
    print(f'>>>> db_name: {db_name}')

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the SentenceTransformer model using BERT-based model
    cache_dir = 'sentence-transformers/all-mpnet-base-v2'  # or other suitable BERT-based model directory
    model = SentenceTransformer(cache_dir)

    # Load JSON data for the current database
    with open(f'{path_to_database_directory}/{db_name}_set1.json', 'r') as file:
        generated_data1 = json.load(file)
        print(f'>>> generated_data1: {len(generated_data1)}')

    # Prepare a list of evidence from the JSON data
    all_evs_embed = []
    ev_list = []
    for k in generated_data1.keys():
        for idx, j in enumerate(generated_data1[k]['actual_evidence']):
            ev = j
            ev_list.append(ev)

    # Encode all evidence using SentenceTransformer
    all_evs_embed = model.encode(ev_list)

    # Create directory for storing embeddings and outputs if it doesn't exist
    if not os.path.exists(f'{output_directory}/{db_name}_embeddings'):
        os.makedirs(f'{output_directory}/{db_name}_embeddings')
        print(f"Subdirectory '{db_name}_embeddings' created successfully.")
    else:
        print(f"Subdirectory '{db_name}_embeddings' already exists.")

    # Save embeddings to a pickle file
    with open(f'{output_directory}/{db_name}_embeddings/{db_name}_all_evidences_embeddings.pkl', 'wb') as file:
        pickle.dump(all_evs_embed, file)

    print('Embeddings saved.')

    # Load embeddings from the pickle file
    with open(f'{output_directory}/{db_name}_embeddings/{db_name}_all_evidences_embeddings.pkl', 'rb') as file:
        all_evs_embed = pickle.load(file)

    recall = 20
    pred_evidences = {}

    # Process each question in the database
    for oidx, k in enumerate(generated_data1.keys()):
        test = generated_data1[k]['question']
        print(oidx + 1)
        ev_scores = {}
        test = re.sub('[^A-Za-z0-9]+', ' ', test)
        test = re.sub(r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b', "100-100-100", test)
        test = re.sub(r'^([\s\d]+)$', "100", test)
        test = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", test)

        # Encode question into embeddings
        all_ques_data1_embd = model.encode(test)

        cnt = -1
        all_evs_list = []
        
        # Calculate similarity scores between question embeddings and evidence embeddings
        for oidx, k2 in enumerate(generated_data1.keys()):
            for idx, j in enumerate(generated_data1[k2]['actual_evidence']):
                cnt += 1
                ev = j
                ev = re.sub('[^A-Za-z0-9]+', ' ', ev.split('refers to')[0])

                if ev.strip() in all_evs_list:
                    continue
                else:
                    all_evs_list.append(ev.strip())

                ev_str = ev.split()
                denominator = get_factorial(len(ev_str))

                ev_list = [' '.join(ev_str[i: j]) for i in range(len(ev_str))
                           for j in range(i + 1, len(ev_str) + 1)]
                ev_list.sort(key=lambda x: x.count(' '), reverse=True)
                len_list = [len(s.split()) + 1 for s in ev_list]

                if len(len_list) == 0:
                    print("len_list is empty, using dummy value for ev_list_weight")
                    ev_list_weight = np.array([1.0])  # Dummy value
                else:
                    ev_list_weight = softmax(len_list)

                all_evidences_data1_embd = all_evs_embed[cnt]

                # Calculate similarity score between evidence and question embeddings
                similarity = np.matmul(all_evidences_data1_embd, all_ques_data1_embd)
                max_rows = np.amax(similarity, axis=0)
                evidence_score = max_rows
                ev_scores[j] = evidence_score

        # Select top evidence based on scores and store in pred_evidences
        top_evs = list(dict(sorted(ev_scores.items(), key=lambda item: item[1])).keys())
        top_evs.reverse()
        top_evs = top_evs[:recall]
        ev_filtered = []
        for i in top_evs:
            ev_filtered.append([i, np.float64(ev_scores[i])])
        pred_evidences[generated_data1[k]['question']] = ev_filtered

        # Save predictions to a JSON file
        with open(f'{output_directory}/{db_name}_embeddings/{db_name}_pred_evidences_iid_matrix.json', 'w') as file:
            json.dump(pred_evidences, file)
            print(f'Saved: {output_directory}/{db_name}_embeddings/{db_name}_pred_evidences_iid_matrix.json')

