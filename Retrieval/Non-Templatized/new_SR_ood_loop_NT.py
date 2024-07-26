import re
import torch  # PyTorch library
import numpy as np
import json
import pickle
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_factorial(n):
    """
    Compute factorial of a number.
    """
    fact = 1
    for i in range(1, n+1):
        fact = fact * i
    return fact

# Define the output folder for saving embeddings and results
folder_2 = 'outputs'

# List of databases to process
db_list = ['debit_card_specializing', 'toxicology', 'financial', 'european_football_2', 'superhero', 'student_club', 'thrombosis_prediction', 'california_schools', 'card_games', 'codebase_community', 'formula_1']

# Initialize SentenceTransformer model
cache_dir = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(cache_dir)

# Loop through each database
for db_name in db_list:
    print(f'>>>> Processing database: {db_name}')

    # Load the first set of generated data (assuming it's evidence data)
    with open(f'/path/to/generated_data/{db_name}_set1.json', 'r') as file:
        generated_data1 = json.load(file)
        print(f'>>> generated_data1: {len(generated_data1)}')

    # Load the second set of generated data (assuming it's test/evaluation data)
    with open(f'/path/to/generated_data/{db_name}_set2.json', 'r') as file:
        generated_data2 = json.load(file)

    # Load precomputed BERT embeddings for all evidences
    with open(f'/path/to/embeddings/{folder_2}/{db_name}_all_evidences_embeddings_BERT.pkl', 'rb') as file:
        all_evs_embed = pickle.load(file)

    # Set recall value for retrieving top evidence
    recall = 20
    pred_evidences = {}

    # Process each question from the second set of generated data
    for oidx, k in enumerate(generated_data2.keys()):
        test = generated_data2[k]['question']
        ev_scores = {}

        for q in test.split():
            try:
                float(q)
                test = test.replace(q,'100')
            except:
                continue

        test_str = test.split()

        # Generate all possible subquestions from the test question
        ques_list = [' '.join(test_str[i: j]) for i in range(len(test_str))
                    for j in range(i + 1, len(test_str) + 1)]
        ques_list.sort(key=lambda x: x.count(' '), reverse=True)

        # Encode all subquestions using SentenceTransformer
        all_ques_data1_embd = model.encode(ques_list)
        all_ques_data1_embd = np.transpose(all_ques_data1_embd)
        
        cnt = -1
        all_evs_list = []

        # Iterate over each evidence in the first set of generated data
        for oidx, k2 in enumerate(generated_data1.keys()):
            for idx, j in enumerate(generated_data1[k2]['actual_evidence']):
                cnt += 1
                ev = j
                ev = re.sub('[^A-Za-z0-9]+', ' ', ev.split('refers to')[0])

                # Ensure unique evidence entries
                if ev.strip() in all_evs_list:
                    continue
                else:
                    all_evs_list.append(ev.strip())

                ev_str = ev.split()
                denominator = get_factorial(len(ev_str))

                ev_list = [' '.join(ev_str[i: j]) for i in range(len(ev_str))
                           for j in range(i + 1, len(ev_str) + 1)]
                ev_list.sort(key=lambda x: x.count(' '), reverse=True)
                len_list = [len(s.split())+1 for s in ev_list]

                if len(len_list) == 0:
                    ev_list_weight = np.array([1.0])  # Dummy value
                else:
                    ev_list_weight = softmax(len_list)

                all_evidences_data1_embd = all_evs_embed[cnt]

                # Compute similarity between evidence and subquestions
                similarity = np.matmul(all_evidences_data1_embd, all_ques_data1_embd)
                max_rows = np.amax(similarity, axis=0)
                evidence_score = max_rows
                ev_scores[j] = evidence_score

        # Retrieve top scored evidence for the current question
        top_evs = list(dict(sorted(ev_scores.items(), key=lambda item: item[1])).keys())
        top_evs.reverse()
        top_evs = top_evs[:recall]
        ev_filtered = []
        for i in top_evs:
            ev_filtered.append([i, np.float64(ev_scores[i])])
        pred_evidences[generated_data2[k]['question']] = ev_filtered

        # Save predicted evidence for the current database
        with open(f'/path/to/output/{folder_2}/{db_name}_pred_evidences_ood_matrix.json', 'w') as file:
            json.dump(pred_evidences, file)
    print(f'Saved: {folder_2}/{db_name}_pred_evidences_ood_matrix.json')
