import re
import torch
import numpy as np
import time
import json
import pickle
import os
import random
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from bert_score import score
from sentence_transformers import SentenceTransformer

def get_embeddings(selected_demos):
    # Placeholder for evidence_to_index_mapping and all_evs_embed
    # Replace with actual implementation details
    embeddings = []
    masks = []
    max_length = 50

    for demo in selected_demos:
        idx = evidence_to_index_mapping[demo]  # Get index from evidence_to_index_mapping
        embedding = all_evs_embed[idx]         # Get corresponding embedding from all_evs_embed
        
        current_length = embedding.shape[1]  # Length of the second dimension
        
        # Truncate or pad the embedding to make its second dimension 50
        if current_length > max_length:
            padded_embedding = embedding[:, :max_length, :]
            mask = np.ones((embedding.shape[0], max_length, 1))
        else:
            padding = ((0, 0), (0, max_length - current_length), (0, 0))  # Pad the second dimension
            padded_embedding = np.pad(embedding, padding, mode='constant')
            mask = np.concatenate((np.ones((embedding.shape[0], current_length, 1)), 
                                   np.zeros((embedding.shape[0], max_length - current_length, 1))), axis=1)
        
        embeddings.append(padded_embedding)
        masks.append(mask)
    
    embeddings_array = np.array(embeddings)
    masks_array = np.array(masks)
    return embeddings_array, masks_array

def set_coverage(query, selected_demos, new_demo):
    # Placeholder for tokenizer and model initialization
    # Replace with actual tokenizer and model setup
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_layer_hidden_states = outputs.last_hidden_state
    query_embedding = last_layer_hidden_states.numpy()

    selected_demos_embedding, masks_array_Z = get_embeddings(selected_demos)
    new_demo_embedding, mask_z_prime = get_embeddings([new_demo])

    query_embedding_broadcasted = np.broadcast_to(query_embedding, (len(selected_demos), query_embedding.shape[1], query_embedding.shape[2]))

    max_scores = []
    for i in range(query_embedding.shape[1]):
        cosine_similarities_Z = []
        for j in range(len(selected_demos)):
            masked_demo = selected_demos_embedding[j] * masks_array_Z[j]
            cosine_sim_Z = cosine_similarity(query_embedding_broadcasted[j], masked_demo)
            cosine_similarities_Z.append(cosine_sim_Z.max())
        max_cosine_sim_Z = max(cosine_similarities_Z)

        masked_new_demo = new_demo_embedding[0] * mask_z_prime[0]
        cosine_sim_z_prime = cosine_similarity(query_embedding[0], masked_new_demo).max()

        max_coverage = max(max_cosine_sim_Z, cosine_sim_z_prime)
        max_scores.append(max_coverage)
    
    set_coverage_score = sum(max_scores)
    return set_coverage_score

def argmax_setcov(T, x_test, Z_curr):
    best_demonstration = None
    max_coverage = float('-inf')

    for demonstration in T:
        coverage = set_coverage(x_test, Z_curr, demonstration)
        if coverage > max_coverage:
            max_coverage = coverage
            best_demonstration = demonstration

    return best_demonstration, max_coverage

def greedy_set_coverage(T: List[str], xtest: str, k: int) -> List[str]:
    T = random.sample(T, len(T))
    Z = []
    Zcurr = []
    curr_cov = float('-inf')

    while len(Z) < k:
        z_star, next_cov = argmax_setcov([item for item in T if item not in Z], xtest, Zcurr)
        
        if next_cov > curr_cov:
            curr_cov = next_cov
            Z.append(z_star)
            Zcurr.append(z_star)
        
    return Z

# Placeholder for cache_dir and tokenizer/model initialization
# Replace with actual paths and model setup
cache_dir = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model = AutoModel.from_pretrained(cache_dir)

# Placeholder for dataset paths and processing logic
# Replace with actual paths and dataset processing logic
db_list = ['debit_card_specializing', 'toxicology', 'financial', 'european_football_2', 'superhero', 'student_club', 'thrombosis_prediction', 'california_schools', 'card_games', 'codebase_community', 'formula_1']

for db_name in db_list:
    print(f'**************** {db_name} ********************')
    
    # Placeholder for dataset loading and processing
    # Replace with actual dataset loading and processing logic
    with open(f'/path/to/{db_name}_set1.json','r') as file:
        generated_data1 = json.load(file)


    # Placeholder for evidence processing and embeddings
    # Replace with actual evidence processing and embedding logic
    all_evs_embed = {}
    ev_list = []
    evidence_to_index_mapping = {}
    index_to_evidence_mapping = {}

    for k in generated_data1.keys():
        for idx, j in enumerate(generated_data1[k]['generic_evidence']):
            ev = j
            # Process ev here if necessary
            ev_list.append(ev)
            evidence_to_index_mapping[ev] = idx
            index_to_evidence_mapping[idx] = ev

    # Placeholder for saving embeddings
    # Replace with actual saving logic or remove if not needed
    with open(f'/path/to/{db_name}_all_evidences_embeddings_BERT.pkl','wb') as file:
        pickle.dump(all_evs_embed, file)

    # Placeholder for loading embeddings
    # Replace with actual loading logic
    with open(f'/path/to/{db_name}_all_evidences_embeddings_BERT.pkl','rb') as file:
        all_evs_embed = pickle.load(file)

    # Placeholder for dataset processing and coverage computation
    # Replace with actual dataset processing and coverage computation logic
    for oidx, k in enumerate(generated_data1.keys()):
        test = generated_data1[k]['question']
        ev_scores = {}
        test = re.sub('[^A-Za-z0-9]+', ' ', test)
        K = 4
        start_time = time.time()

        top_evs = greedy_set_coverage(ev_list, test, K)

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution Time:", execution_time)

        # Placeholder for saving predictions
        # Replace with actual saving logic or remove if not needed
        pred_evidences = {}
        for i in top_evs:
            ev_filtered = [[ev_refers_to_to_ev_complete_mapping[i]]]
            ev_filtered.append([ev_refers_to_to_ev_complete_mapping[i]])
            pred_evidences[generated_data1[k]['question']] = ev_filtered

            with open(f'/path/to/{db_name}_pred_evidences_iid_matrix.json','w') as file:
                json.dump(pred_evidences, file, indent=4)

