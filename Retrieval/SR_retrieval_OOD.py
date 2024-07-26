import re
import torch # the main pytorch library
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import json
import pickle

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_factorial(n):
	fact = 1
 
	for i in range(1, n+1):
	    fact = fact * i
	return fact

folder='CG'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_map={'TP':'thrombosis_prediction', 'CS': 'california_schools', 'CG': 'card_games', 'CC':'codebase_community', 'F1':'formula_1', 'DCS':'debit_card_specializing', 'EF2': 'european_football_2', 'FI': 'financial', 'SC': 'student_club', 'SH': 'superhero', 'TC': 'toxicology'}

folder= ## Key from the folder map for the database you want to work on

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_database_directory_dev = ##PATH TO THE DEV SPLIT DIRECTORY OF BIRDSQL

cache_dir = ## PATH TO THE CACHE DIRECTORY OF MODEL (IF STORED LOCALLY)

model = SentenceTransformer(cache_dir)

with open('IID Domain Statement File','r') as file:
  generated_data1=json.load(file)

with open('OOD Domain Statement File','r') as file:
  generated_data2=json.load(file)

with open('Pre_computed_TDS_embeddings_from_SR_retrieval_IID','rb') as file:
	all_evs_embed=pickle.load(file)

recall=10
pred_evidences={}

## To retrieve the relevant evidences for each question
for oidx,k in enumerate(generated_data2.keys()):
	test=generated_data2[k]['question']
	print(oidx+1)

	ev_scores={}
	test = re.sub('[^A-Za-z0-9]+', ' ', test)
	test = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", test)
	test = re.sub(r'^([\s\d]+)$', "100", test)
	test = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", test)

	for q in test.split():
	  try:
	      float(q)
	      test = test.replace(q,'100')
	  except:
	      continue

	test_str = test.split()

	ques_list = [' '.join(test_str[i: j]) for i in range(len(test_str))
	          for j in range(i + 1, len(test_str) + 1)]

	ques_list.sort(key=lambda x: x.count(' '), reverse=True)
	
	cnt=-1
	all_evs_list=[]
	
	for oidx,k2 in enumerate(generated_data1.keys()):
		for idx,j in enumerate(generated_data1[k2]['generic_evidence']):
			cnt+=1
			ev=j
			ev = re.sub('[^A-Za-z0-9]+', ' ', ev.split('refers to')[0])

			if ev.strip() in all_evs_list:
				continue
			else:
				all_evs_list.append(ev.strip())

			all_evidences_data1_embd = all_evs_embed[cnt]

			start,end = 0,0

			for lidx,l in enumerate(ques_list):
				if len(l.split()) == len(ev.split()) + 2:
					start = lidx
					continue
				if len(l.split()) == len(ev.split()) - 2:
					end = lidx
					break
			if end == 0:
				end = len(ques_list)-1
			
			ques_list  = ques_list[start:end+1]
			all_ques_data1_embd = model.encode(ques_list)
			all_ques_data1_embd=np.transpose(all_ques_data1_embd)

			similarity = np.matmul(all_evidences_data1_embd,all_ques_data1_embd)
			max_rows = np.amax(similarity, axis=0)
		
			evidence_score = max_rows
			ev_scores[j] = evidence_score

	top_evs=list(dict(sorted(ev_scores.items(), key=lambda item: item[1])).keys())
	top_evs.reverse()
	top_evs=top_evs[:recall]
	ev_filtered=[]
	for i in top_evs:
		ev_filtered.append([i, ev_scores[i]])
	
	pred_evidences[generated_data2[k]['question']] = ev_filtered

with open('/pred_evidences_ood_matrix_whole_new','wb') as file:
    pickle.dump(pred_evidences,file)
	