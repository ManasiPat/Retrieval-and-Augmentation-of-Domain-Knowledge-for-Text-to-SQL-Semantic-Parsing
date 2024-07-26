from __future__ import absolute_import

import openai
from openai.error import RateLimitError, InvalidRequestError
import json
import os
import sqlite3
import re
import numpy as np
from tqdm import tqdm, trange
import time
import sqlparse
import random

import torch # the main pytorch library
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_to_database_directory_dev = ##PATH TO THE DEV SPLIT DIRECTORY OF BIRDSQL

cache_dir = ## PATH TO THE CACHE DIRECTORY OF MODEL (IF STORED LOCALLY)

import requests

def init1():
    GOOGLE_API_KEY = "KEY1"
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {
      "temperature": 0,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }
    # model = genai.GenerativeModel('gemini-pro')
    # model_name="gemini-1.5-flash-latest"
    model = genai.GenerativeModel(
      model_name="gemini-pro",
      generation_config=generation_config,
    )

    return model

def init2():
    GOOGLE_API_KEY = "KEY2"
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {
      "temperature": 0,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }
    # model = genai.GenerativeModel('gemini-pro')
    model = genai.GenerativeModel(
      model_name="gemini-pro",
      generation_config=generation_config,
    )

    return model

model = init1()
retry_cnt=0

# def query(payload):
#     global retry_cnt
#     try:
#         response = model.generate_content(payload).text
#         retry_cnt=0
#     except Exception as e:
#         retry_cnt+=1
#         print(f"Exception: {e} occurred")
#         print(f"Waiting for 10 Seconds....")
#         for _ in tqdm(range(10), desc="Second(s) Passed"):
#             time.sleep(1)
#         print(f"Continuing") 
#         if retry_cnt >= 5:
#             return 'NA'
#         response = query(payload)

#     return response

def query(payload):
    global retry_cnt
    global model
    try:
        response = model.generate_content(payload).text
        retry_cnt=0
    except Exception as e:
        retry_cnt+=1
        if retry_cnt >= 5:
            return 'NA'
        if retry_cnt % 2 != 0 and retry_cnt!=0:
            print(f"Exception: {e} occurred")
            print(f"Waiting for 10 Seconds....")
            for _ in tqdm(range(10), desc="Second(s) Passed"):
                time.sleep(1)
            print(f"Continuing")
            model = init1()
            response = query(payload)
        else:
            model = init2()
            response = query(payload)


    return response


method = ##Method to use for retrieval of Templatized Domain Statements From: QS, FS, BR, LR, SR, No DK, All DS T, All DS NT, IID TDS, OOD File

recall = #Number of retrived evidences

folder_map={'TP':'thrombosis_prediction', 'CS': 'california_schools', 'CG': 'card_games', 'CC':'codebase_community', 'F1':'formula_1', 'DCS':'debit_card_specializing', 'EF2': 'european_football_2', 'FI': 'financial', 'SC': 'student_club', 'SH': 'superhero', 'TC': 'toxicology'}

folder= ## Key from the folder map for the database you want to work on


db_name=folder_map[folder]

threshold = #Thershold Value you want to set, keep 1 by default

# Function to get response from OpenAI API call

def get_answer(sys_prompt, prompt, n, temperature=0, answer=[]):

    if len(answer) > 0:            
        messages = [{"role": "system", "content": sys_prompt}]
        for i in range(len(answer)):
            messages.append({"role": "user", "content": prompt[i]})
            messages.append({"role": "assistant", "content": answer[i]})
        messages.append({"role": "user", "content": prompt[-1]})
    else:
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
        ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=temperature,
            messages=messages
            )

    except RateLimitError as e:
        print(f"Exception: {e} occurred at Sample: {n}")
        print(f"Waiting for 2 Seconds....")
        for _ in tqdm(range(2), desc="Second(s) Passed"):
            time.sleep(1)
        print(f"Continuing at Sample: {n}....")
        response = get_answer(sys_prompt, prompt, n, temperature, answer)

    except InvalidRequestError as e:
        print(f"Exception: {e} occurred at Sample: {n}")
        print(f"Waiting for 2 Seconds....")
        for _ in tqdm(range(2), desc="Second(s) Passed"):
            time.sleep(1)
        print(f"Continuing at Sample: {n}....")
        response = get_answer("", prompt, n, temperature, answer)

    except Exception as e:
        print(f"Exception: {e} occurred at Sample: {n}")
        print(f"Waiting for 10 Seconds....")
        for _ in tqdm(range(10), desc="Second(s) Passed"):
            time.sleep(1)
        print(f"Continuing at Sample: {n}")
        response = get_answer(sys_prompt, prompt, n, temperature,answer)
    return response


def get_result(path,string):

    with open(path) as f:
        generated_data1 = json.load(f)

    path_to_database_directory_dev = '/dev_databases'
    path = os.path.join(path_to_database_directory_dev, db_name, db_name + '.sqlite')

    # SQL Generation
    try:
        conn = sqlite3.connect(path)
    except Exception as e:
        print(f'{idx} ; {database_name} : {e}')
    c=conn.cursor()

    corr_lex_iid=0
    corr_zs=0
    corr_actual = 0
    corr_gen=0
    corr_pred_iid = 0
    corr_pred_all = 0
    corr_generic_iid = 0
    corr_generic_all = 0

    flag = 0
    actual = 'NA'
    prediction = 'NA'
    error1 = 'NA'
    error2 = 'NA'


    if method == 'No DK':
        for idx,k in enumerate(generated_data1.keys()):

            # Generating SQL With No evidences
            
            flag = 0
            actual = 'NA'
            prediction = 'NA'
            error1 = 'NA'
            error2 = 'NA'

            try:
                c.execute(generated_data1[k]['GT_SQL'])
                actual = c.fetchall()
                generated_data1[k]['actual_answer'] = actual
            except Exception as e:
                error1 = e
                flag =1
                generated_data1[k]['actual_answer'] = error1

            print(idx+1)
            evidence=''
          
            prompt='Question: '+generated_data1[k]['question']+'Generate a single SQL in SQLite format for the above question. Do not include any extra text, tag or information other than the SQL query itself.'
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
        generated_data1[k]['predicted_output'] = corrected_query
        # print(corrected_query)
        # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
        # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                    generated_data1[k]['predicted_output'] = corrected_query

                    try:
                        corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                    except:
                        pass

            generated_data1[k]['predicted_SQL_actual_evidence'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_actual_evidence'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_actual_evidence'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_zs = corr_zs + 1

        print('corr_zs: ',corr_zs)
        print('accuracy with Zero Shot: ',corr_zs/len(generated_data1.keys()))

        with open("Output Path",'wb') as file:
            pickle.dump(generated_data1,file)


    if method == 'QS':
        for idx,k in enumerate(generated_data1.keys()):

            # Generating SQL With actual evidences
            
            flag = 0
            actual = 'NA'
            prediction = 'NA'
            error1 = 'NA'
            error2 = 'NA'

            try:
                c.execute(generated_data1[k]['GT_SQL'])
                actual = c.fetchall()
                generated_data1[k]['actual_answer'] = actual
            except Exception as e:
                error1 = e
                flag =1
                generated_data1[k]['actual_answer'] = error1

            print(idx+1)
            evidence=''
            for e in generated_data1[k]['actual_evidence']:
                evidence = evidence + e + '\n'
            # prompt='Question: '+generated_data1[k]['question']+ "\nThe required Domain Knowledge:\n"+evidence+'\nGenerate SQL in SQLite format for the above question:'
            prompt='Question: '+generated_data1[k]['question']+ "\nDomain Knowledge statements, which will be useful to generate the query: "+evidence+'Generate a single SQL in SQLite format for the above question. Do not include any extra text, tag or information other than the SQL query itself.'
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                    generated_data1[k]['predicted_output'] = corrected_query

                    try:
                        corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                    except:
                        pass

                generated_data1[k]['predicted_SQL_actual_evidence'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_actual_evidence'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_actual_evidence'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_actual = corr_actual + 1

        print('corr_actual: ',corr_actual)
        print('accuracy with QS evidences: ',corr_actual/ len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)
        
    if method == 'All DS NT':
        for idx,k in enumerate(generated_data1.keys()):
            if k == 'all_generic_evidence_iid':
                continue
            print(idx+1)
            # Generating SQL With generic evidences from all iid evidences
            flag = 0
            prediction = 'NA'
            error2 = 'NA'

            try:
              c.execute(generated_data1[k]['GT_SQL'])
              actual = c.fetchall()
              generated_data1[k]['actual_answer'] = actual
            except Exception as e:
              error1 = e
              flag =1
              generated_data1[k]['actual_answer'] = error1

            if idx == 0:
                evidence = ''
                for key in enumerate(generated_data1.keys()):
                    for e in generated_data1[k]:
                        evidence = evidence + e + '\n'
                
            
            # prompt='Question: '+generated_data1[k]['question']+ "\nThe required Domain Knowledge:\n"+evidence+'Generate SQL in SQLite format for the above question:'
            prompt='Question: '+generated_data1[k]['question']+ "\nDomain Knowledge statements, some of which might or might not be useful to generate the query: "+evidence+'Generate a single SQL in SQLite format for the above question. Do not include any extra text, tag or information other than the SQL query itself.'
            #print('prompt', prompt)
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                generated_data1[k]['predicted_output'] = corrected_query

                try:
                    corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                except:
                    pass

            generated_data1[k]['predicted_SQL_all_generic_evidence_iid'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_all_generic_evidence_iid'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_all_generic_evidence_iid'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_generic_iid = corr_generic_iid + 1

        print('corr_generic_iid: ',corr_generic_iid)
        print('Accuracy with all NTDS evidences: ',corr_generic_iid/len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)


    if method == 'All DS T':
        for idx,k in enumerate(generated_data1.keys()):
            if k == 'all_generic_evidence_iid':
                continue
            print(idx+1)
            # Generating SQL With generic evidences from all iid evidences
            flag = 0
            prediction = 'NA'
            error2 = 'NA'

            try:
              c.execute(generated_data1[k]['GT_SQL'])
              actual = c.fetchall()
              generated_data1[k]['actual_answer'] = actual
            except Exception as e:
              error1 = e
              flag =1
              generated_data1[k]['actual_answer'] = error1

            if idx == 0:
                evidence = ''
                for e in generated_data1['all_generic_evidence_iid']:
                    evidence = evidence + e + '\n'
                
            

            prompt='Question: '+generated_data1[k]['question']+ "\nDomain Knowledge statements, some of which might or might not be useful to generate the query: "+evidence+'Generate a single SQL in SQLite format for the above question. Do not include any extra text, tag or information other than the SQL query itself.'
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                generated_data1[k]['predicted_output'] = corrected_query

                try:
                    corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                except:
                    pass

            generated_data1[k]['predicted_SQL_all_generic_evidence_iid'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_all_generic_evidence_iid'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_all_generic_evidence_iid'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_generic_iid = corr_generic_iid + 1

        print('corr_generic_iid: ',corr_generic_iid)
        print('Accuracy with all TDS evidences: ',corr_generic_iid/len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)

    if method == 'No DK':
        for idx,k in enumerate(generated_data1.keys()):
            print(idx+1)
            # Generating SQL With BR evidences from iid evidences
            flag = 0
            prediction = 'NA'
            error2 = 'NA'

            try:
              c.execute(generated_data1[k]['GT_SQL'])
              actual = c.fetchall()
              generated_data1[k]['actual_answer'] = actual
            except Exception as e:
              error1 = e
              flag =1
              generated_data1[k]['actual_answer'] = error1

            evidence = ''
            for e in generated_data1[k]['predicted_generic_evidence_iid']:
                evidence = evidence + e + '\n'
                
            prompt='Question: '+generated_data1[k]['question']+ "\nDomain Knowledge statements, some of which might or might not be useful to generate the query: "+evidence+'Generate a single SQL in SQLite format for the above question. Do not include any extra text, tag or information other than the SQL query itself.'
        
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                generated_data1[k]['predicted_output'] = corrected_query

                try:
                    corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                except:
                    pass

            generated_data1[k]['predicted_SQL_generic_evidence_iid'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_generic_evidence_iid'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_generic_evidence_iid'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_generic_iid = corr_generic_iid + 1

        print('corr_generic_iid: ',corr_generic_iid)
        print('Accuracy with generic evidences BERT retrieved from evidences from iid set: ',corr_generic_iid/len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)

    if method == 'SR':
        for idx,k in enumerate(generated_data1.keys()):
            print(idx+1)
            # Generating SQL With SR evidences from iid evidences
            flag = 0
            prediction = 'NA'
            error2 = 'NA'

            try:
              c.execute(generated_data1[k]['GT_SQL'])
              actual = c.fetchall()
              generated_data1[k]['actual_answer'] = actual
            except Exception as e:
              error1 = e
              flag =1
              generated_data1[k]['actual_answer'] = error1

            evidence = ''
            for e in generated_data1[k]['pred_evidences_thresholded']:
                evidence = evidence + e + '\n'
                
            

            prompt='Question: '+generated_data1[k]['question']+ "\nDomain Knowledge statements, some of which might or might not be useful to generate the query: "+evidence+'Generate a single SQL in SQLite format for the above question. Do not include any extra text, tag or information other than the SQL query itself.'
    
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                generated_data1[k]['predicted_output'] = corrected_query

                try:
                    corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                except:
                    pass

            generated_data1[k]['predicted_SQL_generic_evidence'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_generic_evidence'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_generic_evidence'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_generic_iid = corr_generic_iid + 1

        print('corr_generic_iid: ',corr_generic_iid)
        print('Accuracy with SR evidences retrieved from evidences: ',corr_generic_iid/len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)

    if method == 'OR':
        for idx,k in enumerate(generated_data1.keys()):
            print(idx+1)
            flag = 0
            prediction = 'NA'
            error2 = 'NA'

            try:
              c.execute(generated_data1[k]['GT_SQL'])
              actual = c.fetchall()
              generated_data1[k]['actual_answer'] = actual
            except Exception as e:
              error1 = e
              flag =1
              generated_data1[k]['actual_answer'] = error1

            evidence = ''
            for e in generated_data1[k]['predicted_generic_evidence_iid_ai']:
                evidence = evidence + e+ '\n'
                
            
            prompt='Question: '+generated_data1[k]['question']+ "\nDomain Knowledge statements, some of which might or might not be useful to generate the query: "+evidence+'Generate SQL in SQLite format for the above question:'
                
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                generated_data1[k]['predicted_output'] = corrected_query

                try:
                    corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                except:
                    pass

            generated_data1[k]['predicted_SQL_generic_evidence_iid_ai'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_generic_evidence_iid_ai'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_generic_evidence_iid_ai'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_generic_iid = corr_generic_iid + 1

        print('corr_generic_iid: ',corr_generic_iid)
        print('Accuracy with LR evidences retrieved from evidences from iid set: ',corr_generic_iid/len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)

    if method == 'LR':
        for idx,k in enumerate(generated_data1.keys()):
            print(idx+1)
            # Generating SQL With generic evidences from iid evidences lex
            flag = 0
            prediction = 'NA'
            error2 = 'NA'

            try:
              c.execute(generated_data1[k]['GT_SQL'])
              actual = c.fetchall()
              generated_data1[k]['actual_answer'] = actual
            except Exception as e:
              error1 = e
              flag =1
              generated_data1[k]['actual_answer'] = error1

            evidence = ''
            for e in generated_data1[k]['predicted_generic_evidence_iid_lex']:
                evidence = evidence + e+ '\n'
            
            prompt='Question: '+generated_data1[k]['question']+ "\nDomain Knowledge statements, some of which might or might not be useful to generate the query: "+evidence+'Generate a single SQL in SQLite format for the above question. Do not include any extra text, tag or information other than the SQL query itself.'
            
            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                generated_data1[k]['predicted_output'] = corrected_query

                try:
                    corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                except:
                    pass

            generated_data1[k]['predicted_SQL_generic_evidence_iid_lex'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_generic_evidence_iid_lex'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_generic_evidence_iid_lex'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_generic_iid = corr_generic_iid + 1

        print('corr_generic_iid_lex: ',corr_generic_iid)
        print('Accuracy with generic evidences retrieved from evidences from iid set Lex: ',corr_generic_iid/len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)



    if method == 'FS':
        for idx,k in enumerate(generated_data1.keys()):
            print(idx+1)

            # Generating SQL With Few Shots
            flag = 0
            prediction = 'NA'
            error2 = 'NA'

            try:
              c.execute(generated_data1[k]['GT_SQL'])
              actual = c.fetchall()
              generated_data1[k]['actual_answer'] = actual
            except Exception as e:
              error1 = e
              flag =1
              generated_data1[k]['actual_answer'] = error1

            prompt='Given are the 4 examples of SQL generation given a question in Natural Language Form. Generate SQL in SQLite format for the final question'

            for ex in generated_data1[k]['predicted_iid_shots_ood_ai']:
                prompt+='Question: '+ex[0]+'\nSQL: '+ex[1]
            
            prompt+='Question: '+generated_data1[k]['question']+ "\nSQL:"

            corrected_query = query({
    "inputs": sys_prompt+prompt,
})      
            generated_data1[k]['predicted_output'] = corrected_query
            # print(corrected_query)
            # print(corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0])
            # input()
            try:
                corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
            except:
                pass_it=True

            if not pass_it:
                try:
                    c.execute(corrected_query)
                except Exception as e:
                    corrected_query = query({
            "inputs": sys_prompt+prompt+"Correct the generated SQL Syntax considering the following syntax error- "+str(e)+". Do not include any text or explaination other than the SQL query itself in the generated output.",
        })          
                generated_data1[k]['predicted_output'] = corrected_query

                try:
                    corrected_query = corrected_query[0]['generated_text'].split('\n\n')[-1].split(';')[0]
                except:
                    pass

            generated_data1[k]['predicted_SQL_few_shots_iid_ai'] = corrected_query


            try:
                c.execute(corrected_query)
                prediction = c.fetchall()
                generated_data1[k]['predicted_answer_few_shots_iid_ai'] = prediction
            except Exception as e:
                error2 = e
                flag =1
                generated_data1[k]['predicted_answer_few_shots_iid_ai'] = error2
                

            if flag == 0:
                if(actual == prediction):
                    corr_generic_iid = corr_generic_iid + 1

        print('corr_few_shots_iid: ',corr_generic_iid)
        print('Accuracy with few_shots retrieved from evidences from iid set: ',corr_generic_iid/len(generated_data1.keys()))

        with open("Output_Path",'wb') as file:
            pickle.dump(generated_data1,file)

if folder == 'TP':
    sys_prompt= '''You are a database administrator and have designed the following database for thrombosis prediction whose schema is represented as:

    CREATE TABLE Examination (
      ID INTEGER PRIMARY KEY, 
      `Examination Date` DATE,
      `aCL IgG` REAL,
      `aCL IgM` REAL,
      ANA INTEGER ,
      `ANA Pattern` TEXT ,
      `aCL IgA` INTEGER ,
      Diagnosis TEXT ,
      KCT TEXT ,
      RVVT TEXT ,
      LAC TEXT ,
      Symptoms TEXT ,
      Thrombosis INTEGER ,
    );
     
    CREATE TABLE Patient ( 
      ID INTEGER  PRIMARY KEY,
      SEX TEXT ,
      Birthday DATE,
      Description DATE ,
      `First Date` DATE ,
      Admission TEXT ,
      Diagnosis TEXT  -- disease names;
    );

    CREATE TABLE Laboratory (
      ID INTEGER PRIMARY KEY,
      Date DATE PRIMARY KEY,
      GOT INTEGER ,
      GPT INTEGER ,
      LDH INTEGER ,
      ALP INTEGER ,
      TP REAL ,
      ALB REAL ,
      UA REAL ,
      UN INTEGER ,
      CRE REAL ,
      `T-BIL` REAL ,
      `T-CHO` INTEGER ,
      TG INTEGER ,
      CPK INTEGER ,
      GLU INTEGER ,
      WBC REAL ,
      RBC REAL ,
      HGB REAL ,
      HCT REAL ,
      PLT INTEGER ,
      PT REAL ,
      APTT INTEGER ,
      FG REAL ,
      PIC INTEGER ,
      TAT INTEGER ,
      TAT2 INTEGER ,
      `U-PRO` TEXT ,
      IGG INTEGER ,
      IGA INTEGER ,
      IGM INTEGER ,
      CRP TEXT ,
      RA TEXT ,
      RF TEXT ,
      C3 INTEGER ,
      C4 INTEGER ,
      RNP TEXT ,
      SM TEXT ,
      SC170 TEXT ,
      SSA TEXT ,
      SSB TEXT ,
      CENTROMEA TEXT ,
      DNA TEXT ,
      DNA-II INTEGER ,
    );
    '''

    few_shots = '''
Example:
Question: What is the percentage of female patient were born after 1930?
Question Specific domain statement: female refers to Sex = 'F'
Generic Domain Statement for the question specific statement: 'female' refers to Patient.SEX = 'F'

Example:
Question: For patient born between Year 1930 to 1940, how many percent of them were inpatient?
Question Specific domain statement: patient born between Year 1930 to 1940 refers to Birthday BETWEEN '1930-12-31' AND '1940-01-01'
Generic Domain Statement for the question specific statement: 'patient born between year 100 and 100' refers to STRFTIME('%Y', Patient.Birthday) BETWEEN '100' AND '100'

Example:
Question: What is the percentage of female patient had total protein not within the normal range?
Question Specific domain statement:  calculation = DIVIDE((ID where sex = 'F' and TP < '6.0' or TP > '8.5'), COUNT(ID)) * 100
Generic Domain Statement for the question specific statement: 'percentage of female patient' refers to CAST(SUM(CASE WHEN T1.SEX = 'F' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*)

Example:
Question:  What is the ratio of outpatient to inpatient followed up treatment among all the 'SLE' diagnosed patient?
Question Specific domain statement: DIVIDE(COUNT(ID) where Diagnosis = 'SLE' and Admission = '+', COUNT(ID) where Diagnosis = 'SLE' and Admission = '-')
Generic Domain Statement for the question specific statement: 'the ratio of outpatient to inpatient followed up treatment' refers to SUM(CASE WHEN Admission = '+' THEN 1 ELSE 0 END) / SUM(CASE WHEN Admission = '-' THEN 1 ELSE 0 END)

Example:
Question: What was the age of the youngest patient when they initially inmumvpn1.tcs.com/lnxarrived at the hospital?
Question Specific domain statement:   the youngest patient refers to MAX(YEAR(Birthday))
Generic Domain Statement for the question specific statement: 'youngest patient' refers to STRFTIME('%Y', Patient.`First Date`) - STRFTIME('%Y', Patient.Birthday)

Example:
Question: For in-patient age 50 and above, what is their average anti-cardiolipin antibody (IgG) concentration?
Question Specific domain statement: age 50 and above refers to SUBTRACT(year(current_timestamp), year(Birthday)) > '50'
Generic Domain Statement for the question specific statement: 'age 100 and above' refers to STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', Patient.Birthday) > 100

Example:
Question: What is the ratio of male to female patients among all those with abnormal uric acid counts?
Question Specific domain statement:  abnormal uric acid refers to UA < = '8.0' where SEX = 'M', UA > '6.5' where SEX = 'F'
Generic Domain Statement for the question specific statement: 'ratio of male to female patients among all those with abnormal uric acid counts' refers to CAST(SUM(CASE WHEN T2.UA <= 8.0 AND T1.SEX = 'M' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.UA <= 6.5 AND T1.SEX = 'F' THEN 1 ELSE 0 END)

Example:
Question: What was the anti-nucleus antibody concentration level for the patient id 3605340 on 1996/12/2?
Question Specific domain statement: 1996/12/2 refers to `Examination Date` = '1996-12-02'
Generic Domain Statement for the question specific statement: 'on Date 100/100/100' refers to Examination.`Examination Date` = '100-100-100'

Example:
Question: Was the patient a man or a women whose ALT glutamic pylvic transaminase status got 9 on 1992-6-12?
Question Specific domain statement: ALT glutamic pylvic transaminase status got 9 GPT = '9'
Generic Domain Statement for the question specific statement: 'ALT glutamic pylvic transaminase status got 100' refres to Laboratory.GPT = '100'

Example:
Question: What is the percentage of female patient were born after 1930?
Question Specific domain statement: patient who were born after 1930 refers to year(Birthday) > '1930'
Generic Domain Statement for the question specific statement: 'Patient who were born after year 100' refers to STRFTIME('%Y', Patient.Birthday) > '100'

Example:
Question: '''


if folder == 'CS':
    sys_prompt= '''You are a database administrator and have designed the following database for California Schools whose schema is represented as:

    CREATE TABLE frpm (
      CDSCode INTEGER PRIMARY KEY,
      `Academic Year` INTEGER,
      `County Code` INTEGER,
      `District Code` INTEGER,
      `School Code` INTEGER,
      `County Name` TEXT,
      `District Name` TEXT,
      `School Name` TEXT,
      `District Type` TEXT,
      `School Type` TEXT,
      `Educational Option Type` TEXT,
      `NSLP Provision Status` TEXT,
      `Charter School (Y/N)` INTEGER,
      `Charter School Number` TEXT,
      `Charter Funding Type` TEXT,
      IRC INTEGER,
      `Low Grade` TEXT,
      `High Grade` TEXT,
      `Enrollment (K-12)` REAL,
      `Free Meal Count (K-12)` REAL,
      `FRPM Count (K-12)` REAL,
      `Enrollment (Ages 5-17)` REAL,
      `Free Meal Count (Ages 5-17)` REAL,
      `Percent (%) Eligible Free (K-12)` REAL,
      `Percent (%) Eligible Free (Ages 5-17)` REAL,
      `Percent (%) Eligible FRPM (Ages 5-17)` REAL,
      `2013-14 CALPADS Fall 1 Certification Status` REAL
    );

    CREATE TABLE satscores (
      cds TEXT PRIMARY KEY,
      rtype TEXT,
      sname TEXT,
      dname TEXT,
      cname TEXT,
      enroll12 INTEGER,
      NumTstTakr INTEGER,
      AvgScrRead INTEGER,
      AvgScrMath INTEGER,
      AvgScrWrite INTEGER,
      NumGE1500 INTEGER
    );

    CREATE TABLE schools (
      CDSCode TEXT PRIMARY KEY,
      NCESDist TEXT,
      NCESSchool TEXT,
      StatusType TEXT,
      County TEXT,
      District TEXT,
      School TEXT,
      Street TEXT,
      StreetAbr TEXT,
      City TEXT,
      Zip TEXT,
      State TEXT,
      MailStreet TEXT,
      MailStrAbr TEXT,
      MailCity TEXT,
      MailZip TEXT,
      MailState TEXT,
      Phone TEXT,
      Ext TEXT,
      Website TEXT,
      OpenDate DATE,
      ClosedDate DATE,
      Charter INTEGER,
      CharterNum TEXT,
      FundingType TEXT,
      DOC TEXT,
      DOCType TEXT,
      SOC TEXT,
      SOCType TEXT,
      EdOpsCode TEXT,
      EdOpsName TEXT,
      EILCode TEXT,
      EILName TEXT,
      GSoffered TEXT,
      GSserved TEXT,
      Virtual TEXT,
      Magnet TEXT,
      Latitude REAL,
      Longitude REAL,
      AdmFName1 TEXT,
      AdmLName1 TEXT,
      AdmEmail1 TEXT,
      AdmFName2 TEXT,
      AdmLName2 TEXT,
      AdmEmail2 TEXT,
      AdmFName3 TEXT,
      AdmLName3 TEXT,
      AdmEmail3 TEXT,
      LastUpdate DATE
    );
    '''

    few_shots='''
Example:
Question: List the names of schools with more than 30 difference in enrollements between K-12 and ages 5-17? Please also give the full street adress of the schools.
Question Specific domain statement: Difference in enrollment = `Enrollment (K-12)` - `Enrollment (Ages 5-17)`
Generic domain statement for the question specific statement: 'difference in enrollements between `Enrollment (K-12)` and `Enrollment (Ages 5-17)`' refers to frpm.`Enrollment (K-12)` - frpm.`Enrollment (Ages 5-17)`

Example:
Question: Please list the codes of the schools with a total enrollment of over 500.
Question Specific domain statement: Total enrollment can be represented by `Enrollment (K-12)` + `Enrollment (Ages 5-17)` > 500
Generic domain statement for the question specific statement: 'the codes of the schools with a total enrollment of over (100)' refres to `frpm.Enrollment (K-12)` + frpm.`Enrollment (Ages 5-17)` > 100

Example:
Question: Among the schools with an SAT excellence rate of over 0.3, what is the highest eligible free rate for students aged 5-17?
Question Specific domain statement: Excellence rate = NumGE1500 / NumTstTakr
Generic domain statement for the question specific statement: 'schools with an SAT excellence rate of over (100)' refers to satscores.NumGE1500 / satscores.NumTstTakr > 100

Example:
Question: What is the average math score of the school with the lowest average score for all subjects, and in which county is it located?
Question Specific domain statement: Average score for all subjects can be computed by AvgScrMath + AvgScrRead + AvgScrWrite
Generic domain statement for the question specific statement: 'average score for all subjects' refers to satscores.AvgScrMath + satscores.AvgScrRead + satscores.AvgScrWrite

Example:
Question: Between 1/1/2000 to 12/31/2005, how many directly funded schools opened in the county of Stanislaus?
Question Specific domain statement: Directly funded schools refers to FundingType = 'Directly Funded'
Generic domain statement for the question specific statement: 'directly funded schools' refers to schools.FundingType = 'Directly Funded'

Example:
Question: Name elementary schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?
Question Specific domain statement: Average of average math = sum(average math scores) / count(schools).
Generic domain statement for the question specific statement: 'the average of average math score for SAT' refers to CAST(SUM(schools.AvgScrMath) AS REAL) / COUNT(schools.cds)

Example:
Question: What is the average score in writing for the schools that were opened after 1991 and closed before 2000? List the school names along with the score. Also, list the communication number of the schools if there is any.
Question Specific domain statement: Communication number refers to phone number.
Generic domain statement for the question specific statement: 'Communication number' refers to schools.Phone

Example:
Question: What is the Percent (%) Eligible Free (K-12) in the school administered by an administrator whose first name is Alusine. List the district code of the school.
Question Specific domain statement: Percent (%) Eligible Free (K-12) = `FRPM Count (K-12)` / `Enrollment (K-12)` * 100%
Generic domain statement for the question specific statement: 'the Percent (%) Eligible Free (K-12) in the school' refers to frpm.`FRPM Count (K-12)` * 100 / frpm.`Enrollment (K-12)` 

Example:
Question: Find the average difference between K-12 enrollment and 15-17 enrollment of schools that are locally funded? List the names, K-12 enrollment, and 15-17 enrollment of schools which has a difference above this average.
Question Specific domain statement: 'Difference between K-12 enrollment and 15-17 enrollment can be computed by `Enrollment (K-12)` - `Enrollment (Ages 5-17)`
Generic domain statement for the question specific statement: 'difference between K-12 enrollment and 15-17 enrollment of schools' refers to frpm.`Enrollment (K-12)` - frpm.`Enrollment (Ages 5-17)`

Example:
Question: Of the schools that offers a magnet program serving a grade span of Kindergarten to 8th grade, how many offers Multiple Provision Types? List the number of cities that offers a Kindergarten to 8th grade span and indicate how many schools are there serving such grade span for each city.
Question Specific domain statement: Kindergarten to 8th grade refers to K-8
Generic domain statement for the question specific statement: 'a grade span of Kindergarten to 8th grade' refers to schools.GSoffered = 'K-8'

Example:
Question: 
'''

if folder == 'CG':
    sys_prompt= '''You are a database administrator and have designed the following database for Card Games whose schema is represented as:

    CREATE TABLE cards (
     `id` INTEGER PRIMARY KEY,
     `artist` TEXT,
     `asciiName` TEXT,
     `availability` TEXT,
     `borderColor` TEXT,
     `cardKingdomFoilId` TEXT,
     `cardKingdomId` TEXT,
     `colorIdentity` TEXT,
     `colorIndicator` TEXT,
     `colors` TEXT,
     `convertedManaCost` REAL,
     `duelDeck` TEXT,
     `edhrecRank` INTEGER,
     `faceConvertedManaCost` REAL,
     `faceName` TEXT,
     `flavorName` TEXT,
     `flavorText` TEXT,
     `frameEffects` TEXT,
     `frameVersion` TEXT,
     `hand` TEXT,
     `hasAlternativeDeckLimit` INTEGER,
     `hasContentWarning` INTEGER,
     `hasFoil` INTEGER,
     `hasNonFoil` INTEGER,
     `isAlternative` INTEGER,
     `isFullArt` INTEGER,
     `isOnlineOnly` INTEGER,
     `isOversized` INTEGER,
     `isPromo` INTEGER,
     `isReprint` INTEGER,
     `isReserved` INTEGER,
     `isStarter` INTEGER,
     `isStorySpotlight` INTEGER,
     `isTextless` INTEGER,
     `isTimeshifted` INTEGER,
     `keywords` TEXT,
     `layout` TEXT,
     `leadershipSkills` TEXT,
     `life` TEXT,
     `loyalty` TEXT,
     `manaCost` TEXT,
     `mcmId` TEXT,
     `mcmMetaId` TEXT,
     `mtgArenaId` TEXT,
     `mtgjsonV4Id` TEXT,
     `mtgoFoilId` TEXT,
     `mtgoId` TEXT,
     `multiverseId` TEXT,
     `name` TEXT,
     `number` TEXT,
     `originalReleaseDate` TEXT,
     `originalText` TEXT,
     `originalType` TEXT,
     `otherFaceIds` TEXT,
     `power` TEXT,
     `printings` TEXT,
     `promoTypes` TEXT,
     `purchaseUrls` TEXT,
     `rarity` TEXT,
     `scryfallId` TEXT,
     `scryfallIllustrationId` TEXT,
     `scryfallOracleId` TEXT,
     `setCode` TEXT,
     `side` TEXT,
     `subtypes` TEXT,
     `supertypes` TEXT,
     `tcgplayerProductId` TEXT,
     `text` TEXT,
     `toughness` TEXT,
     `type` TEXT,
     `types` TEXT,
     `uuid` TEXT,
     `variations` TEXT,
     `watermark` TEXT
    );

    CREATE TABLE foreign_data (
     `id` INTEGER PRIMARY KEY,
     `flavorText` TEXT,
     `language` TEXT,
     `multiverseid` INTEGER,
     `name` TEXT,
     `text` TEXT,
     `type` TEXT,
     `uuid` TEXT
    );

    CREATE TABLE legalities (
     `id` INTEGER PRIMARY KEY,
    `format` TEXT,
     `status` TEXT,
     `uuid` TEXT
    );

    CREATE TABLE rulings (
      `id` INTEGER PRIMARY KEY, 
      `date` DATE, 
      `text` TEXT, 
      `uuid` TEXT
    );

    CREATE TABLE Sets (
     `id` INTEGER PRIMARY KEY,
     `baseSetSize` INTEGER,
     `block` TEXT,
     `booster` TEXT,
     `code` TEXT,
     `isFoilOnly` INTEGER,
     `isForeignOnly` INTEGER,
     `isNonFoilOnly` INTEGER,
     `isOnlineOnly` INTEGER,
     `isPartialPreview` INTEGER,
     `keyruneCode` TEXT,
     `mcmId` INTEGER,
     `mcmIdExtras` INTEGER,
     `mcmName` TEXT,
     `mtgoCode` TEXT,
     `name` TEXT,
     `parentCode` TEXT,
     `releaseDate` DATE,
     `tcgplayerGroupId` INTEGER,
     `totalSetSize` INTEGER,
     `type` TEXT
    );

    CREATE TABLE Set_translations (
      `id` INTEGER PRIMARY KEY, 
      `language` TEXT, 
      `setCode` TEXT, 
      `translation` TEXT
    );
    '''
    few_shots='''
Example:
Question: List the card with value that cost more converted mana for the face.
Question Specific domain statement: more converted mana for the face refers to Max(faceConvertedManaCost)
Generic domain statement for the question specific statement: 'value that cost more converted mana for the face' refers to ORDER BY cards.faceConvertedManaCost LIMIT 1

Example:
Question: Name all cards with 2015 frame style ranking below 210 on EDHRec.
Question Specific domain statement: below 210 on EDHRec refers to EDHRec < 210
Generic domain statement for the question specific statement: 'frame style ranking below (100) on EDHRec' refers to cards.edhrecRank < 100

Example:
Question: Name the card and artist with the most ruling information. Also state if the card is a promotional printing.
Question Specific domain statement: with the most ruling information refers to Max(count(rulings.uuid))
Generic domain statement for the question specific statement: 'artist with the most ruling information' refers to ORDER BY COUNT(DISTINCT cards.uuid) DESC LIMIT 1

Example:
Question: List the keyrune code for the set whose code is 'PKHC'.
Question Specific domain statement: keyrune code refers to keyruneCode
Generic domain statement for the question specific statement: 'keyrune code' refers to sets.keyruneCode

Example:
Question: For the set of cards with "Ancestor\'s Chosen" in it, is there a Korean version of it?
Question Specific domain statement: set of cards with "Ancestor\'s Chosen" in it refers to name = \'Ancestor\'s Chosen\'
Generic domain statement for the question specific statement: 'the set of cards with "Ancestor\'s Chosen" in it' refers to cards.name = "Ancestor\'s Chosen"

Example:
Question: How many cards in the set Coldsnap have a black border color?
Question Specific domain statement: card set Coldsnap refers to name = 'Coldsnap'
Generic domain statement for the question specific statement: 'cards in the set Coldsnap' refers to sets.name = 'Coldsnap'

Example:
Question: What's the magic card market name for the set which was released on 2017/6/9?
Question Specific domain statement: the set which was released on 2017/6/9 refers to releaseDate = '2017-06-09'
Generic domain statement for the question specific statement: 'the set which was released on 100/100/100' refers to sets.releaseDate = '100-100-100'

Example:
Question: How many cards with original type of "Summon - Angel" have subtype other than "Angel"?'
Question Specific domain statement: subtype other than Angel refers to subtypes is not 'Angel'
Generic domain statement for the question specific statement: 'have subtype other than "Angel"' refers to cards.subtypes != 'Angel'

Example:
Question: How many cards are oversized, reprinted, and printed for promotions?
Question Specific domain statement: printed for promotions refers to isPromo = 1
Generic domain statement for the question specific statement: 'printed for promotions' refers to cards.isPromo = 1

Example:
Question: How many sets are available just in Japanese and not in Magic: The Gathering Online?
Question Specific domain statement: not in Magic: The Gathering Online refers to mtgoCode is null or mtgoCode = ''
Generic domain statement for the question specific statement: 'not in Magic: The Gathering Online' refers to set_translations.mtgoCode IS NULL

Example:
Question: '''

if folder == 'CC':
    sys_prompt= '''You are a database administrator and have designed the following database for Codebase Community whose schema is represented as:

    CREATE TABLE badges (
     `Id` INTEGER PRIMARY KEY,
     `UserId` INTEGER, 
     `Name` TEXT, 
     `Date` DATETIME
    );

    CREATE TABLE comments (
     `Id` INTEGER PRIMARY KEY, 
     `PostId` INTEGER, 
     `Score` INTEGER, 
     `Text` TEXT, 
     `CreationDate` DATETIME, 
     `UserId` INTEGER, 
     `UserDisplayName` TEXT
    );

    CREATE TABLE postHistory (
     `Id` INTEGER PRIMARY KEY,
     `PostHistoryTypeId` INTEGER,
     `PostId` INTEGER,
     `RevisionGUID` INTEGER,
     `CreationDate` DATETIME,
     `UserId` INTEGER,
     `Text` TEXT,
     `Comment` TEXT,
     `UserDisplayName` TEXT
    );

    CREATE TABLE postLinks (
      `Id` INTEGER PRIMARY KEY, 
      `CreationDate` DATETIME, 
      `PostId` INTEGER, 
      `RelatedPostId` INTEGER, 
      `LinkTypeId` INTEGER
    );

    CREATE TABLE posts (
     `Id` INTEGER PRIMARY KEY, 
     `PostTypeId` INTEGER,
     `AcceptedAnswerId` INTEGER,
     `CreaionDate` DATETIME,
     `Score` INTEGER,
     `ViewCount` INTEGER,
     `Body` TEXT,
     `OwnerUserId` INTEGER,
     `LasActivityDate` DATETIME,
     `Title` TEXT,
     `Tags` TEXT,
     `AnswerCount` INTEGER,
     `CommentCount` INTEGER,
     `FavoriteCount` INTEGER,
     `LastEditorUserId` INTEGER,
     `LastEditDate` DATETIME,
     `CommunityOwnedDate` DATETIME,
     `ParentId` INTEGER,
     `ClosedDate` DATEFORMAT,
     `OwnerDisplayName` TEXT,
     `LastEditorDisplayName` TEXT
    );

    CREATE TABLE tags (
      `Id` INTEGER PRIMARY KEY,  
      `TagName` TEXT, 
      `Count` INTEGER, 
      `ExcerptPostId` INTEGER, 
      `WikiPostId` TEXT
    );

    CREATE TABLE users (
      `Id` INTEGER PRIMARY KEY,  
     `Reputation` INTEGER,
     `CreationDate` DATETIME,
     `DisplayName` TEXT,
     `LastAccessDate` DATETIME,
     `WebsiteUrl` TEXT,
     `Location` TEXT,
     `AboutMe` TEXT,
     `Views` INTEGER,
     `UpVotes` INTEGER,
     `DownVotes` INTEGER,
     `AccountId` INTEGER,
     `Age` INTEGER,
     `ProfileImageUrl` TEXT
    );

    CREATE TABLE votes (
      `Id` INTEGER PRIMARY KEY,
      `PostId` INTEGER, 
      `VoteTypeId` INTEGER, 
      `CreationDate` DATETIME, 
      `UserId` INTEGER, 
      `BountyAmount` INTEGER
    );
    '''
    few_shots='''
Example:
Question: List out the age of users who located in Vienna, Austria obtained the badge?
Question Specific domain statement: "Vienna, Austria" is the Location
Generic domain statement for the question specific statement: 'users who located in Vienna, Austria' refers to users.Location = 'Vienna, Austria'

Example:
Question: User No.3025 gave a comment at 20:29:39 on 2014/4/23 to a post, how many favorite counts did that post get?
Question Specific domain statement: comment at 20:29:39 on 2014/4/23 refers to CreationDate = '2014/4/23 20:29:39
Generic domain statement for the question specific statement: 'gave a comment at 100:100:100 on 100/100/100 to a post' refers comments.CreationDate = '100-100-100 100:100:100'

Example:
Question: Among the tags with tag ID below 15, how many of them have 20 count of posts and below?
Question Specific domain statement: have 20 count of posts and below refers to Count < = 20
Generic domain statement for the question specific statement: 'tags that have (100) count of posts and below' refers to Count < = 100

Example:
Question: What is the owner's display name of the most popular post?
Question Specific domain statement: the most popular post refers to MAX(ViewCount)
Generic domain statement for the question specific statement: 'owner's most popular post' refers to ORDER BY posts.ViewCount DESC LIMIT 1

Example:
Question: Please list the display names of all the users whose accounts were created in the year 2014.
Question Specific domain statement: account created in the year 2014 refers to year(CreationDate) = 2014
Generic domain statement for the question specific statement: 'users accounts were created in the year 100' refers to STRFTIME('%Y', users.CreationDate) = '100'

Example:
Question: What is the name of tags used by John Stauffer's?
Question Specific domain statement: DisplayName = 'John Stauffer'
Generic domain statement for the question specific statement: 'tags used by John Stauffer's' refers to T1.DisplayName = 'John Stauffer'

Example:
Question: What is the display name of the user who acquired the first Archeologist badge?
Question Specific domain statement: acquired the first refers to MIN(Date)
Generic domain statement for the question specific statement: 'the user who acquired the first badge' refers to ORDER BY badges.Date LIMIT 1

Example:
Question: How many posts were created on 21st July, 2010?
Question Specific domain statement: created on 21st July, 2010 refers to CreationDate BETWEEN '2010-07-21 00:00:00' and '2012-07-21 23:59:59'
Generic domain statement for the question specific statement: 'posts were created on 21st July, 100' refers to date(postHistory.CreationDate) = '100-07-21'

Example:
Question: Write down the related posts titles and link type IDs of the post "What are principal component scores?".
Question Specific domain statement: Title = 'What are principal component scores?
Generic domain statement for the question specific statement: 'the post "What are principal component scores?"' refers to posts.Title = 'What are principal component scores?'

Example:
Question: How many users last accessed the website after 2014/9/1?
Question Specific domain statement: not in Magic: last accessed after 2014/9/1 refers to LastAccessDate > '2014-09-01 00:00:00'
Generic domain statement for the question specific statement: 'users last accessed the website after 100/9/1' refers to date(users.LastAccessDate) > '100-09-01'

Example:
Question: '''

if folder == 'F1':
    sys_prompt= '''You are a database administrator and have designed the following database for Formula 1 whose schema is represented as:

    CREATE TABLE circuits (
     `circuitId` INTEGER PRIMARY KEY,
     `circuitRef` TEXT,
     `name` TEXT,
     `location` TEXT,
     `country` TEXT,
     `lat` REAL,
     `lng` REAL,
     `alt` INTEGER,
     `url` TEXT
    );

    CREATE TABLE constructors (
     `constructorId` INTEGER PRIMARY KEY, 
     `constructorRef` TEXT, 
     `name` TEXT, 
     `nationality` TEXT, 
     `url` TEXT
    );

    CREATE TABLE drivers (
     `driverId` INTEGER PRIMARY KEY,
     `driverRef` TEXT,
     `number` INTEGER,
     `code` TEXT,
     `forename` TEXT,
     `surname` TEXT,
     `dob` DATE,
     `nationality` TEXT,
     `url` TEXT
    );

    CREATE TABLE seasons (
      `year` INTEGER PRIMARY KEY,
      `url` TEXT
    );

    CREATE TABLE races (
     `raceId` INTEGER PRIMARY KEY,
     `year` INTEGER, 
     `round` INTEGER, 
     `circuitId` INTEGER, 
     `name` TEXT, 
     `date` DATE, 
     `time` TEXT, 
     `url` TEXT
    );

    CREATE TABLE constructorResults (
     `constructorResultsId` INTEGER PRIMARY KEY,
     `raceId` INTEGER, 
     `constructorId` INTEGER, 
     `points` REAL, 
     `status` TEXT
    );

    CREATE TABLE constructorStandings (
     `constructorStandingsId` INTEGER PRIMARY KEY,
     `raceId` INTEGER,
     `constructorId` INTEGER,
     `points` INTEGER,
     `position` INTEGER,
     `positionText` TEXT,
     `wins` INTEGER
    );

    CREATE TABLE driverStandings (
     `driverStandingsId` INTEGER PRIMARY KEY,
     `raceId` INTEGER,
     `driverId` INTEGER,
     `points` REAL,
     `position` INTEGER,
     `positionText` TEXT,
     `wins` INTEGER
    );

    CREATE TABLE lapTimes (
     `raceId` INTEGER PRIMARY KEY, 
     `driverId` INTEGER, 
     `lap` INTEGER, 
     `position` INTEGER, 
     `time` TEXT, 
     `milliseconds` INTEGER
    );

    CREATE TABLE pitStops (
     `raceId` INTEGER PRIMARY KEY,  
     `driverId` INTEGER, 
     `stop` INTEGER, 
     `lap` INTEGER, 
     `time` TEXT, 
     `duration` TEXT, 
     `milliseconds` INTEGER
    );

    CREATE TABLE qualifying (
     `qualifyId` INTEGER PRIMARY KEY,
     `raceId` INTEGER,
     `driverId` INTEGER,
     `constructorId` INTEGER,
     `number` INTEGER,
     `position` INTEGER,
     `q1` TEXT,
     `q2` TEXT,
     `q3` TEXT
    );

    CREATE TABLE status (
     `statusId` INTEGER PRIMARY KEY, 
     `status` TEXT
    );

    CREATE TABLE results (
     `resultId` INTEGER PRIMARY KEY, 
     `raceId` INTEGER,
     `driverId` INTEGER,
     `constructorId` INTEGER,
     `number` INTEGER,
     `grid` INTEGER,
     `position` INTEGER,
     `positionText` TEXT,
     `positionOrder` INTEGER,
     `points` REAL,
     `laps` INTEGER,
     `time` TEXT,
     `milliseconds` INTEGER,
     `fastestLap` INTEGER,
     `rank` INTEGER,
     `fastestLapTime` TEXT,
     `fastestLapSpeed` TEXT,
     `statusId` INTEGER
    );
    '''
    few_shots='''
Example:
Question: In which country can I find the circuit with the highest altitude?
Question Specific domain statement: highest altitude refers to max(alt)
Generic domain statement for the question specific statement: 'with the highest altitude' refers to ORDER BY circuits.alt DESC LIMIT 1

Example:
Question: Please list the reference names of the drivers who are eliminated in the first period in race number 18.
Question Specific domain statement: driver reference name refers to driverRef
Generic domain statement for the question specific statement: 'reference names of the drivers' refers to drivers.driverRef

Example: 
Question: What's the reference name of Marina Bay Street Circuit?
Question Specific domain statement: reference name refers to circuitRef
Generic domain statement for the question specific statement: 'the reference name' refers to circuits.circuitRef

Example:
Question: How many times did Michael Schumacher won from races hosted in Sepang International Circuit?
Question Specific domain statement: win from races refers to max(points)
Generic domain statement for the question specific statement: 'How many times driver won from races' refers to SUM(driverStandings.wins)

Example:
Question: How many drivers managed to finish the race in the 2008 Australian Grand Prix?
Question Specific domain statement: managed to finish the race refers to time is not null
Generic domain statement for the question specific statement: 'drivers managed to finish the race' refers to results.time IS NOT NULL

Example:
Question: What is the surname of the driver with the best lap time in race number 19 in the second period?
Question Specific domain statement: second qualifying period refers to q2
Generic domain statement for the question specific statement: 'second qualifying period' refers to qualifying.q2

Example:
Question: For the driver who set the fastest lap speed in race No.933, where does he come from?
Question Specific domain statement: fastest lap speed refers to MIN(fastestLapSpeed)
Generic domain statement for the question specific statement: 'the fastest lap speed in race' refers to ORDER BY results.fastestLapSpeed DESC LIMIT 1

Example:
Question: Please give the link of the website that shows more information about the circuits the Spanish Grand Prix used in 2009.
Question Specific domain statement: link of the website refers to url
Generic domain statement for the question specific statement: 'the link of the website' refers to circuits.url

Example:
Question: How many accidents did the driver who had the highest number accidents in the Canadian Grand Prix have?
Question Specific domain statement: highest number of accidents refers to MAX(statusID)
Generic domain statement for the question specific statement: 'the highest number accidents' refers to status.statusId = 3

Example:
Question: For all the drivers who finished the game in race No. 872, who is the youngest?
Question Specific domain statement: the youngest is a driver where MAX(dob)
Generic domain statement for the question specific statement: 'the youngest driver' refers to ORDER BY drivers.dob DESC LIMIT 1

Example:
Question: '''

if folder == 'DCS':
    sys_prompt=''' CREATE TABLE customers(
    CustomerID INTEGER PRIMARY KEY,
    Segment TEXT,
    Currency TEXT 
);

CREATE TABLE gasstations(
    GasStationID INTEGER PRIMARY KEY,
    ChainID INTEGER,
    Country TEXT,
    Segment TEXT  
);

CREATE TABLE products(
    ProductID INTEGER PRIMARY KEY,
    Description TEXT  
);

CREATE TABLE yearmonth(
    CustomerID INTEGER PRIMARY KEY,
    Date INTEGER PRIMARY KEY,
    Consumption REAL,
             
);

CREATE TABLE "transactions_1k"(
    TransactionID INTEGER PRIMARY KEY,
    Date DATE,
    Time TEXT,
    CustomerID INTEGER,
    CardID INTEGER,
    GasStationID INTEGER,
    ProductID INTEGER,
    Amount INTEGER,
    Price REAL
); '''
    few_shots= '''
Example:
Question: What is the ratio of costumers who pay in EUR against customers who pay in CZK?
Question Specific domain statement: ratio of costumers who pay in EUR against customers who pay in CZK = count(Currency = 'EUR') / count(Currency = 'CZK').
Generic domain statement for the question specific statement: 'customers who pay in EUR' refers to SUM(IIF(customers.Currency = 'EUR', 1, 0))


Example:
Question: What was the difference in gas consumption between CZK-paying customers and EUR-paying customers in 2012?
Question Specific domain statement: Year 2012 can be presented as Between 201201 And 201212, which means between January and December in 2012
Generic domain statement for the question specific statement: 'Year {1000}' refers to SUBSTRING(yearmonth.Date, 1, 4) = '1000'


Example:
Question: Which year recorded the most consumption of gas paid in CZK?
Question Specific domain statement: The first 4 strings of the values in the table yearmonth can represent year.
Generic domain statement for the question specific statement: 'Which year' refers to SUBSTRING(yearmonth.Date, 1, 4)


Example:
Question: How much did customer 6 consume in total between August and November 2013?
Question Specific domain statement: Between August And November 2013 refers to Between 201308 And 201311
Generic domain statement for the question specific statement: 'Between August And November 100' refers to 'yearmonth.Date BETWEEN 10008 And 10011'


Example:
Question: Which client ID consumed the most in September 2013?
Question Specific domain statement: September 2013 refers to yearmonth.date = '201309'
Generic domain statement for the question specific statement: 'September 100' refers to yearmonth.Date = '10009'


Example:
Question: What is the biggest monthly consumption of the customers who use euro as their currency?
Question Specific domain statement: Monthly consumption = SUM(consumption) / 12
Generic domain statement for the question specific statement: 'biggest monthly consumption of customers' refers to ORDER BY yearmonth.Consumption DESC LIMIT 1


Example:
Question: How many transactions were paid in EUR in the morning of 2012/8/26?
Question Specific domain statement: '2012/8/26' can be represented by '2012-08-26'
Generic domain statement for the question specific statement: 'Date in format 100/100/100' refers to transactions_1k.Date = '100-100-100'

Example:
Question: How many transactions were paid in EUR in the morning of 2012/8/26?
Question Specific domain statement:  The morning refers to the time before '13:00:00'
Generic domain statement for the question specific statement: 'The morning' refers to transactions_1k.Time < '13:00:00'

Example:
Question: Which are the top five best selling products? Please state the full name of them.
Question Specific domain statement: Description of products contains full name
Generic domain statement for the question specific statement: 'full name of products' refers to products.Description

Example:
Question: Who is the top spending customer and how much is the average price per single item purchased by this customer? What currency was being used?
Question Specific domain statement: verage price per single item = price / amount
Generic domain statement for the question specific statement: 'average price per single item purchased by this customer' refers to SUM(transactions_1k.Price / transactions_1k.Amount)
'''

if folder == 'TC':
    sys_prompt='''CREATE TABLE atom(
   atom_id TEXT PRIMARY KEY,
   molecule_id TEXT,
   element TEXT,
);

CREATE TABLE bond (
   bond_id TEXT PRIMARY KEY,
   molecule_id TEXT,
   bond_type  TEXT,
);

CREATE TABLE connected  (
   atom_id TEXT PRIMARY KEY,
   atom_id2 TEXT PRIMARY KEY,
   bond_id TEXT,
);

CREATE TABLE molecule(
   molecule_id TEXT PRIMARY KEY,
   label TEXT,
);'''

    few_shots='''Example:
Question: What elements are in the TR004_8_9 bond atoms?
Question Specific domain statement:  element = ‘o’ means Oxygen, element = ’s’ means Sulfur
Generic domain statement for the question specific statement: 'Oxygen' refers to atom.element = 'o'

Example:
Question: Which type of label is the most numerous in atoms with hydrogen?
Question Specific domain statement:  label most numerous in atoms refers to MAX(COUNT(label))
Generic domain statement for the question specific statement: 'the most numerous label' refers to MAX(COUNT(molecule.label))

Example:
Question: Which type of label is the most numerous in atoms with hydrogen?
Question Specific domain statement:  label = '+' mean molecules are carcinogenic
Generic domain statement for the question specific statement: 'carcinogenic' refers to molecule.label = '+'

Example:
Question: Which type of label is the most numerous in atoms with hydrogen?
Question Specific domain statement:  label = '-' means molecules are non-carcinogenic
Generic domain statement for the question specific statement: 'non-carcinogenic' refers to molecule.label = '-'

Example:
Question: Tellurium is in what type of bond?
Question Specific domain statement: type of bond refers to bond_type
Generic domain statement for the question specific statement: 'type of bond' refers to bond.bond_type


Example:
Question: What type of bond is there between the atoms TR004_8 and TR004_20?
Question Specific domain statement:  between the atoms TR004_8 and TR004_20 refers to atom_id between atom_id = 'TR004_8' and atom_id = 'TR004_20'
Generic domain statement for the question specific statement:  between the atoms TR004_8 and TR004_20 refers to atom_id between atom_id = 'TR004_8' and atom_id = 'TR004_20'


Example:
Question: What percentage of carcinogenic-type molecules does not contain fluorine?
Question Specific domain statement:  percentage = DIVIDE(SUM(element = 'f'), COUNT(molecule_id)) as percent where label = '+'
Generic domain statement for the question specific statement: 'percentage of molecules does not contain fluorine' refers to DIVIDE(COUNT(DISTINCT CASE WHEN element <> 'f' THEN molecule_id ELSE NULL END), COUNT(DISTINCT molecule_id)) * 100


Example:
Question: What is the percentage of molecules that are carcinogenic?
Question Specific domain statement:  percentage = DIVIDE(SUM(label = '+'), COUNT(molecule_id)) as percent
Generic domain statement for the question specific statement: 'count of molecules that are carcinogenic' refers to COUNT(CASE WHEN T.label = '+' THEN T.molecule_id ELSE NULL END)

Example:
Question: What is the atom ID of double bonded carbon in TR012 molecule?
Question Specific domain statement:  double bond refers to bond_type = ' = '
Generic domain statement for the question specific statement: 'double bonded' refers to bond.bond_type = '='

Example:
Question: List the atom ID of the carcinogenic molecule that contains oxygen?
Question Specific domain statement: label = '+' mean molecules are carcinogenic
Generic domain statement for the question specific statement: 'carcinogenic molecule' refers to molecule.label = '+'
'''

if folder == 'FI':
    sys_prompt='''CREATE TABLE account(
    account_id INTEGER PRIMARY KEY,
    district_id INTEGER,
    frequency TEXT,
    date DATE
);

CREATE TABLE card(
    card_id INTEGER PRIMARY KEY,
    disp_id INTEGER,
    type TEXT,
    issued DATE,
);

CREATE TABLE client(
    client_id INTEGER PRIMARY KEY,
    gender TEXT,
    birth_date DATE,
    district_id INTEGER,
);

CREATE TABLE disp(
    disp_id INTEGER PRIMARY KEY,
    client_id INTEGER,
    account_id INTEGER,
    type TEXT,
);

CREATE TABLE district(
    district_id INTEGER PRIMARY KEY,
    A2 TEXT,
    A3 TEXT,
    A4 TEXT,
    A5 TEXT,
    A6 TEXT,
    A7 TEXT,
    A8 INTEGER,
    A9 INTEGER              ,
    A10 REAL,
    A11 INTEGER,
    A12 REAL,
    A13 REAL,
    A14 INTEGER,
    A15 INTEGER,
    A16 INTEGER
);

CREATE TABLE loan(
    loan_id INTEGER primary key,
    account_id INTEGER,
    date DATE,
    amount INTEGER,
    duration INTEGER,
    payments REAL,
    status TEXT,
    foreign key (account_id) references account (account_id)
);

CREATE TABLE `order`(
    order_id INTEGER primary key,
    account_id INTEGER,
    bank_to TEXT,
    account_to INTEGER,
    amount REAL,
    k_symbol TEXT,
    foreign key (account_id) references account (account_id)
);

CREATE TABLE trans(
    trans_id INTEGER primary key,
    account_id INTEGER,
    date DATE,
    type TEXT,
    operation TEXT,
    amount INTEGER,
    balance INTEGER,
    k_symbol TEXT,
    bank TEXT,
    account INTEGER,
    foreign key (account_id) references account (account_id)
);  
'''
    few_shots = '''Example:
Question: How many accounts who choose issuance after transaction are staying in East Bohemia region?
Question Specific domain statement: A3 contains the data of region
Generic domain statement for the question specific statement: 'East Bohemia region' refers to district.A3 = 'East Bohemia'

Example:
Question: How many accounts who have region in Prague are eligible for loans?
Question Specific domain statement: A3 contains the data of region
Generic domain statement for the question specific statement: 'region in Prague' refers to district.A3 = 'Prague'

Example:
Question: The average unemployment ratio of 1995 and 1996, which one has higher percentage?
Question Specific domain statement: A12 refers to unemploymant rate 1995
Generic domain statement for the question specific statement: 'average unemployment ratio of year 1995' refers to AVG(district.A12)

Example:
Question: The average unemployment ratio of 1995 and 1996, which one has higher percentage?
Question Specific domain statement:  A13 refers to unemploymant rate 1996
Generic domain statement for the question specific statement: 'average unemployment ratio of year 1996' refers to AVG(district.A13)

Example:
Question: How many male customers who are living in North Bohemia have average salary greater than 8000?
Question Specific domain statement: Male means that gender = 'M'
Generic domain statement for the question specific statement: 'male customers' refers to client.gender = 'M'


Example:
Question: List out the account numbers of female clients who are oldest and has lowest average salary, calculate the gap between this lowest average salary with the highest average salary?
Question Specific domain statement: Female means gender = 'F'
Generic domain statement for the question specific statement: 'female clients' refers to client.gender = 'F'

Example:
Question: How many customers who choose statement of weekly issuance are Owner?
Question Specific domain statement: 'POPLATEK TYDNE' stands for weekly issuance
Generic domain statement for the question specific statement: 'weekly issuance' refers to account.frequency = 'POPLATEK TYDNE'

Example:
Question: Among the accounts who have loan validity more than 12 months, list out the accounts that have the highest approved amount and have account opening date in 1993.
Question Specific domain statement: Loan validity more than 12 months refers to duration > 12
Generic domain statement for the question specific statement: 'Loan validity more than {100} months' refers to loan.duration > 100

Example:
Question: What is the gender of the oldest client who opened his/her account in the highest average salary branch?
Question Specific domain statement: Earlier birthdate refers to older age
Generic domain statement for the question specific statement: 'oldest client' refers to client.birth_date ASC LIMIT 1

Example:
Question: How many loan accounts are for pre-payment of duration of 24 months with weekly issuance of statement.
Question Specific domain statement: Frequency = 'POPLATEK TYDNE' referes to weekly statement
Generic domain statement for the question specific statement: 'weekly issuance of statement' refers to account.Frequency = 'POPLATEK TYDNE'
'''

if folder == 'EF':
    sys_prompt = '''CREATE TABLE "Player_Attributes" (
     id     INTEGER PRIMARY KEY,
     player_fifa_api_id     INTEGER,
     player_api_id  INTEGER,
     date   TEXT,
     overall_rating     INTEGER,
     potential  INTEGER,
     preferred_foot     TEXT,
     attacking_work_rate    TEXT,
     defensive_work_rate    TEXT,
     crossing   INTEGER,
     finishing  INTEGER,
     heading_accuracy   INTEGER,
     short_passing  INTEGER,
     volleys    INTEGER,
     dribbling  INTEGER,
     curve  INTEGER,
     free_kick_accuracy     INTEGER,
     long_passing   INTEGER,
     ball_control   INTEGER,
     acceleration   INTEGER,
     sprint_speed   INTEGER,
     agility    INTEGER,
     reactions  INTEGER,
     balance    INTEGER,
     shot_power     INTEGER,
     jumping    INTEGER,
     stamina    INTEGER,
     strength   INTEGER,
     long_shots     INTEGER,
     aggression     INTEGER,
     interceptions  INTEGER,
     positioning    INTEGER,
     vision     INTEGER,
     penalties  INTEGER,
     marking    INTEGER,
     standing_tackle    INTEGER,
     sliding_tackle     INTEGER,
     gk_diving  INTEGER,
     gk_handling    INTEGER,
     gk_kicking     INTEGER,
     gk_positioning     INTEGER,
     gk_reflexes    INTEGER 
);

CREATE TABLE  Player  (
     id     INTEGER PRIMARY KEY,
     player_api_id  INTEGER,
     player_name    TEXT,
     player_fifa_api_id     INTEGER,
     birthday   TEXT,
     height     INTEGER,
     weight     INTEGER 
);

CREATE TABLE  Match  (
     id     INTEGER PRIMARY KEY,
     country_id     INTEGER,
     league_id  INTEGER,
     season     TEXT,
     stage  INTEGER,
     date   TEXT,
     match_api_id   INTEGER  ,
     home_team_api_id   INTEGER,
     away_team_api_id   INTEGER,
     home_team_goal     INTEGER,
     away_team_goal     INTEGER,
     home_player_X1     INTEGER,
     home_player_X2     INTEGER,
     home_player_X3     INTEGER,
     home_player_X4     INTEGER,
     home_player_X5     INTEGER,
     home_player_X6     INTEGER,
     home_player_X7     INTEGER,
     home_player_X8     INTEGER,
     home_player_X9     INTEGER,
     home_player_X10    INTEGER,
     home_player_X11    INTEGER,
     away_player_X1     INTEGER,
     away_player_X2     INTEGER,
     away_player_X3     INTEGER,
     away_player_X4     INTEGER,
     away_player_X5     INTEGER,
     away_player_X6     INTEGER,
     away_player_X7     INTEGER,
     away_player_X8     INTEGER,
     away_player_X9     INTEGER,
     away_player_X10    INTEGER,
     away_player_X11    INTEGER,
     home_player_Y1     INTEGER,
     home_player_Y2     INTEGER,
     home_player_Y3     INTEGER,
     home_player_Y4     INTEGER,
     home_player_Y5     INTEGER,
     home_player_Y6     INTEGER,
     home_player_Y7     INTEGER,
     home_player_Y8     INTEGER,
     home_player_Y9     INTEGER,
     home_player_Y10    INTEGER,
     home_player_Y11    INTEGER,
     away_player_Y1     INTEGER,
     away_player_Y2     INTEGER,
     away_player_Y3     INTEGER,
     away_player_Y4     INTEGER,
     away_player_Y5     INTEGER,
     away_player_Y6     INTEGER,
     away_player_Y7     INTEGER,
     away_player_Y8     INTEGER,
     away_player_Y9     INTEGER,
     away_player_Y10    INTEGER,
     away_player_Y11    INTEGER,
     home_player_1  INTEGER,
     home_player_2  INTEGER,
     home_player_3  INTEGER,
     home_player_4  INTEGER,
     home_player_5  INTEGER,
     home_player_6  INTEGER,
     home_player_7  INTEGER,
     home_player_8  INTEGER,
     home_player_9  INTEGER,
     home_player_10     INTEGER,
     home_player_11     INTEGER,
     away_player_1  INTEGER,
     away_player_2  INTEGER,
     away_player_3  INTEGER,
     away_player_4  INTEGER,
     away_player_5  INTEGER,
     away_player_6  INTEGER,
     away_player_7  INTEGER,
     away_player_8  INTEGER,
     away_player_9  INTEGER,
     away_player_10     INTEGER,
     away_player_11     INTEGER,
     goal   TEXT,
     shoton     TEXT,
     shotoff    TEXT,
     foulcommit     TEXT,
     card   TEXT,
     cross  TEXT,
     corner     TEXT,
     possession     TEXT,
     B365H  NUMERIC,
     B365D  NUMERIC,
     B365A  NUMERIC,
     BWH    NUMERIC,
     BWD    NUMERIC,
     BWA    NUMERIC,
     IWH    NUMERIC,
     IWD    NUMERIC,
     IWA    NUMERIC,
     LBH    NUMERIC,
     LBD    NUMERIC,
     LBA    NUMERIC,
     PSH    NUMERIC,
     PSD    NUMERIC,
     PSA    NUMERIC,
     WHH    NUMERIC,
     WHD    NUMERIC,
     WHA    NUMERIC,
     SJH    NUMERIC,
     SJD    NUMERIC,
     SJA    NUMERIC,
     VCH    NUMERIC,
     VCD    NUMERIC,
     VCA    NUMERIC,
     GBH    NUMERIC,
     GBD    NUMERIC,
     GBA    NUMERIC,
     BSH    NUMERIC,
     BSD    NUMERIC,
     BSA    NUMERIC
);

CREATE TABLE  League  (
     id     INTEGER PRIMARY KEY,
     country_id     INTEGER,
     name   TEXT
);

CREATE TABLE  Country  (
     id     INTEGER PRIMARY KEY,
     name   TEXT 
);

CREATE TABLE "Team" (
     id     INTEGER PRIMARY KEY,
     team_api_id    INTEGER  ,
     team_fifa_api_id   INTEGER,
     team_long_name     TEXT,
     team_short_name    TEXT 
);

CREATE TABLE  Team_Attributes  (
     id     INTEGER PRIMARY KEY,
     team_fifa_api_id   INTEGER,
     team_api_id    INTEGER,
     date   TEXT,
     buildUpPlaySpeed   INTEGER,
     buildUpPlaySpeedClass  TEXT,
     buildUpPlayDribbling   INTEGER,
     buildUpPlayDribblingClass  TEXT,
     buildUpPlayPassing     INTEGER,
     buildUpPlayPassingClass    TEXT,
     buildUpPlayPositioningClass    TEXT,
     chanceCreationPassing  INTEGER,
     chanceCreationPassingClass     TEXT,
     chanceCreationCrossing     INTEGER,
     chanceCreationCrossingClass    TEXT,
     chanceCreationShooting     INTEGER,
     chanceCreationShootingClass    TEXT,
     chanceCreationPositioningClass     TEXT,
     defencePressure    INTEGER,
     defencePressureClass   TEXT,
     defenceAggression  INTEGER,
     defenceAggressionClass     TEXT,
     defenceTeamWidth   INTEGER,
     defenceTeamWidthClass  TEXT,
     defenceDefenderLineClass   TEXT
);
'''
    few_shots ='''Example:
Question: What is the percentage of players that are under 180 cm who have an overall strength of more than 70?
Question Specific domain statement: percentage refers to DIVIDE(COUNT(height < 180 AND overall_rating > 70),COUNT(id)) * 100%
Generic domain statement for the question specific statement: 'percentage of players that who have an overall strength of more than {100}' refers to CAST(COUNT(CASE WHEN Player_Attributes.overall_rating > 100 THEN t1.id ELSE NULL END) AS REAL) * 100 / COUNT(Player.id)

Example:
Question: What is the height of the tallest player? Indicate his name.
Question Specific domain statement: tallest player refers to MAX(height)
Generic domain statement for the question specific statement: 'height of the tallest player' refers to ORDER BY Player.height DESC LIMIT 1

Example:
Question: Which player has the highest overall rating? Indicate the player's api id.
Question Specific domain statement: highest overall rating refers to MAX(overall_rating)
Generic domain statement for the question specific statement: 'highest overall rating' refers to ORDER BY Player_Attributes.overall_rating DESC LIMIT 1

Example:
Question: Which home team had lost the fewest matches in the 2016 season?
Question Specific domain statement: home team that had lost the fewest matches refers to MIN(SUBTRACT(home_team_goal, away_team_goal))
Generic domain statement for the question specific statement: 'home team that had lost the fewest matches' refers to Match.home_team_goal - Match.away_team_goal < 0 ORDER BY Match.home_team_goal - Match.away_team_goal DESC LIMIT 1

Example:
Question: Which home team had lost the fewest matches in the 2016 season?
Question Specific domain statement:  2016 season refers to season = '2015/2016'
Generic domain statement for the question specific statement: 'the 2016 season' refers to Match.season = '2015/2016'

Example:
Question: In Scotland Premier League, which away team won the most during the 2010 season?
Question Specific domain statement: Scotland Premier League refers to League.name = 'Scotland Premier League'
Generic domain statement for the question specific statement: 'Scotland Premier League' refers to League.name = 'Scotland Premier League'

Example:
Question: In Scotland Premier League, which away team won the most during the 2010 season?
Question Specific domain statement:  away team refers to away_team_api_id
Generic domain statement for the question specific statement: 'which away team' refers to Match.away_team_api_id


Example:
Question: What are the speed in which attacks are put together of the top 4 teams with the highest build Up Play Speed and whose tendency/ frequency of dribbling is little?
Question Specific domain statement: highest build up play speed refers to MAX(buildUpPlaySpeed)
Generic domain statement for the question specific statement: top 4 teams with the highest build up play speed refers to ORDER BY Team_Attributes.buildUpPlayDribbling ASC LIMIT 4


Example:
Question: At present, calculate for the player's age who have a sprint speed of no less than 97 between 2013 to 2015.
Question Specific domain statement: players age at present = SUBTRACT((DATETIME(), birthday))
Generic domain statement for the question specific statement: 'player's age at present' refers to DATETIME() - Player.birthday

Example:
Question: Please provide the full name of the away team that scored the most goals.
Question Specific domain statement:  scored the most goals refers to MAX(COUNT(away_team_goal))
Generic domain statement for the question specific statement: 'the away team that scored the most goals' refers to ORDER BY Match.away_team_goal DESC LIMIT 1
'''

if folder == 'SH':
    sys_prompt = '''CREATE TABLE alignment (
    id INTEGER primary key,
    alignment TEXT
)

CREATE TABLE attribute (
    id INTEGER primary key,
    attribute_name TEXT
)

CREATE TABLE colour (
    id INTEGER primary key,
    colour TEXT
)

CREATE TABLE gender (
    id INTEGER primary key,
    gender TEXT
)

CREATE TABLE publisher (
    id INTEGER primary key,
    publisher_name TEXT
)

CREATE TABLE race (
    id INTEGER primary key,
    race TEXT
)

CREATE TABLE superhero (
    id INTEGER primary key,
    superhero_name TEXT,
    full_name TEXT,
    gender_id INTEGER,
    eye_colour_id INTEGER,
    hair_colour_id INTEGER,
    skin_colour_id INTEGER,
    race_id INTEGER,
    publisher_id INTEGER,
    alignment_id INTEGER,
    height_cm INTEGER,
    weight_kg INTEGER
)

CREATE TABLE hero_attribute (
    hero_id  INTEGER,
    attribute_id    INTEGER,
    attribute_value INTEGER
)

CREATE TABLE superpower (
    id  INTEGER primary key,
    power_name TEXT
)

CREATE TABLE hero_power (
    hero_id  INTEGER,
    power_id INTEGER,
    foreign key (hero_id) references superhero(id),
    foreign key (power_id) references superpower(id)
)   
'''
    few_shots='''
Example:
Question: Please list all the superpowers of 3-D Man.
Question Specific domain statement: 3-D Man refers to superhero_name = '3-D Man'
Generic domain statement for the question specific statement: '3-D Man' refers to superhero.superhero_name = '3-D Man'


Example:
Question: How many superheroes have the super power of "Super Strength"?
Question Specific domain statement: super power of "Super Strength" refers to power_name = 'Super Strength'
Generic domain statement for the question specific statement: super power of Super Strength refers to superpower.power_name = 'Super Strength'

Example:
Question: Among the superheroes with the super power of "Super Strength", how many of them have a height of over 200cm?
Question Specific domain statement:  a height of over 200cm refers to height_cm > 200
Generic domain statement for the question specific statement: 'height of over {100} cm' refers to superhero.height_cm > 100

Example:
Question: Among the superheroes with blue eyes, how many of them have the super power of "Agility"?
Question Specific domain statement: blue eyes refers to colour = 'Blue' and eye_colour_id = colour.id
Generic domain statement for the question specific statement: 'superheroes with blue eyes' refers to colour.colour = 'Blue'

Example:
Question: Please list the superhero names of all the superheroes that have blue eyes and blond hair.
Question Specific domain statement: blue eyes refers to colour = 'Blue' and eye_colour_id = colour.id
Generic domain statement for the question specific statement: 'superheroes that have blue eyes' refers to colour.colour = 'Blue'


Example:
Question: How many superheroes are published by Marvel Comics?
Question Specific domain statement: published by Marvel Comics refers to publisher_name = 'Marvel Comics'
Generic domain statement for the question specific statement: 'published by Marvel Comics' refers to publisher.publisher_name = 'Marvel Comics'

Example:
Question: Please give the full name of the tallest hero published by Marvel Comics.
Question Specific domain statement: the tallest hero refers to MAX(height_cm)
Generic domain statement for the question specific statement: 'the tallest hero' refers to ORDER BY superhero.height_cm DESC LIMIT 1

Example:
Question: Among the superheroes from Marvel Comics, how many of them have blue eyes?
Question Specific domain statement: the superheroes from Marvel Comics refers to publisher_name = 'Marvel Comics'
Generic domain statement for the question specific statement: 'the superheroes from Marvel Comics' refers to publisher.publisher_name = 'Marvel Comics'

Example:
Question: Provide the full name of the superhero named Alien.
Question Specific domain statement: FALSE
Generic domain statement for the question specific statement: 'superhero named Alien' refers to superhero.superhero_name = 'Alien'.

Example:
Question: In superheroes with weight less than 100, list the full name of the superheroes with brown eyes.
Question Specific domain statement: weight less than 100 refers to weight_kg < 100
Generic domain statement for the question specific statement: 'weight less than {100}' refers to superhero.weight_kg < 100
'''

if folder == 'SC':
    sys_prompt='''CREATE TABLE event (
    event_id   TEXT primary key,
    event_name TEXT,
    event_date TEXT,
    type       TEXT,
     es      TEXT,
    location   TEXT,
    status     TEXT 
)

CREATE TABLE major (
    major_id   TEXT primary key,
    major_name TEXT,
    department TEXT,
    college    TEXT 
)

CREATE TABLE zip_code (
    zip_code    INTEGER primary key,
    type        TEXT,
    city        TEXT,
    county      TEXT,
    state       TEXT,
    short_state TEXT 
)

CREATE TABLE "attendance" (
    link_to_event  TEXT primary key,
    link_to_member TEXT primary key,

)

CREATE TABLE "budget" (
    budget_id     TEXT primary key,
    category      TEXT,
    spent         FLOAT,
    remaining     FLOAT,
    amount        INTEGER,
    event_status  TEXT,
    link_to_event TEXT 
)

CREATE TABLE "expense" (
    expense_id          TEXT primary key,
    expense_description TEXT,
    expense_date        TEXT,
    cost                REAL,
    approved            TEXT,
    link_to_member      TEXT,
    link_to_budget      TEXT 
)

CREATE TABLE "income" (
    income_id TEXT primary key,
    date_received  TEXT,
    amount INTEGER,
    source TEXT,
     es TEXT,
    link_to_member TEXT 
)

CREATE TABLE "member" (
    member_id TEXT primary key,
    first_name    TEXT,
    last_name     TEXT,
    email         TEXT,
    position      TEXT,
    t_shirt_size  TEXT,
    phone         TEXT,
    zip           INTEGER,
    link_to_major TEXT 
)
'''
    few_shots='''
Example:
Question: What's Angela Sanders's major?
Question Specific domain statement: major refers to major_name
Generic domain statement for the question specific statement: 'Angela Sanders's major' refers to major.major_name.

Example:
Question: Please list the full names of the students in the Student_Club that come from the Art and Design Department.
Question Specific domain statement: full name refers to first_name, last_name
Generic domain statement for the question specific statement: 'list the full names' refers to member.first_name AND member.last_name

Example:
Question: How many students of the Student_Club have attended the event "Women's Soccer"?
Question Specific domain statement: Women's Soccer is an event name
Generic domain statement for the question specific statement: 'the event Women's Soccer' refers to event.event_name = 'Women's Soccer'

Example:
Question: What is the event that has the highest attendance of the students from the Student_Club?
Question Specific domain statement: event with highest attendance refers to MAX(COUNT(link_to_event))
Generic domain statement for the question specific statement: 'highest attendance of the students from the Student_Club' refers to ORDER BY COUNT(attendance.link_to_event) DESC LIMIT 1

Example:
Question: Which college is the vice president of the Student_Club from?
Question Specific domain statement: Vice President is a position of the Student Club
Generic domain statement for the question specific statement: 'Vice President of the Student Club' refers to member.position LIKE 'vice president'

Example:
Question: How many events of the Student_Club did Sacha Harrison attend in 2019?
Question Specific domain statement: events attended in 2019 refers to YEAR(event_date) = 2019
Generic domain statement for the question specific statement: 'events attended in the year {1000}' refers to SUBSTR(event.event_date, 1, 4) = '1000'

Example:
Question: How many members of the Student_Club have majored Environmental Engineering?

Question Specific domain statement: 'Environmental Engineering' is the major name
Generic domain statement for the question specific statement: 'majored Environmental Engineering' refers to major.major_name = 'Environmental Engineering'

Example:
Question: What is the amount of the funds that the Vice President received?
Question Specific domain statement: 'Vice President' is a position of Student Club
Generic domain statement for the question specific statement: the Vice President refers to member.position = 'Vice President'.


Example:
Question: Was each expense in October Meeting on October 8, 2019 approved?
Question Specific domain statement: event_name = 'October Meeting' where event_date = '2019-10-08'
Generic domain statement for the question specific statement: 'in October Meeting' refers to event.event_name = 'October Meeting'

Example:
Question: Calculate the difference of the total amount spent in all events by the Student_Club in year 2019 and 2020.
Question Specific domain statement: SUBTRACT(spent where YEAR(expense_date) = 2020, spent where YEAR(expense_date) = 2019)
Generic domain statement for the question specific statement: 'The total amount spent in all events by the Student_Club in year {1000}' refers to SUM(CASE WHEN SUBSTR(event.event_date, 1, 4) = '2019' THEN budget.spent ELSE 0 END)
    '''

evi_sys_prompt= '''You have been given a few examples of question and the question specific domain statement, which is required to generate SQL query for the given question. The schema for the database relevant to the examples is given below:
''' + sys_prompt+ '''Now, for the given question and the question specific domain statement, You need to create a generic statement depicting the domain knowledge , if required.
Provide the statement in one sentence only. Replace float values to '(100)' if appear seperately.
Generate the generic statement of the following form: "Relevant part of NL Question" refers to "corresponding SQL Syntax evidence" for each sample.'''



with open('IID Split.json') as f:
  data1 = json.load(f)

with open('OOD Split.json') as f:
  data2 = json.load(f)

# Generating generic evidences from actual evidences
if method == 'IID TDS':
    generated_data1 ={}
    for idx, da in enumerate(data1.keys()):
        print(idx+1)
        d=data1[da]
        generated_data1[idx] = {}
        generated_data1[idx]['question'] = d['question']
        generated_data1[idx]['GT_SQL'] = d['GT_SQL']
        generated_data1[idx]['actual_evidence'] = []
        generated_data1[idx]['generic_evidence'] = []
       
        for e in d['actual_evidence']:
            if len(e) > 2:
                generated_data1[idx]['actual_evidence'].append(e)
                prompt= few_shots + d['question']+'\nQuestion Specific domain statement: '+e+"\nGeneric domain statement for the question specific statement:"
                env = get_answer(evi_sys_prompt, prompt,idx)['choices'][0]['message']['content']
                generated_data1[idx]['generic_evidence'].append(env)
        with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/iid_data_Generic_indomian_WOSQL.json','w') as file:
            json.dump(generated_data1,file)


if method == 'OOD File':
    generated_data2 ={}
    for idx, d in enumerate(data2):
        print(idx+1)
        
        generated_data2[idx] = {}
        generated_data2[idx]['question'] = d['question']
        generated_data2[idx]['GT_SQL'] = d['SQL']
        generated_data2[idx]['actual_evidence'] = []
        generated_data2[idx]['generic_evidence'] = []
       
        for e in d['evidence'].split(';'):
            if len(e) > 2:
                generated_data2[idx]['actual_evidence'].append(e)
        with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/ood_data_Generic.json','w') as file:
            json.dump(generated_data2,file)


with open('IID Domain Statement File','r') as file:
  generated_data1=json.load(file)
with open('OOD File generated above') as f:
    generated_data2 = json.load(f)

all_generic_evidences_data1 = []
all_query_specific_evidences_data1 = []


manual_ev = open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/column_refersto.txt','r')
man_data=manual_ev.readlines()

inter_generic_map={}

for k in generated_data1.keys():
  for idx,j in enumerate(generated_data1[k]['generic_evidence']):
    if j.strip() not in all_generic_evidences_data1:
        all_generic_evidences_data1.append(j.strip().lower())

path = os.path.join(path_to_database_directory_dev, db_name, db_name + '.sqlite')
try:
    conn = sqlite3.connect(path)
except Exception as e:
    print(f'{idx} ; {db_name} : {e}')
c = conn.cursor()
c.execute("select * from sqlite_master WHERE type='table'")
tables = [v[2] for v in c.fetchall() if v[2] != "sqlite_sequence"]

for t in tables:
    c.execute("SELECT * from "+t)
    columns = list(map(lambda x: x[0], c.description))
    for col in columns:
        try:
            c.execute("SELECT DISTINCT "+col+" FROM "+t)
        except:
            continue
        values=c.fetchall()
        # print()
        # input()
        all_generic_evidences_data1.append(col+ ' refers to representative values like '+ re.sub('[^A-Za-z0-9]+', ' ', str(values[:4])).strip().replace(' ',','))


all_generic_evidences = all_generic_evidences_data1.copy()

cnt=0
if folder=='TP':
    for i in man_data:
        if i=='\n':
            break
        else:
            all_generic_evidences.append(i.lower())
cnt=1
if folder=='CS':
    for i in man_data:
        if i=='\n':
            cnt-=1
            if cnt==0:
                break
        elif cnt==0:
            all_generic_evidences.append(i.lower())
cnt=2
if folder=='CG':
    for i in man_data:
        if i=='\n':
            cnt-=1
            if cnt==0:
                break
        elif cnt==0:
            all_generic_evidences.append(i.lower())

all_generic_evidences_data1 = all_generic_evidences


if method == 'All DS T':
    generated_data1['all_generic_evidence_iid']=all_generic_evidences_data1
    with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/iid_data_all_ge.json','w') as file:
        json.dump(generated_data1,file)
    generated_data2['all_generic_evidence_iid']=all_generic_evidences_data1
    with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/ood_data_all_ge.json','w') as file:
        json.dump(generated_data2,file)


# ###########Taking partial Generic Evidence###########
if method == 'LR':
    all_generic_evidences_data1_lex=[]
    refer_map={}

    for j in all_generic_evidences:
        if j.strip().split('refers to')[0].replace("'",'') not in all_generic_evidences_data1_lex:
            all_generic_evidences_data1_lex.append(re.sub('[^A-Za-z0-9]+', ' ', j.strip().split('refers to')[0]))
            refer_map[re.sub('[^A-Za-z0-9]+', ' ', j.strip().split('refers to')[0])] = j

    stop_words = set(stopwords.words('english'))

    evidence_vocab={}
    filter_org={}

    def intersection(lst1, lst2):
      lst3 = [value for value in lst1 if value in lst2]
      return list(set(lst3))

    question_ev={}


    for k in generated_data1.keys():
      ev_score={}
      question=generated_data1[k]['question'].lower()
      question = re.sub('[^A-Za-z0-9]+', ' ', question)
      question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
      question = re.sub(r'^([\s\d]+)$', "100", question)
      question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

      for q in question.split():
        try:
            float(q)
            question = question.replace(q,'100')
        except:
            continue

      question_ev[question]=[]
      
      word_tokens=question.split()
      filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

      for word in filtered_sentence:
        for ev in all_generic_evidences_data1_lex:
          word_tokens=ev.split()
          ev_filter=[w.lower() for w in word_tokens if not w.lower() in stop_words]
          if word in  ev_filter:
            ev_score[ev]=len(intersection(filtered_sentence,ev_filter))/(len(ev_filter)+len(filtered_sentence))
            if ev_score[ev] == 0:
                del ev_score[ev]
           
      top_evs=list(dict(sorted(ev_score.items(), key=lambda item: item[1])).keys())
      top_evs.reverse()
      top_evs=top_evs[:recall]
      generated_data1[k]['predicted_generic_evidence_iid_lex']=[]
      for j in top_evs:
        generated_data1[k]['predicted_generic_evidence_iid_lex'].append(refer_map[j])

    with open('IID_lex_retr_output.json','w') as file:
        json.dump(generated_data1,file)

    evidence_vocab={}
    filter_org={}
    question_ev={}


    for k in generated_data2.keys():
      ev_score={}
      question=generated_data2[k]['question'].lower()
      question = re.sub('[^A-Za-z0-9]+', ' ', question)
      question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
      question = re.sub(r'^([\s\d]+)$', "100", question)
      question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

      for q in question.split():
        try:
            float(q)
            question = question.replace(q,'100')
        except:
            continue

      question_ev[question]=[]
    
      word_tokens=question.split()
      filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

      for word in filtered_sentence:
        for ev in all_generic_evidences_data1_lex:
          word_tokens=ev.split()
          ev_filter=[w.lower() for w in word_tokens if not w.lower() in stop_words]
          if word in  ev_filter:
            ev_score[ev]=len(intersection(filtered_sentence,ev_filter))/(len(ev_filter)+len(filtered_sentence))
            if ev_score[ev] == 0:
                del ev_score[ev]

      top_evs=list(dict(sorted(ev_score.items(), key=lambda item: item[1])).keys())
      top_evs.reverse()
      top_evs=top_evs[:recall]
      generated_data2[k]['predicted_generic_evidence_iid_lex']=[]
      for j in top_evs:
        generated_data2[k]['predicted_generic_evidence_iid_lex'].append(refer_map[j])

    with open('OOD_output_lex_retrieved.json','w') as file:
        json.dump(generated_data2,file)

if method == 'OR':
    def get_embed(line):
      response = openai.Embedding.create(
          input=line,
          model="text-embedding-ada-002"
      )
      return response.data


###Retrieving similar evidences using BERT embeddings 2

if method == 'BR T':
    all_generic_evidences_data1_OQ=[]
    refer_map={}

    for j in all_generic_evidences:
        if re.sub('[^A-Za-z0-9]+', ' ', j.strip().split('refers to')[0]) not in all_generic_evidences_data1_OQ:
            all_generic_evidences_data1_OQ.append(re.sub('[^A-Za-z0-9]+', ' ', j.strip().split('refers to')[0]))
            refer_map[re.sub('[^A-Za-z0-9]+', ' ', j.strip().split('refers to')[0])] = j

    model = SentenceTransformer(cache_dir)
    all_generic_evidences_data1_embd = model.encode(all_generic_evidences_data1_OQ)        

if method == 'BR NT':
    all_NT_evidences=[]
    for k in generated_data1.keys():
      for idx,j in enumerate(generated_data1[k]['actual_evidence']):
        if j.strip() not in all_generic_evidences_data1:
            all_NT_evidences.append(j.strip().lower())
    model = SentenceTransformer(cache_dir)
    all_generic_evidences_data1_embd = model.encode(all_NT_evidences) 

# # # # # # # # # # # # # # # # # # # Retrieving similar evidences using OpenAI embeddings

if method == 'OR':
    all_generic_evidences_data1_embd_ai = get_embed(all_generic_evidences_data1)   
    for k in generated_data1.keys():

    question = generated_data1[k]['question'].lower()
    question = re.sub('[^A-Za-z0-9]+', ' ', question)
    question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
    question = re.sub(r'^([\s\d]+)$', "100", question)
    question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

    for q in question.split():
        try:
            float(q)
            question = question.replace(q,'100')
        except:
            continue

    question_embedding_ai = get_embed(question)[0].embedding

    generated_data1[k]['predicted_generic_evidence_iid_ai'] = []

    oa_embeddings=[]
    for oa_embed in all_generic_evidences_data1_embd_ai:
        oa_embeddings.append(oa_embed.embedding)
    similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

    for i in (similarity_ai.argsort()[-recall:][::-1]):
        generated_data1[k]['predicted_generic_evidence_iid_ai'].append(all_generic_evidences_data1[i])

    with open('Output_OpenAI_retrieval_IID.json','w') as file:
        json.dump(generated_data1,file)
    
    for k in generated_data2.keys():

        question = generated_data2[k]['question'].lower()
        question = re.sub('[^A-Za-z0-9]+', ' ', question)
        question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
        question = re.sub(r'^([\s\d]+)$', "100", question)
        question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

        for q in question.split():
            try:
                float(q)
                question = question.replace(q,'100')
            except:
                continue

        question_embedding = model.encode(question)

        question_embedding_ai = get_embed(question)[0].embedding

        generated_data2[k]['predicted_generic_evidence_iid_ai'] = [] #Only IID generic evidences

        oa_embeddings=[]
        for oa_embed in all_generic_evidences_data1_embd_ai:
            oa_embeddings.append(oa_embed.embedding)
        similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

        for i in (similarity_ai.argsort()[-recall:][::-1]):
            generated_data2[k]['predicted_generic_evidence_iid_ai'].append(all_generic_evidences_data1[i])


    with open('Output_OpenAI_retrieval_OOD.json','w') as file:
        json.dump(generated_data2,file)   


if method == 'BR':
    model = SentenceTransformer(cache_dir)
    all_generic_evidences_data1_embd = model.encode(all_generic_evidences_data1)   


    for k in generated_data1.keys():

        question = generated_data1[k]['question'].lower()
        question = re.sub('[^A-Za-z0-9]+', ' ', question)
        question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
        question = re.sub(r'^([\s\d]+)$', "100", question)
        question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

        for q in question.split():
            try:
                float(q)
                question = question.replace(q,'100')
            except:
                continue

        question_embedding = model.encode(question)

        generated_data1[k]['predicted_generic_evidence_iid'] = []
        
        similarity = np.matmul(all_generic_evidences_data1_embd,question_embedding)

        for i in (similarity.argsort()[-recall:][::-1]):
            generated_data1[k]['predicted_generic_evidence_iid'].append(all_generic_evidences_data1[i])


    with open('Outpur_BERT_retrieval_IID.json','w') as file:
        json.dump(generated_data1,file)


    for k in generated_data2.keys():

        question = generated_data2[k]['question'].lower()
        question = re.sub('[^A-Za-z0-9]+', ' ', question)
        question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
        question = re.sub(r'^([\s\d]+)$', "100", question)
        question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

        for q in question.split():
            try:
                float(q)
                question = question.replace(q,'100')
            except:
                continue

        question_embedding = model.encode(question)

        generated_data2[k]['predicted_generic_evidence_iid'] = [] #Only IID generic evidences
        
        similarity = np.matmul(all_generic_evidences_data1_embd,question_embedding)

        for i in (similarity.argsort()[-recall:][::-1]):
            generated_data2[k]['predicted_generic_evidence_iid'].append(all_generic_evidences_data1[i])


    with open('Ouput_BERT_retrieval_OOD.json','w') as file:
        json.dump(generated_data2,file)



### Function for OpenAI based retrieval where numbers are converted to constants in question and TDS
def bf_oi(generated_data1, all_generic_evidences):
    
    all_generic_evidences_data1=[]
    for j in all_generic_evidences:
        if j.strip().split('refers to')[0].replace("'",'') not in all_generic_evidences_data1:
            j=re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", j)
            j=re.sub(r'^([\s\d]+)$', "100", j)
            all_generic_evidences_data1.append(j.strip().split('refers to')[0].replace("'",''))
        
    all_generic_evidences_data1_embd_ai = get_embed(all_generic_evidences_data1)        

    for k in generated_data1.keys():
        question = generated_data1[k]['question']
        question=re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
        question=re.sub(r'^([\s\d]+)$', "100", question)
        question_embedding_ai = get_embed(question)[0].embedding

        generated_data1[k]['predicted_generic_evidence_iid_ai'] = []
        
        oa_embeddings=[]
        for oa_embed in all_generic_evidences_data1_embd_ai:
            oa_embeddings.append(oa_embed.embedding)
        similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

        for i in (similarity_ai.argsort()[-recall:][::-1]):
            generated_data1[k]['predicted_generic_evidence_iid_ai'].append(all_generic_evidences_data1[i])


    with open('ood_data_retr_OpenAI_const_refersto.json','w') as file:
        json.dump(generated_data1,file)

### Function for OpenAI based retrieval taking only part before refersto
def bf_oi_2(generated_data1, all_generic_evidences):
    
    all_generic_evidences_data1=[]
    for j in all_generic_evidences:
        if j.strip().split('refers to')[0].replace("'",'') not in all_generic_evidences_data1:
            all_generic_evidences_data1.append(j.strip().split('refers to')[0].replace("'",''))
        
    all_generic_evidences_data1_embd_ai = get_embed(all_generic_evidences_data1)        

    for k in generated_data1.keys():
        question = generated_data1[k]['question']
        question_embedding_ai = get_embed(question)[0].embedding

        generated_data1[k]['predicted_generic_evidence_iid_ai'] = []
        
        oa_embeddings=[]
        for oa_embed in all_generic_evidences_data1_embd_ai:
            oa_embeddings.append(oa_embed.embedding)
        similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

        for i in (similarity_ai.argsort()[-recall:][::-1]):
            generated_data1[k]['predicted_generic_evidence_iid_ai'].append(all_generic_evidences_data1[i])


    with open('/iid_data_retr_OpenAI_refersto_only.json','w') as file:
        json.dump(generated_data1,file)

## Function for combining both Lex and OpenAI based retrieval where we get top 10 DS using Lex and then final 4 
def LexPlusOpenAI(all_generic_evidences, generated_data1):
    all_generic_evidences_data1_lex=[]
    refer_map={}

    for j in all_generic_evidences:
        if j.strip().split('refers to')[0].replace("'",'') not in all_generic_evidences_data1_lex:
            all_generic_evidences_data1_lex.append(j.strip().split('refers to')[0].replace("'",''))
            refer_map[j.strip().split('refers to')[0].replace("'",'')] = j


    stop_words = set(stopwords.words('english'))

    evidence_vocab={}
    filter_org={}
    recall=10
    def intersection(lst1, lst2):
      lst3 = [value for value in lst1 if value in lst2]
      return lst3


    question_ev={}


    for k in generated_data1.keys():
      ev_score={}
      question=generated_data1[k]['question'].lower()
      question_ev[question]=[]
      word_tokens=question.split()
      filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]


      for word in filtered_sentence:
        for ev in all_generic_evidences_data1_lex:
          word_tokens=ev.split()
          ev_filter=[w for w in word_tokens if not w.lower() in stop_words]
          if word in  ev_filter:
            ev_score[ev]=len(intersection(filtered_sentence,ev_filter))/len(ev_filter)
            
      top_evs=list(dict(sorted(ev_score.items(), key=lambda item: item[1])).keys())
      top_evs.reverse()
      top_evs=top_evs[:recall]
      generated_data1[k]['predicted_generic_evidence_iid_lex']=[]
      for j in top_evs:
        generated_data1[k]['predicted_generic_evidence_iid_lex'].append(j)

    recall=4        

    for k in generated_data1.keys():
        if len(generated_data1[k]['predicted_generic_evidence_iid_lex']) > 0:
            all_generic_evidences_data1_embd_ai = get_embed(generated_data1[k]['predicted_generic_evidence_iid_lex'])
        else:
            generated_data1[k]['predicted_generic_evidence_iid_ai']=['NA']
            continue
        question = generated_data1[k]['question']
        question_embedding_ai = get_embed(question)[0].embeddingpredicted_generic_evidence_iid_ai

        generated_data1[k]['predicted_generic_evidence_iid_ai'] = []
        
        oa_embeddings=[]
        for oa_embed in all_generic_evidences_data1_embd_ai:
            oa_embeddings.append(oa_embed.embedding)
        similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

        top_evs=[]
        # try:
        for i in (similarity_ai.argsort()[-recall:][::-1]):
            top_evs.append(generated_data1[k]['predicted_generic_evidence_iid_lex'][i])
        # except:
        #     print(generated_data1[k]['predicted_generic_evidence_iid_lex'])
        #     print(i)
        #     input()

        for j in top_evs:
            generated_data1[k]['predicted_generic_evidence_iid_ai'].append(refer_map[j])


    with open('ood_data_retr_lexPlusOI.json','w') as file:
        json.dump(generated_data1,file)


## Function for retrieving unique set of DS (2 using Lex and 2 using OR)
def LexPlusOpenAI_2(all_generic_evidences, generated_data1):
    all_generic_evidences_data1_lex=[]
    refer_map={}

    for j in all_generic_evidences:
        if j.strip().split('refers to')[0].replace("'",'') not in all_generic_evidences_data1_lex:
            all_generic_evidences_data1_lex.append(j.strip().split('refers to')[0].replace("'",''))
            refer_map[j.strip().split('refers to')[0].replace("'",'')] = j


    stop_words = set(stopwords.words('english'))

    evidence_vocab={}
    filter_org={}
    recall=2
    def intersection(lst1, lst2):
      lst3 = [value for value in lst1 if value in lst2]
      return lst3


    question_ev={}


    for k in generated_data1.keys():
      ev_score={}
      question=generated_data1[k]['question'].lower()
      question_ev[question]=[]
      word_tokens=question.split()
      filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]


      for word in filtered_sentence:
        for ev in all_generic_evidences_data1_lex:
          word_tokens=ev.split()
          ev_filter=[w for w in word_tokens if not w.lower() in stop_words]
          if word in  ev_filter:
            ev_score[ev]=len(intersection(filtered_sentence,ev_filter))/len(ev_filter)
            
      top_evs=list(dict(sorted(ev_score.items(), key=lambda item: item[1])).keys())
      top_evs.reverse()
      top_evs=top_evs[:recall]
      generated_data1[k]['predicted_generic_evidence_iid_ai']=[]
      for j in top_evs:
        generated_data1[k]['predicted_generic_evidence_iid_ai'].append(refer_map[j])
       
    all_generic_evidences_data1_embd_ai = get_embed(all_generic_evidences)        
    for k in generated_data1.keys():
        
        question = generated_data1[k]['question'].lower()
        question_embedding_ai = get_embed(question)[0].embedding
        
        oa_embeddings=[]
        for oa_embed in all_generic_evidences_data1_embd_ai:
            oa_embeddings.append(oa_embed.embedding)
        similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

        for i in (similarity_ai.argsort()[-recall:][::-1]):
            generated_data1[k]['predicted_generic_evidence_iid_ai'].append(all_generic_evidences[i])

    with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/iid_data_retr_lexPlusOI_2.json','w') as file:
        json.dump(generated_data1,file)

## Function to get similar few shots from the IID set for queries in OOD set
def similarInIID(generated_data1,generated_data2):
    oa_embeddings=[]
    for k2 in generated_data1.keys():
        oa_embeddings.append(get_embed(generated_data1[k2]['question'])[0].embedding)

    for k in generated_data2.keys():

        question_embedding_ai = get_embed(generated_data2[k]['question'])[0].embedding

        generated_data2[k]['predicted_iid_shots_ood_ai'] = []

        similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

        for i in (similarity_ai.argsort()[-recall:][::-1]):
            generated_data2[k]['predicted_iid_shots_ood_ai'].append([generated_data1[str(i)]['question'],generated_data1[str(i)]['GT_SQL']])

    with open('iid_shots_ood_ai.json','w') as file:
        json.dump(generated_data2,file)

## Function to get Random few shots from the IID set for queries in OOD set
def randomIID(generated_data1,generated_data2):

    key_list=list(generated_data1.keys())
    random.seed(7)
    random.shuffle(key_list)
    for k in generated_data2.keys():
        generated_data2[k]['predicted_iid_shots_ood_ai'] = []
        for i in key_list[:4]:
                generated_data2[k]['predicted_iid_shots_ood_ai'].append([generated_data1[str(i)]['question'],generated_data1[str(i)]['GT_SQL']])

    with open('/iid_shots_ood_random.json','w') as file:
        json.dump(generated_data2,file)

## Function to get BERT based retrieved evidneces after decomposing the questions and converting constants and dates from the question to placeholders
def decompose_BERT(all_generic_evidences, question):
    recall=10

    question = re.sub('[^A-Za-z0-9]+', ' ', question)
    question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
    question = re.sub(r'^([\s\d]+)$', "100", question)
    question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

    for q in question.split():
        try:
            float(q)
            question = question.replace(q,'100')
        except:
            continue

    question_embedding = model.encode(question)

    predicted_generic_evidence_iid = []
    
    similarity = np.matmul(all_generic_evidences_data1_embd,question_embedding)
    
    similarity[::-1].sort()

    for i,j in zip(similarity.argsort()[-recall:][::-1],similarity):
        predicted_generic_evidence_iid.append([all_generic_evidences_data1[i],j])

    print(predicted_generic_evidence_iid)


## Function to get OpenAI based retrieved evidneces after decomposing the questions and converting constants and dates from the question to placeholders
def decompose_OpenAI(all_generic_evidences, question):
    recall=10

    question = re.sub('[^A-Za-z0-9]+', ' ', question)
    question = re.sub("\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d\d|20\d\d)\b", "100-100-100", question)
    question = re.sub(r'^([\s\d]+)$', "100", question)
    question = re.sub(r'\d{2}:\d{2}:\d{2}', "100:100:100", question)

    for q in question.split():
        try:
            float(q)
            question = question.replace(q,'100')
        except:
            continue

    question_embedding_ai = get_embed(question)[0].embedding

    oa_embeddings=[]
    for oa_embed in all_generic_evidences_data1_embd_ai:
        oa_embeddings.append(oa_embed.embedding)
    similarity_ai = np.matmul(oa_embeddings,question_embedding_ai)

    predicted_generic_evidence_iid = []
    
    similarity_ai[::-1].sort()

    for i,j in zip(similarity_ai.argsort()[-recall:][::-1],similarity_ai):
        predicted_generic_evidence_iid.append([all_generic_evidences_data1[i],j])

    print(predicted_generic_evidence_iid)

## Function that returns the lexical overlap between two lists
def lex_overlap(lst1,lst2):
  lst3 = [value for value in lst1 if value in lst2]
  return len(list(set(lst3)))

## Function that selects the retrieved evidences using SR with a soecified threshold and recall for Templatized Dmain Statements
def get_filtered_evidences(path, threshold,split):
    with open(path,'rb') as file:
        evs=pickle.load(file)

    if split == 'iid':
        for k in generated_data1.keys():
            generated_data1[k]['pred_evidences_thresholded'] = []
            for ev in evs[generated_data1[k]['question']]:
                # if ev[1] >= threshold:
                generated_data1[k]['pred_evidences_thresholded'].append(ev[0])
                if len(generated_data1[k]['pred_evidences_thresholded']) >= 4:
                    break
            # generated_data1[k]['pred_evidences_thresholded'] = generated_data1[k]['pred_evidences_thresholded'][:10]
        with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/'+split+'_whole_thr_'+str(threshold)+'.json','w') as file:
            json.dump(generated_data1,file)
    else:
        for k in generated_data2.keys():
            generated_data2[k]['pred_evidences_thresholded'] = []
            for ev in evs[generated_data2[k]['question']]:
                # if ev[1] >= threshold:
                generated_data2[k]['pred_evidences_thresholded'].append(ev[0])
                if len(generated_data2[k]['pred_evidences_thresholded']) >= 4:
                    break
            # generated_data2[k]['pred_evidences_thresholded'] = generated_data2[k]['pred_evidences_thresholded'][:10]
        with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/'+split+'_whole_thr_'+str(threshold)+'.json','w') as file:
            json.dump(generated_data2,file)

## Function that selects the retrieved evidences using SR with a soecified threshold and recall for Non-Templatized Dmain Statements
def get_filtered_evidences_NT(path, threshold,split):
    with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/gen_act_map.json','r') as file:
        ev_map=json.load(file)
    with open(path) as file:
        evs=json.load(file)

    if split == 'iid':
        for k in generated_data1.keys():
            generated_data1[k]['pred_evidences_thresholded'] = []
            for ev in evs[generated_data1[k]['question']]:
                if ev[1] >= threshold:
                    try:
                        generated_data1[k]['pred_evidences_thresholded'].append(ev_map[ev[0]])
                    except:
                        continue
                if len(generated_data1[k]['pred_evidences_thresholded']) >= 10:
                    break
        with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/'+split+'_NT_thr_'+str(threshold)+'.json','w') as file:
            json.dump(generated_data1,file)
    else:
        for k in generated_data2.keys():
            generated_data2[k]['pred_evidences_thresholded'] = []
            for ev in evs[generated_data2[k]['question']]:
                if ev[1] >= threshold:
                    try:
                        generated_data2[k]['pred_evidences_thresholded'].append(ev_map[ev[0]])
                    except:
                        continue
                if len(generated_data2[k]['pred_evidences_thresholded']) >= 10:
                    break
        with open('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/'+split+'_NT_thr_'+str(threshold)+'.json','w') as file:
            json.dump(generated_data2,file)

## Function to get Semantic Overlap between manually generated TDS and through LLM
def get_intersection(path, generated_data1):
    with open(path) as file:
        in_domain=json.load(file)

    intersection=0
    tot=0
    for i,j in zip(in_domain,generated_data1):
        for k,l in zip(generated_data1[j]['generic_evidence'],in_domain[i]['generic_evidence']):
            # intersection+= lex_overlap(k.split(),l.split())/len(set(k.split()))
            intersection += np.matmul(model.encode(k),model.encode(l))
            # print(intersection)
            # input()
            tot+=1
    print(intersection/tot)

## Function to get Lexical Overlap between manually generated TDS and through LLM
def get_intersection(path, generated_data1):
    with open(path) as file:
        in_domain=json.load(file)

    intersection=0
    tot=0
    for i,j in zip(in_domain,generated_data1):
        for k,l in zip(generated_data1[j]['generic_evidence'],in_domain[i]['generic_evidence']):
            intersection+= lex_overlap(k.split(),l.split())/len(set(k.split()))
            tot+=1
    print(intersection/tot)

## Function Call to get the SR retrieved evidences along with the threshold
if method == 'SR':
    get_filtered_evidences('pred_evidences_ood_matrix_whole',threshold,'ood')

## Function call to get the SR retrieved Non-Templatized evidences along with the threshold
if method == 'BR NT':
    get_filtered_evidences_NT('pred_evidences_ood_matrix.json',threshold,'ood')

if method == 'FS R':
    randomIID(generated_data1,generated_data2)

# #Generate SQLs and get Results
## Results for IID set
# print('Results on the iid set: '+str(threshold))
# get_result('/ood_data_retr_lex.json','iid')
## Reaults for OOD set
# print('Results on the ood set: '+str(threshold))
# get_result('/home/hw1639188/Documents/project/NLtoSQL/Manasi Code/'+folder+'/'+'ood'+'_whole_thr_'+str(threshold)+'.json','ood') 