##To run the codes with templatized domain statements

1. BM 25 iid and ood results
python bm25.py

2. Marcos based dense retrieval for IN and OUT sets, respectively
python dense_retriever_marcos_iid.py; dense_retriever_marcos_ood.py

3. Roberta based dense retrieval for IN and OUT sets, respectively
python sbst_iid.py; sbst_ood.py

4. Coverage based retrieval for IN and OUT sets, respectively
python sBSR_iid.py; python sBSR_ood.py



To run the codes with non-templatized domain statements

1. BM 25 iid and ood results
python NT/bm25_NT.py

2. Marcos based dense retrieval for IN and OUT sets, respectively
python NT/dense_retriever_marcos_iid_NT.py; NT/dense_retriever_marcos_ood_NT.py

3. Roberta based dense retrieval for IN and OUT sets, respectively
python NT/sbst_iid_NT.py; NT/sbst_ood_NT.py

4. Coverage based retrieval for IN and OUT sets, respectively
python NT/sBSR_iid_NT.py; python NT/sBSR_ood_NT.py

5. Our proposed approach based retrieval for IN and OUT sets, respectively
python NT/new_SR_loop_NT.py; python NT/new_SR_ood_loop_NT.py
