# To run the codes with templatized domain statements

1. BM 25 iid and ood results run
  
   python bm25.py

2. Marcos based dense retrieval for IN and OUT sets, respectively run

   python dense_retriever_marcos_iid.py; dense_retriever_marcos_ood.py

3. Roberta based dense retrieval for IN and OUT sets, respectively run

    python sbst_iid.py; sbst_ood.py

4. Coverage based retrieval for IN and OUT sets, respectively

    python sBSR_iid.py; python sBSR_ood.py



# To run the codes with non-templatized domain statements

1. BM25 IN and OUT results run

  python NT/bm25_NT.py

2. Marcos based dense retrieval for IN and OUT sets, respectively, run
 
   python NT/dense_retriever_marcos_iid_NT.py; NT/dense_retriever_marcos_ood_NT.py

3. Roberta based dense retrieval for IN and OUT sets, respectively run

   python NT/sbst_iid_NT.py; NT/sbst_ood_NT.py

5. Coverage based retrieval for IN and OUT sets, respectively run

   python NT/sBSR_iid_NT.py; python NT/sBSR_ood_NT.py

6. Our proposed approach based retrieval for IN and OUT sets, respectively run

    python NT/new_SR_loop_NT.py; python NT/new_SR_ood_loop_NT.py
