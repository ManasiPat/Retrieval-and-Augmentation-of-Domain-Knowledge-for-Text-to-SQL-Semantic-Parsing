# Retrieval-and-Augmentation-of-Domain-Knowledge-for-Text-to-SQL-Semantic-Parsing

To get the retrieved evidences for IN and OUT split respectively, refer to ReadMe.md file in Retrieval folder. 

Run openai_gen_Code.py file to get the results using GPT3.5 Turbo.

Run hugging_face_SQL_Gen.py file to get the results using Mixtral, SQL Coder and LLama based results (use approach model paths from hugging face).

Run gemini_SQL_gen.py file to get the results using Gemini 1.5 Flash.

For getting the results using a particular method, select the method from the list of methods commented after the method variable is defined in the file.
Provide the appropriate paths and the OpenAI api key to run the code.

The files required for running the code are provided. Files Provided:

1. Split Files for IID and OOD splits respectievly
   
2. Templatized Domain Statements Files
   
3. Manually curated Value and Column Description Files
