# Natural Language Processing for Job Advertisement Classification

# 1) Basic Text Pre-processing
input data: files in input_data folder \n
file: basic_text_preprocessing.ipynb

- Tokenizing the job advertisement list
- Removal of stopwords, single character words, most common words, and words that only appear one time
- The vocabulary filtering statistics as follow:
  - vocab count before filtering: 9834
  - after single character words removal: 9808 (less 26 vocab)
  - after stopwords removal: 9404 (less 404 vocab)
  - after words that only appear once removal: 5218 (less 4186 vocab)
  - after most commonn 50 words removal: 5168 (less 50 vocab)
- Through the unrelated vocabulary removal, the lexical diversity ratio has increased from 0.053 to 0.064.
- The lexical diversity ratio statistics is as follows.
  - vocab count before filtering: 0.053
  - after single character words removal: 0.054
  - after stopwords removal: 0.088
  - after words that only appear once removal: 0.051
  - after most commonn 50 words removal: 0.065
 
# 2) Generating Feature Representations for Job Advertisement Descriptions
input data: files generated from step 1 (job_adv.txt, job_adv_json.txt, job_adv_string.txt, vocab.txt) \n
file: generating_feature_representations_for_job_adv_desc.ipynb
