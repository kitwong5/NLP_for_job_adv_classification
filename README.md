# Natural Language Processing for Job Advertisement Classification

# 1) Basic Text Pre-processing
input data: files in input_data folder <br>
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
input data: files generated from step 1 (job_adv.txt, job_adv_json.txt, job_adv_string.txt, vocab.txt) <br>
file: generating_feature_representations_for_job_adv_desc.ipynb

1) Language model comparisons

To find out which language model performs the best on the job advertisement descriptions. Modeling with in-house trained and pre-trained language models has been executed with weighted and un-weighted word vectors. Below are the models used in this language model comparison activity.

FastText: model was Initialised with using 300 dimention vectors.
GloVe: the text-encoded with 300-dimensional vectors size (glove.6B.300d.txt) was used.
Word2Vec: embeddings pre-trained from Google news 300 dataset. [4]
Doc2Vec: vector size of 300 was used to fit the Doc2Vec mode.

Below are the summarize of the model performance:
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/a31a72aa-9d8e-4973-92da-dd8e74c41fd1)

Findings:
Pre-trained Word2Vec has the best performance (highest Accuracy, lowest MAE, RMSE) compared to other models on the job advertisement descriptions.
Overall, the in-house trained models do not performance as well as the pre-trained models.
For the weighted vs. unweighted version of model, mixed results were obtained. For example, FastText in-house trained model got better MAE, RMSE in the weighted version, however pre-trained GloVe and Word2Vec got better MAE, RMSE in the unweighted version instead.

2) Does more information provide higher accuracy?

Un-weighted word embeddings:
Doc2Vec and Word2Vec language models were used to evaluate if more information provided can enhance the model performance. Execution of the logistic regression model was performed for the Title of the job advertisement, description of the job advertisement, and concatenation of the title and description of the job advertisement. Below is the evaluation result.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/37bc6036-bf3e-49c5-925f-baa5f702a418)

In both Doc2Vec and WordVec models with un-weighted embedding vector, using concatenation of the title and description has achieved the highest model performance (the lowest MAE/RMSE and the highest accuracy) when compared to modeling with title or description alone. The result demonstrated that when more information cooperates for word embedding, it can improve the classification model accuracy.

Weighted word embeddings:
Word2Vec language model was used to evaluate if more information provided can enhance the model performance for weighted word embeddings. Below is the evaluation result of the model performance.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/4b68967d-c400-4337-9cb5-acd52b804ca9)

The weighted word embedding model execution findings were not in line with the results from the un-weighted one. In the weighted word embeddings, concatenation of the title and description resulted in lowering the model performance. The findings showed that adding more information did not able to achieve higher accuracy for modeling with weighted word embeddings.




