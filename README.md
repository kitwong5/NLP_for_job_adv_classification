# Natural Language Processing for Job Advertisement Classification

# 1) Basic Text Pre-processing
Input Data: files in input_data folder <br>
File: basic_text_preprocessing.ipynb

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
 
# 2) Generating Feature Representations and perform Classification for Job Advertisement Descriptions
Input Data: files generated from step 1 (job_adv.txt, job_adv_json.txt, job_adv_string.txt, vocab.txt) <br>
File: generating_feature_representations_for_job_adv_desc.ipynb

1) Language model comparisons

To find out which language model performs the best on the job advertisement descriptions. Modeling with in-house trained and pre-trained language models has been executed with weighted and un-weighted word vectors. Below are the models used in this language model comparison activity.

FastText: model was Initialised with using 300 dimention vectors.
GloVe: the text-encoded with 300-dimensional vectors size (glove.6B.300d.txt) was used.
Word2Vec: embeddings pre-trained from Google news 300 dataset. [4]
Doc2Vec: vector size of 300 was used to fit the Doc2Vec mode.

Below are the summarize of the model performance:
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/a31a72aa-9d8e-4973-92da-dd8e74c41fd1)

Findings: <br>
Pre-trained Word2Vec has the best performance (highest Accuracy, lowest MAE, RMSE) compared to other models on the job advertisement descriptions.
Overall, the in-house trained models do not performance as well as the pre-trained models.
For the weighted vs. unweighted version of model, mixed results were obtained. For example, FastText in-house trained model got better MAE, RMSE in the weighted version, however pre-trained GloVe and Word2Vec got better MAE, RMSE in the unweighted version instead.

2) Does more information provide higher accuracy?

Un-weighted word embeddings:
Doc2Vec and Word2Vec language models were used to evaluate if more information provided can enhance the model performance. Execution of the logistic regression model was performed for the Title of the job advertisement, description of the job advertisement, and concatenation of the title and description of the job advertisement. Below is the evaluation result.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/37bc6036-bf3e-49c5-925f-baa5f702a418)

In both Doc2Vec and WordVec models with un-weighted embedding vector, using concatenation of the title and description has achieved the highest model performance (the lowest MAE/RMSE and the highest accuracy) when compared to modeling with title or description alone. The result demonstrated that when more information cooperates for word embedding, it can improve the classification model accuracy.

Weighted word embeddings: <br>
Word2Vec language model was used to evaluate if more information provided can enhance the model performance for weighted word embeddings. Below is the evaluation result of the model performance.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/4b68967d-c400-4337-9cb5-acd52b804ca9)

The weighted word embedding model execution findings were not in line with the results from the un-weighted one. In the weighted word embeddings, concatenation of the title and description resulted in lowering the model performance. The findings showed that adding more information did not able to achieve higher accuracy for modeling with weighted word embeddings.

# 3)  Web App for Job Advertisement Classification
Setup Setps: <br>
- save Fast Text model bbcFT_dvs as descFT.model.wv.vectors_ngrams.npy and place the file inside the '\web_app\code' folder 
- place all files inside the folder '\web_app\code' to your python vitural directory
- open cmd, activate python, and cd to your python vitural directory
- run the follow steps in cmd, to activate the app.py program
  - set FLASK_APP=app
  - flask run
- the web page should available at the follow link: http://127.0.0.1:5000/
- if file size of 'descFT.model.wv.vectors_ngrams.npy' is too large to save.  The vector size of FastText model has changed form 300 to 30.

The application contains 3 pages, the home page, the job seekers page, and the employers page.  On the top of the application, there is a menu bar that allows users to easily access these 3 pages.   The home page is the landing page that users will see when the application is first loaded.  The main purpose of the home page is to let users have quick access to both the job seeker page and the employer's page.  The Home page also listed out the job seeker category, so that job seekers can quickly access the job adv page for a specific category.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/e4fdbfa0-de3e-4b91-8fdb-8e19438ecf80)

In the Job Seekers page. there is a sub-menu bar available on the top of the page for users to view the job adv by category. The page listed all the related job adv with Titles and a brief description. If users click on the title, it will bring them to another page to show them the full details of the job advertisements.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/cacde0ac-32b7-4afa-94bd-79b79132bdfb)

In the employer page, it allows employer users to input a new job advisement.   For the job category field, users can either select an existing category from the drop-down box or they can key in a new category in the same input box.  All input fields are set to be mandatory because they are all required fields.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/bd5274d4-8453-457e-bebe-f65be7125897)

Once users fill out the form and press the save button. The application will then use the job description field to perform the prediction of the advertisement category.  FastText learning word embedding model and logistic regression model were used for the prediction.  The prediction result will be shown at the bottom of the page.  
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/15fbc359-e5b2-4f86-97a9-f9de215493db)

If users feel like the prediction category, they can make changes to the category field.  Then press the save button again to record the change.  The updated job information can now be searched on the home page or on the Job seekers page.
![image](https://github.com/kitwong5/NLP_for_job_adv_classification/assets/142315009/3fcc584a-53d6-425e-9d34-f19be1cfbb92)







