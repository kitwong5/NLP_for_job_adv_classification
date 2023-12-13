from flask import Flask, render_template, request
from gensim.models.fasttext import FastText
from itertools import chain
import pickle
import pandas as pd
import os
import numpy as np
import json

# reference: L.Gallagher (2023) ‘Week 10 - Activies’ [Lecture Notes, COSC2820 Advanced Programming for Data Science], RMIT University, Melbourne
def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs

# Category and Category_id mapping
def map_cat(cat_id, df_input):
    category = ""
    df_cat = df_input.drop_duplicates(['categoryidx', 'category'])
    for i, row in df_cat.iterrows():
        if row['categoryidx'] == cat_id:
            category = row['category']
            return category

def map_catid(cat_name, df_input):
    category_id = None
    df_cat = df_input.drop_duplicates(['categoryidx', 'category'])
    for i, row in df_cat.iterrows():
        if row['category'] == cat_name:
            category_id = row['categoryidx']
            return category_id


app = Flask(__name__)

@app.route('/')
def index():
    # Open job adv data file for reading
    df = pd.read_csv('job_adv.txt', sep=" ")
    df_catid = df.drop_duplicates(['categoryidx', 'category'])

    return render_template('home.html', category_id_ls=df_catid['categoryidx'], df_catid=df_catid,category_len = len(df_catid['category']))

@app.route('/job_seekers/<cat_index>', methods=['GET'])
def job_seekers(cat_index = 9):
    # Open job adv data file for reading
    df = pd.read_csv('job_adv.txt', sep=" ")
    df_catid = df.drop_duplicates(['categoryidx', 'category'])
    ##category_ls = df['category'].unique().tolist()
    return render_template('job_seekers.html', df_title=df['title'], df_desc=df['description'], df_cat=df['categoryidx'], len = len(df), cat_index = int(cat_index), df_catid=df_catid, category_len = len(df_catid['category']))

@app.route('/job_seekers_index/<job_index>', methods=['GET', 'POST'])
def job_seekers_index(job_index):
    # Open job adv data file for reading
    df = pd.read_csv('job_adv.txt', sep=" ")
    return render_template('job_seekers_index.html', 
                           job_index = job_index, 
                           title=df.iloc[int(job_index)]['title'], 
                           desc=df.iloc[int(job_index)]['description'],
                           comp=df.iloc[int(job_index)]['company'],
                           webindex=df.iloc[int(job_index)]['webindex']   )

@app.route('/employers', methods=['GET', 'POST'])
def employers():
    # Open job adv data file for reading
    df = pd.read_csv('job_adv.txt', sep=" ")
    category_ls = df['category'].unique().tolist()
    if request.method == 'POST':
        # Read the content
        f_title = request.form['title']
        f_desc = request.form['description']
        f_comp = request.form['company']
        f_cat = request.form['category']
        f_webid = request.form['webid']
        f_index = request.form['df_index']

        # reference: L.Gallagher (2023) ‘Week 10 - Activies’ [Lecture Notes, COSC2820 Advanced Programming for Data Science], RMIT University, Melbourne
        # Tokenize the content of the .txt file so as to input to the saved model
        # Here, as an example,  we just do a very simple tokenization
        tokenized_data = f_desc.split(' ')

        # Load the FastText model
        descFT = FastText.load("descFT.model")
        descFT_wv= descFT.wv

        # Generate vector representation of the tokenized data
        descFT_dvs = docvecs(descFT_wv, [tokenized_data])

        # Load the LR model
        pkl_filename = "descFT_LR.pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict the label of tokenized_data
        y_pred = model.predict(descFT_dvs)
        y_pred = y_pred[0]

        y_pred_cat = map_cat(y_pred, df)
        # Set the predicted message
        y_pred
        predicted_message = "Prediction of this job's category is {}.".format(y_pred_cat)

        # prepare words list
        tk_ls = list(tokenized_data)

        # find category id with the input category name
        f_cat_id = map_catid(f_cat, df)
        # if input category name is a new value, create a new category id 
        if f_cat_id == None:
            f_cat_id = df['categoryidx'].max() + 1



        
        if f_webid == '' or f_webid == None:
            # create new web index for the new job adv, add the entry to dataframe
            webindex = df['webindex'].max() + 1
            df_index = df.index.max() + 1
            insert_row = {'advertisement':[f_title+str(webindex)+f_comp+f_desc], 
               'categoryidx':[f_cat_id], 
               'filename':[None],
               'title':[f_title], 
               'webindex':[webindex], 
               'company':[f_comp], 
               'description':[f_desc], 
               'tk_adv5':[tk_ls], 
               'category':[f_cat]}
            df = pd.concat([df,  pd.DataFrame(insert_row)], ignore_index=True)
        else:
            # update user changes of the new job entry    
            webindex = f_webid
            df_index = f_index
            insert_row = {'advertisement':[f_title+str(webindex)+f_comp+f_desc], 
               'categoryidx':[f_cat_id], 
               'filename':[None],
               'title':[f_title], 
               'webindex':[webindex], 
               'company':[f_comp], 
               'description':[f_desc], 
               'tk_adv5':[tk_ls], 
               'category':[f_cat]}
            df = pd.concat([df[:-1],  pd.DataFrame(insert_row)], ignore_index=True)

        # save changes to file
        df.to_csv('job_adv.txt', index=None, sep=' ')
        # go back to employer page
        return render_template('employers.html', category_ls=category_ls, category_ls_len=len(category_ls), category=f_cat, company=f_comp, title=f_title, description=f_desc, predicted_message=predicted_message, webindex=webindex, df_index=df_index)
    else:    
        return render_template('employers.html', category_ls=category_ls, category_ls_len=len(category_ls))

