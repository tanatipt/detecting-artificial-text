import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import re
import os
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
from spacy_syllables import SpacySyllables
from collections import Counter
import string
import math
from lexical_diversity import lex_div as ld
from functionwords import FunctionWords
import sys


# Initialising a robust scaler
scaler = RobustScaler()
# Initialising the list of function words
fw = FunctionWords(function_words_list='english')
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after="tagger")
# A list storing all POS tags from the universal POS tagsets
pos_tags = ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB", "ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ", "PUNCT", "SYM", "X"]
# Initialising a one-hot encoder
enc = OneHotEncoder(sparse_output=False)

def transform_label(y):
    """
    
    Given a list of labels, this function transforms each label into a one-hot encoding

    Args:
        y (pandas.Series): A series of string representing the author of each news article 

    Returns:
        pandas.DataFrame: A dataframe of the one-hot encoding of the label for each news article
    """
    y_ohe = enc.transform(y).astype(int)
    return pd.DataFrame(y_ohe, columns=enc.get_feature_names_out())

def compute_stylo(row):
    """
    
    Given an article's text, this function computes the stylometric features for the article.

    Args:
        row (pandas.Series): The Series of the given article w/o the stylometric features

    Returns:
        pandas.Series : The Series of the given article with the stylometric features
    """
    # Retrieving the text of the article
    original_text = row['text']
    # Tranforming the text into a Spacy Doc object
    doc = nlp(original_text)
    
    # Stores the length of each word in the article
    words_len = []
    # Stores the length of each sentence in the article
    sentences_len = []
    # Store the syllable count of each word in the article
    syllable_count = []
    # Store the POS tag count in each sentence of the article
    pos_counts = {k : [] for k in pos_tags}
    # Store the number of stopwords in the article
    sw_counts = []
    # Store the number of function words in the article
    fw_counts = []
    # Store the lemma of each word in the article
    lemmas  = []
    # Store the number of punctuation in the article
    total_punctuation = 0
    
    # Iterating through each sentence in the article
    for sent in doc.sents:
        # Store the POS tag count in the current sentence
        sentence_tags_count = {k : 0 for k in pos_tags}
        # Store the number of stopwords in the current sentence
        sentence_sw_count = 0
        # Store the number of function words in the current sentence
        sentence_fw_count = 0
        # Store the number of words in the sentence
        sentence_length = 0
        
        # Iterating through each word in the current sentence
        for word in sent:  
            
            # Recording the POS tag of the current word
            if word.pos_ in sentence_tags_count:
                sentence_tags_count[word.pos_] += 1
            else:
                sentence_tags_count["X"] += 1
            
            # Checking if the current word is a function word
            if word.lower_ in fw.function_words:
                    sentence_fw_count += 1
            
            # Checking if the current word is a stop word 
            if word.is_stop:
                sentence_sw_count += 1
            
            if not word.is_punct and not word.is_space and word._.syllables_count is not None:
                # Recording the lemma of the current word
                lemmas.append(word.lemma_)
                # Recording the length of the current word
                words_len.append(len(word))
                sentence_length += 1
                # Recording the syllables count of the current word
                syllable_count.append(word._.syllables_count)
            
            # Recording the number of punctuations used in the current sentence
            total_punctuation += sum(1 for c in word.text if c in string.punctuation)
            

        for pos_tag, sentence_tag_count in sentence_tags_count.items():
            pos_counts[pos_tag].append(sentence_tag_count)  
        
        sw_counts.append(sentence_sw_count)
        fw_counts.append(sentence_fw_count)
        sentences_len.append(sentence_length)
    
    # Count the number of times each lemmas was used in the article
    counts = Counter(lemmas)
    
    # Computing the Hapax Legomenon and Dislegemena
    legomena = [lemma for lemma in counts if counts[lemma] == 1]
    dislegomina = [lemma for lemma in counts if counts[lemma] == 2]
    # Computing the number of different lemmas used in the article
    vocab_count = len(set(lemmas))
    
    # Appending the total number of word, sentences , stopwords, function words and punctuations used in the sentence to the Series
    row["total_word"] = len(words_len)
    row["total_sentence"] = len(sentences_len)
    row["total_sw"] = sum(sw_counts)
    row["total_fw"] = sum(fw_counts)
    row["total_punctuation"] = total_punctuation
    
    # Appending the average character and syllable per word to the Series. (In addition to its standard deviation)
    row["char_per_word_m"] , row["char_per_word_std"] = np.mean(words_len), np.std(words_len)
    row["syllable_per_word_m"], row["syllable_per_word_std"] = np.mean(syllable_count), np.std(syllable_count)
   
   # Appending the average word, stopword and function word per sentence to the Series. (In addition to its standard deviation)
    row["word_per_sentence_m"] , row["word_per_sentence_std"] = np.mean(sentences_len), np.std(sentences_len)
    row["sw_per_sentence_m"] , row["sw_per_sentence_std"] = np.mean(sw_counts), np.std(sw_counts)
    row["fw_per_sentence_m"] , row["fw_per_sentence_std"] = np.mean(fw_counts), np.std(fw_counts)
    
    # For each POS tag in the universal POS tagset, we append the average POS tag count per sentence to the Series. (In addition to its standard deviation)
    pos_m, pos_std = {pos_tag : np.mean(pos_counts[pos_tag]) for pos_tag in pos_tags}, {pos_tag : np.std(pos_counts[pos_tag]) for pos_tag in pos_tags}
    for pos_tag in pos_tags:
        row[f"{pos_tag}_m"] = pos_m[pos_tag]
        row[f"{pos_tag}_std"] = pos_std[pos_tag]
    
    # Appending the number of Hapax Legomenon and Dislegemena in the article to the Series
    row["legomena_count"] = len(legomena)
    row["dislegomina_count"] = len(dislegomina)
    
    # Appending the MATTR, Honores Measure and Sichel Measure of the article to the Series
    row["mttr"] = ld.maas_ttr(lemmas)
    row["honores_measure"] = 100 * math.log(row["total_word"] / (1 - (row["legomena_count"] / vocab_count)) if row["legomena_count"]  < vocab_count else sys.maxsize) 
    row["sichel_measure"] = row["dislegomina_count"] / vocab_count
    
    # Appending the Flesh Reading Ease score, Flesh-Kincaid Grading Level and Gunning Fog Index of the article to the Series
    row["fr_score"] = 206.835 - 1.015 * row["word_per_sentence_m"] - 84.6 * row["syllable_per_word_m"]
    row["fkg_score"] = 0.39 * row["word_per_sentence_m"] + 11.8 * row["syllable_per_word_m"] - 15.59
    row["gunning_fox"] = 0.4 * (row["word_per_sentence_m"] + 100 * (len([syllable for syllable in syllable_count if syllable >= 3]) / row["total_word"]))
    
    return row


def load_data():
    """
    
    Load the news articles from the MAGE testbed that have been written by humans or generated
    by the 8 LLMs :  LLama 65B , GPT-3.5 Turbo, Text Davinci-003, Flan-T5 XXL, GPT-J ,GPT-NeoX , OPT 30B,
    BigScience T0 11B.

    Returns:
        pandas.DataFrame: A dataframe consisting of news articles generated by the selected authors, along with its label.
    """
    dataset = pd.DataFrame(columns=['text', 'label'])  
    # Iterating through each file in the source_data directory
    for file in os.listdir("source_data/"):
        # Obtaining the author that generated the articles in a given CSV file
        label = os.path.splitext(file.split("_",maxsplit=1)[1])[0]
        df = pd.read_csv(f"source_data/{file}").drop("index", axis=1)
        df['label'] = label
        dataset = pd.concat([dataset, df], ignore_index=True)
    
    # Performing white space reduction on each news articles
    dataset['text'] = dataset['text'].apply(lambda x : re.sub(r"\s+" , " ",x).strip())
    # Computing the stylometric features for each news article in the dataframe
    dataset = dataset.reset_index(drop=True).apply(compute_stylo, axis = 1)
    
    return dataset


# A dataframe containing the news articles from the MAGE testbed generated by our selected authors
dataset = load_data()
# Obtaining the name of the columns that contain the stylometric features
stylo_columns = dataset.columns[list(dataset.columns).index("label")+1:]
# Fitting our one-hot encoder on the label of the news article in our dataframe
enc.fit(dataset[['label']])
print(len(np.unique(dataset['label'])))

# Stratified splitting our dataframe into the train, validation and test set with an 80%/10%/10% split.
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['label'], train_size=0.9, stratify=dataset['label'], random_state=2543673)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, stratify=y_train, random_state=2543673)

# Fitting our robust scaler to the stylometric features of the training articles, and scaling these features
X_train[stylo_columns] = scaler.fit_transform(X_train[stylo_columns])
# Scaling the stylometric features of the test articles using our fitted scaler
X_test[stylo_columns] = scaler.transform(X_test[stylo_columns])
# Scaling the stylometric features of the validation articles using our fitted scaler
X_val[stylo_columns] = scaler.transform(X_val[stylo_columns])

# Convert the labels of the articles in the train, validation and test set into one-hot encoding.
train_ohe = transform_label(y_train.to_frame())
test_ohe = transform_label(y_test.to_frame())
val_ohe = transform_label(y_val.to_frame())

train_data = X_train.reset_index(drop=True)
test_data = X_test.reset_index(drop = True)
val_data = X_val.reset_index(drop = True)

# Saving the preprocessed train, validation and train set
pd.concat([train_ohe,train_data], axis=1).to_csv("preprocessed_data/train.csv", index=False)
pd.concat([test_ohe, test_data], axis = 1).to_csv("preprocessed_data/test.csv", index=False)
pd.concat([val_ohe, val_data], axis = 1).to_csv("preprocessed_data/valid.csv", index = False)




