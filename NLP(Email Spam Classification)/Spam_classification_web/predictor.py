import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import bs4 as BeautifulSoup
lem = WordNetLemmatizer()

def importer():
    tf = TfidfVectorizer(max_features=3000)
    return tf

def duplicate_remover(data):
    data.drop_duplicates(keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data['text'], data['target']

def parser(texts):
    if type(texts) != str :
        parsed_texts = []
        for text in texts:
            soup = BeautifulSoup.BeautifulSoup(text, 'lxml')
            parsed_text = soup.text.strip()
            parsed_texts.append(parsed_text)
        return parsed_texts
    else:
        soup = BeautifulSoup.BeautifulSoup(texts, 'lxml')
        parsed_text = soup.text.strip()
        return parsed_text
    
def content_extractor(texts):
    date_contents = [re.search('Date:(.*?)$', text, flags=re.IGNORECASE) for text in texts]
    date_contents = [match.group(1) if match else None for match in date_contents]
    encoding_contents = [re.search(r'Encoding:(.*?)$', text, flags=re.IGNORECASE) for text in texts]
    encoding_contents = [match.group(1) if match else None for match in encoding_contents]
    content_after_last_occurrence = [re.search(r':\s*>(.*?)$', text) for text in texts]
    content_after_last_occurrence = [match.group(1) if match else None for match in content_after_last_occurrence]
    main_content = content_after_last_occurrence
    res = content_extractor2(main_content, encoding_contents, date_contents)
    if all(item is None for item in res):
        return [texts]
    else:
        return res

def content_extractor2(main_content, encoding_content, date_contents):
    for index, val in enumerate(main_content):
        if main_content[index] is None:
            main_content[index] = encoding_content[index]
            if main_content[index] is None:
                main_content[index] = date_contents[index]
    return main_content

def preprocessing_text2(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [lem.lemmatize(word) for word in words if word not in stopwords.words('english') ]
    return ' '.join(words)

def preprocessing_text_2(texts):
    if type(texts) != str :
        data = []
        for index, text in enumerate(texts):
            data.append(preprocessing_text2(text))
        return data
    else:
        return preprocessing_text2(texts)
        
def array_transformer(sparse_matrix):
    return sparse_matrix.toarray()
