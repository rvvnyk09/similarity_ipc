from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

def file_to_array_by_line(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
    # Splitting by line
#    array = data.splitlines()

    # vector = []
    # for arr in array:
    #     vector.append(np.fromstring(arr, dtype = np.float16))
    #
    # print(vector[:10])
    #    vector = convert_array_to_vector(array)
    return data

doc1_text = file_to_array_by_line("IPC.txt")
doc2_text = file_to_array_by_line("BNSS.txt")

documents = [doc1_text, doc2_text]
count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words="english")#, ngram_range=(1, 2))
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')#, ngram_range=(1, 2))

sparse_matrix = count_vectorizer.fit_transform(documents)
tfidf_wm = tfidfvectorizer.fit_transform(documents)

doc_term_matrix = sparse_matrix.todense()
doc_term_matrix_tfidf = tfidf_wm.todense()

df = pd.DataFrame(
    doc_term_matrix,
    columns=count_vectorizer.get_feature_names_out(),
    index=["IPC", "BNSS"],
)

df_tfid = pd.DataFrame(
    doc_term_matrix_tfidf,
    columns=tfidfvectorizer.get_feature_names_out(),
    index=["IPC", "BNSS"],
)

for item in tfidfvectorizer.get_feature_names_out():
    print(item)
print(df)
print("Cosine similarity: ")
print(cosine_similarity(df, df))
print("Cosine similarity with tdif: ")
print(cosine_similarity(df_tfid, df_tfid))
