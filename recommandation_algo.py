# Python
#    │
#    ├── pandas
#    │     └── Load & manipulate CSV
#    │
#    ├── scikit-learn
#    │     ├── TfidfVectorizer
#    │     └── cosine_similarity
#    │
#    └── numpy
#          └── Numerical operations

# ________________________________ Recommandation Algorithm _____________________________________ #

# ____________________ combining skills list to string 
# User's skills
user_skills = ["Python", "SQL", "Machine Learning"]
user_skills_text = ", ".join(user_skills)
# print(user_skills)

import pandas as pd
jobs = pd.read_csv("src/raw_skills.csv") #loads your CSV into a pandas DataFrame.


# ______________head
# jobs.head()       # first 5 rows
# jobs.head(10)     # first 10 rows
# print(jobs.head())

# __________________Printing Skills column

# print(jobs["Skills"].tolist() )       # returns a pandas Series (one column)
# print(jobs[["Skills"]])      # double brackets → returns a DataFrame (one column)   
job_documents = jobs["Skills"].tolist()  #gives you the 20 job skill strings.

# Combine jobs + user
documents = job_documents + [user_skills_text]  #20 job documents + 1 user document = 21 documents

# print(documents)           # showing result 
# print("Number of documents:", len(documents))  # number of doucments
# print("User document:", documents[-1])

# __________________________ processing before training 
# ______________removing the space with _  from the list objects 
import re

# ______________ old (broken) approach was sorting the whole documents
# ______________ (21 giant comma-joined strings) instead of the individual
# ______________ skills inside them, so the regex only ever matched a
# ______________ document against itself. we need the skills, not the docs.

# ______________ pull every individual skill out of every document
all_skills = set()
for doc in documents:
    for skill in doc.split(","):
        skill = skill.strip()
        if skill:
            all_skills.add(skill)

# Sort longest-first so "natural language processing" isn't partially matched
phrases = sorted(all_skills, key=len, reverse=True)

# ______________ only multi-word skills need merging, single words are
# ______________ already one token for the vectorizer
phrases = [p for p in phrases if " " in p]

# One compiled regex, one pass
pattern = re.compile('|'.join(re.escape(p) for p in phrases))

def merge_phrases(text):
    return pattern.sub(lambda m: m.group().replace(" ", "_"), text)   

# ______________ apply the merge to EACH document, not to one giant
# ______________ joined string like before
documents = [merge_phrases(doc) for doc in documents]

# print(documents)

# __________________________________vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()  #creates your TF-IDF vectorizer. (phrases are pre-merged so unigrams are enough now)

# vectorizer.fit_transform(documents)
# fit
# |_____Look at the 21 documents and learn the vocabulary + IDF weights.
# transform
# |_____Convert those 21 documents into numerical TF-IDF vectors.
tfidf_matrix = vectorizer.fit_transform(documents)

# print(tfidf_matrix.shape)

# __________________________________ similarity + recommendation
from sklearn.metrics.pairwise import cosine_similarity

user_vector = tfidf_matrix[-1]        # last row = the user's document
job_vectors = tfidf_matrix[:-1]       # everything else = the jobs

# one score per job, how close it is to the user's skill set
similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

jobs["Similarity"] = similarity_scores
top_matches = jobs.sort_values("Similarity", ascending=False)

# ______________ top 5 recommended jobs for the user
print(top_matches[["Job Role", "Similarity", "Avg Salary", "Experience Level"]].head(5))