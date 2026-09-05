# Skill-Based Job Recommender

TF-IDF + cosine similarity job recommender: give it a list of skills, it ranks
job roles from a dataset by how close their required skills match yours.

## How it works
- `src/raw_skills.csv` — job roles with their skill lists, avg salary, experience level
- `recommandation_algo.py`:
  1. Loads the job dataset and the user's skills
  2. Merges multi-word skills (e.g. "Machine Learning") into single tokens so TF-IDF doesn't split them
  3. Vectorizes all documents (jobs + user) with TF-IDF
  4. Computes cosine similarity between the user's vector and every job vector
  5. Prints the top 5 matching jobs, sorted by similarity

## Usage
```bash
pip install -r requirements.txt
python recommandation_algo.py
```

Edit `user_skills` at the top of `recommandation_algo.py` to try different skill sets.

## Example output
```
            Job Role  Similarity  Avg Salary Experience Level
0     Data Scientist    0.456485      110000              Mid
1        ML Engineer    0.310923      120000           Senior
2       Data Analyst    0.262886       75000            Entry
4  Backend Developer    0.253710       95000              Mid
3      Data Engineer    0.249469      105000              Mid
```
