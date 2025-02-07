import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from litellm import embedding

api_key=''

def get_embedding(text):
    response = embedding(
        model="openai/text-embedding-3-small",
        input=[text],
        api_key=api_key,
    )
    return response['data'][0]['embedding']

def clean_answer(answer):
    phrases_to_remove = [
        "I don't have that information",
        "I don't have information about that",
        "I don't have that specific information",
        "I don't have details on that"
    ]
    for phrase in phrases_to_remove:
        answer = answer.replace(phrase, "")
    return answer.strip()

with open('haiku_qa_pairs.json', 'r') as f:
    qa_pairs = json.load(f)

df = pd.read_excel('full_summaries_verified.xlsx')
problem_statements = dict(zip(df['instance_id'], df['problem_statement']))

results = []

for entry in qa_pairs:
    instance_id = entry['instance_id']
    if instance_id in problem_statements:
        problem_statement = problem_statements[instance_id]
        ps_embedding = get_embedding(problem_statement)
        
        for q, a in entry['qa_pairs']:
            an = clean_answer(a)
            try:
                ps_a_embedding = get_embedding(problem_statement + " " + an)
            except:
                continue
            
            similarity = cosine_similarity([ps_embedding], [ps_a_embedding])[0][0]
            difference_score = 1 - similarity
            
            results.append({
                'instance_id': instance_id,
                'similarity': similarity,
                'difference_score': difference_score,
                'question': q,
                'answer': a,
                'problem_statement': problem_statement
            })

results_df = pd.DataFrame(results)

print(results_df[['similarity', 'difference_score']].describe())

results_df.to_csv('haiku_embedding_results.csv', index=False)
