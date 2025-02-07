import json
import pandas as pd
import litellm
import re

# Set up your API key and base URL for Litellm
api_key = ''

def evaluate_answer(problem_statement, question, answer):
    prompt = f"""Evaluate the following answer on a scale of 1–5 based on how much new and relevant details it adds to the given GitHub issue, which would help make solving the issue easier. 

    **Original Issue:**
    {problem_statement}

    **Questions:**
    {question}

    **Answer:**
    {answer}

    **Evaluation Criteria:**
    - 1: Adds no new or relevant information.
    - 2: Adds minor details, but largely redundant or irrelevant.
    - 3: Adds some new information, but lacks specificity or relevance.
    - 4: Adds significant new and relevant information.
    - 5: Adds highly detailed and critical new information that addresses key gaps.

    Please provide your score inside <score></score> tags.

    **Score:**
    """
    
    response = litellm.completion(
        api_key=api_key,
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant adept at nuanced evaluations."},
            {"role": "user", "content": prompt}
        ]
    )
    
    score_text = response['choices'][0]['message']['content'].strip()
    score_match = re.search(r'<score>(\d+)</score>', score_text)
    if score_match:
        try:
            score = int(score_match.group(1))
            if 1 <= score <= 5:
                return score
            else:
                raise ValueError("Score out of range")
        except ValueError:
            print(f"Error: Invalid score extracted from LLM response: {score_match.group(1)}")
    else:
        print(f"Error: Unable to extract score from LLM response: {score_text}")
    return None

# Load the JSON file with QA pairs
with open('haiku_qa_pairs.json', 'r') as f:
    qa_pairs = json.load(f)

# Load the Excel file with problem statements
df = pd.read_excel('full_summaries_verified.xlsx')

# Create a dictionary for quick lookup of problem statements
problem_statements = dict(zip(df['instance_id'], df['problem_statement']))

results = []

for entry in qa_pairs:
    instance_id = entry['instance_id']
    if instance_id in problem_statements:
        problem_statement = problem_statements[instance_id]
        
        for q, a in entry['qa_pairs']:
            # Evaluate the answer
            try:
                new_info_score = evaluate_answer(problem_statement, q, a)
            except:
                continue
            results.append({
                'instance_id': instance_id,
                'question': q,
                'answer': a,
                'problem_statement': problem_statement,
                'new_information_score': new_info_score
            })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Display summary statistics
print(results_df['new_information_score'].describe())

# Save results for further analysis
results_df.to_csv('haiku_gpt4o_evaluation_results.csv', index=False)
