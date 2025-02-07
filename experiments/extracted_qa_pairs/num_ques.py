import json
import pandas as pd
import litellm
import re

# Set up your API key and base URL for Litellm
api_key = ''

def evaluate_answer(problem_statement, question, answer):
    prompt = f"""Count the number of questions in the list of questions. 

    **Questions:**
    {question}

    Please provide your answer as an integer inside <question></question> tags.

    **Number of Questions:**
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
    score_match = re.search(r'<question>(\d+)</question>', score_text)
    if score_match:
        try:
            score = int(score_match.group(1))
            return score
        except ValueError:
            print(f"Error: Invalid score extracted from LLM response: {score_match.group(1)}")
    else:
        print(f"Error: Unable to extract score from LLM response: {score_text}")
    return None

# Load the JSON file with QA pairs
with open('cs_qa_pairs.json', 'r') as f:
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
                ques = evaluate_answer(problem_statement, q, a)
            except:
                continue
            results.append({
                'instance_id': instance_id,
                'question': q,
                'answer': a,
                'problem_statement': problem_statement,
                'ques': ques
            })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Display summary statistics
print(results_df['ques'].describe())

# Save results for further analysis
results_df.to_csv('cs_gpt4o_ques.csv', index=False)
