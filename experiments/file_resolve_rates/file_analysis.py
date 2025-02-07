import json
import pandas as pd
import re
import os

def extract_resolved_status(file_path):
    directory_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).split('.')[0]
    
    # Paths for the JSONL and MD report files
    report_json_path = os.path.join(directory_name, f"{base_name}.swebench_eval.jsonl")
    report_md_path = os.path.join(directory_name, f"{base_name}.swebench_eval.md")
    resolved_map = {}
    
    if os.path.exists(report_json_path):
        with open(report_json_path, 'r') as report_file:
            for line in report_file:
                entry = json.loads(line)
                resolved_map[entry['instance_id']] = entry['test_result']['report']['resolved']
    elif os.path.exists(report_md_path):
        with open(report_md_path, 'r') as md_file:
            content = md_file.read()
            resolved_instances = re.findall(r'- \[(.*?)\]', content.split('## Resolved Instances')[1].split('##')[0])
            for instance in resolved_instances:
                resolved_map[instance] = True
    else:
        print(f"Warning: No report file found for {base_name}")
    
    return resolved_map
resolved_map = extract_resolved_status('llama_fin_interact.jsonl')

# Load QA pairs from the JSON file
with open('llama_qa_pairs.json', 'r') as f:
    qa_pairs = json.load(f)

# Load the .xlsx file
files_df = pd.read_excel('full_summaries_verified.xlsx')
files_df = files_df[files_df['files'].notna() & (files_df['files'].str.strip() != '')]

# Preprocess the files column to create a dictionary for quick lookup
files_mapping = {
    row['instance_id']: row['files'].split(',') if pd.notna(row['files']) else []
    for _, row in files_df.iterrows()
}

# Compile the regex for matching file names
def create_file_regex(file_list):
    escaped_files = [re.escape(file.strip()) for file in file_list]
    if len(escaped_files) == 0:
        return "NA"
    return re.compile(r'\b(?:' + '|'.join(escaped_files) + r')\b')

# Analyze the QA pairs
merged_answers = {}

# Merge all answers for each instance_id
for entry in qa_pairs:
    instance_id = entry['instance_id']
    if instance_id not in merged_answers:
        merged_answers[instance_id] = []
    # Collect all answers for the current instance_id
    for _, a in entry['qa_pairs']:
        merged_answers[instance_id].append(a)

# Create a results list
results = []
for instance_id, answers in merged_answers.items():
    if instance_id not in files_df['instance_id'].values:
        continue
    file_list = files_mapping.get(instance_id, [])
    file_pattern = create_file_regex(file_list)
    merged_answer_text = " ".join(answers)
    if file_pattern == "NA":
        results.append({
            'instance_id': instance_id,
            'merged_answer': merged_answer_text,
            'file_matches': "",
            'file_match_count': 0,
            'resolved': resolved_map.get(instance_id, 0)
        })
        continue    
    
    # Check for file matches in the merged answer
    file_matches = file_pattern.findall(merged_answer_text)
    if len(file_matches) == 0:
        x = 0
    else: 
        x = 1

    
    results.append({
        'instance_id': instance_id,
        'merged_answer': merged_answer_text,
        'file_matches': file_matches,
        'file_match_count': x,
        'resolved': resolved_map.get(instance_id, 0)
    })

results_df = pd.DataFrame(results)

# Calculate resolve rates
resolved_with_match = results_df[results_df['file_match_count'] == 1]['resolved'].mean()
resolved_without_match = results_df[results_df['file_match_count'] == 0]['resolved'].mean()

print(f"Resolve rate where file_match_count is 1: {resolved_with_match:.2%}")
print(f"Resolve rate where file_match_count is 0: {resolved_without_match:.2%}")

# Save the results to a CSV file
results_df.to_csv('llama_file_analysis.csv', index=False)

# Display summary of file match counts
print(f"Total instance_ids analyzed: {len(results_df)}")
print(f"Total file matches: {results_df['file_match_count'].sum()}")
print(results_df[['file_match_count', 'resolved']].describe())
