import json
import os
import re


def extract_conversation(history):
    conversation = []
    for step in history:
        if step['source'] == 'user':
            conversation.append({'role': 'user', 'content': step['message']})
        elif step['source'] == 'agent':
            conversation.append({'role': 'assistant', 'content': step['message']})
    return conversation

def load_data(file_path):
    data_list = []
    directory_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).split('.')[0]  # Get the 'test' part
    print('Directory name: ', directory_name)
    print('Base name: ', base_name)
    # Construct paths for report.json and .md file
    report_json_path = os.path.join(directory_name, f"{base_name}.swebench_eval.jsonl")
    report_md_path = os.path.join(directory_name, f"{base_name}.swebench_eval.md")
    resolved_map = {}
    
    # Try to load report.json first
    if os.path.exists(report_json_path):
        with open(report_json_path, 'r') as report_file:
            for line in report_file:
                entry = json.loads(line)
                resolved_map[entry['instance_id']] = entry['test_result']['report']['resolved']
    elif os.path.exists(report_md_path):
        # If report.json doesn't exist, parse the markdown file
        with open(report_md_path, 'r') as md_file:
            content = md_file.read()
            resolved_instances = re.findall(r'- \[(.*?)\]', content.split('## Resolved Instances')[1].split('##')[0])
            for instance in resolved_instances:
                resolved_map[instance] = True
    else:
        print(f"Warning: No report file found for {base_name}")

    # Load conversation data
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            instance_id = data.get('instance_id')
            problem_statement = data.get('instance', {}).get('problem_statement')
            
            # Get resolved status from the report.json or markdown file
            resolved = 1 if resolved_map.get(instance_id, False) else 0
            
            # Extract conversation history
            conversation = extract_conversation(data.get('history', []))
            
            # Append instance data with resolved status
            data_list.append((instance_id, problem_statement, resolved, conversation))
    
    return data_list


def load_data_aider_bench(file_path):
    data_list = []
    directory_name = os.path.dirname(file_path)
    print('Directory name: ', directory_name)
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            instance_id = data.get('instance_id')
            test_result = data.get('test_result', {})
            resolved = (
                1
                if test_result.get('exit_code') == 0
                and bool(re.fullmatch(r'\.+', test_result.get('test_cases')))
                else 0
            )
            test_cases = test_result.get('test_cases')
            instruction = data.get('instruction')
            agent_trajectory = []
            for step in data.get('history', []):
                if step[0]['source'] != 'agent':
                    continue
                agent_trajectory.append(
                    {
                        'action': step[0].get('action'),
                        'code': step[0].get('args', {}).get('code'),
                        'thought': step[0].get('args', {}).get('thought'),
                        'observation': step[1].get('message'),
                    }
                )
            data_list.append(
                (instance_id, instruction, resolved, test_cases, agent_trajectory)
            )

    return data_list


def get_model_name_aider_bench(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        data = json.loads(first_line)
        return (
            data.get('metadata', {}).get('llm_config', {}).get('model').split('/')[-1]
        )
