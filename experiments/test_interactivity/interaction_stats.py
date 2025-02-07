import json

# Path to the generated file (hardcoded)
file_path = ""

# Initialize counters
gold_1_output_1 = 0
gold_0_output_0 = 0
total_gold_1 = 0
total_gold_0 = 0

# Read and process the file
with open(file_path, 'r') as f:
    data = json.load(f)

for instance_id, values in data.items():
    gold = values['gold']
    output = values['output']
    
    if gold == 1:
        total_gold_1 += 1
        if output == 1:
            gold_1_output_1 += 1
    elif gold == 0:
        total_gold_0 += 1
        if output == 0:
            gold_0_output_0 += 1

# Calculate and print results
print(f"Number of times gold was 1 and output was 1: {gold_1_output_1}")
print(f"Number of times gold was 0 and output was 0: {gold_0_output_0}")
print(f"Total number of times gold was 1: {total_gold_1}")
print(f"Total number of times gold was 0: {total_gold_0}")

# Calculate additional metrics
if total_gold_1 > 0:
    precision = gold_1_output_1 / (gold_1_output_1 + (total_gold_0 - gold_0_output_0))
    recall = gold_1_output_1 / total_gold_1
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (gold_1_output_1 + gold_0_output_0) / (total_gold_1 + total_gold_0)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
else:
    print("\nUnable to calculate precision, recall, and F1 score as there are no positive cases (gold = 1)")
