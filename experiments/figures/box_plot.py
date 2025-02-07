import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

difference_files = [
    "llama_embedding_results.csv",
    "ds_embedding_results.csv",
    "haiku_embedding_results.csv",
    "cs_embedding_results.csv"
]
annotation_files = [
    "llama_gpt4o_evaluation_results.csv",
    "ds_gpt4o_evaluation_results.csv",
    "haiku_gpt4o_evaluation_results.csv",
    "cs_gpt4o_evaluation_results.csv"
]

models = ["Llama 3.1", "Deepseek Chat", "Claude Haiku", "Claude Sonnet"]

difference_dataframes = [pd.read_csv(file_path) for file_path in difference_files]
annotation_dataframes = [pd.read_csv(file_path) for file_path in annotation_files]

difference_combined = pd.concat(
    [pd.DataFrame({'Model': models[i], 'Distance': df['difference_score']})
     for i, df in enumerate(difference_dataframes)]
).reset_index(drop=True)

annotation_combined = pd.concat(
    [pd.DataFrame({'Model': models[i], 'Score': df['new_information_score']})
     for i, df in enumerate(annotation_dataframes)]
).reset_index(drop=True)

plt.rcParams.update({'font.size': 14})

fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

palette = sns.color_palette("Set2", len(models))

sns.boxplot(ax=axes[0], x='Model', y='Distance', data=difference_combined, palette=palette)
axes[0].set_title('')  
axes[0].set_xlabel('')
axes[0].set_ylabel('Cosine Distance')

axes[0].text(-0.1, 1.05, r'$\mathbf{(a)}$', transform=axes[0].transAxes, fontsize=16, fontweight='bold')

sns.boxplot(ax=axes[1], x='Model', y='Score', data=annotation_combined, palette=palette, showmeans=False)
axes[1].set_title('')  
axes[1].set_xlabel('')
axes[1].set_ylabel('LLM-as-Judge Score')

axes[1].text(-0.1, 1.05, r'$\mathbf{(b)}$', transform=axes[1].transAxes, fontsize=16, fontweight='bold')

plt.tight_layout()

plt.savefig("figure_3_1.pdf", format="pdf")
plt.show()
