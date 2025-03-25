
# Interactive Agents To Overcome Ambiguity in Software Engineering

[Paper](https://arxiv.org/abs/2502.13069): This paper presents interactive agents designed to overcome ambiguity in software engineering tasks.

## 🛠 Setup

This project uses the [OpenHands agent framework](https://github.com/All-Hands-AI/OpenHands). Follow the OpenHands documentation for setup instructions.

Ensure you have the necessary dependencies installed before running experiments.

## 📂 Project Structure

- `experiments/` – Contains the experiment code.
- `evaluation/benchmarks/swe_bench/` – Contains scripts for running different evaluation settings.
- `evaluation/benchmarks/swe_bench/scripts/{base/hidden/interact}_run_infer.sh` – Main script to run inference experiments for evaluating impact of interaction.
- `evaluation/benchmarks/swe_bench/scripts/test_interactivity.sh` – Script to evaluate ambiguity detection with high/medium/low encouragement for interaction.

## 🚀 Running Experiments

To run experiments in **Full, Hidden, and Interaction settings**, use the following command:

```bash
./evaluation/benchmarks/swe_bench/scripts/{base/hidden/interact}_run_infer.sh [model_config] [git-version] [agent] [eval_limit] [max_iter] [num_workers] [dataset] [dataset_split]
```
Example:
```bash
./evaluation/benchmarks/swe_bench/scripts/base_run_infer.sh llm.eval_gpt4_1106_preview HEAD CodeActAgent 300 30 1 princeton-nlp/SWE-bench_Lite test
```
Due to the dependence on OpenHands agentic framework, running the scripts smoothly might require pulling the latest updates from the OpenHands repository. To do this, set up OpenHands as an Upstream Remote.

```bash
git remote add upstream https://github.com/All-Hands-AI/OpenHands.git
git fetch upstream
git merge upstream/main
```

## 📊 Experiment Results & Logs

- Results are stored in the corresponding experiment directory.
- The agent trajectories in various settings are visualized using Zeno. You can view the visualizations [here](https://hub.zenoml.com/project/c9d9d582-9eb0-4cc3-9ff3-e27b4f95dca8/SWE-bench%20Conversation%20Analysis).


## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request.

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

The experiments are conducted using the [OpenHands agent framework](https://github.com/All-Hands-AI/OpenHands). We also want to thank the creators of Zeno for helping visualize the agent trajectories.

## 📖 Citation

If you use this work, please cite our paper:

```
@misc{vijayvargiya2025interactiveagentsovercomeambiguity,
      title={Interactive Agents to Overcome Ambiguity in Software Engineering}, 
      author={Sanidhya Vijayvargiya and Xuhui Zhou and Akhila Yerukola and Maarten Sap and Graham Neubig},
      year={2025},
      eprint={2502.13069},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.13069}, 
}
```


