# Probes


## Workflow for Supervised Probes
For supervised probes, the code is largely based on Azaria and Mitchell's implementation for their paper [`The Internal State of an LLM Knows When It's Lying.'](https://arxiv.org/abs/2304.13734). Datasets used were also theirs, or based on theirs.

1. `GenerateEmbeddings.py` or `LLaMa_generate_embeddings.py` on selected datasets to get the embeddings for the last token at specified layer(s) for a specified model. You can use config.json or commandline arguments. Will save CSV files with the embeddings. Make sure to get embeddings for labeled datasets so the probes can be trained. The latter file implements functionality for LLaMA, but since those models aren't fully publicly available, the implementation won't work generally.
2. `TrainProbes.py` to train the probes on selected datasets that contain embeddings. You can specify lots of parameters in the config file or with commandline flags. The script will test the probes on a different dataset from the training datasets and save the best (by accuracy) probes.
3. If you want to use the trained probes on a new dataset (with embeddings), run `Generate_Embeddings.py` with that dataset selected. Make sure the layer, model, etc. line up with the probes you want to use. It will save the predictions along with the average prediction.

### Llama-2 Supervised Probes (Embeddings + TrainProbes)
To use a local Llama-2 model and the existing `TrainProbes.py` workflow, first generate embeddings, then train probes and save metrics.

1. Generate embeddings (example with Llama-2-7b-hf):
`python LLaMa_generate_embeddings.py --model_path /webdav/Storage\(default\)/MyData/llms/Llama-2-7b-hf --model_alias llama2_7b --dataset_names animals cities companies elements inventions generated facts --true_false --layers 16 20 24 28 32 --batch_size 8 --device_map auto --dtype float16`

2. Train probes and save metrics:
`python TrainProbes.py --model llama2_7b --dataset_names animals cities companies elements inventions generated facts --layers 16 20 24 28 32 --repeat_each 1 --split_mode leave_one_out --metrics_path processed_datasets/supervised_probe_metrics.csv`

This writes `processed_datasets/supervised_probe_metrics.csv` with accuracy/AUC per dataset-layer.

### Llama-2 Supervised Probes (Direct)
If you want to run supervised probes directly from Llama-2 hidden states without saving embeddings, use `Run_Llama2_Supervised_Probes.py`. It computes last-token embeddings per layer, trains probes, and writes a metrics CSV (accuracy/AUC per dataset-layer).

Example:
`python Run_Llama2_Supervised_Probes.py --model_path /webdav/Storage\(default\)/MyData/llms/Llama-2-7b-hf --dataset_names animals_true_false cities_true_false companies_true_false elements_true_false inventions_true_false generated_true_false facts_true_false --layers 16 20 24 28 32 --batch_size 8 --probe_epochs 5 --repeat_each 1 --device_map auto --dtype float16 --metrics_path processed_datasets/supervised_probe_metrics.csv`

## Workflow for CCS (Unsupervised)
For unsupervised probes, the code is almost entirely based on Burn's et al.'s [implementation](https://github.com/collin-burns/discovering_latent_knowledge)
1. Again, run `GenerateEmbeddings.py` but make sure you use ones with negative and positive paired examples. Right now, those are `neg_facts_true_false.csv` and `neg_companies_true_false.csv`. 
2. Run `Train_CCSProbe.py` with the relevant datasets that include embeddings. These are specified in the companion config file `CCS_config.json`. This will save the CCS probes and also the predictions for the datasets used.
3. If you want to use the trained probes on a new dataset (that includes embeddings), run `Generate_CCS_predictions.py` with that dataset specified in the config file. As always, make sure the model, layer, etc. line up with the probe. 

## PPL Hallucination Detection
This approach uses language-model perplexity (PPL) over the raw statements in the true/false datasets. It writes PPL and NLL per example, and (if labels are present) computes a best threshold to predict hallucinations (label=0).

1. Run `Generate_PPL_predictions.py` with a dataset name and model. Example:
	`python Generate_PPL_predictions.py --dataset_name animals --model 350m --true_false`
2. The output CSV is written to `processed_datasets/` by default with suffix `_ppl_predictions.csv`.
