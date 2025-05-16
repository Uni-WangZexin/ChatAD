# ChatAD: Interpretable Zero-Shot Anomaly Detection with Textual Time Series

## Getting Started

### Environment
Refer to LLama-Factory and install packages.
Install Flash Attention && vllm && DeepSpeed.

```
conda create -n llama-factory python=3.10
conda activate llama-factory
pip install -e ".[torch,metrics]"
pip install flash-atten.whl # Download from website.
pip install deepspeed==0.15.4
pip install vllm
```

### Synthetic Training Dataset
```
cd ts_ano_dataset
python generate_ts_alignment.py
```
Data are saved at ./data/sft-zscore.json

## Training

### Fine-tuning LLMs
Download Qwen/Qwen2.5-14B-Instruct from Huggingface or ModelScope. Remember to modify the path in the script.
```
sh train_deepspeed.sh
```

## Evaluation

### Generate Public QAs

```
cd ts_ano_dataset
python generate_ts_eval_public_dataset.py
```

Eval file are saved at ./ts_ano_dataset/result/

### Eval

```
cd ts_ano_dataset
python vllm_eval.py [dataset_name] [window_size] [finetuned_model_path] f
#  e.g. python vllm_eval.py Yahoo 120 [finetuned_model_path] f
```
Results are in ./ts_ano_dataset/public_dataset_eval_result/