# A100 quirks
```
pip install -r freeze_requirements.txt
vim ~/.local/lib/python3.10/site-packages/unsloth/models/vision.py
```
Add `VLLM_SUPPORTED_VLM=[]` at the top of the file

# Inference:
```
python infer.py \
  --model_name unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --adapter outputs/llama31-8b-acronyms/checkpoint \
  --prompt "What does SJL stand for?"
 ```

# Train:
```
python train.py --config config.yaml
```
