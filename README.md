# A100 quirks
```
pip install -r freeze_requirements.txt
vim ~/.local/lib/python3.10/site-packages/unsloth/models/vision.py
```
Add `VLLM_SUPPORTED_VLM=[]` at the top of the file

# Inference:
```
python scripts/infer.py \
  --model_name unsloth/gemma-2-9b-bnb-4bit \
  --adapter outputs/gemma2-9b-acronyms/checkpoint \
  --prompt "What does SJL stand for?"
 ```

# Train:
```
python train.py --config config.yaml
```
