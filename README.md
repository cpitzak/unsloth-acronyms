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
