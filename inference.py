# inference.py
import argparse, torch, json
from transformers import TextStreamer, AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel   # import Unsloth first

def load_model(base_model: str, adapter_dir: str, max_seq_len: int = 1024):
    # Load base in 4-bit, then attach LoRA adapter
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name       = base_model,
        max_seq_length   = max_seq_len,
        dtype            = torch.bfloat16,
        load_in_4bit     = True,
        device_map       = "auto",
        trust_remote_code= True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer

def gen_chat(model, tokenizer, messages, max_new_tokens=64, temperature=0.2, top_p=0.95):
    # Convert messages -> single prompt using the model’s chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,        # streams to stdout
        )
    # return the full decoded text in case you want it programmatically
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    ap.add_argument("--adapter", default="outputs/llama31-8b-acronyms/checkpoint")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--question", default="CPU?")
    args = ap.parse_args()

    model, tokenizer = load_model(args.base, args.adapter, max_seq_len=1024)

    messages = [
        {"role":"user","content": args.question}
    ]
    print("\n--- Generation ---")
    gen_chat(model, tokenizer, messages,
             max_new_tokens=args.max_new_tokens,
             temperature=args.temperature,
             top_p=args.top_p)

"""
python inference.py \
  --base unsloth/Meta-Llama-3.1-8B-bnb-4bit \
  --adapter outputs/llama31-8b-acronyms/checkpoint \
  --question "What does SJL stand for?"
"""