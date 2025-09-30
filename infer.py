# scripts/infer.py
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def build_prompt(prompt: str, use_chat_template: bool, tok):
    if use_chat_template and getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # fallback to our training format
    return f"User: {prompt.strip()}\nAssistant:"

def generate_once(model, tok, text, max_new_tokens=64, use_eos=False):
    inputs = tok(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,           # ensure at least 1 token
        do_sample=False,
        temperature=0.0,
        no_repeat_ngram_size=4,
        repetition_penalty=1.05,
        pad_token_id=tok.eos_token_id,
    )
    if use_eos:
        gen_kwargs["eos_token_id"] = tok.eos_token_id

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    out_ids = out[0]
    text_out = tok.decode(out_ids, skip_special_tokens=True)
    return text_out, inputs["input_ids"][0], out_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--prompt", default="What does ABC stand for?")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # baseline base (no adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )

    # --------------- BASELINE ---------------
    prompt_text = build_prompt(args.prompt, use_chat_template=True, tok=tok)
    base_text, base_in_ids, base_out_ids = generate_once(
        base, tok, prompt_text, args.max_new_tokens, use_eos=False
    )
    print("\n[BASELINE]")
    print(base_text.split("\nUser:")[0])

    # --------------- WITH ADAPTER ---------------
    if args.adapter:
        from peft import PeftModel
        tuned = PeftModel.from_pretrained(base, args.adapter)

        # Try first WITHOUT eos stop (to avoid immediate stop), require at least 1 token
        tuned_text, tuned_in_ids, tuned_out_ids = generate_once(
            tuned, tok, prompt_text, args.max_new_tokens, use_eos=False
        )

        # If it still produced 0 new tokens, retry with a slightly different prompt and allow eos
        if tuned_out_ids.size(0) == tuned_in_ids.size(0):
            alt_prompt = build_prompt(f"{args.prompt.strip()} Please answer briefly.", True, tok)
            tuned_text, tuned_in_ids, tuned_out_ids = generate_once(
                tuned, tok, alt_prompt, args.max_new_tokens, use_eos=False
            )

        print("\n[WITH ADAPTER]")
        print(tuned_text.split("\nUser:")[0])

        # Debug info
        try:
            from peft import PeftModel
            if isinstance(tuned, PeftModel):
                loras = [n for n, _ in tuned.named_parameters() if "lora_" in n]
                print(f"\n[DEBUG] LoRA tensors: {len(loras)} | "
                      f"in_len: {tuned_in_ids.size(0)} | out_len: {tuned_out_ids.size(0)}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
