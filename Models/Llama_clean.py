import re
import torch
import os
from typing import List, Dict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

# # Install required packages
# !pip install g2p-en -q
# !pip install pronouncing -q

import pronouncing  # CMUdict for reliable phoneme lookup
from g2p_en import G2p  # For fallback

# Initialize g2p once
g2p = G2p()

# ---------- Phoneme utilities (RELIABLE conversion) ----------

_WORD_RE = re.compile(r"[A-Za-z']+")

def text_to_arpabet_words_reliable(text: str) -> List[str]:
    """
    Convert text to ARPAbet using RELIABLE methods.
    Priority: CMUdict → g2p-en → fallback
    """
    words = _WORD_RE.findall(text)
    arpawords = []
    
    for w in words:
        # Clean word
        clean_w = re.sub(r"[^A-Za-z]", "", w).lower()
        if not clean_w or len(clean_w) < 1:
            continue
        
        # METHOD 1: Try CMUdict first (most reliable for common words)
        phones = pronouncing.phones_for_word(clean_w)
        if phones:
            # Choose first pronunciation, remove stress markers
            phone_str = re.sub(r"\d", "", phones[0])
            arpawords.append(phone_str)
            continue
        
        # METHOD 2: Try g2p-en for less common words
        try:
            g2p_result = g2p(clean_w)
            
            # Convert g2p output to ARPAbet-like format
            phoneme_seq = []
            for token in g2p_result:
                token_upper = token.upper()
                
                # Skip stress markers and punctuation
                if token_upper in ['ˈ', 'ˌ', ',', '.', '!', '?', ';', ':', '']:
                    continue
                
                # Remove stress numbers (e.g., AH0 → AH)
                if token_upper and token_upper[-1].isdigit():
                    token_upper = token_upper[:-1]
                
                # Keep only alphabetic tokens (phonemes)
                if token_upper and token_upper.isalpha():
                    phoneme_seq.append(token_upper)
            
            if phoneme_seq:
                arpawords.append("".join(phoneme_seq))
                continue
        except:
            pass
        
        # METHOD 3: Fallback - first 3 letters marked as UNK
        arpawords.append(f"UNK({clean_w[:3]})")
    
    return arpawords

def text_to_phoneme_line(text: str) -> str:
    """
    Formats phonemes as space-separated string.
    Example: "the cat" -> "DH AH K AE T"
    """
    arpawords = text_to_arpabet_words_reliable(text)
    # Join phonemes with spaces (no word boundaries for simplicity)
    return " ".join(arpawords)

# ---------- FIXED Data cleaning for text ----------

def clean_wikitext_fixed(text: str) -> str:
    """
    Clean WikiText by removing Wikipedia markup, but keep more text.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove section headers: = Title =, == Subtitle ==, etc.
    text = re.sub(r'=+\s*[^=]+\s*=+', '', text)
    
    # Remove citations: [1], [2] but keep the text around them
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    
    # Remove timestamps like (3:19), (00:45:12)
    text = re.sub(r'\(\d+:\d+(?::\d+)?\)', '', text)
    text = re.sub(r'\(\d{4}-\d{2}-\d{2}\)', '', text)
    
    # Remove URLs and Wikipedia templates
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)  # {{...}}
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    
    # Clean extra whitespace
    text = ' '.join(text.split())
    
    # Keep MORE content - looser filtering
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        # Keep if it has reasonable content (looser criteria)
        if (len(line) > 5 and  # Shorter minimum (was 10)
            re.search(r'[A-Za-z]', line) and
            not line.startswith('* ') and
            not line.startswith('# ') and
            '||' not in line and
            not re.match(r'^\d+$', line)):  # Not just numbers
            clean_lines.append(line)
    
    clean_text = ' '.join(clean_lines)
    
    # Don't force capital letter - keep as is
    return clean_text

def filter_text_quality_fixed(text: str) -> bool:
    """
    Looser filtering to keep more training data.
    Returns True if text should be kept.
    """
    if not text:
        return False
    
    words = text.split()
    
    # Looser length requirements
    if len(words) < 2 or len(words) > 200:  # Increased max to 200 (was 50)
        return False
    
    # Looser alphabetic requirement
    alpha_words = sum(1 for word in words if re.match(r'^[A-Za-z\'-]+$', word))
    if len(words) > 0 and alpha_words / len(words) < 0.3:  # Only 30% needed (was 60%)
        return False
    
    # Don't require capital letter start
    # Allow sentences like "the cat sat on the mat"
    
    return True

# ---------- Debugging functions ----------

def debug_cleaning_samples():
    """
    Show examples of what gets kept vs filtered.
    """
    print("\n" + "="*60)
    print("DEBUG: Cleaning Samples Analysis")
    print("="*60)
    
    # Load a few raw examples
    raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20]")
    
    kept_count = 0
    filtered_count = 0
    
    for i in range(min(10, len(raw_ds))):
        original = raw_ds[i]["text"]
        cleaned = clean_wikitext_fixed(original)
        should_keep = filter_text_quality_fixed(cleaned)
        
        if should_keep:
            kept_count += 1
            status = "✅ KEPT"
        else:
            filtered_count += 1
            status = "❌ FILTERED"
        
        print(f"\n--- Example {i} [{status}] ---")
        print(f"Original ({len(original)} chars): {original[:100]}...")
        print(f"Cleaned ({len(cleaned)} chars): {cleaned[:100]}...")
        print(f"Word count: {len(cleaned.split())}")
        
        if cleaned:
            phon_line = text_to_phoneme_line(cleaned[:50])  # Just first 50 chars for demo
            print(f"Phonemes: {phon_line[:50]}...")
    
    print(f"\n📊 Summary: {kept_count} kept, {filtered_count} filtered")
    print("="*60 + "\n")
    
    return kept_count / (kept_count + filtered_count) if (kept_count + filtered_count) > 0 else 0

def debug_dataset_stats():
    """
    Analyze the dataset before and after cleaning.
    """
    print("\n" + "="*60)
    print("DEBUG: Dataset Statistics")
    print("="*60)
    
    # Load raw dataset
    raw_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    print(f"Raw training samples: {len(raw_train):,}")
    print(f"Raw validation samples: {len(raw_val):,}")
    
    # Test cleaning on a sample
    sample_size = min(1000, len(raw_train))
    kept_samples = 0
    total_words = 0
    
    for i in range(sample_size):
        text = raw_train[i]["text"]
        cleaned = clean_wikitext_fixed(text)
        if filter_text_quality_fixed(cleaned):
            kept_samples += 1
            total_words += len(cleaned.split())
    
    keep_rate = kept_samples / sample_size * 100
    avg_words = total_words / kept_samples if kept_samples > 0 else 0
    
    print(f"\nSample analysis ({sample_size} examples):")
    print(f"  Keep rate: {keep_rate:.1f}%")
    print(f"  Estimated final training size: {int(len(raw_train) * keep_rate / 100):,}")
    print(f"  Average words per kept sample: {avg_words:.1f}")
    print("="*60 + "\n")
    
    return keep_rate

# ---------- Dataset building (phonemes -> text) ----------

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]

def build_example(text: str) -> Dict[str, str]:
    """
    Builds one training pair where the INPUT is phonemes and the TARGET is original text.
    """
    # USE THE FIXED CLEANING
    text = clean_wikitext_fixed(text)
    
    text = (text or "").strip()
    if not text or not filter_text_quality_fixed(text):
        return None

    phon_line = text_to_phoneme_line(text)

    prompt = (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phon_line}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )
    target = text  # what we want the model to generate
    full = prompt + target
    return {"prompt": prompt, "target": target, "full": full}

def prepare_split(split: str):
    base = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    base = base.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)

    def mapper(batch):
        # Clean batch texts first
        cleaned_batch = []
        for t in batch["text"]:
            cleaned = clean_wikitext_fixed(t)
            if cleaned and filter_text_quality_fixed(cleaned):
                cleaned_batch.append(cleaned)
        
        outs = [build_example(t) for t in cleaned_batch]
        # filter Nones (empty lines etc.)
        outs = [o for o in outs if o is not None]
        if not outs:
            return {"prompt": [], "target": [], "full": []}
        return {
            "prompt": [o["prompt"] for o in outs],
            "target": [o["target"] for o in outs],
            "full":   [o["full"]   for o in outs],
        }

    ds = base.map(mapper, batched=True, remove_columns=base.column_names)
    return ds

# ---------- Tokenization with prefix-masked labels ----------

def make_tokenize_fn(tokenizer, max_length: int = 512, min_target_tokens: int = 4):
    """
    Ensures every example has at least `min_target_tokens` supervised tokens.
    We encode prompt and target separately, then truncate the prompt to leave room.
    """
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "pad_token_id must be set"

    def _tok(batch):
        input_ids_batch, attn_batch, labels_batch = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            # Encode WITHOUT adding extra special tokens
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t = tokenizer(target, add_special_tokens=False)["input_ids"]

            # Skip pathological cases
            if len(t) == 0:
                continue

            # Reserve space for target
            max_prompt_len = max_length - min_target_tokens
            if max_prompt_len <= 0:
                continue

            # Truncate prompt to leave room
            if len(p) > max_prompt_len:
                p = p[:max_prompt_len]

            # Fit as much target as possible, but keep at least min_target_tokens
            space = max_length - len(p)
            if space < min_target_tokens:
                # Even after truncating prompt, no room left → skip example
                continue
            t = t[:space]

            ids = p + t
            attn = [1] * len(ids)
            labs = ([-100] * len(p)) + t[:]  # supervise only target

            # Pad to max_length
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids  += [pad_id] * pad_len
                attn += [0] * pad_len
                labs += [-100] * pad_len

            input_ids_batch.append(ids)
            attn_batch.append(attn)
            labels_batch.append(labs)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "labels": labels_batch,
        }
    return _tok

def debug_supervision(ds, name):
    import numpy as np
    import random
    n = min(2000, len(ds))
    cnt = 0
    for i in range(n):
        labs = ds[i]["labels"]
        if any(l != -100 for l in labs):
            cnt += 1
    print(f"[{name}] examples with at least 1 supervised token: {cnt}/{n}")

# ---------- Custom collator (keep labels, just pad if needed) ----------

class CausalLMDataCollator(DataCollatorWithPadding):
    """
    Uses tokenizer padding for inputs. Expects 'labels' already provided;
    pads labels with -100 to match input length.
    """
    def __call__(self, features):
        labels = [f["labels"] for f in features]
        for f in features:
            f.pop("labels")
        batch = super().__call__(features)

        max_len = batch["input_ids"].shape[1]
        padded = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded.append(lab)
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch

# ---------- Find latest checkpoint ----------

def find_latest_checkpoint(checkpoint_dir="./results_lora"):
    """
    Find the latest checkpoint in the given directory.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(checkpoint_dir, item)))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]

# ---------- Test phoneme conversion quality ----------

def test_phoneme_conversion():
    """
    Test phoneme conversion quality.
    """
    print("\n" + "="*60)
    print("Testing Phoneme Conversion Quality")
    print("="*60)
    
    test_phrases = [
        "the cat sat on the mat",
        "hello world",
        "artificial intelligence",
        "development",
        "positive sales",
        "Valkyria Chronicles",
        "this is a test"
    ]
    
    for phrase in test_phrases:
        phonemes = text_to_phoneme_line(phrase)
        print(f"\n'{phrase}'")
        print(f"→ {phonemes}")
        
        # Count UNK tokens
        unk_count = phonemes.count("UNK(")
        if unk_count > 0:
            print(f"  ⚠️  {unk_count} UNK tokens")
    
    print("="*60 + "\n")

# ---------- Main training with LoRA ----------

def main():
    # 0) Test phoneme conversion quality first
    test_phoneme_conversion()
    
    # 1) DEBUG FIRST - Check cleaning before full processing
    print("Running debugging analysis...")
    debug_cleaning_samples()
    keep_rate = debug_dataset_stats()
    
    if keep_rate < 10:  # If keeping less than 10%
        print("⚠️ WARNING: Keeping less than 10% of data. Filters might be too strict!")
        print("Consider adjusting cleaning parameters.")
    
    # 2) Build dataset
    print("\n" + "="*60)
    print("Building phonemes → text dataset...")
    print("Using RELIABLE phoneme conversion (CMUdict + g2p-en)")
    
    train_ds = prepare_split("train")
    val_ds   = prepare_split("validation")
    
    print(f"✅ Training samples after cleaning: {len(train_ds):,}")
    print(f"✅ Validation samples after cleaning: {len(val_ds):,}")
    
    if len(train_ds) < 10000:
        print(f"⚠️ WARNING: Only {len(train_ds):,} training samples.")
        print("This might be too small for good LLM fine-tuning.")
        print("Consider using less strict filtering.")
    
    # Show some cleaned examples
    print("\n=== Sample Cleaned Training Examples ===")
    for i in range(min(3, len(train_ds))):
        example = train_ds[i]
        if "target" in example and len(example["target"]) > 0:
            print(f"\nExample {i}:")
            print(f"Target text: {example['target'][:100]}...")
            if "prompt" in example:
                phon_part = example["prompt"].split("\n")[2] if "\n" in example["prompt"] else ""
                print(f"Phonemes: {phon_part[:50]}...")
    
    # Ask for confirmation before proceeding
    print("\n" + "="*60)
    response = input(f"Proceed with training on {len(train_ds):,} samples? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # 3) Tokenizer & special tokens
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Add tags and ensure a pad token
    special = {"additional_special_tokens": TAGS}
    added = tokenizer.add_special_tokens(special)
    print(f"Added {added} special tokens")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # LLaMA convention

    # 4) Tokenize with prompt-masked labels
    print("Tokenizing dataset with masked labels...")
    tok_fn = make_tokenize_fn(tokenizer, max_length=512)
    tokenized_train = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val   = val_ds.map(tok_fn,   batched=True, remove_columns=val_ds.column_names)

    # 5) Load base model, resize embeddings, then wrap with LoRA
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to {len(tokenizer)} tokens")

    # --- LoRA config (typical for LLaMA) ---
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # attention
            "gate_proj", "up_proj", "down_proj"          # MLP
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check

    # 6) Collator & args
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results_lora_reliable",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        learning_rate=2e-4,          # a bit higher is common for (Q)LoRA
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=1000,  # Save every 1000 steps
        save_total_limit=5,
        fp16=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        report_to="none",
    )

    vocab = model.get_input_embeddings().num_embeddings
    print(f"Tokenizer vocab: {len(tokenizer)}, Model vocab: {vocab}")
    debug_supervision(tokenized_train, "train")
    debug_supervision(tokenized_val, "val")

    # 7) Setup Trainer
    print("\nSetting up Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Find and use latest checkpoint
    checkpoint_path = find_latest_checkpoint("/content/VALLR/results_lora")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"✅ Found checkpoint at: {checkpoint_path}")
        
        # Check if it's valid
        required_files = ["trainer_state.json", "adapter_model.safetensors", "config.json"]
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(checkpoint_path, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️ Missing files: {missing_files}")
            print("⚠️ Will start fresh training instead")
            resume_checkpoint = None
        else:
            resume_checkpoint = checkpoint_path
            # Extract step number from checkpoint path
            step = os.path.basename(checkpoint_path).split("-")[1]
            print(f"✅ Checkpoint is valid, resuming from step {step}")
    else:
        print("ℹ️ No valid checkpoint found, starting fresh")
        resume_checkpoint = None

    print("\nStarting training with RELIABLE phoneme conversion...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 8) Save adapters (default) + tokenizer
    print("\nSaving the LoRA adapters...")
    trainer.save_model("./llama_phonemes_to_text_lora_reliable")  # saves PEFT adapters
    tokenizer.save_pretrained("./llama_phonemes_to_text_lora_reliable")

    # --- Optional: export a merged full model (fp16, larger) ---
    print("Merging LoRA into base weights for export...")
    merged = model.merge_and_unload()
    merged.save_pretrained("./llama_phonemes_to_text_lora_merged_reliable")
    tokenizer.save_pretrained("./llama_phonemes_to_text_lora_merged_reliable")

    # 9) Evaluate
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # 10) Quick test of the model
    print("\n=== Quick Test ===")
    test_phonemes = ["DH", "AH", "K", "AE", "T"]  # "the cat"
    phoneme_str = " ".join(test_phonemes)
    prompt = f"<S2S>\n<PHONEMES>\n{phoneme_str}\n</PHONEMES>\n<TEXT>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7, do_sample=True)
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Input phonemes: {phoneme_str}")
    print(f"Generated: {generated}")

if __name__ == "__main__":
    main()
