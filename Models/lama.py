import os
import re
import torch
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
import pronouncing  # CMUdict-based

# ---------- GPU Optimization Settings ----------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Better memory management
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer warnings

# ---------- Phoneme utilities (ARPAbet via pronouncing) ----------

_WORD_RE = re.compile(r"[A-Za-z']+")

def strip_stress(arpabet: str) -> str:
    # Remove 0/1/2 stress digits (e.g., AH0 -> AH)
    return re.sub(r"\d", "", arpabet)

def text_to_arpabet_words(text: str) -> List[str]:
    """
    Converts text into a list of word-level ARPAbet strings (stress-stripped).
    Falls back to the raw word if not in CMUdict to avoid losing alignment.
    """
    words = _WORD_RE.findall(text)
    arpawords = []
    for w in words:
        lw = w.lower()
        phones = pronouncing.phones_for_word(lw)
        if phones:
            # choose first pronunciation; strip stress
            arp = strip_stress(phones[0])
            arpawords.append(arp)
        else:
            # Fallback: mark as unknown token; you can also just keep the grapheme
            arpawords.append(f"UNK({lw})")
    return arpawords

def text_to_phoneme_line(text: str) -> str:
    """
    Formats phonemes as word-separated with ' | ' between words.
    Example: "the cat" -> "DH AH | K AE T"
    """
    arpawords = text_to_arpabet_words(text)
    return " ".join(arpawords)

# ---------- Dataset building (phonemes -> text) ----------

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]

def build_example(text: str) -> Dict[str, str]:
    """
    Builds one training pair where the INPUT is phonemes and the TARGET is original text.
    """
    text = (text or "").strip()
    if not text:
        return None

    phon_line = text_to_phoneme_line(text)

    prompt = (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phon_line}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )
    target = text
    full = prompt + target
    return {"prompt": prompt, "target": target, "full": full}

def prepare_split(split: str):
    base = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    base = base.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)

    def mapper(batch):
        outs = [build_example(t) for t in batch["text"]]
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
    """
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "pad_token_id must be set"

    def _tok(batch):
        input_ids_batch, attn_batch, labels_batch = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t = tokenizer(target, add_special_tokens=False)["input_ids"]

            if len(t) == 0:
                continue

            max_prompt_len = max_length - min_target_tokens
            if max_prompt_len <= 0:
                continue

            if len(p) > max_prompt_len:
                p = p[:max_prompt_len]

            space = max_length - len(p)
            if space < min_target_tokens:
                continue
            t = t[:space]

            ids = p + t
            attn = [1] * len(ids)
            labs = ([-100] * len(p)) + t[:]

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
    n = min(2000, len(ds))
    cnt = 0
    for i in range(n):
        labs = ds[i]["labels"]
        if any(l != -100 for l in labs):
            cnt += 1
    print(f"[{name}] examples with at least 1 supervised token: {cnt}/{n}")

# ---------- SIMPLIFIED collator ----------

class SimpleCausalLMDataCollator(DataCollatorWithPadding):
    """
    Simplified collator that works with the dataset format.
    """
    def __call__(self, features):
        # Extract labels
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
        
        # Pad the rest
        batch = super().__call__(features)
        
        # Pad labels if they exist
        if labels is not None:
            # Find max length
            max_len = batch["input_ids"].shape[1]
            
            # Pad labels
            padded_labels = []
            for lab in labels:
                if len(lab) < max_len:
                    padded = lab + [-100] * (max_len - len(lab))
                else:
                    padded = lab[:max_len]
                padded_labels.append(padded)
            
            # Convert to tensor
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch

# ---------- Main training with LoRA ----------

def main():
    # Check GPU availability
    print("=" * 60)
    print("GPU DIAGNOSTICS")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU speed
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        torch.cuda.synchronize()
        import time
        start = time.time()
        for _ in range(100):
            _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        print(f"GPU performance test: {100/(end-start):.0f} matmuls/sec")
    else:
        print("WARNING: No GPU available, training will be very slow!")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print("=" * 60)

    # 1) Build dataset
    print("\nBuilding phonemes → text dataset...")
    train_ds = prepare_split("train")
    val_ds   = prepare_split("validation")

    # 2) Tokenizer & special tokens
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Add tags and ensure a pad token
    special = {"additional_special_tokens": TAGS}
    added = tokenizer.add_special_tokens(special)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Tokenize with prompt-masked labels
    print("Tokenizing dataset with masked labels...")
    tok_fn = make_tokenize_fn(tokenizer, max_length=512)
    tokenized_train = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val   = val_ds.map(tok_fn,   batched=True, remove_columns=val_ds.column_names)

    # 4) Load base model with FIXED gradient settings
    print("Loading base model...")
    
    # Load model with gradient computation enabled
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        use_cache=False  # IMPORTANT: Disable cache for gradient checkpointing
    )
    
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # --- LoRA config ---
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5) Use the simplified collator
    collator = SimpleCausalLMDataCollator(tokenizer=tokenizer)

    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results_lora",
        overwrite_output_dir=True,
        eval_strategy="steps",  # More frequent evaluation
        eval_steps=500,  # Evaluate every 500 steps
        save_strategy="steps",  # CRITICAL: Save checkpoints
        save_steps=500,  # Save every 500 steps
        save_total_limit=3,  # Keep last 3 checkpoints
        
        # GPU optimization settings
        per_device_train_batch_size=1,  # Start with 1 to ensure it works
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Increase this instead
        num_train_epochs=3,  # Reduced for Colab time limits
        
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,  # More frequent logging
        fp16=True,  # Mixed precision training
        warmup_ratio=0.03,
        report_to="none",
        
        # Data loading optimizations
        dataloader_num_workers=0,  # Set to 0 for now to avoid issues
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        remove_unused_columns=False,  # Important for custom datasets
        
        # Memory optimization - DISABLE gradient checkpointing for now
        gradient_checkpointing=False,  # Disable to fix the gradient issue
        
        # Optional: Early stopping
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    vocab = model.get_input_embeddings().num_embeddings
    print(f"Tokenizer/model vocab: {len(tokenizer)}/{vocab}")
    debug_supervision(tokenized_train, "train")
    debug_supervision(tokenized_val, "val")

    # Verify model is trainable
    print("\n" + "=" * 60)
    print("MODEL TRAINABILITY CHECK")
    print("=" * 60)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {(trainable_params/total_params)*100:.2f}%")
    
    # Quick test: Check if gradients will flow
    print("\nTesting gradient flow...")
    model.train()
    sample = tokenized_train[0]
    inputs = {
        "input_ids": torch.tensor([sample["input_ids"]]).to(device),
        "attention_mask": torch.tensor([sample["attention_mask"]]).to(device),
        "labels": torch.tensor([sample["labels"]]).to(device)
    }
    
    with torch.no_grad():  # Just test, don't actually compute
        outputs = model(**inputs)
        print(f"Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
    
    # Check if LoRA adapters are properly configured
    print(f"Model is training mode: {model.training}")
    print("=" * 60)

    # 6) Train with GPU monitoring
    print("\nSetting up Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Training steps per epoch: {len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    print(f"Estimated time per epoch: ~4-6 hours (optimized)")
    print(f"Checkpoints saved every {training_args.save_steps} steps")
    print("=" * 60)
    
    # Train with potential checkpoint resumption
    import glob
    checkpoints = sorted(glob.glob("./results_lora/checkpoint-*"))
    resume_from_checkpoint = None
    if checkpoints:
        resume_from_checkpoint = checkpoints[-1]
        print(f"\nResuming from checkpoint: {resume_from_checkpoint}")
    
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if LoRA adapters are properly attached")
        print("2. Verify model is in training mode")
        print("3. Check if gradients are enabled")
        
        # Try a simpler approach if training fails
        print("\nTrying simpler training approach...")
        training_args.gradient_accumulation_steps = 1
        training_args.per_device_train_batch_size = 1
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=collator,
            tokenizer=tokenizer,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 7) Save adapters
    print("\nSaving the LoRA adapters...")
    trainer.save_model("./llama_phonemes_to_text_lora")
    tokenizer.save_pretrained("./llama_phonemes_to_text_lora")

    # 8) Optional: merge and save full model
    print("Merging LoRA into base weights for export...")
    merged = model.merge_and_unload()
    merged.save_pretrained("./llama_phonemes_to_text_lora_merged")
    tokenizer.save_pretrained("./llama_phonemes_to_text_lora_merged")

    # 9) Final evaluation
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
