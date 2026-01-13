#!/usr/bin/env python3
"""
Merge LoRA adapters with base LLaMA model for inference.
This creates a standalone model that can be used without PEFT.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

# Special tokens that were added during training (from Models/Llama.py)
SPECIAL_TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]

def merge_lora_adapters(
    base_model_id="meta-llama/Llama-3.2-3B-Instruct",
    adapter_path="./results_lora/checkpoint-10000",
    output_path="./llama_merged_model",
    device="cuda"
):
    """
    Merge LoRA adapters into base model.
    
    Args:
        base_model_id: Hugging Face model ID or local path to base model
        adapter_path: Path to LoRA adapter checkpoint
        output_path: Where to save the merged model
        device: 'cuda' or 'cpu'
    """
    
    print("="*60)
    print("Merging LoRA Adapters into Base Model")
    print("="*60)
    
    # Check if adapter path exists
    if not os.path.exists(adapter_path):
        print(f"❌ Error: Adapter path not found: {adapter_path}")
        return False
    
    # Step 1: Load tokenizer from the adapter checkpoint (has the special tokens)
    print(f"\n📥 Loading tokenizer from adapter: {adapter_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return False
    
    # Step 2: Load base model
    print(f"\n📥 Loading base model: {base_model_id}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        print(f"✓ Base model loaded (original vocab size: {base_model.config.vocab_size})")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        print("\nTip: Make sure you're logged into Hugging Face:")
        print("  huggingface-cli login")
        return False
    
    # Step 3: Resize embeddings to match the trained model
    # The trained model had special tokens added, so we need to resize
    print(f"\n🔧 Resizing model embeddings: {base_model.config.vocab_size} → {len(tokenizer)}")
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"✓ Embeddings resized to {len(tokenizer)}")
    
    # Step 4: Load LoRA adapters
    print(f"\n📥 Loading LoRA adapters from: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✓ LoRA adapters loaded")
    except Exception as e:
        print(f"❌ Error loading LoRA adapters: {e}")
        return False
    
    # Step 5: Merge LoRA weights into base model
    print("\n🔀 Merging LoRA weights into base model...")
    try:
        merged_model = model.merge_and_unload()
        print("✓ Merge completed")
    except Exception as e:
        print(f"❌ Error merging model: {e}")
        return False
    
    # Step 6: Save the merged model
    print(f"\n💾 Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    try:
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("✓ Model saved successfully")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False
    
    # Print model info
    model_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(output_path)
        for filename in filenames
    ) / (1024 ** 3)  # Convert to GB
    
    print("\n" + "="*60)
    print("✅ Merge Complete!")
    print("="*60)
    print(f"📁 Output location: {output_path}")
    print(f"📦 Model size: {model_size:.2f} GB")
    print(f"📝 Vocabulary size: {len(tokenizer)} tokens")
    print(f"🏷️  Special tokens: {SPECIAL_TAGS}")
    print(f"\n🚀 Next step: Run inference with:")
    print(f"  export VALLR_LLM_PATH={output_path}")
    print(f"  python main.py --mode infer --version V1 \\")
    print(f"    --save_model_path VALLR.path \\")
    print(f"    --videos_root your_video.mp4")
    print("="*60 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base LLaMA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge the checkpoint
  python merge_lora_adapters.py
  
  # Specify custom paths
  python merge_lora_adapters.py \\
    --adapter-path ./results_lora/checkpoint-10000 \\
    --output-path ./my_merged_model
  
  # Use CPU (if no GPU available)
  python merge_lora_adapters.py --device cpu
        """
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default='meta-llama/Llama-3.2-3B-Instruct',
        help='Base model ID from Hugging Face or local path'
    )
    parser.add_argument(
        '--adapter-path',
        type=str,
        default='./results_lora/checkpoint-10000',
        help='Path to LoRA adapter checkpoint'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./llama_merged_model',
        help='Where to save the merged model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for merging'
    )
    
    args = parser.parse_args()
    
    success = merge_lora_adapters(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        device=args.device
    )
    
    if not success:
        print("\n❌ Merge failed. Please check the errors above.")
        exit(1)
    else:
        print("✅ Success! Your model is ready for inference.")
        exit(0)


if __name__ == "__main__":
    main()
