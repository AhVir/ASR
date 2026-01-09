# VALLR Inference Guide

## Quick Start

### 1. Running Basic Inference (Stage 1 Only: Video → Phonemes)

```bash
python main.py --mode infer \
  --version V1 \
  --save_model_path VALLR.path \
  --videos_root /path/to/your/video.mp4
```

**Output:** Phoneme sequence (e.g., `P HH TH R`)

---

### 2. Running Full Inference (Stage 1 + Stage 2: Video → Phonemes → Text)

```bash
# Set the LLM model path
export VALLR_LLM_PATH=/path/to/llama_phonemes_to_text_lora

python main.py --mode infer \
  --version V1 \
  --save_model_path VALLR.path \
  --videos_root /path/to/your/video.mp4
```

**Output:** Full text transcription

---

## What's Happening Now?

### Previous Issue
The code was showing:
```python
Inferences ([['P', '<pad>', 'HH', '<pad>', 'TH', 'R', '<pad>', '<pad>']], ...)
```

This was **only Stage 1** output (raw phonemes with padding).

### What's Fixed

#### ✅ 1. **Fixed `torch.load` Warning**
- Added `weights_only=False` parameter to suppress the security warning
- For production use with untrusted models, set `weights_only=True`

#### ✅ 2. **Added CTC Decoding**
The new `ctc_decode_phonemes()` function:
- Removes consecutive duplicate phonemes
- Removes blank/padding tokens
- Produces clean phoneme sequences

**Before CTC:**
```
['P', '<pad>', 'HH', '<pad>', 'TH', 'R', '<pad>', '<pad>']
```

**After CTC:**
```
['P', 'HH', 'TH', 'R']
```

#### ✅ 3. **Integrated Stage 2 LLM**
The `phonemes_to_text_llm()` function:
- Loads a fine-tuned LLaMA model
- Converts phoneme sequences to readable text
- Uses the format: `<S2S>\n<PHONEMES>\nP HH TH R\n</PHONEMES>\n<TEXT>\n`
- Generates: "either" or "path" (depending on context)

#### ✅ 4. **Added Face Detection & Cropping**
The updated `load_videos()` function:
- Automatically detects faces using MediaPipe
- Crops and centers the mouth region
- Falls back to full frame if face detection fails
- Normalizes frames for better model performance

---

## Video Preparation

### Option 1: Automatic (Recommended)

Use the helper script to check and prepare your video:

```bash
# Check if video is compatible
python prepare_video_for_inference.py --video your_video.mp4

# Test face detection
python prepare_video_for_inference.py --video your_video.mp4 --test-face --preview

# Preprocess video if needed
python prepare_video_for_inference.py --video your_video.mp4 --preprocess
```

### Option 2: Manual Requirements

Your video should:
- **Minimum 16 frames** (preferably more)
- **Show a clear frontal face** 
- **Focus on the speaker's mouth area**
- **Good lighting** (avoid shadows on face)
- **Resolution**: Any (will be resized to 224×224)
- **Format**: MP4, AVI, MOV, MKV

### What Happens During Preprocessing?

1. **Video Loading**: Frames are sampled evenly (16 frames by default)
2. **Face Detection**: MediaPipe detects face landmarks
3. **Face Cropping**: Crops and centers the mouth region
4. **Normalization**: Resizes to 224×224, converts to tensors
5. **Inference**: Passes through VideoMAE + Wav2Vec2 model

---

## Understanding the Output

### Stage 1 Output (Phonemes)

```
Raw phonemes (with padding): ['P', '<pad>', 'HH', '<pad>', 'TH', 'R', '<pad>', ...]
Decoded phonemes (CTC): ['P', 'HH', 'TH', 'R']
```

**Phoneme Types (ARPAbet):**
- Vowels: `AA`, `AE`, `AH`, `AO`, `AW`, `AY`, `EH`, `ER`, `EY`, `IH`, `IY`, `OW`, `OY`, `UH`, `UW`
- Consonants: `B`, `CH`, `D`, `DH`, `F`, `G`, `HH`, `JH`, `K`, `L`, `M`, `N`, `NG`, `P`, `R`, `S`, `SH`, `T`, `TH`, `V`, `W`, `Y`, `Z`, `ZH`

### Stage 2 Output (Text)

```
Text: either
```

The LLM model uses linguistic context to convert phonemes to proper words.

---

## Complete Example

```bash
# Step 1: Check your video
python prepare_video_for_inference.py --video sample.mp4 --test-face

# Step 2: Run inference (Stage 1 only)
python main.py --mode infer \
  --version V1 \
  --save_model_path VALLR.path \
  --videos_root sample.mp4

# Expected output:
# ============================================================
# VALLR Two-Stage Inference
# ============================================================
# 
# [Stage 1] Video to Phonemes
# Loading video: sample.mp4
# ✓ Face cropper initialized
# Video loaded: shape=torch.Size([1, 16, 3, 224, 224])
# ✓ Face detected in 14/16 frames
# Loading Stage 1 model from VALLR.path...
# Model loaded from VALLR.path
# Model output: logits shape=torch.Size([1, 256, 41])
# 
# Raw phonemes (with padding): ['P', '<pad>', 'HH', '<pad>', 'TH', ...]
# Decoded phonemes (CTC): ['P', 'HH', 'TH', 'R']
# 
# [Stage 2] Phonemes to Text
# Stage 2 (LLM) not available. Returning phoneme sequence only.
# 
# ============================================================
# RESULTS:
# ============================================================
# Phonemes: P HH TH R
# Text: P HH TH R
# ============================================================

# Step 3: Run with Stage 2 (if you have the LLM model)
export VALLR_LLM_PATH=./llama_phonemes_to_text_lora_merged
python main.py --mode infer \
  --version V1 \
  --save_model_path VALLR.path \
  --videos_root sample.mp4

# Expected output with Stage 2:
# [Stage 2] Phonemes to Text
# Loading Stage 2 LLM from ./llama_phonemes_to_text_lora_merged...
# Stage 2 input: P HH TH R
# 
# ============================================================
# RESULTS:
# ============================================================
# Phonemes: P HH TH R
# Text: either
# ============================================================
```

---

## Training Stage 2 LLM

If you want to train your own Stage 2 model:

```bash
python Models/Llama.py
```

This will:
1. Load WikiText-2 dataset
2. Convert text to phonemes
3. Fine-tune LLaMA-3.2-3B with LoRA
4. Save to `./llama_phonemes_to_text_lora_merged/`

---

## Troubleshooting

### Problem: "No faces detected"
**Solution:** 
- Ensure video shows a clear frontal face
- Improve lighting
- Use `--test-face --preview` to see what's detected
- Inference will still work, but accuracy may be lower

### Problem: "Not enough frames"
**Solution:**
- Video needs at least 16 frames
- Use longer video clips
- Check video isn't corrupted

### Problem: "Model file not found"
**Solution:**
- Ensure `VALLR.path` exists or use correct `--save_model_path`
- Download the pre-trained model from the paper's repository

### Problem: "Out of memory"
**Solution:**
- Reduce video resolution
- Use CPU instead: `--device cpu` (slower but works)
- Close other applications

### Problem: "Stage 2 LLM not working"
**Solution:**
- Train the LLM model first using `Models/Llama.py`
- Or download a pre-trained Stage 2 model
- Check `VALLR_LLM_PATH` points to a valid directory

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                    INPUT VIDEO                      │
│               (any length, any resolution)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Video Preprocessing  │
         │  - Face detection     │
         │  - Mouth cropping     │
         │  - Resize to 224×224  │
         │  - Sample 16 frames   │
         └──────────┬────────────┘
                    │
                    ▼
       ┌────────────────────────────┐
       │       STAGE 1 MODEL        │
       │   (VideoMAE + Wav2Vec2)    │
       │    Video → Phonemes        │
       └──────────┬─────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  CTC Decoding  │
         │  Clean phonemes│
         └────────┬───────┘
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
 [Optional]              [Required]
┌──────────────┐      ┌──────────────┐
│  STAGE 2     │      │  PHONEMES    │
│  LLM Model   │      │  Output      │
│  Phonemes    │      │  (if no LLM) │
│  → Text      │      └──────────────┘
└──────┬───────┘
       │
       ▼
  ┌──────────┐
  │   TEXT   │
  │  Output  │
  └──────────┘
```

---

## Citation

If you use VALLR, please cite:

```bibtex
@inproceedings{thomas2025vallr,
  title={VALLR: Visual ASR Language Model for Lip Reading},
  author={Thomas, Marshall and Fish, Edward and Bowden, Richard},
  booktitle={ICCV},
  year={2025}
}
```
