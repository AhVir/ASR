# 🚀 VALLR Quick Reference

## ⚡ Run Inference (One-Liner)

### Stage 1 Only (Video → Phonemes)
```bash
python main.py --mode infer --version V1 --save_model_path VALLR.path --videos_root video.mp4
```

### Stage 1 + 2 (Video → Phonemes → Text)
```bash
VALLR_LLM_PATH=/path/to/llm python main.py --mode infer --version V1 --save_model_path VALLR.path --videos_root video.mp4
```

---

## 📹 Prepare Your Video

### Check Video Compatibility
```bash
python prepare_video_for_inference.py --video your_video.mp4
```

### Test Face Detection
```bash
python prepare_video_for_inference.py --video your_video.mp4 --test-face --preview
```

---

## 📊 Understanding Output

### What You See Now (Fixed)
```
============================================================
[Stage 1] Video to Phonemes
✓ Face detected in 14/16 frames
Decoded phonemes (CTC): ['P', 'HH', 'TH', 'R']

[Stage 2] Phonemes to Text
Text: either
============================================================
```

### What You Had Before (Your Issue)
```
Inferences ([['P', '<pad>', 'HH', '<pad>', 'TH', 'R', '<pad>']], ...)
```

---

## 🎯 What Changed?

| Issue | Solution |
|-------|----------|
| ⚠️ torch.load warning | ✅ Added `weights_only=False` |
| Raw phonemes with `<pad>` | ✅ Added CTC decoding |
| No text output | ✅ Integrated Stage 2 LLM |
| No face detection | ✅ Added automatic face cropping |

---

## 📁 New Files

1. **`prepare_video_for_inference.py`** - Video compatibility checker
2. **`INFERENCE_GUIDE.md`** - Complete usage guide
3. **`CHANGES_SUMMARY.md`** - Detailed changes documentation

---

## 🔧 Video Requirements

- ✅ **Minimum:** 16 frames
- ✅ **Format:** MP4, AVI, MOV, MKV
- ✅ **Content:** Clear frontal face
- ✅ **Lighting:** Good visibility
- ✅ **Resolution:** Any (auto-resized to 224×224)

---

## 🤔 FAQ

**Q: Where is `VALLR.path` from?**  
A: It should be your trained model file (`.pth` or `.pt`). If you don't have it, train the model first.

**Q: Do I need Stage 2?**  
A: No. Stage 1 gives phonemes, Stage 2 converts to text. Both are useful!

**Q: What if no face is detected?**  
A: Inference still works, but accuracy may be lower. Use full frames as fallback.

**Q: Can I use CPU?**  
A: Yes! Add `--device cpu` (slower but works).

---

## 📚 Full Documentation

- **Quick Start:** This file
- **Complete Guide:** [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- **What Changed:** [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
- **Original README:** [README.md](README.md)

---

## 🎓 Architecture

```
Video → Face Detection → VideoMAE+Wav2Vec2 → Phonemes → LLM → Text
        [Preprocessing]   [Stage 1]           [CTC]      [Stage 2]
```

---

**Made with ❤️ for VALLR users**
