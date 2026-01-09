#!/usr/bin/env python3
"""
Test script to validate VALLR inference setup.
Checks all components without running full inference.
"""

import sys
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def test_imports():
    """Test if all required packages are installed."""
    print_header("Testing Imports")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'decord': 'Decord (video loading)',
        'cv2': 'OpenCV',
        'face_cropper': 'Face Cropper (local)',
        'numpy': 'NumPy',
    }
    
    success = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name} - MISSING")
            print(f"  Error: {e}")
            success = False
    
    return success

def test_files():
    """Test if required files exist."""
    print_header("Testing Required Files")
    
    required_files = [
        'main.py',
        'config.py',
        'face_cropper.py',
        'Models/VALLR.py',
        'Models/ML_VALLR.py',
        'Models/Llama.py',
        'Data/dataset.py',
        'prepare_video_for_inference.py',
        'INFERENCE_GUIDE.md',
    ]
    
    success = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            success = False
    
    return success

def test_functions():
    """Test if new functions are properly integrated."""
    print_header("Testing Code Integration")
    
    try:
        # Import main module
        sys.path.insert(0, os.getcwd())
        import main
        
        # Check if new functions exist
        functions = [
            'ctc_decode_phonemes',
            'phonemes_to_text_llm',
            'run_inference',
            'load_videos',
        ]
        
        success = True
        for func_name in functions:
            if hasattr(main, func_name):
                print(f"✓ Function '{func_name}' exists")
            else:
                print(f"✗ Function '{func_name}' - MISSING")
                success = False
        
        return success
        
    except Exception as e:
        print(f"✗ Error importing main.py: {e}")
        return False

def test_ctc_decode():
    """Test CTC decoding function."""
    print_header("Testing CTC Decoding")
    
    try:
        sys.path.insert(0, os.getcwd())
        from main import ctc_decode_phonemes
        
        # Test data
        reverse_vocab = {0: '<pad>', 1: 'P', 2: 'HH', 3: 'TH', 4: 'R'}
        test_indices = [1, 0, 2, 0, 0, 3, 4, 0, 0]
        
        # Run CTC decode
        decoded = ctc_decode_phonemes(test_indices, reverse_vocab, blank_idx=0)
        
        expected = ['P', 'HH', 'TH', 'R']
        if decoded == expected:
            print(f"✓ CTC decode working correctly")
            print(f"  Input:  {test_indices}")
            print(f"  Output: {decoded}")
            return True
        else:
            print(f"✗ CTC decode incorrect")
            print(f"  Expected: {expected}")
            print(f"  Got:      {decoded}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing CTC decode: {e}")
        return False

def test_vocab():
    """Test phoneme vocabulary."""
    print_header("Testing Phoneme Vocabulary")
    
    try:
        sys.path.insert(0, os.getcwd())
        from config import get_vocab
        
        vocab = get_vocab()
        
        print(f"✓ Vocabulary loaded: {len(vocab)} phonemes")
        print(f"  Sample phonemes: {list(vocab.keys())[:10]}")
        
        # Check for required tokens
        if '<pad>' in vocab:
            print(f"✓ Padding token exists: '<pad>' = {vocab['<pad>']}")
        else:
            print(f"⚠ Warning: No '<pad>' token found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing vocabulary: {e}")
        return False

def test_model_path():
    """Check if model file exists."""
    print_header("Testing Model File")
    
    model_files = ['VALLR.path', 'model.pth', 'model.pt']
    
    found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✓ Model file found: {model_file} ({size_mb:.1f} MB)")
            found = True
            break
    
    if not found:
        print(f"⚠ Warning: No model file found")
        print(f"  Looked for: {', '.join(model_files)}")
        print(f"  You'll need to train the model or download pre-trained weights")
    
    return found

def test_gpu():
    """Test GPU availability."""
    print_header("Testing GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  PyTorch version: {torch.__version__}")
            return True
        else:
            print(f"⚠ No GPU available - will use CPU")
            print(f"  Note: CPU inference is much slower")
            return False
            
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  VALLR INFERENCE VALIDATION")
    print("="*60)
    
    results = {
        'Imports': test_imports(),
        'Files': test_files(),
        'Functions': test_functions(),
        'CTC Decode': test_ctc_decode(),
        'Vocabulary': test_vocab(),
        'Model File': test_model_path(),
        'GPU': test_gpu(),
    }
    
    # Summary
    print_header("Summary")
    
    total = len(results)
    passed = sum(results.values())
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} - {test}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run inference.")
        print("\nNext steps:")
        print("  1. Prepare your video:")
        print("     python prepare_video_for_inference.py --video your_video.mp4")
        print("\n  2. Run inference:")
        print("     python main.py --mode infer --version V1 \\")
        print("       --save_model_path VALLR.path \\")
        print("       --videos_root your_video.mp4")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        
        if not results['Imports']:
            print("\n  Fix missing imports:")
            print("    pip install -r requirements.txt")
        
        if not results['Model File']:
            print("\n  Get the model file:")
            print("    - Train the model using the training code")
            print("    - Or download pre-trained weights")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
