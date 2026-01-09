#!/usr/bin/env python3
"""
Helper script to prepare and test videos for VALLR inference.
This script checks if your video is compatible and optionally preprocesses it.
"""

import os
import sys
import cv2
import argparse
from face_cropper import FaceCropper
from decord import VideoReader, cpu
import numpy as np

def check_video_info(video_path):
    """Check basic video information."""
    print(f"\n{'='*60}")
    print(f"Checking video: {video_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found!")
        return False
    
    try:
        # Load video
        vr = VideoReader(video_path, ctx=cpu(0))
        cap = cv2.VideoCapture(video_path)
        
        # Get video info
        frame_count = len(vr)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"✓ Video Information:")
        print(f"  - Frames: {frame_count}")
        print(f"  - FPS: {fps:.2f}")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Duration: {duration:.2f} seconds")
        
        # Check requirements
        print(f"\n✓ Requirements Check:")
        min_frames = 16
        if frame_count >= min_frames:
            print(f"  ✓ Frame count ({frame_count}) >= minimum ({min_frames})")
        else:
            print(f"  ⚠ Frame count ({frame_count}) < minimum ({min_frames})")
            print(f"    Video may not work properly!")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error reading video: {e}")
        return False

def test_face_detection(video_path, show_preview=False):
    """Test face detection on the video."""
    print(f"\n{'='*60}")
    print("Testing Face Detection")
    print(f"{'='*60}\n")
    
    try:
        # Initialize face cropper
        face_cropper = FaceCropper(
            min_face_detector_confidence=0.5,
            face_detector_model_selection="SHORT_RANGE",
            landmark_detector_static_image_mode="STATIC_MODE",
            min_landmark_detector_confidence=0.5
        )
        print("✓ Face cropper initialized")
        
        # Load video
        vr = VideoReader(video_path, ctx=cpu(0))
        
        # Test on first, middle, and last frames
        test_indices = [0, len(vr)//2, len(vr)-1]
        detected_faces = 0
        
        for idx in test_indices:
            frame = vr[idx].asnumpy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            faces = face_cropper.get_faces(frame_rgb, remove_background=False, correct_roll=True)
            
            if faces:
                detected_faces += 1
                print(f"  ✓ Frame {idx}: Face detected")
                
                if show_preview:
                    # Save preview
                    preview_path = f"face_preview_frame_{idx}.jpg"
                    cv2.imwrite(preview_path, cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR))
                    print(f"    Saved preview: {preview_path}")
            else:
                print(f"  ✗ Frame {idx}: No face detected")
        
        print(f"\n✓ Detection Summary: {detected_faces}/{len(test_indices)} frames with faces")
        
        if detected_faces == 0:
            print("\n⚠ WARNING: No faces detected in test frames!")
            print("  Tips:")
            print("  - Ensure the video shows a clear frontal view of the face")
            print("  - Check if lighting is adequate")
            print("  - The face should be reasonably large in the frame")
            print("  - You can still run inference, but results may be poor")
        
        return detected_faces > 0
        
    except Exception as e:
        print(f"❌ Error during face detection: {e}")
        return False

def preprocess_video(input_path, output_path=None, target_fps=25):
    """
    Preprocess video for optimal inference.
    - Convert to standard format
    - Adjust FPS if needed
    - Resize if extremely large
    """
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_preprocessed.mp4"
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Video")
    print(f"{'='*60}\n")
    
    try:
        cap = cv2.VideoCapture(input_path)
        
        # Get original properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine if we need to resize (if resolution is very large)
        max_dim = 720
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"  Resizing: {width}x{height} -> {new_width}x{new_height}")
        else:
            new_width, new_height = width, height
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize if needed
            if (new_width, new_height) != (width, height):
                frame = cv2.resize(frame, (new_width, new_height))
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"✓ Preprocessed video saved: {output_path}")
        print(f"  - Frames: {frame_count}")
        print(f"  - FPS: {target_fps}")
        print(f"  - Resolution: {new_width}x{new_height}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error preprocessing video: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and test videos for VALLR inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check video compatibility
  python prepare_video_for_inference.py --video my_video.mp4
  
  # Check with face detection test
  python prepare_video_for_inference.py --video my_video.mp4 --test-face
  
  # Preprocess video
  python prepare_video_for_inference.py --video my_video.mp4 --preprocess
  
  # Full check with preview images
  python prepare_video_for_inference.py --video my_video.mp4 --test-face --preview
        """
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--test-face', action='store_true',
                       help='Test face detection on sample frames')
    parser.add_argument('--preview', action='store_true',
                       help='Save preview images of detected faces')
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess video to standard format')
    parser.add_argument('--output', type=str,
                       help='Output path for preprocessed video')
    
    args = parser.parse_args()
    
    # Check video info
    if not check_video_info(args.video):
        return
    
    # Test face detection if requested
    if args.test_face:
        test_face_detection(args.video, show_preview=args.preview)
    
    # Preprocess if requested
    if args.preprocess:
        preprocess_video(args.video, args.output)
    
    # Final instructions
    print(f"\n{'='*60}")
    print("Next Steps")
    print(f"{'='*60}\n")
    print("To run inference on this video:")
    print(f"\n  python main.py --mode infer \\")
    print(f"    --version V1 \\")
    print(f"    --save_model_path /path/to/model.pth \\")
    print(f"    --videos_root {args.video}")
    print("\nOptional: Enable Stage 2 (LLM) for phoneme-to-text conversion:")
    print(f"  export VALLR_LLM_PATH=/path/to/llm/model")
    print(f"  python main.py --mode infer ...\n")

if __name__ == "__main__":
    main()
