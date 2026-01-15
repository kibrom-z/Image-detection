"""
Main entry point for Face Detection Application
"""

import cv2
import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from detection import FaceDetector


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Face Detection using OpenCV')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'],
                       default='image', help='Detection mode: image, video, or webcam')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--output', type=str, help='Output path for processed image/video')
    parser.add_argument('--cascade', type=str, help='Path to custom Haar Cascade XML file')
    parser.add_argument('--scale-factor', type=float, default=1.1,
                       help='Scale factor for face detection (default: 1.1)')
    parser.add_argument('--min-neighbors', type=int, default=5,
                       help='Minimum neighbors for face detection (default: 5)')
    parser.add_argument('--min-size', type=int, nargs=2, default=[30, 30],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Minimum face size (default: 30 30)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window for video processing')
    
    args = parser.parse_args()
    
    # Initialize face detector
    try:
        detector = FaceDetector(cascade_path=args.cascade)
        print("Face detector initialized successfully!")
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        return
    
    # Prepare detection parameters
    detect_params = {
        'scale_factor': args.scale_factor,
        'min_neighbors': args.min_neighbors,
        'min_size': tuple(args.min_size)
    }
    
    # Process based on mode
    if args.mode == 'image':
        if not args.input:
            print("Error: --input is required for image mode")
            return
        
        try:
            annotated_image, faces = detector.process_image(
                args.input,
                output_path=args.output,
                **detect_params
            )
            
            # Display result
            if not args.output:  # Only show if not saving
                cv2.imshow('Face Detection Result', annotated_image)
                print(f"Detected {len(faces)} face(s). Press any key to close.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"Error processing image: {e}")
    
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input is required for video mode")
            return
        
        try:
            detector.process_video(
                args.input,
                output_path=args.output,
                show_preview=not args.no_preview,
                **detect_params
            )
        except Exception as e:
            print(f"Error processing video: {e}")
    
    elif args.mode == 'webcam':
        try:
            detector.process_webcam(**detect_params)
        except Exception as e:
            print(f"Error processing webcam: {e}")


def example_image_detection():
    """Example: Detect faces in an image."""
    print("=== Example: Image Face Detection ===")
    
    # Initialize detector
    detector = FaceDetector()
    
    # Example image path (you should replace this with your actual image path)
    image_path = "../data/image/sample.jpg"
    
    if os.path.exists(image_path):
        annotated_image, faces = detector.process_image(
            image_path,
            output_path="../data/image/output.jpg"
        )
        print(f"Detected {len(faces)} face(s)")
    else:
        print(f"Image not found: {image_path}")
        print("Please place an image in data/image/ directory")


def example_webcam_detection():
    """Example: Real-time face detection from webcam."""
    print("=== Example: Webcam Face Detection ===")
    print("Press 'q' to quit")
    
    detector = FaceDetector()
    detector.process_webcam()


if __name__ == "__main__":
    # If no command-line arguments, run example
    if len(sys.argv) == 1:
        print("Running example: Image Face Detection")
        print("For command-line usage, run: python main.py --help")
        print()
        example_image_detection()
    else:
        main()
