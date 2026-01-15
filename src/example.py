"""
Example script demonstrating face detection capabilities
"""

import cv2
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from detection import FaceDetector


def example_1_image_detection():
    """Example 1: Detect faces in a single image"""
    print("\n" + "="*50)
    print("Example 1: Image Face Detection")
    print("="*50)
    
    detector = FaceDetector()
    
    # Try to find an image in the data/image directory
    image_dir = Path(__file__).parent.parent / "data" / "image"
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    if image_files:
        image_path = str(image_files[0])
        print(f"Processing: {image_path}")
        
        output_path = str(image_dir / "detected_output.jpg")
        annotated_image, faces = detector.process_image(image_path, output_path)
        
        print(f"✓ Detected {len(faces)} face(s)")
        print(f"✓ Output saved to: {output_path}")
        
        # Display the result
        cv2.imshow('Face Detection Result', annotated_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No images found in {image_dir}")
        print("Please add an image file (.jpg, .png, .jpeg) to the data/image/ directory")


def example_2_batch_processing():
    """Example 2: Process multiple images"""
    print("\n" + "="*50)
    print("Example 2: Batch Image Processing")
    print("="*50)
    
    detector = FaceDetector()
    image_dir = Path(__file__).parent.parent / "data" / "image"
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        try:
            output_path = str(image_dir / f"detected_{image_path.name}")
            annotated_image, faces = detector.process_image(str(image_path), output_path)
            print(f"  ✓ Detected {len(faces)} face(s)")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_3_custom_parameters():
    """Example 3: Using custom detection parameters"""
    print("\n" + "="*50)
    print("Example 3: Custom Detection Parameters")
    print("="*50)
    
    detector = FaceDetector()
    image_dir = Path(__file__).parent.parent / "data" / "image"
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    image_path = str(image_files[0])
    print(f"Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Try different parameter combinations
    parameter_sets = [
        {"scale_factor": 1.1, "min_neighbors": 5, "min_size": (30, 30), "name": "Default"},
        {"scale_factor": 1.2, "min_neighbors": 6, "min_size": (50, 50), "name": "Strict"},
        {"scale_factor": 1.05, "min_neighbors": 3, "min_size": (20, 20), "name": "Sensitive"},
    ]
    
    for params in parameter_sets:
        name = params.pop("name")
        faces = detector.detect_faces(image, **params)
        print(f"  {name} parameters: {len(faces)} face(s) detected")


def example_4_webcam_detection():
    """Example 4: Real-time webcam detection"""
    print("\n" + "="*50)
    print("Example 4: Webcam Face Detection")
    print("="*50)
    print("Press 'q' to quit")
    
    detector = FaceDetector()
    
    try:
        detector.process_webcam()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and accessible")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Face Detection Examples")
    print("="*60)
    
    # Initialize detector to check if it works
    try:
        detector = FaceDetector()
        print("✓ Face detector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize face detector: {e}")
        return
    
    # Run examples
    examples = [
        ("1", "Image Detection", example_1_image_detection),
        ("2", "Batch Processing", example_2_batch_processing),
        ("3", "Custom Parameters", example_3_custom_parameters),
        ("4", "Webcam Detection", example_4_webcam_detection),
    ]
    
    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]
        for num, name, func in examples:
            if num == example_num:
                func()
                return
        print(f"Example {example_num} not found")
    else:
        # Run all examples
        for num, name, func in examples:
            try:
                func()
            except KeyboardInterrupt:
                print("\n\nStopped by user")
                break
            except Exception as e:
                print(f"\nError in {name}: {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
