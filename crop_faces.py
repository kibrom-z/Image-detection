"""
Script to detect faces in images and crop only the face regions
Saves cropped faces to data/image/ folder
"""

import cv2
import os
import sys
from pathlib import Path
import argparse


def crop_faces_from_image(image_path: str, output_dir: str = "data/image", 
                          padding: int = 20, min_size: tuple = (50, 50)):
    """
    Detect faces in an image and crop each face, saving them separately.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save cropped faces
        padding: Extra pixels to add around each face (default: 20)
        min_size: Minimum face size to crop (width, height)
    
    Returns:
        List of paths to saved cropped face images
    """
    # Initialize face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"Error: Could not load face cascade classifier")
        return []
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return []
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return []
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size
    )
    
    if len(faces) == 0:
        print(f"No faces detected in: {image_path}")
        return []
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    saved_files = []
    
    # Crop and save each face
    for i, (x, y, w, h) in enumerate(faces):
        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        # Crop face
        face_crop = image[y_start:y_end, x_start:x_end]
        
        # Generate output filename
        output_filename = f"{base_name}_face_{i+1}.jpg"
        output_filepath = output_path / output_filename
        
        # Save cropped face
        cv2.imwrite(str(output_filepath), face_crop)
        saved_files.append(str(output_filepath))
        
        print(f"  [OK] Cropped face {i+1}: {output_filename} ({w}x{h} pixels)")
    
    return saved_files


def crop_faces_from_directory(input_dir: str, output_dir: str = "data/image", 
                             recursive: bool = False, padding: int = 20):
    """
    Process all images in a directory and crop faces from each.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped faces
        recursive: Whether to search subdirectories
        padding: Extra pixels around each face
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # Find all image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if recursive:
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    else:
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"No image files found in: {input_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    print("="*60)
    
    total_faces = 0
    processed_images = 0
    
    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")
        try:
            saved_files = crop_faces_from_image(str(img_file), output_dir, padding)
            if saved_files:
                total_faces += len(saved_files)
                processed_images += 1
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"  - Processed images: {processed_images}/{len(image_files)}")
    print(f"  - Total faces cropped: {total_faces}")
    print(f"  - Saved to: {output_dir}")
    print("="*60)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Crop faces from images and save to data/image/ folder'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file or directory path')
    parser.add_argument('--output', type=str, default='data/image',
                       help='Output directory for cropped faces (default: data/image)')
    parser.add_argument('--padding', type=int, default=20,
                       help='Padding pixels around face (default: 20)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories recursively (for directory input)')
    parser.add_argument('--min-size', type=int, nargs=2, default=[50, 50],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Minimum face size to crop (default: 50 50)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image file
        print(f"Cropping faces from: {args.input}")
        print("="*60)
        saved_files = crop_faces_from_image(
            args.input,
            args.output,
            padding=args.padding,
            min_size=tuple(args.min_size)
        )
        
        if saved_files:
            print(f"\n[OK] Successfully cropped {len(saved_files)} face(s)")
        else:
            print("\n[INFO] No faces were cropped")
    
    elif input_path.is_dir():
        # Directory of images
        print(f"Processing images from directory: {args.input}")
        crop_faces_from_directory(
            args.input,
            args.output,
            recursive=args.recursive,
            padding=args.padding
        )
    
    else:
        print(f"Error: Input path does not exist: {args.input}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode - show usage examples
        print("="*60)
        print("Face Cropping Tool")
        print("="*60)
        print("\nUsage examples:")
        print("\n1. Crop faces from a single image:")
        print("   python crop_faces.py --input path/to/photo.jpg")
        print("\n2. Crop faces from all images in a directory:")
        print("   python crop_faces.py --input path/to/photos/")
        print("\n3. Crop with custom padding:")
        print("   python crop_faces.py --input photo.jpg --padding 30")
        print("\n4. Process directory recursively:")
        print("   python crop_faces.py --input photos/ --recursive")
        print("\n5. Custom output directory:")
        print("   python crop_faces.py --input photo.jpg --output my_faces/")
        print("\n" + "="*60)
        print("\nFor help: python crop_faces.py --help")
    else:
        main()
