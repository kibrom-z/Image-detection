"""
Script to create sample images for face detection testing
Creates test images programmatically (no internet required)
"""

import os
from pathlib import Path
import cv2
import numpy as np


def create_sample_images():
    """
    Create sample images using OpenCV (no internet required).
    Generates test images with face-like patterns for testing the detection system.
    """
    image_dir = Path("data/image")
    image_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample test images...")
    
    # Sample 1: Simple face pattern
    img1 = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Face outline (ellipse)
    cv2.ellipse(img1, (200, 200), (120, 150), 0, 0, 360, (220, 200, 180), -1)
    cv2.ellipse(img1, (200, 200), (120, 150), 0, 0, 360, (180, 160, 140), 3)
    
    # Eyes
    cv2.circle(img1, (170, 170), 15, (50, 50, 50), -1)
    cv2.circle(img1, (230, 170), 15, (50, 50, 50), -1)
    
    # Nose
    cv2.ellipse(img1, (200, 200), (10, 20), 0, 0, 360, (150, 130, 110), -1)
    
    # Mouth
    cv2.ellipse(img1, (200, 240), (30, 15), 0, 0, 180, (100, 50, 50), 3)
    
    cv2.imwrite(str(image_dir / "sample_face_1.jpg"), img1)
    print("[OK] Created: sample_face_1.jpg")
    
    # Sample 2: Larger face pattern
    img2 = np.ones((500, 500, 3), dtype=np.uint8) * 240
    cv2.ellipse(img2, (250, 250), (150, 180), 0, 0, 360, (210, 190, 170), -1)
    cv2.ellipse(img2, (250, 250), (150, 180), 0, 0, 360, (170, 150, 130), 3)
    cv2.circle(img2, (220, 220), 18, (40, 40, 40), -1)
    cv2.circle(img2, (280, 220), 18, (40, 40, 40), -1)
    cv2.ellipse(img2, (250, 280), (35, 18), 0, 0, 180, (90, 40, 40), 3)
    
    cv2.imwrite(str(image_dir / "sample_face_2.jpg"), img2)
    print("[OK] Created: sample_face_2.jpg")
    
    # Sample 3: Multiple faces (smaller)
    img3 = np.ones((600, 800, 3), dtype=np.uint8) * 250
    
    # Face 1 (left)
    cv2.ellipse(img3, (200, 300), (100, 130), 0, 0, 360, (215, 195, 175), -1)
    cv2.ellipse(img3, (200, 300), (100, 130), 0, 0, 360, (175, 155, 135), 2)
    cv2.circle(img3, (180, 280), 12, (45, 45, 45), -1)
    cv2.circle(img3, (220, 280), 12, (45, 45, 45), -1)
    cv2.ellipse(img3, (200, 320), (25, 12), 0, 0, 180, (95, 45, 45), 2)
    
    # Face 2 (right)
    cv2.ellipse(img3, (600, 300), (100, 130), 0, 0, 360, (225, 205, 185), -1)
    cv2.ellipse(img3, (600, 300), (100, 130), 0, 0, 360, (185, 165, 145), 2)
    cv2.circle(img3, (580, 280), 12, (45, 45, 45), -1)
    cv2.circle(img3, (620, 280), 12, (45, 45, 45), -1)
    cv2.ellipse(img3, (600, 320), (25, 12), 0, 0, 180, (95, 45, 45), 2)
    
    cv2.imwrite(str(image_dir / "sample_multiple_faces.jpg"), img3)
    print("[OK] Created: sample_multiple_faces.jpg")
    
    # Sample 4: Portrait orientation
    img4 = np.ones((600, 400, 3), dtype=np.uint8) * 245
    cv2.ellipse(img4, (200, 300), (110, 140), 0, 0, 360, (218, 198, 178), -1)
    cv2.ellipse(img4, (200, 300), (110, 140), 0, 0, 360, (178, 158, 138), 3)
    cv2.circle(img4, (175, 270), 14, (48, 48, 48), -1)
    cv2.circle(img4, (225, 270), 14, (48, 48, 48), -1)
    cv2.ellipse(img4, (200, 310), (28, 14), 0, 0, 180, (98, 48, 48), 3)
    
    cv2.imwrite(str(image_dir / "sample_portrait.jpg"), img4)
    print("[OK] Created: sample_portrait.jpg")


def main():
    """Main function to create sample images."""
    print("="*60)
    print("Creating Sample Images for Face Detection")
    print("="*60)
    
    # Create sample images locally (no internet required)
    create_sample_images()
    
    # Check what we have
    image_dir = Path("data/image")
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    print("\n" + "="*60)
    print(f"[OK] Sample images created in: {image_dir.absolute()}")
    print(f"[OK] Total images: {len(image_files)}")
    print("\nCreated files:")
    for img_file in sorted(image_files):
        file_size = img_file.stat().st_size / 1024  # Size in KB
        print(f"  - {img_file.name} ({file_size:.1f} KB)")
    print("="*60)
    
    print("\nNote: These are synthetic test images.")
    print("For better face detection results, add real photos with faces!")
    print("\nYou can:")
    print("  1. Copy photos from your phone/camera to data/image/")
    print("  2. Download free stock photos from sites like Unsplash, Pexels")
    print("  3. Use the sample images to test the detection system")
    
    print("\n[READY] Test with: python src/main.py --mode image --input data/image/sample_face_1.jpg")


if __name__ == "__main__":
    main()
