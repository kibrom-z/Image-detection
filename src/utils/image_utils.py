"""
Utility functions for image processing
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (BGR format)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def resize_image(image: np.ndarray, width: Optional[int] = None, 
                height: Optional[int] = None, scale: float = 1.0) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Input image
        width: Target width (None to maintain aspect ratio)
        height: Target height (None to maintain aspect ratio)
        scale: Scale factor (if width and height are None)
    
    Returns:
        Resized image
    """
    if width is not None and height is not None:
        return cv2.resize(image, (width, height))
    elif width is not None:
        h = int(image.shape[0] * width / image.shape[1])
        return cv2.resize(image, (width, h))
    elif height is not None:
        w = int(image.shape[1] * height / image.shape[0])
        return cv2.resize(image, (w, height))
    else:
        w = int(image.shape[1] * scale)
        h = int(image.shape[0] * scale)
        return cv2.resize(image, (w, h))


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    
    Args:
        image: BGR image
    
    Returns:
        RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR.
    
    Args:
        image: RGB image
    
    Returns:
        BGR image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
