"""
Face Detection Module using OpenCV Haar Cascade Classifier
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Face detection class using OpenCV's Haar Cascade classifier.
    Supports detection in images, videos, and webcam streams.
    """
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the face detector.
        
        Args:
            cascade_path: Path to Haar Cascade XML file. If None, uses default OpenCV frontal face cascade.
        """
        if cascade_path is None:
            # Use default OpenCV frontal face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar Cascade file not found: {cascade_path}")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load Haar Cascade from: {cascade_path}")
    
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                     min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size. Objects smaller than this are ignored.
        
        Returns:
            List of tuples (x, y, width, height) representing detected face bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: Input image
            faces: List of face bounding boxes (x, y, width, height)
            color: BGR color tuple for bounding boxes
            thickness: Thickness of bounding box lines
        
        Returns:
            Image with bounding boxes drawn
        """
        result_image = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
            # Add label
            cv2.putText(result_image, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return result_image
    
    def detect_and_draw(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detect faces and draw bounding boxes in one step.
        
        Args:
            image: Input image
            **kwargs: Additional arguments passed to detect_faces()
        
        Returns:
            Tuple of (annotated_image, list_of_faces)
        """
        faces = self.detect_faces(image, **kwargs)
        annotated_image = self.draw_faces(image, faces)
        return annotated_image, faces
    
    def process_image(self, image_path: str, output_path: Optional[str] = None, 
                     **kwargs) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image. If None, image is not saved.
            **kwargs: Additional arguments passed to detect_faces()
        
        Returns:
            Tuple of (annotated_image, list_of_faces)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        annotated_image, faces = self.detect_and_draw(image, **kwargs)
        
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Output saved to: {output_path}")
        
        print(f"Detected {len(faces)} face(s) in {image_path}")
        
        return annotated_image, faces
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_preview: bool = True, **kwargs):
        """
        Process a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video. If None, video is not saved.
            show_preview: Whether to show preview window
            **kwargs: Additional arguments passed to detect_faces()
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_faces = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_frame, faces = self.detect_and_draw(frame, **kwargs)
                total_faces += len(faces)
                frame_count += 1
                
                if writer:
                    writer.write(annotated_frame)
                
                if show_preview:
                    cv2.imshow('Face Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames. Average faces per frame: {total_faces / frame_count:.2f}")
        if output_path:
            print(f"Output video saved to: {output_path}")
    
    def process_webcam(self, camera_index: int = 0, **kwargs):
        """
        Process webcam stream in real-time.
        
        Args:
            camera_index: Camera device index (default: 0)
            **kwargs: Additional arguments passed to detect_faces()
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_index}")
        
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_frame, faces = self.detect_and_draw(frame, **kwargs)
                
                # Display face count
                cv2.putText(annotated_frame, f'Faces: {len(faces)}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Face Detection - Webcam', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
