# Face Detection Project using OpenCV

A complete face detection system built with OpenCV's Haar Cascade classifier. This project supports face detection in images, videos, and real-time webcam streams.

## Features

- ✅ Face detection in static images
- ✅ Face detection in video files
- ✅ Real-time face detection from webcam
- ✅ Face cropping tool - extract only face regions from photos
- ✅ Customizable detection parameters
- ✅ Support for custom Haar Cascade classifiers
- ✅ Easy-to-use Python API
- ✅ Command-line interface

## Project Structure

```
image_detection_project/
├── data/
│   ├── image/          # Input/output images
│   ├── models/         # Custom Haar Cascade models
│   └── videos/         # Input/output videos
├── src/
│   ├── main.py         # Main entry point with CLI
│   ├── detection.py    # Face detection module
│   ├── models/         # Model files
│   └── utils/          # Utility functions
│       ├── __init__.py
│       └── image_utils.py
├── requirements.txt    # Python dependencies
└── Readme.md          # This file
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd image_detection_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface

#### Detect faces in an image:
```bash
python src/main.py --mode image --input data/image/photo.jpg --output data/image/output.jpg
```

#### Detect faces in a video:
```bash
python src/main.py --mode video --input data/videos/video.mp4 --output data/videos/output.mp4
```

#### Real-time webcam detection:
```bash
python src/main.py --mode webcam
```

#### Advanced options:
```bash
python src/main.py --mode image \
    --input data/image/photo.jpg \
    --output data/image/output.jpg \
    --scale-factor 1.2 \
    --min-neighbors 6 \
    --min-size 50 50
```

### Python API

#### Basic usage:
```python
from src.detection import FaceDetector

# Initialize detector
detector = FaceDetector()

# Detect faces in an image
annotated_image, faces = detector.process_image(
    'data/image/photo.jpg',
    output_path='data/image/output.jpg'
)

print(f"Detected {len(faces)} face(s)")
```

#### Custom detection parameters:
```python
detector = FaceDetector()

# Detect with custom parameters
faces = detector.detect_faces(
    image,
    scale_factor=1.2,
    min_neighbors=6,
    min_size=(50, 50)
)

# Draw bounding boxes
annotated_image = detector.draw_faces(image, faces)
```

#### Video processing:
```python
detector = FaceDetector()

# Process video file
detector.process_video(
    'data/videos/input.mp4',
    output_path='data/videos/output.mp4',
    show_preview=True
)
```

#### Webcam detection:
```python
detector = FaceDetector()

# Real-time webcam detection
detector.process_webcam()
```

## Detection Parameters

- **scale_factor** (default: 1.1): How much the image size is reduced at each scale. Lower values detect more faces but are slower.
- **min_neighbors** (default: 5): Minimum number of neighbors required for a detection. Higher values reduce false positives.
- **min_size** (default: (30, 30)): Minimum face size in pixels. Smaller values detect smaller faces.

## Custom Haar Cascade Models

You can use custom Haar Cascade XML files:

```python
detector = FaceDetector(cascade_path='data/models/custom_cascade.xml')
```

Or via command-line:
```bash
python src/main.py --mode image --input photo.jpg --cascade data/models/custom_cascade.xml
```

## Examples

### Example 1: Simple image detection
```python
from src.detection import FaceDetector

detector = FaceDetector()
annotated_image, faces = detector.process_image('photo.jpg')
cv2.imshow('Result', annotated_image)
cv2.waitKey(0)
```

### Example 2: Batch processing images
```python
import os
from src.detection import FaceDetector

detector = FaceDetector()
image_dir = 'data/image/'

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        output_path = os.path.join(image_dir, f'detected_{filename}')
        detector.process_image(image_path, output_path)
```

### Example 3: Real-time detection with statistics
```python
from src.detection import FaceDetector
import cv2

detector = FaceDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated_frame, faces = detector.detect_and_draw(frame)
    
    # Display face count
    cv2.putText(annotated_frame, f'Faces: {len(faces)}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Face Cropping Tool

The project includes a utility script to crop faces from images and save them to `data/image/` folder. This is useful for creating a dataset of face images.

### Crop faces from a single image:
```bash
python crop_faces.py --input path/to/photo.jpg
```

### Crop faces from all images in a directory:
```bash
python crop_faces.py --input path/to/photos/
```

### Advanced options:
```bash
# Custom padding around faces
python crop_faces.py --input photo.jpg --padding 30

# Process directory recursively
python crop_faces.py --input photos/ --recursive

# Custom output directory
python crop_faces.py --input photo.jpg --output my_faces/

# Minimum face size filter
python crop_faces.py --input photo.jpg --min-size 100 100
```

### Example workflow:
1. **Place photos in a folder** (e.g., `my_photos/`)
2. **Run the cropping tool:**
   ```bash
   python crop_faces.py --input my_photos/ --output data/image/
   ```
3. **All cropped faces will be saved** to `data/image/` with names like:
   - `photo1_face_1.jpg`
   - `photo1_face_2.jpg` (if multiple faces)
   - `photo2_face_1.jpg`
   - etc.

The cropped faces can then be used for face detection testing or as a dataset.

## Dataset Classification (face / no_face)

You can automatically **move** images into dataset folders based on face detection:

- `data/dataset/face/`: images that contain **≥1 face** with size **≥ 80x80**
- `data/dataset/no_face/`: all other images

### Dry-run (recommended first)

```bash
python classify_dataset.py --dry-run
```

### Move files (real run)

```bash
python classify_dataset.py
```

### Custom options

```bash
python classify_dataset.py --min-face-size 80 80 --scale-factor 1.1 --min-neighbors 5
```

## Troubleshooting

### Issue: "Haar Cascade file not found"
- **Solution**: The default cascade is included with OpenCV. If using a custom cascade, ensure the path is correct.

### Issue: Poor detection accuracy
- **Solution**: Adjust `scale_factor` (try 1.05-1.3) and `min_neighbors` (try 3-7). Ensure good lighting in images.

### Issue: Webcam not working
- **Solution**: Check camera permissions and try different camera indices (0, 1, 2, etc.).

### Issue: Video processing is slow
- **Solution**: Increase `min_size` to skip smaller faces, or reduce video resolution.

## Requirements

- Python 3.7+
- OpenCV 4.8.0+
- NumPy 1.24.0+

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Acknowledgments

- Built with OpenCV's Haar Cascade classifier
- Uses the default frontal face cascade included with OpenCV
