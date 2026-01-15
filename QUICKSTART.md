# Quick Start Guide - How to Run Your Face Detection Project

## Step 1: Install Dependencies

First, make sure you have Python installed (3.7 or higher). Then install the required packages:

```bash
pip install -r requirements.txt
```

Or if you prefer to install individually:
```bash
pip install opencv-python numpy
```

## Step 2: Choose How to Run

You have **3 ways** to run the project:

---

## Method 1: Run Examples (Easiest - Recommended for First Time)

Run the example script to see all features:

```bash
python src/example.py
```

This will run all examples automatically. To run a specific example:
```bash
python src/example.py 1    # Image detection
python src/example.py 2    # Batch processing
python src/example.py 3    # Custom parameters
python src/example.py 4    # Webcam detection
```

---

## Method 2: Command-Line Interface (CLI)

### Detect Faces in an Image

**Basic usage:**
```bash
python src/main.py --mode image --input data/image/your_photo.jpg --output data/image/output.jpg
```

**Without saving (just display):**
```bash
python src/main.py --mode image --input data/image/your_photo.jpg
```

### Detect Faces in a Video

```bash
python src/main.py --mode video --input data/videos/your_video.mp4 --output data/videos/output.mp4
```

**Without saving (just preview):**
```bash
python src/main.py --mode video --input data/videos/your_video.mp4 --no-preview
```

### Real-Time Webcam Detection

```bash
python src/main.py --mode webcam
```
Press **'q'** to quit.

### Advanced Options

```bash
python src/main.py --mode image \
    --input data/image/photo.jpg \
    --output data/image/output.jpg \
    --scale-factor 1.2 \
    --min-neighbors 6 \
    --min-size 50 50
```

---

## Method 3: Python Script (For Custom Usage)

Create your own Python script:

```python
from src.detection import FaceDetector
import cv2

# Initialize detector
detector = FaceDetector()

# Detect faces in an image
annotated_image, faces = detector.process_image(
    'data/image/photo.jpg',
    output_path='data/image/output.jpg'
)

print(f"Detected {len(faces)} face(s)")

# Display result
cv2.imshow('Result', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Save this as `my_script.py` and run:
```bash
python my_script.py
```

---

## Quick Test (No Image Needed)

To test if everything works, run the default example:

```bash
python src/main.py
```

This will try to find images in `data/image/` directory. If no images are found, it will show an error message telling you where to place images.

---

## Troubleshooting

### "Module not found" error
- Make sure you're in the project root directory
- Install dependencies: `pip install -r requirements.txt`

### "Image not found" error
- Place your images in `data/image/` folder
- Use the correct file path (relative or absolute)

### Webcam not working
- Check if your camera is connected
- Try different camera indices: modify `camera_index` in the code (default is 0)

### No faces detected
- Try adjusting parameters: `--scale-factor 1.05 --min-neighbors 3`
- Ensure good lighting in the image/video
- Make sure faces are clearly visible

---

## Example Workflow

1. **First time setup:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test with example:**
   ```bash
   python src/example.py
   ```

3. **Detect faces in your image:**
   ```bash
   # Place your image in data/image/ folder first
   python src/main.py --mode image --input data/image/my_photo.jpg --output data/image/result.jpg
   ```

4. **Try webcam:**
   ```bash
   python src/main.py --mode webcam
   ```

---

## Need Help?

Check the full documentation in `Readme.md` for more details and advanced usage.
