# Pose Detection

Experiments in detecting people and body poses in images.

| File | What it does |
|---|---|
| `main.py` | Pose landmarks on a test image (`insan.jpg`) with **MediaPipe Pose**. |
| `main_opencv.py`, `main_simple.py` | Lightweight person detection with plain OpenCV (no MediaPipe). |
| `main_pose_detection.py` | Pose estimation on a single image with drawing of the detected body. |
| `advanced_pose_detection.py` | Combines several detectors, estimates body type (upper body / full body), detection quality and common problems (cropped image, lying position, tiny subject). |
| `batch_pose_analysis.py` | Runs the analysis on every image in a folder and writes a text report. |

## Run

```bash
pip install opencv-python mediapipe numpy
python main.py                      # needs a test image named insan.jpg
python batch_pose_analysis.py       # analyse the images in ./img and write a report
```

Put your own test images into an `img/` folder (the original photos are not included for privacy).

## Türkçe özet

Görüntüde insan / vücut pozu tespiti denemeleri (OpenCV, MediaPipe) ve bir klasördeki tüm resimleri
analiz edip rapor çıkaran araç. Kişilerin fotoğrafları gizlilik için eklenmedi.
