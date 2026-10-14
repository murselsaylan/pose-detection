import cv2
import mediapipe as mp

# Mediapipe pose modülünü başlat
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Görseli yükle
image = cv2.imread("insan.jpg")

# Resim yüklenip yüklenmediğini kontrol et
if image is None:
    print("Hata: Resim dosyası yüklenemedi. Dosya yolu kontrol edin.")
    exit()

print("Resim başarıyla yüklendi, boyut:", image.shape)

# Mediapipe pose modülünü başlat
try:
    pose = mp_pose.Pose(static_image_mode=True)
    print("MediaPipe pose modülü başarıyla başlatıldı")
except Exception as e:
    print(f"MediaPipe hatası: {e}")
    exit()

# BGR'den RGB'ye çevir (mediapipe RGB ile çalışır)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Tahmin yap
results = pose.process(image_rgb)

# Eklemleri ve bağlantıları çiz
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        image,                    # çizim yapılacak yer
        results.pose_landmarks,  # poz verileri
        mp_pose.POSE_CONNECTIONS,  # hangi noktalar birbirine bağlanacak
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
    )

# Görseli göster
cv2.imshow("Vücut Tespiti", image)
cv2.waitKey(0)
cv2.destroyAllWindows()