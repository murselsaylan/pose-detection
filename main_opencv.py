import cv2
import numpy as np

# Görseli yükle
image = cv2.imread("insan.jpg")

if image is None:
    print("Hata: Resim dosyası yüklenemedi.")
    exit()

print("Resim başarıyla yüklendi, boyut:", image.shape)

# Basit yüz tespiti örneği (MediaPipe alternatifi)
# Haar Cascade yüz dedektörü kullanacağız
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yüzleri tespit et
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Tespit edilen yüzleri çerçevelendirme
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, 'Yuz', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

print(f"{len(faces)} yüz tespit edildi.")

# Görseli göster
cv2.imshow("Yuz Tespiti", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Program başarıyla tamamlandı!")
