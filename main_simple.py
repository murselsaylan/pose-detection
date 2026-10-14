import cv2
import numpy as np

# Görseli yükle
image = cv2.imread("insan.jpg")

if image is None:
    print("Hata: Resim dosyası yüklenemedi.")
    exit()

print("Resim başarıyla yüklendi, boyut:", image.shape)

# Basit kenar tespiti ve kontur bulma
# Gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültüyü azalt
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Kenar tespiti
edges = cv2.Canny(blurred, 50, 150)

# Konturları bul
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# En büyük konturları bul (muhtemelen insan vücudu)
if contours:
    # Konturları boyuta göre sırala
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # En büyük 3 konturu çiz
    for i, contour in enumerate(contours[:3]):
        # Kontur alanı yeterince büyükse çiz
        if cv2.contourArea(contour) > 500:
            # Konturun etrafında dikdörtgen çiz
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Kontur çiz
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
            
            # Etiket ekle
            cv2.putText(image, f'Nesne {i+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

print(f"{len([c for c in contours if cv2.contourArea(c) > 500])} büyük nesne tespit edildi.")

# Görseli göster
cv2.imshow("Nesne Tespiti", image)
print("Gösterilen pencereyi kapatmak için herhangi bir tuşa basın...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Program başarıyla tamamlandı!")
