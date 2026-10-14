import cv2
import numpy as np

# Görseli yükle
image = cv2.imread("insan.jpg")

if image is None:
    print("Hata: Resim dosyası yüklenemedi.")
    exit()

print("Resim başarıyla yüklendi, boyut:", image.shape)

# Vücut tespiti için basit algoritma
# İnsan vücudunun genel şeklini tespit etmeye çalışacağız

# Gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültüyü azalt
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Kenar tespiti (daha hassas parametreler)
edges = cv2.Canny(blurred, 30, 80)

# Morfolojik işlemler ile kenarları birleştir
kernel = np.ones((3,3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges = cv2.dilate(edges, kernel, iterations=2)

# Konturları bul
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# En büyük konturlari filtrele
min_area = 1000  # Minimum alan
min_aspect_ratio = 0.3  # En - boy oranı (insan vücudu için)
max_aspect_ratio = 3.0

potential_body_parts = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        # Sınırlayıcı dikdörtgen
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            potential_body_parts.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio
            })

# Vücut parçalarını büyüklüğe göre sırala
potential_body_parts.sort(key=lambda x: x['area'], reverse=True)

# Basit vücut iskelet sistemi simülasyonu
def draw_body_skeleton(image, body_parts):
    if len(body_parts) >= 1:
        # Ana vücut (en büyük parça)
        main_body = body_parts[0]
        x, y, w, h = main_body['bbox']
        
        # Vücut merkezini hesapla
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Baş tahmini (vücudun üst kısmı)
        head_y = y - h // 4
        head_x = center_x
        
        # Basit iskelet noktaları
        # Omuzlar
        left_shoulder = (center_x - w // 3, y + h // 6)
        right_shoulder = (center_x + w // 3, y + h // 6)
        
        # Kalça
        left_hip = (center_x - w // 4, y + 2 * h // 3)
        right_hip = (center_x + w // 4, y + 2 * h // 3)
        
        # Dirsekler (tahmin)
        left_elbow = (center_x - w // 2, y + h // 3)
        right_elbow = (center_x + w // 2, y + h // 3)
        
        # Eller (tahmin)
        left_hand = (center_x - 2 * w // 3, y + h // 2)
        right_hand = (center_x + 2 * w // 3, y + h // 2)
        
        # Dizler (tahmin)
        left_knee = (center_x - w // 6, y + 5 * h // 6)
        right_knee = (center_x + w // 6, y + 5 * h // 6)
        
        # Ayaklar
        left_foot = (center_x - w // 4, y + h)
        right_foot = (center_x + w // 4, y + h)
        
        # Baş
        cv2.circle(image, (head_x, head_y), 15, (0, 255, 255), -1)
        cv2.putText(image, 'Bas', (head_x - 15, head_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Vücut eklemleri
        joints = [
            (left_shoulder, 'Sol Omuz'),
            (right_shoulder, 'Sag Omuz'), 
            (left_hip, 'Sol Kalca'),
            (right_hip, 'Sag Kalca'),
            (left_elbow, 'Sol Dirsek'),
            (right_elbow, 'Sag Dirsek'),
            (left_hand, 'Sol El'),
            (right_hand, 'Sag El'),
            (left_knee, 'Sol Diz'),
            (right_knee, 'Sag Diz'),
            (left_foot, 'Sol Ayak'),
            (right_foot, 'Sag Ayak')
        ]
        
        # Eklemleri çiz
        for joint, label in joints:
            cv2.circle(image, joint, 5, (0, 255, 0), -1)
            cv2.putText(image, label, (joint[0] + 10, joint[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Vücut bağlantıları (iskelet çizgileri)
        connections = [
            # Gövde
            (left_shoulder, right_shoulder),  # Omuz çizgisi
            (left_shoulder, left_hip),        # Sol gövde
            (right_shoulder, right_hip),      # Sağ gövde
            (left_hip, right_hip),            # Kalça çizgisi
            
            # Kollar
            (left_shoulder, left_elbow),      # Sol üst kol
            (left_elbow, left_hand),          # Sol alt kol
            (right_shoulder, right_elbow),    # Sağ üst kol
            (right_elbow, right_hand),        # Sağ alt kol
            
            # Bacaklar
            (left_hip, left_knee),            # Sol üst bacak
            (left_knee, left_foot),           # Sol alt bacak
            (right_hip, right_knee),          # Sağ üst bacak
            (right_knee, right_foot),         # Sağ alt bacak
            
            # Baş-gövde bağlantısı
            ((head_x, head_y), (center_x, y + h // 6))
        ]
        
        # Bağlantı çizgilerini çiz
        for start, end in connections:
            cv2.line(image, start, end, (255, 0, 0), 2)
        
        # Ana vücut konturunu çiz
        cv2.drawContours(image, [main_body['contour']], -1, (255, 255, 0), 2)

# Tespit edilen vücut parçaları
if potential_body_parts:
    print(f"{len(potential_body_parts)} potansiyel vücut parçası tespit edildi.")
    draw_body_skeleton(image, potential_body_parts)
else:
    print("Vücut parçası tespit edilemedi. Parametreleri ayarlayın.")

# Görseli göster
cv2.imshow("Vucut Tespiti ve Iskelet", image)
print("Gösterilen pencereyi kapatmak için herhangi bir tuşa basın...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Program başarıyla tamamlandı!")
