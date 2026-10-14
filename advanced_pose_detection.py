import cv2
import numpy as np

def detect_human_pose(image_path):
    """
    Gelişmiş vücut tespit algoritması
    
    Algoritma Açıklaması:
    1. Resmi gri tonlamaya çevirir
    2. Histogram eşitleme ile kontrast artırır
    3. Morfolojik işlemlerle gürültüyü azaltır
    4. Kenar tespiti yapar
    5. Konturları bulup insan vücuduna uygun olanları filtreler
    6. Vücut oranlarına göre eklem noktalarını tahmin eder
    """
    
    # Görseli yükle
    image = cv2.imread(image_path)
    if image is None:
        print("Hata: Resim dosyası yüklenemedi.")
        return None
    
    print(f"Resim boyutu: {image.shape}")
    original = image.copy()
    
    # 1. Ön İşleme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitleme (kontrast artırma)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Kenar Tespiti ve Morfolojik İşlemler
    # Adaptive threshold (değişken eşikleme)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # 3. Kontur Analizi
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # İnsan vücudu kriterlerine göre filtreleme
    body_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Çok küçük alanları eleme
            continue
            
        # Sınırlayıcı dikdörtgen
        x, y, w, h = cv2.boundingRect(contour)
        
        # İnsan vücudu oranları (en-boy oranı)
        aspect_ratio = h / w if w > 0 else 0
        
        # İnsan vücudu genellikle 1.5-3.5 en-boy oranına sahiptir
        if 0.3 <= aspect_ratio <= 4.0:
            # Konturin düzgünlüğünü kontrol et
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Çok karmaşık şekilleri eleme
            if solidity > 0.3:
                body_candidates.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity
                })
    
    # En uygun vücut adayını seç (en büyük alan)
    if not body_candidates:
        print("Vücut tespit edilemedi!")
        return image
    
    best_body = max(body_candidates, key=lambda x: x['area'])
    x, y, w, h = best_body['bbox']
    
    print(f"Tespit edilen vücut: En-boy oranı={best_body['aspect_ratio']:.2f}, "
          f"Alan={best_body['area']}, Düzgünlük={best_body['solidity']:.2f}")
    
    # 4. Akıllı Vücut Parçası Analizi ve Eklem Tespiti
    # Fotoğraftaki görünen vücut kısmını analiz et
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Vücudun hangi kısmının görüldüğünü analiz et
    # En-boy oranına ve konuma göre vücut kısmını belirle
    
    # Analiz sonuçlarına göre düzeltilmiş parametreler
    is_upper_body_only = best_body['aspect_ratio'] > 1.8  # Artırıldı: 1.4 → 1.8
    is_head_shoulders_only = best_body['aspect_ratio'] > 3.2  # Artırıldı: 2.8 → 3.2  
    is_full_body = best_body['aspect_ratio'] <= 1.8  # Artırıldı: 1.4 → 1.8
    
    # Yatay/Panoramik resim kontrolü ekle
    is_horizontal_crop = best_body['aspect_ratio'] < 0.8  # Çok yatay resimler
    
    visible_joints = []
    visible_connections = []
    
    if is_horizontal_crop:
        print("Tespit: Yatay/Panoramik resim - Özel işlem gerekiyor")
        # Yatay resimler için basit eklem yerleştirme
        head_center = (x + w//10, y + h//4)
        head_radius = min(w // 20, h // 6)
        
        neck = (x + w//8, y + h//3)
        left_shoulder = (x + w//6, y + h//2)
        right_shoulder = (x + w//4, y + h//2)
        chest_center = (x + w//5, y + 2*h//3)
        
        # Kollar yatay pozisyonda
        left_elbow = (x + w//3, y + h//2)
        right_elbow = (x + w//2, y + h//2)
        left_hand = (x + 2*w//3, y + h//2)
        right_hand = (x + 3*w//4, y + h//2)
        
        visible_joints = [
            (head_center, 'BAS', (0, 255, 255), head_radius, True),
            (neck, 'BOYUN', (255, 0, 255), 4, False),
            (left_shoulder, 'SOL OMUZ', (0, 255, 0), 5, False),
            (right_shoulder, 'SAG OMUZ', (0, 255, 0), 5, False),
            (chest_center, 'GOGUS', (255, 100, 0), 5, False),
            (left_elbow, 'SOL DIRSEK', (0, 255, 0), 4, False),
            (right_elbow, 'SAG DIRSEK', (0, 255, 0), 4, False),
            (left_hand, 'SOL EL', (0, 255, 0), 4, False),
            (right_hand, 'SAG EL', (0, 255, 0), 4, False)
        ]
        
        visible_connections = [
            (neck, left_shoulder, (255, 255, 255)),
            (neck, right_shoulder, (255, 255, 255)),
            (left_shoulder, right_shoulder, (255, 255, 255)),
            (left_shoulder, chest_center, (255, 255, 255)),
            (right_shoulder, chest_center, (255, 255, 255)),
            (left_shoulder, left_elbow, (0, 255, 0)),
            (left_elbow, left_hand, (0, 255, 0)),
            (right_shoulder, right_elbow, (0, 255, 0)),
            (right_elbow, right_hand, (0, 255, 0))
        ]
        
    elif is_head_shoulders_only:
        print("Tespit: Sadece baş ve omuzlar görünüyor")
        # Sadece baş, boyun, omuzlar
        neck = (center_x, y + h // 4)
        head_center = (center_x, y - h // 8)
        head_radius = min(w // 4, h // 6)
        
        left_shoulder = (center_x - w // 3, y + h // 3)
        right_shoulder = (center_x + w // 3, y + h // 3)
        
        visible_joints = [
            (head_center, 'BAS', (0, 255, 255), head_radius, True),  # True = daire çiz
            (neck, 'BOYUN', (255, 0, 255), 4, False),
            (left_shoulder, 'SOL OMUZ', (0, 255, 0), 4, False),
            (right_shoulder, 'SAG OMUZ', (0, 255, 0), 4, False)
        ]
        
        visible_connections = [
            (neck, left_shoulder, (255, 255, 255)),
            (neck, right_shoulder, (255, 255, 255)),
            (left_shoulder, right_shoulder, (255, 255, 255))
        ]
        
    elif is_upper_body_only:
        print("Tespit: Üst vücut (belden yukarı) görünüyor")
        
        # Gerçek kontur analizi ile eklem noktalarını bul
        # Konturun en üst, sol, sağ ve alt noktalarını bul
        contour = best_body['contour']
        
        # Konturun ekstrem noktalarını bul
        topmost = tuple(contour[contour[:,:,1].argmin()][0])      # En üst nokta (baş)
        bottommost = tuple(contour[contour[:,:,1].argmax()][0])   # En alt nokta
        leftmost = tuple(contour[contour[:,:,0].argmin()][0])     # En sol nokta
        rightmost = tuple(contour[contour[:,:,0].argmax()][0])    # En sağ nokta
        
        # Konturun çeşitli yükseklik seviyelerindeki genişlik noktalarını analiz et
        height_levels = []
        for i in range(0, 10):  # 10 seviyeye böl
            level_y = y + (i * h // 10)
            level_points = []
            
            # Bu seviyedeki kontur noktalarını bul
            for point in contour:
                if abs(point[0][1] - level_y) < h//20:  # Tolerans
                    level_points.append(point[0])
            
            if level_points:
                level_points.sort(key=lambda p: p[0])  # X koordinatına göre sırala
                left_point = level_points[0]
                right_point = level_points[-1]
                width = right_point[0] - left_point[0]
                height_levels.append({
                    'y': level_y,
                    'left': left_point,
                    'right': right_point,
                    'width': width,
                    'center': ((left_point[0] + right_point[0])//2, level_y)
                })
        
        # Akıllı eklem tespiti
        if len(height_levels) >= 5:
            # Baş (en üst seviye)
            head_center = topmost
            head_radius = min(w // 8, h // 12)
            
            # Boyun (üst %20 seviyesinde)
            neck_level = next((hl for hl in height_levels if hl['y'] > y + h//5), height_levels[1])
            neck = neck_level['center']
            
            # Omuzlar - en geniş seviyeyi bul (genellikle omuz seviyesi)
            widest_level = max(height_levels[:6], key=lambda hl: hl['width'])
            left_shoulder = widest_level['left']
            right_shoulder = widest_level['right']
            
            # Göğüs merkezi (omuz seviyesinin biraz altı)
            chest_level_idx = min(len(height_levels)-1, height_levels.index(widest_level) + 2)
            chest_center = height_levels[chest_level_idx]['center']
            
            # Dirsek tespiti - konturun çıkıntı noktalarını bul
            # Orta seviyede en dış noktalar muhtemelen dirsekler
            mid_level_idx = len(height_levels) // 2
            if mid_level_idx < len(height_levels):
                elbow_level = height_levels[mid_level_idx]
                left_elbow = elbow_level['left']
                right_elbow = elbow_level['right']
            else:
                # Fallback
                left_elbow = (leftmost[0], center_y)
                right_elbow = (rightmost[0], center_y)
            
            # El tespiti - alt seviyedeki en dış noktalar
            if len(height_levels) >= 3:
                hand_level = height_levels[-3]  # Alt seviyelerden biri
                left_hand = hand_level['left']
                right_hand = hand_level['right']
            else:
                left_hand = (leftmost[0], bottommost[1] - h//10)
                right_hand = (rightmost[0], bottommost[1] - h//10)
        
        else:
            # Fallback - basit hesaplama
            head_center = (center_x, max(0, y - h // 12))
            head_radius = min(w // 6, h // 10)
            neck = (center_x, y + h // 8)
            left_shoulder = (center_x - w // 3, y + h // 4)
            right_shoulder = (center_x + w // 3, y + h // 4)
            chest_center = (center_x, y + h // 2)
            left_elbow = leftmost
            right_elbow = rightmost
            left_hand = (leftmost[0], bottommost[1] - h//10)
            right_hand = (rightmost[0], bottommost[1] - h//10)
        
        visible_joints = [
            (head_center, 'BAS', (0, 255, 255), head_radius, True),
            (neck, 'BOYUN', (255, 0, 255), 4, False),
            (left_shoulder, 'SOL OMUZ', (0, 255, 0), 5, False),
            (right_shoulder, 'SAG OMUZ', (0, 255, 0), 5, False),
            (chest_center, 'GOGUS', (255, 100, 0), 5, False),
            (left_elbow, 'SOL DIRSEK', (0, 255, 0), 4, False),
            (right_elbow, 'SAG DIRSEK', (0, 255, 0), 4, False),
            (left_hand, 'SOL EL', (0, 255, 0), 4, False),
            (right_hand, 'SAG EL', (0, 255, 0), 4, False)
        ]
        
        visible_connections = [
            # Gövde
            (neck, left_shoulder, (255, 255, 255)),
            (neck, right_shoulder, (255, 255, 255)),
            (left_shoulder, right_shoulder, (255, 255, 255)),
            (left_shoulder, chest_center, (255, 255, 255)),
            (right_shoulder, chest_center, (255, 255, 255)),
            
            # Kollar - gerçek pozisyona göre
            (left_shoulder, left_elbow, (0, 255, 0)),
            (left_elbow, left_hand, (0, 255, 0)),
            (right_shoulder, right_elbow, (0, 255, 0)),
            (right_elbow, right_hand, (0, 255, 0))
        ]
        
    else:  # Tam vücut
        print("Tespit: Tam vücut görünüyor")
        # Tüm eklemler (orijinal kod)
        head_center = (center_x, y - h // 16)
        head_radius = min(w // 6, h // 12)
        
        neck = (center_x, y + h // 20)
        
        shoulder_y = y + h // 6
        left_shoulder = (center_x - w // 3, shoulder_y)
        right_shoulder = (center_x + w // 3, shoulder_y)
        
        chest_center = (center_x, y + h // 4)
        
        hip_y = y + h // 2
        left_hip = (center_x - w // 6, hip_y)
        right_hip = (center_x + w // 6, hip_y)
        
        elbow_y = (shoulder_y + hip_y) // 2
        left_elbow = (center_x - w // 2, elbow_y)
        right_elbow = (center_x + w // 2, elbow_y)
        
        hand_y = hip_y
        left_hand = (center_x - 2 * w // 3, hand_y)
        right_hand = (center_x + 2 * w // 3, hand_y)
        
        knee_y = y + 3 * h // 4
        left_knee = (center_x - w // 8, knee_y)
        right_knee = (center_x + w // 8, knee_y)
        
        foot_y = y + h - h // 20
        left_foot = (center_x - w // 4, foot_y)
        right_foot = (center_x + w // 4, foot_y)
        
        visible_joints = [
            (head_center, 'BAS', (0, 255, 255), head_radius, True),
            (neck, 'BOYUN', (255, 0, 255), 4, False),
            (left_shoulder, 'SOL OMUZ', (0, 255, 0), 5, False),
            (right_shoulder, 'SAG OMUZ', (0, 255, 0), 5, False),
            (chest_center, 'GOGUS', (255, 100, 0), 5, False),
            (left_elbow, 'SOL DIRSEK', (0, 255, 0), 4, False),
            (right_elbow, 'SAG DIRSEK', (0, 255, 0), 4, False),
            (left_hand, 'SOL EL', (0, 255, 0), 4, False),
            (right_hand, 'SAG EL', (0, 255, 0), 4, False),
            (left_hip, 'SOL KALCA', (255, 0, 0), 5, False),
            (right_hip, 'SAG KALCA', (255, 0, 0), 5, False),
            (left_knee, 'SOL DIZ', (255, 0, 0), 4, False),
            (right_knee, 'SAG DIZ', (255, 0, 0), 4, False),
            (left_foot, 'SOL AYAK', (255, 0, 0), 4, False),
            (right_foot, 'SAG AYAK', (255, 0, 0), 4, False)
        ]
        
        visible_connections = [
            # Gövde
            (neck, left_shoulder, (255, 255, 255)),
            (neck, right_shoulder, (255, 255, 255)),
            (left_shoulder, right_shoulder, (255, 255, 255)),
            (left_shoulder, chest_center, (255, 255, 255)),
            (right_shoulder, chest_center, (255, 255, 255)),
            (chest_center, left_hip, (255, 255, 255)),
            (chest_center, right_hip, (255, 255, 255)),
            (left_hip, right_hip, (255, 255, 255)),
            
            # Kollar
            (left_shoulder, left_elbow, (0, 255, 0)),
            (left_elbow, left_hand, (0, 255, 0)),
            (right_shoulder, right_elbow, (0, 255, 0)),
            (right_elbow, right_hand, (0, 255, 0)),
            
            # Bacaklar
            (left_hip, left_knee, (255, 0, 0)),
            (left_knee, left_foot, (255, 0, 0)),
            (right_hip, right_knee, (255, 0, 0)),
            (right_knee, right_foot, (255, 0, 0))
        ]
    
    # 5. Çizim İşlemleri
    
    # Vücut konturunu çiz
    cv2.drawContours(image, [best_body['contour']], -1, (0, 255, 255), 2)
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    # Görünen eklemleri çiz
    for joint_data in visible_joints:
        if len(joint_data) == 5:  # Yeni format
            joint, label, color, size, is_circle = joint_data
            if is_circle:  # Baş için büyük daire
                cv2.circle(image, joint, size, color, 2)
                cv2.putText(image, label, (joint[0] - 15, joint[1] - size - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:  # Normal eklemler
                cv2.circle(image, joint, size, color, -1)
                cv2.putText(image, label, (joint[0] + 8, joint[1] - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Görünen bağlantıları çiz
    for start, end, color in visible_connections:
        cv2.line(image, start, end, color, 2)
    
    # Vücut tipi bilgisi
    body_type = ""
    if is_horizontal_crop:
        body_type = "YATAY/PANORAMİK"
    elif is_head_shoulders_only:
        body_type = "BAS-OMUZ"
    elif is_upper_body_only:
        body_type = "UST VUCUT"
    else:
        body_type = "TAM VUCUT"
    
    # Bilgi metni
    cv2.putText(image, f'Tespit: {body_type} - Alan: {best_body["area"]:.0f}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f'En-Boy Orani: {best_body["aspect_ratio"]:.2f}', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f'Gorunen Eklem Sayisi: {len(visible_joints)}', 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image

# Ana program
if __name__ == "__main__":
    print("=== GELİŞMİŞ VÜCUT TESPİT ALGORITMASI ===")
    print("\nAlgoritma Nasıl Çalışır:")
    print("1. Histogram eşitleme ile kontrast artırır")
    print("2. Adaptive threshold ile kenar tespiti yapar")
    print("3. Morfolojik işlemlerle gürültüyü temizler")
    print("4. Konturları bulup insan vücudu kriterlerine göre filtreler:")
    print("   - En-boy oranı: 0.3-4.0 (yatay resimler dahil)")
    print("   - Minimum alan: 500 piksel")
    print("   - Düzgünlük oranı: >0.3 (çok karmaşık şekilleri eleme)")
    print("5. Akıllı vücut analizi:")
    print("   - En-boy < 0.8: Yatay/Panoramik resim")
    print("   - En-boy > 3.2: Sadece baş-omuz")
    print("   - En-boy > 1.8: Üst vücut (belden yukarı)")
    print("   - En-boy ≤ 1.8: Tam vücut")
    print("6. Gerçek kontur analizi ile eklem tespiti:")
    print("   - Konturun ekstrem noktalarını bulur (en üst, alt, sol, sağ)")
    print("   - Farklı yükseklik seviyelerindeki genişlikleri analiz eder")
    print("   - En geniş seviye = omuz seviyesi")
    print("   - Çıkıntı noktalar = dirsek/el pozisyonları")
    print("7. Gerçek vücut pozisyonuna uygun iskelet çizer\n")
    
    result = detect_human_pose("img/insan.jpg")
    
    if result is not None:
        cv2.imshow("Gelismis Vucut Tespiti", result)
        print("Pencereyi kapatmak için herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Program başarıyla tamamlandı!")
