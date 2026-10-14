import cv2
import numpy as np
import os
import glob

def detect_human_pose_batch_analysis(img_folder):
    """
    Tüm fotoğraflar üzerinde pose detection analizi yapar ve sonuçları karşılaştırır
    """
    
    # Desteklenen resim formatları
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    # Tüm resim dosyalarını bul
    for extension in image_extensions:
        pattern = os.path.join(img_folder, extension)
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"HATA: {img_folder} klasöründe resim dosyası bulunamadı!")
        return
    
    print(f"=== TOPLU ANALİZ BAŞLIYOR ===")
    print(f"Klasör: {img_folder}")
    print(f"Bulunan resim sayısı: {len(image_files)}")
    print("=" * 60)
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n[{i}/{len(image_files)}] İşleniyor: {filename}")
        
        try:
            # Resmi yükle
            image = cv2.imread(image_path)
            if image is None:
                print(f"   HATA: {filename} yüklenemedi!")
                continue
            
            # Algoritma analizi
            analysis_result = analyze_single_image(image, filename)
            results.append(analysis_result)
            
            # Kısa özet
            print(f"   Boyut: {analysis_result['dimensions']}")
            print(f"   En-boy oranı: {analysis_result['aspect_ratio']:.2f}")
            print(f"   Vücut tipi: {analysis_result['body_type']}")
            print(f"   Tespit kalitesi: {analysis_result['detection_quality']}")
            
        except Exception as e:
            print(f"   HATA: {filename} işlenirken hata oluştu: {str(e)}")
    
    # Genel analiz raporu
    print("\n" + "=" * 60)
    print("=== GENEL ANALİZ RAPORU ===")
    
    if results:
        # İstatistikler
        total_images = len(results)
        successful_detections = sum(1 for r in results if r['detection_success'])
        
        print(f"Toplam işlenen resim: {total_images}")
        print(f"Başarılı tespit: {successful_detections} ({successful_detections/total_images*100:.1f}%)")
        
        # Vücut tipi dağılımı
        body_types = {}
        for result in results:
            bt = result['body_type']
            body_types[bt] = body_types.get(bt, 0) + 1
        
        print("\nVücut tipi dağılımı:")
        for body_type, count in body_types.items():
            print(f"  {body_type}: {count} adet ({count/total_images*100:.1f}%)")
        
        # En-boy oranı analizi
        aspect_ratios = [r['aspect_ratio'] for r in results if r['detection_success']]
        if aspect_ratios:
            print(f"\nEn-boy oranı istatistikleri:")
            print(f"  Minimum: {min(aspect_ratios):.2f}")
            print(f"  Maksimum: {max(aspect_ratios):.2f}")
            print(f"  Ortalama: {np.mean(aspect_ratios):.2f}")
        
        # Tespit kalitesi analizi
        quality_counts = {}
        for result in results:
            quality = result['detection_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        print("\nTespit kalitesi dağılımı:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} adet ({count/total_images*100:.1f}%)")
        
        # Sorunlu dosyalar
        poor_detections = [r for r in results if r['detection_quality'] == 'ZAYIF']
        if poor_detections:
            print(f"\nSorunlu tespitler ({len(poor_detections)} adet):")
            for result in poor_detections:
                print(f"  {result['filename']}: {result['issues']}")
        
        # Öneriler
        print("\n=== ALGORİTMA GELİŞTİRME ÖNERİLERİ ===")
        
        # En-boy oranı eşikleri analizi
        upper_body_count = sum(1 for r in results if r['body_type'] == 'ÜST VÜCUT')
        full_body_count = sum(1 for r in results if r['body_type'] == 'TAM VÜCUT')
        
        if aspect_ratios:
            avg_ratio = np.mean(aspect_ratios)
            if avg_ratio > 1.8:
                print("1. En-boy oranı eşiklerini yükseltmek gerekebilir")
            elif avg_ratio < 1.3:
                print("1. En-boy oranı eşiklerini düşürmek gerekebilir")
        
        if len(poor_detections) > total_images * 0.3:
            print("2. Kontur filtreleme parametrelerini iyileştirmek gerekiyor")
        
        if upper_body_count > full_body_count * 2:
            print("3. Veri setinde çoğunlukla üst vücut fotoğrafları var - parametreler buna göre ayarlanabilir")
    
    return results

def analyze_single_image(image, filename):
    """
    Tek bir resim için detaylı analiz yapar
    """
    result = {
        'filename': filename,
        'dimensions': f"{image.shape[1]}x{image.shape[0]}",
        'detection_success': False,
        'body_type': 'BİLİNMEYEN',
        'aspect_ratio': 0.0,
        'detection_quality': 'ZAYIF',
        'issues': []
    }
    
    try:
        # Ön işleme
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Kenar tespiti
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Kontur analizi
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            result['issues'].append("Kontur bulunamadı")
            return result
        
        # En büyük konturu bul
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Temel kontroller
        if area < 500:
            result['issues'].append("Kontur alanı çok küçük")
            return result
        
        # En-boy oranı
        aspect_ratio = h / w if w > 0 else 0
        result['aspect_ratio'] = aspect_ratio
        
        if not (1.2 <= aspect_ratio <= 4.0):
            result['issues'].append(f"En-boy oranı normal aralık dışında: {aspect_ratio:.2f}")
        
        # Düzgünlük kontrol
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < 0.3:
            result['issues'].append(f"Şekil çok karmaşık (düzgünlük: {solidity:.2f})")
        
        # Vücut tipi belirleme
        if aspect_ratio > 2.8:
            result['body_type'] = 'BAŞ-OMUZ'
        elif aspect_ratio > 1.4:
            result['body_type'] = 'ÜST VÜCUT'
        else:
            result['body_type'] = 'TAM VÜCUT'
        
        # Tespit kalitesi belirleme
        quality_score = 0
        
        # Alan büyüklüğü (resmin ne kadarını kaplıyor)
        image_area = image.shape[0] * image.shape[1]
        coverage = area / image_area
        if coverage > 0.1:
            quality_score += 1
        
        # En-boy oranı makullüğü
        if 1.3 <= aspect_ratio <= 3.0:
            quality_score += 1
        
        # Düzgünlük
        if solidity > 0.4:
            quality_score += 1
        
        # Boyut uygunluğu
        if w > 50 and h > 100:
            quality_score += 1
        
        if quality_score >= 3:
            result['detection_quality'] = 'İYİ'
        elif quality_score >= 2:
            result['detection_quality'] = 'ORTA'
        else:
            result['detection_quality'] = 'ZAYIF'
        
        result['detection_success'] = True
        
        # Ek sorun tespitleri
        if coverage < 0.05:
            result['issues'].append("Vücut resimde çok küçük görünüyor")
        
        if w < 30 or h < 60:
            result['issues'].append("Tespit edilen vücut çok küçük")
        
        if aspect_ratio > 4.0:
            result['issues'].append("Sadece baş/boyun görünüyor olabilir")
        elif aspect_ratio < 1.2:
            result['issues'].append("Yatay pozisyon veya kırpılmış resim olabilir")
        
    except Exception as e:
        result['issues'].append(f"İşleme hatası: {str(e)}")
    
    return result

if __name__ == "__main__":
    # img klasörünü kontrol et
    img_folder = "img"
    
    if not os.path.exists(img_folder):
        print(f"HATA: '{img_folder}' klasörü bulunamadı!")
        print("Mevcut klasörde bir 'img' klasörü oluşturun ve test resimlerini oraya koyun.")
    else:
        results = detect_human_pose_batch_analysis(img_folder)
        
        # Sonuçları dosyaya kaydet
        if results:
            with open("pose_detection_analysis_report.txt", "w", encoding="utf-8") as f:
                f.write("POSE DETECTION ANALIZ RAPORU\n")
                f.write("=" * 40 + "\n\n")
                
                for result in results:
                    f.write(f"Dosya: {result['filename']}\n")
                    f.write(f"Boyut: {result['dimensions']}\n")
                    f.write(f"En-boy oranı: {result['aspect_ratio']:.2f}\n")
                    f.write(f"Vücut tipi: {result['body_type']}\n")
                    f.write(f"Tespit kalitesi: {result['detection_quality']}\n")
                    if result['issues']:
                        f.write(f"Sorunlar: {', '.join(result['issues'])}\n")
                    f.write("-" * 30 + "\n")
            
            print(f"\nDetaylı rapor 'pose_detection_analysis_report.txt' dosyasına kaydedildi.")
        
        print("\nAnaliz tamamlandı!")
