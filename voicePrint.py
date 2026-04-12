import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


# ============================================================
# SES KAYDETME FONKSİYONU
# ============================================================

def kaydet(kayit_adi):
    """Mikrofondan ses kaydeder ve .wav dosyası olarak kaydeder."""
    sure = 10
    orneklem_hizi = 22050

    print("🎙️ Kayıt başlıyor... 10 saniye konuş!")
    kayit = sd.rec(
        frames=int(sure * orneklem_hizi),
        samplerate=orneklem_hizi,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("✅ Kayıt bitti! Dosyaya kaydediliyor...")

    dosya_yolu = f"{kayit_adi}.wav"
    sf.write(dosya_yolu, kayit, orneklem_hizi)
    print(f"💾 '{dosya_yolu}' olarak kaydedildi!")

    return dosya_yolu


# ============================================================
# ADIM 1: ÖZELLİK ÇIKARMA FONKSİYONU
# ============================================================
# Bir sesten tüm ham özellikleri çıkarır ve bir liste olarak döndürür.
# Her özellik bir matristir: (özellik_sayısı, zaman_adımları)
# Ortalamaları burada ALMIYORUZ — bu işi vektore_cevir() yapacak.
# ============================================================

def ozellikleri_cikar(ses, sr):
    """
    Bir ses verisinden tüm akustik özellikleri çıkarır.

    Parametreler:
        ses → NumPy dizisi (librosa.load ile yüklenen ses verisi)
        sr  → Örnekleme hızı (sampling rate)

    Döndürür:
        ozellikler → liste: her eleman bir özellik matrisi
    """

    # MFCC + türevleri (13+13+13 = 39 değer)
    mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=13)           # (13, t)
    delta_mfcc = librosa.feature.delta(mfcc)                        # (13, t)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)              # (13, t)

    # Spectral (frekans) özellikleri
    spectral_centroid = librosa.feature.spectral_centroid(y=ses, sr=sr)   # (1, t)
    spectral_bw = librosa.feature.spectral_bandwidth(y=ses, sr=sr)       # (1, t)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=ses, sr=sr)    # (1, t)
    spectral_contrast = librosa.feature.spectral_contrast(y=ses, sr=sr)  # (7, t)
    spectral_flatness = librosa.feature.spectral_flatness(y=ses)         # (1, t)

    # Zaman bazlı özellikler
    zcr = librosa.feature.zero_crossing_rate(y=ses)    # (1, t)
    rms = librosa.feature.rms(y=ses)                   # (1, t)

    # Tonal özellik
    chroma = librosa.feature.chroma_stft(y=ses, sr=sr)  # (12, t)

    # Hepsini bir listede topla ve döndür
    return [
        mfcc, delta_mfcc, delta2_mfcc,
        spectral_centroid, spectral_bw, spectral_rolloff,
        spectral_contrast, spectral_flatness,
        zcr, rms,
        chroma,
    ]


# ============================================================
# ADIM 2: NORMALİZASYON FONKSİYONU (GRUP BAZLI)
# ============================================================
# PROBLEM:
#   spectral_rolloff ortalaması  ≈ 5000    (çok büyük sayı)
#   zcr ortalaması               ≈ 0.05    (çok küçük sayı)
#
#   Tek bir cosine similarity hesaplarken 5000 her şeyi EZİYOR,
#   zcr'nin hiç etkisi KALMIYOR!
#
# ÇÖZÜM: Grup bazlı L2 Normalization
#   Her özellik grubunu (MFCC, chroma, vb.) KENDİ büyüklüğüne böl.
#   L2 norm = vektörün uzunluğu = sqrt(x1² + x2² + ... + xn²)
#
#   Böylece her grup birim uzunluğa (1.0) getirilir ve
#   hiçbir grup diğerini ezemez.
# ============================================================

def grup_l2_normalize(ozellik_listesi):
    """
    Her özellik grubunun ortalamasını ve standart sapmasını alır.
    Çok değerli grupları L2 normalize eder.
    Tek değerli grupları ham bırakır.

    Ayrıca her grubun orijinal büyüklüğünü (L2 norm) kaydeder.
    Bu büyüklük = güvenilirlik göstergesi:
      Büyük norm → güvenilir veri (ör: MFCC ~370)
      Küçük norm → gürültü, rastgele (ör: Delta MFCC ort ~2)
    """

    parcalar = []
    grup_boyutlari = []
    grup_normlari = []     # Orijinal büyüklükler (güvenilirlik)

    for ozellik in ozellik_listesi:
        ort = np.mean(ozellik, axis=1)
        std = np.std(ozellik, axis=1)

        # Orijinal büyüklükleri kaydet (normalize ETMEDEN ÖNCE)
        ort_norm = np.linalg.norm(ort)
        std_norm = np.linalg.norm(std)

        if len(ort) > 1:
            # Çok değerli grup → L2 normalize
            if ort_norm > 0:
                ort = ort / ort_norm
            if std_norm > 0:
                std = std / std_norm

        parcalar.append(ort)
        parcalar.append(std)
        grup_boyutlari.append(len(ort))
        grup_boyutlari.append(len(std))
        grup_normlari.append(ort_norm)
        grup_normlari.append(std_norm)

    parmak_izi = np.concatenate(parcalar)
    return parmak_izi, grup_boyutlari, grup_normlari


# ============================================================
# ADIM 3: VEKTÖRE ÇEVİRME FONKSİYONU
# ============================================================

def vektore_cevir(ozellik_listesi):
    """
    Özellik matrislerinin listesini alıp normalize parmak izi vektörüne çevirir.
    """
    parmak_izi, grup_boyutlari, grup_normlari = grup_l2_normalize(ozellik_listesi)
    return parmak_izi, grup_boyutlari, grup_normlari


# ============================================================
# ADIM 4: PARMAK İZİ OLUŞTURMA (Ana fonksiyon)
# ============================================================

def parmak_izi_olustur(ses, sr):
    """
    Bir ses verisinden normalize edilmiş parmak izi çıkarır.
    """
    ozellikler = ozellikleri_cikar(ses, sr)
    parmak_izi, grup_boyutlari, grup_normlari = vektore_cevir(ozellikler)
    return parmak_izi, grup_boyutlari, grup_normlari


# ============================================================
# SESLERİ KARŞILAŞTIRMA FONKSİYONU (GELİŞMİŞ)
# ============================================================
# ESKI YÖNTEM: Tek büyük vektörde cosine similarity → %99.7
#   Sorun: büyük ölçekli özellikler küçükleri eziyor
#
# YENİ YÖNTEM: Her grup için AYRI cosine similarity hesapla,
#   sonra ortalamasını al. Böylece her grubun eşit oy hakkı var!
#
#   Örnek:  MFCC benzerliği     = %98   → 1 oy
#           Delta MFCC benzerliği = %-53  → 1 oy
#           Chroma benzerliği    = %99   → 1 oy
#           ...
#           Genel sonuç = ortalamaları
# ============================================================

# Özellik grubu adları (teşhis çıktısı için)
OZELLIK_ADLARI = [
    "MFCC ort", "MFCC std",
    "Delta MFCC ort", "Delta MFCC std",
    "Delta2 MFCC ort", "Delta2 MFCC std",
    "Centroid ort", "Centroid std",
    "Bandwidth ort", "Bandwidth std",
    "Rolloff ort", "Rolloff std",
    "Contrast ort", "Contrast std",
    "Flatness ort", "Flatness std",
    "ZCR ort", "ZCR std",
    "RMS ort", "RMS std",
    "Chroma ort", "Chroma std",
]

# Güvenilirlik eşiği: L2 normu bundan küçük olan gruplar
# gürültü sayılır ve skora KATILMAZ.
# Delta MFCC ortalaması ~2-5 civarı (gürültü), MFCC ortalaması ~370 (güvenilir)
GUVENILIRLIK_ESIGI = 10.0

def sesleri_karsilastir(dosya_1, dosya_2):
    """
    Iki ses dosyasini grup bazli karsilastirir.
    Guvenilir gruplarin skorlarini ortalar.
    Gurultu olan gruplari atlar.
    """

    # Dosyalari yukle
    ses_1, sr1 = librosa.load(dosya_1, sr=22050)
    ses_2, sr2 = librosa.load(dosya_2, sr=22050)

    # Parmak izlerini olustur
    parmak_izi_1, grup_boyutlari, normlar_1 = parmak_izi_olustur(ses_1, sr1)
    parmak_izi_2, _, normlar_2 = parmak_izi_olustur(ses_2, sr2)

    # --- GRUP BAZLI KARSILASTIRMA ---
    grup_skorlari = []
    idx = 0

    print(f"\n{'='*55}")
    print(f"  DETAYLI ANALIZ RAPORU")
    print(f"{'='*55}")

    for i, boyut in enumerate(grup_boyutlari):
        grup_1 = parmak_izi_1[idx:idx + boyut]
        grup_2 = parmak_izi_2[idx:idx + boyut]

        # Guvenilirlik kontrolu: iki sesin de normu yeterince buyuk mu?
        # Kucuk norm = sifira yakin deger = rastgele yon = GURULTU
        norm_1 = normlar_1[i]
        norm_2 = normlar_2[i]
        guvenilir = (norm_1 > GUVENILIRLIK_ESIGI) and (norm_2 > GUVENILIRLIK_ESIGI)

        if not guvenilir and boyut > 1:
            # Gurultu grubu → skora KATMA
            ad = OZELLIK_ADLARI[i] if i < len(OZELLIK_ADLARI) else f"Grup {i}"
            print(f"  {ad:<20}   --   [ATLA: dusuk sinyal]")
            idx += boyut
            continue

        if boyut == 1:
            # Tek degerli ozellikler → yuzdesel fark
            buyuk = max(abs(grup_1[0]), abs(grup_2[0]))
            if buyuk > 0:
                skor = 1.0 - abs(grup_1[0] - grup_2[0]) / buyuk
            else:
                skor = 1.0
        else:
            # Cok degerli gruplar → cosine similarity
            skor = 1 - cosine(grup_1, grup_2)

        grup_skorlari.append(skor)

        # Grup adini yazdir
        ad = OZELLIK_ADLARI[i] if i < len(OZELLIK_ADLARI) else f"Grup {i}"
        bar = "#" * max(0, int(skor * 20))
        print(f"  {ad:<20} {skor*100:5.1f}%  {bar}")

        idx += boyut

    # Genel skor = sadece guvenilir gruplarin ortalamasi
    genel_benzerlik = np.mean(grup_skorlari) if grup_skorlari else 0.0

    print(f"\n{'='*55}")
    print(f"  GENEL BENZERLIK:  %{genel_benzerlik * 100:.1f}")
    print(f"  Kullanilan grup:   {len(grup_skorlari)} / {len(grup_boyutlari)}")
    print(f"{'='*55}")

    if genel_benzerlik > 0.80:
        print("  >> Buyuk ihtimalle AYNI KISI!")
    elif genel_benzerlik > 0.55:
        print("  >> Benzer ama kesin degil.")
    else:
        print("  >> Muhtemelen FARKLI kisiler.")

    return genel_benzerlik


# ============================================================
# ANA MENÜ
# ============================================================
# if __name__ == "__main__" ne demek?
#   → Bu dosya doğrudan çalıştırıldığında (python voicePrint.py)
#     menü kodu çalışır.
#   → Ama başka bir dosyadan import edildiğinde
#     (from voicePrint import ozellikleri_cikar) menü ÇALIŞMAZ,
#     sadece fonksiyonlar kullanılabilir.
# ============================================================

if __name__ == "__main__":
    dosya_1 = None
    dosya_2 = None

    while True:
        print("\n" + "=" * 40)
        print("Ses Parmak Izi Uygulamasi")
        print("=" * 40)
        print("1 - Ilk Sesi Kaydet")
        print("2 - Ikinci Sesi Kaydet")
        print("3 - Sesleri Karsilastir")
        print("4 - Cikis")

        secim = input("Seciminizi yapiniz: ")

        if secim == "1":
            kayit_adi = input("Kaydiniza bir isim veriniz: ")
            dosya_1 = kaydet(kayit_adi)

        elif secim == "2":
            kayit_adi = input("Kaydiniza bir isim veriniz: ")
            dosya_2 = kaydet(kayit_adi)

        elif secim == "3":
            if dosya_1 is None or dosya_2 is None:
                print("Once iki ses kaydi da yapmalisin! (Secenek 1 ve 2)")
            else:
                sesleri_karsilastir(dosya_1, dosya_2)

        elif secim == "4":
            print("Cikis yapiliyor...")
            break

        else:
            print("Gecersiz secim! 1-4 arasi bir sayi gir.")
