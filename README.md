# 🎙️ VoicePrint: Ses Parmak İzi ve Karşılaştırma Sistemi

Bu proje, mikrofondan alınan ses kayıtlarının akustik özelliklerini analiz ederek, iki farklı ses kaydının aynı kişiye ait olup olmadığını tespit etmeye çalışan Python tabanlı bir ses işleme aracıdır.

## 🚀 Özellikler

* **Kapsamlı Akustik Analiz:** Ses verisinden MFCC, Spectral Centroid, Rolloff, Zero Crossing Rate (ZCR) ve Chroma gibi detaylı özellikleri çıkarır (`librosa` altyapısı ile).
* **Grup Bazlı L2 Normalizasyon:** Farklı veri ölçeklerine sahip özelliklerin (örneğin ~5000 değerindeki Rolloff ile ~0.05 değerindeki ZCR) birbirini ezmesini önleyerek adil bir karşılaştırma sunar.
* **Akıllı Gürültü Filtreleme:** Güvenilirlik eşiğinin (L2 Norm) altında kalan anlamsız verileri tespit eder ve analiz dışı bırakarak hata payını düşürür.
* **Detaylı Raporlama:** Terminal üzerinden çalışırken her bir akustik özelliğin benzerlik oranını yüzdesel olarak ve bar grafiği formatında raporlar.

## 🤖 Geliştirme Süreci ve AI Destekli Kodlama

Bu projenin hayata geçirilmesi sürecinde; algoritmaların yapılandırılması, kodun modüler hale getirilmesi ve özellikle farklı ölçekteki akustik verilerin birbirini ezmesini engelleyen "Grup Bazlı L2 Normalizasyon" mantığının matematiksel olarak kurgulanması aşamalarında modern **Yapay Zeka (AI)** araçlarından aktif olarak destek alınmıştır. Geliştirme süreci, insan mantığı ile AI asistanlığının ortak bir ürünüdür.

## 🧰 Kullanılan Teknolojiler

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-FF6633?style=for-the-badge)

## ⚙️ Kurulum

Projeyi çalıştırmak için gerekli kütüphaneleri sisteminize yükleyin:

```bash
pip install numpy sounddevice soundfile librosa scipy matplotlib
