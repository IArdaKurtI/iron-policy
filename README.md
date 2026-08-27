# Iron Polcy v7

Iron Polcy v7, Leo ve T-90 adlı iki yapay zekâ tankını eğiten ve maçlarını aynı
uygulama penceresinde gösteren iki boyutlu bir simülasyondur.

## Windows'ta çalıştırma

1. GitHub'da `Code` → `Download ZIP` seçeneğiyle projeyi indirin.
2. ZIP dosyasını normal bir klasöre çıkarın. Programı ZIP'in içinden açmayın.
3. `iron polcy v7.vbs` dosyasına çift tıklayın.

İlk açılışta gerekli paketler otomatik kurulur; internet hızına göre birkaç dakika
sürebilir. Kurulumdan sonra uygulama açılır ve özel ikonlu `Iron Polcy v7`
kısayolu hem proje klasöründe hem Windows masaüstünde otomatik oluşturulur. Sonraki
açılışlarda masaüstündeki bu kısayol kullanılabilir.

Alternatif olarak `BASLAT.bat` dosyasına çift tıklayabilirsiniz. Bilgisayarda
Python 3.10, 3.11, 3.12 veya 3.13 bulunmalıdır. Python yoksa
[python.org](https://www.python.org/downloads/) üzerinden kurulabilir; kurulumda
`Add Python to PATH` seçeneğini işaretleyin.

## macOS ve Linux'ta çalıştırma

Python 3.10–3.13 kurulu olmalıdır. Proje klasöründe şu komutları çalıştırın:

```sh
sh KUR.sh
sh BASLAT.sh
```

İlk komut yalnızca ilk kurulum veya bağımlılıklar değiştiğinde gereklidir.

## Uygulamada neler var?

- `Hazır modelleri izle`: Projeyle gelen Leo ve T-90 modellerinin maçını açar.
- `Son eğittiğim modelleri izle`: En son tamamlanan eğitimin modellerini açar.
- `Eğitim seçenekleri`: Kısa kontrol, davranış, pilot veya tam eğitim başlatır.
- `Eğitim kayıtları`: Kayıtların ilerlemesini, boyutunu ve tarihini gösterir.
- `Eğitim durumunu göster`: Yalnızca son veya aktif eğitimin durumunu gösterir.
- `Programı teknik olarak kontrol et`: Uygulamanın temel parçalarını otomatik sınar.

Eğitim durumu ile teknik kontrol sonucu ayrı ekranlarda tutulur; biri diğerinin
durumunu değiştirmez. Uygulama açılırken ana menü hazır olana kadar aynı pencerede
`Uygulama yükleniyor…` bilgisi gösterilir.

Eğitim kayıtları ekranındaki `Aç` düğmesi ilgili klasörü gösterir. `Sil` düğmesi
onay aldıktan sonra kaydı işletim sisteminin Geri Dönüşüm Kutusu/Çöp klasörüne
taşır. Devam eden eğitim yanlışlıkla silinemez.

Uygulama eğitim sırasında kapatılırsa önce onay ister. Çıkış onaylandığında eğitim
ve ona bağlı işlemler birlikte durur; arkada çalışmaya devam etmez.

## Eğitim seçenekleri

- Hızlı kontrol: 16.384 adım, 1 seed
- Davranış eğitimi: 200.000 adım, 1 seed
- Pilot eğitim: seed başına 1 milyon adım, 3 seed
- Tam eğitim: seed başına 5 milyon adım, 5 seed

Eğitim ayarları bilgisayara göre gizlice değiştirilmez. Tank hareket hızları,
reload süreleri, PPO epoch sayısı ve kayıtlı deney düzeni bütün bilgisayarlarda
aynıdır. Uygulama görüntüsüz eğitimi CPU üzerinde çalıştırır; CUDA veya NVIDIA
ekran kartı zorunlu değildir.

## Klasör ve GitHub taşınabilirliği

Proje klasörü başka yere taşınabilir veya yeniden adlandırılabilir. Başlatıcılar
bulundukları klasörü otomatik algılar; sabit kullanıcı veya masaüstü adresi
kullanmaz. Windows kısayolu her başlatmada güncel konuma göre yenilenir. OneDrive
altındaki bir yedek kopya, çalışan yerel masaüstü kısayolunun hedefini değiştiremez.

GitHub'a gönderilmemesi gereken dosyalar `.gitignore` tarafından otomatik dışlanır:

- `.venv`: Bilgisayara özel Python ortamı
- `runs_v7`: Kullanıcının eğitim kayıtları
- `*.lnk`: Bilgisayara özel Windows kısayolu
- önbellek ve geçici dosyalar

`models_v7` klasöründeki küçük hazır modeller projeye dahildir; böylece yeni bir
kullanıcı eğitim yapmadan hemen maç izleyebilir.

## Geliştirici kullanımı

Windows test komutu:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

macOS/Linux test komutu:

```sh
./.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

GitHub Actions, her gönderimde Windows ve Ubuntu üzerinde Python 3.12 ve 3.13 ile
aynı testleri otomatik çalıştırır.

Komut satırından kısa eğitim örneği:

```powershell
.\.venv\Scripts\python.exe train_v7.py --total-timesteps 16384 --n-envs 2 --n-steps 256 --batch-size 256
```

Sonuçlar otomatik olarak `runs_v7` klasörüne yazılır.

Eğitim tamamlandığında final modellerin normal değerlendirmesine ek olarak kayıtlı
checkpoint'ler otomatik cross-play değerlendirmesine alınır. Örneğin tam eğitimde
`0M`, `1M`, `2M`, `3M`, `4M` ve `final` modelleri birbirleriyle oynatılır. Sonuçlar
ilgili seed'in `logs/checkpoint_crossplay` klasöründeki `episodes.csv`,
`evaluation_v7.json` ve `crossplay_matrix.png` dosyalarına yazılır. Bu işlem eski
politikalara karşı performans düşüşünü ve politika döngülerini görünür kılar; eğitimi
historical self-play'e dönüştürmez. Gerçek historical self-play için eski modellerin
eğitim sırasındaki rakip havuzuna da katılması gerekir.

## Deneysel kapsam ve bilinen sınırlamalar

- Observation 23 boyutludur. Ajan, kendisine en yakın düşman mermisinin konumunu ve
  hızını görür; aynı anda havadaki ikinci ve sonraki mermiler observation'a girmez.
  Bu nedenle ortam ajan açısından kısmen gözlemlenebilir (POMDP) kabul edilmelidir.
  Mevcut model uyumluluğunu korumak için observation boyutu değiştirilmemiştir.
- Projectile evasion metriği yeni kayıtlarda Leo ve T-90 için ayrı mermi mesafeleri
  kullanır. Eski CSV dosyaları açılabilir fakat tek ortak mesafe içeren eski
  `projectile_evasion_ratio` sonuçları bilimsel karşılaştırmada kullanılmamalıdır.
- Failure Memory, normalize edilmiş 23 özellik üzerinde ağırlıksız Öklid uzaklığı
  kullanır ve varsayılan olarak kapalıdır. Etkisi, aynı seed'lerde `off` ve `entropy`
  deneyleri karşılaştırılmadan kanıtlanmış kabul edilmemelidir.
- Ana deney profili `minimal`dır. `shaped` profilde ideal mesafeye yaklaşma ödülü iki
  ajana birden verildiği için yalnızca karşılaştırma/ablation amacıyla tutulur.
- Mevcut eğitim eşzamanlı co-evolution'dır; historical self-play değildir.

## Not

Bu proje gerçek tank veya silah sistemi testi değildir. Deneysel bir yapay zekâ
simülasyonudur.
