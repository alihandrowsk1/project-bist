
# Project-BIST

## Proje Hakkında

Bu proje, Borsa Istanbul'daki hisse senetlerinin fiyatlarını tahminlemek için kullanılan bir makine öğrenimi modelini içerir. Proje, çeşitli özellikler ve veri analizi teknikleri kullanarak gelecekteki fiyat değişikliklerini tahminlemek için tasarlanmıştır.

## Nasıl Çalışır?

1. **Veri Toplama:** Proje, Borsa Istanbul'dan hisse senedi verilerini toplamak için API'lar kullanır.
2. **Veri Ön İşleme:** Elde edilen veriler, eksik değerlerin doldurulması, özellik mühendisliği ve ölçeklendirme gibi ön işleme adımlarından geçirilir.
3. **Modelleme:** Ön işlenmiş veriler, prophet makine öğrenimi algoritmasıyla eğitilir. Bu adımda, zaman serisi tahminleme tekniği kullanılır.
4. **Değerlendirme:** Eğitilen modelin performansı, uygun değerlendirme metrikleri kullanılarak değerlendirilir.
5. **Tahminler:** Model, gelecekteki hisse senedi fiyatlarını tahminlemek için kullanılabilir. Tahminler, kullanıcıya sunulabilir veya başka bir uygulamada kullanılabilir.

## Kullanım

Repo'yu indirip istemci ile (cmd, powershell) dosya konumundan **model_prophet.py** isimli dosyayı python aracılığıyla çalıştırıp tahminlemek istenilen hisseleri (tek seferde en fazla 5 adet) borsa koduna göre tahminleyip grafiklerini inceleyebilirsiniz. (örn; GARAN : Garanti Bankası, SISE : Şişe ve Cam Fabrikaları, THYAO : Türk Hava Yolları) 

## Katkıda Bulunma

Proje geliştirme sürecine katkıda bulunmak isterseniz, lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını inceleyin. Bu dosya, projeye katkı sağlamak isteyenler için rehberlik sağlar ve nasıl katkıda bulunacakları hakkında detaylı bilgi içerir. Projeye katkıda bulunmadan önce mevcut açık konuları (issues) kontrol etmeyi ve yeni bir özelliği geliştirmek için önceden tartışma başlatmayı öneririm.


## Lisans

Bu proje, MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.


## İletişim

Eğer bir sorunla karşılaşırsanız veya proje hakkında başka sorularınız varsa, lütfen bana ulaşın.

E-posta: alihan.ozturk@icloud.com
