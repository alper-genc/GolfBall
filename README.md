# Golf Topu Aerodinamik Simulasyonu (Sade Surum)

Bu klasor, golf topu dimple parametreleri icin:
- ucus simulasyonu yapar,
- bir parametre taramasi calistirir,
- en iyi tasarimi otomatik raporlar.

Amac: **kafa karistirmadan**, hizli sekilde "hangi parametreler daha iyi?" sorusuna cevap vermek.

---

## 1) Neler var?

- `golf_simulator.py`  
  Ana script. RK4 ile ucus simulasyonu yapar ve tum tasarimlari tarar.
- `results.csv`  
  Tum denenen tasarimlarin sonucu (script calistiktan sonra olusur).
- `best_design_summary.md`  
  En iyi tasarimin ozet raporu (script calistiktan sonra olusur).

---

## 2) Calistirma

Terminalde bu klasore gelin ve:

```bash
python golf_simulator.py
```

Calisma bittiginde konsolda en iyi parametreler yazilir.

### Streamlit gorsel arayuz

Simulasyonu ekran uzerinden gormek icin:

```bash
streamlit run streamlit_app.py
```

Arayuzde:
- launch speed / angle / spin degistirilebilir,
- makale uyumlu test modu secilebilir,
- en iyi tasarimlar tablo halinde gorulur,
- secilen tasarimin ucus egrisi cizilir.

---

## 3) Modelin kullandigi temel denklemler

Scriptte standart aerodinamik ifadeler kullanilir:

- `F_D = 0.5 * rho * U^2 * A * C_D`
- `F_L = 0.5 * rho * U^2 * A * C_L`
- `Re = rho * U * d / mu`
- `Sp = pi * d * N / U`

`C_D` ve `C_L`, literatur trendlerine uygun sade bir surrogate (yaklasik) model ile hesaplanir.

---

## 4) Makale uyumlu test modlari

Yazilim makale mantigina gore 4 modda calisir:

1. `paper_depth_only`  
   Derinlik etkisi testi. Sadece `k/d` degisir, ayni dimple ailesi korunur.
2. `paper_occupancy_only`  
   Kaplama etkisi testi. Derinlik sabit tutulur, occupancy ailesi degisir.
3. `paper_volume_only`  
   VDR egilim testi. Literaturdeki tasarimlar uzerinden hacim orani davranisi incelenir.
4. `paper_literature_grid`  
   Tum literatur tasarimlari (15 adet). Yapay capraz kombinasyon yoktur.

Sabit referans degerler:
- `k/d = 0.00455`
- `occupancy = 0.812`
- `volume_ratio = 0.0111`
- `dimple_count = 314`
- `dimple_diameter_mm = 3.5`

---

## 5) Sonraki adim (onerilen)

Bu surum, hizli karar vermek icin idealdir. Tezde daha guclu dogrulama icin:

1. Wind tunnel / mevcut deney verilerinizi `results.csv` ile karsilastirin.
2. Gerekirse `model_cd_cl()` fonksiyonundaki katsayilari kalibre edin.
3. Son asamada en iyi 5 tasarim icin CFD dogrulamasi yapin.

Not:
- `spin=0` iken sistem wind-tunnel mantigina gore `Cd` odakli siralama yapar.

