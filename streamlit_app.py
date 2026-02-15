from __future__ import annotations

from bisect import bisect_left
import time
import altair as alt
import pandas as pd
import streamlit as st

from golf_simulator import (
    DimpleDesign,
    LaunchCondition,
    model_cd_cl,
    reynolds_number,
    run_robust_search,
    run_search_with_launch,
    simulate_flight_with_path,
)


st.set_page_config(page_title="Golf Ball Simulator", layout="wide")
st.title("Golf Topu Simulasyonu - Basit Anlatim")
st.caption(
    "Amac: hangi dimple ayari daha iyi performans veriyor, bunu kolayca gormek. "
    "Senaryo secimi: Ruzgar Tuneli veya Insan Vurusu."
)


MODE_LABELS = {
    "paper_depth_only": "Derinlik etkisi testi (tek degisken)",
    "paper_occupancy_only": "Kaplama orani etkisi testi (tek degisken)",
    "paper_volume_only": "VDR egilim testi (literatur seti)",
    "paper_literature_grid": "Tum literatur tasarimlari (15 adet)",
}

MODE_LONG_INFO = {
    "paper_depth_only": "Ne degisiyor: sadece dimple derinligi. Ne sabit: ayni dimple ailesi (O=81.2%, ND=476).",
    "paper_occupancy_only": "Ne degisiyor: sadece kaplama ailesi (occupancy gruplari). Ne sabit: derinlik D/d=4.55e-3.",
    "paper_volume_only": "Ne degisiyor: VDR trendini gormek icin 15 literatur tasarimi birlikte incelenir.",
    "paper_literature_grid": "Ne degisiyor: makaledeki 15 tasarimin tamami. Yapay kombinasyon yoktur.",
}

CONTEXTS = {
    "wind_tunnel": {
        "label": "Ruzgar Tuneli Modu",
        "description": "Wind tunnel kosullari: hiz 20-140 km/h, no-spin odakli.",
        "speed_unit": "km/h",
        "speed_min_mps": 20.0 / 3.6,
        "speed_max_mps": 140.0 / 3.6,
        "speed_default_mps": 100.0 / 3.6,
        "spin_min": 0,
        "spin_max": 2000,
        "spin_default": 0,
    },
    "human_flight": {
        "label": "Insan Vurusu Modu (PGA referans)",
        "description": "Ucus kosullari: PGA ortalama launch degerleri etrafinda hiz/spin.",
        "speed_unit": "m/s",
        "speed_min_mps": 45.0,
        "speed_max_mps": 85.0,
        "speed_default_mps": 74.0,
        "spin_min": 1500,
        "spin_max": 4500,
        "spin_default": 2685,
    },
}

PRESETS = {
    "wind_onerilen": {
        "label": "Ruzgar Tuneli / Onerilen (100 km/h, no-spin)",
        "context": "wind_tunnel",
        "mode": "paper_depth_only",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Basit",
    },
    "wind_min_hiz": {
        "label": "Ruzgar Tuneli / Minimum hiz (20 km/h)",
        "context": "wind_tunnel",
        "mode": "paper_depth_only",
        "speed_mps": 20.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Basit",
    },
    "wind_maks_hiz": {
        "label": "Ruzgar Tuneli / Maksimum hiz (140 km/h)",
        "context": "wind_tunnel",
        "mode": "paper_depth_only",
        "speed_mps": 140.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Basit",
    },
    "wind_occupancy": {
        "label": "Ruzgar Tuneli / Occupancy karsilastirma",
        "context": "wind_tunnel",
        "mode": "paper_occupancy_only",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Basit",
    },
    "wind_vdr": {
        "label": "Ruzgar Tuneli / VDR egilimi",
        "context": "wind_tunnel",
        "mode": "paper_volume_only",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Basit",
    },
    "wind_tam_grid": {
        "label": "Ruzgar Tuneli / Tum literatur tasarimlari",
        "context": "wind_tunnel",
        "mode": "paper_literature_grid",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 10,
        "view_mode": "Teknik",
    },
    "human_pga_ort": {
        "label": "Insan Vurusu / PGA ortalama (74 m/s, 2685 rpm)",
        "context": "human_flight",
        "mode": "paper_literature_grid",
        "speed_mps": 74.0,
        "launch_angle_deg": 11.2,
        "spin_rpm": 2685,
        "top_n": 10,
        "view_mode": "Teknik",
    },
    "human_dusuk_spin": {
        "label": "Insan Vurusu / Dusuk spin (74 m/s, 2200 rpm)",
        "context": "human_flight",
        "mode": "paper_literature_grid",
        "speed_mps": 74.0,
        "launch_angle_deg": 11.2,
        "spin_rpm": 2200,
        "top_n": 10,
        "view_mode": "Teknik",
    },
    "human_yuksek_spin": {
        "label": "Insan Vurusu / Yuksek spin (74 m/s, 3200 rpm)",
        "context": "human_flight",
        "mode": "paper_literature_grid",
        "speed_mps": 74.0,
        "launch_angle_deg": 11.2,
        "spin_rpm": 3200,
        "top_n": 10,
        "view_mode": "Teknik",
    },
}


def ensure_session_defaults() -> None:
    p = PRESETS["wind_onerilen"]
    st.session_state.setdefault("context", p["context"])
    st.session_state.setdefault("input_mode", "Hazır Ayar")
    st.session_state.setdefault("preset_key", "wind_onerilen")
    st.session_state.setdefault("applied_preset_key", "")
    st.session_state.setdefault("mode", p["mode"])
    st.session_state.setdefault("speed_mps", p["speed_mps"])
    st.session_state.setdefault("launch_angle_deg", p["launch_angle_deg"])
    st.session_state.setdefault("spin_rpm", p["spin_rpm"])
    st.session_state.setdefault("top_n", p["top_n"])
    st.session_state.setdefault("view_mode", p["view_mode"])
    st.session_state.setdefault("auto_recalc", False)
    st.session_state.setdefault("recalc_message", "")


def to_percent(value: float) -> str:
    return f"%{value * 100:.1f}"


def result_comment(row: pd.Series) -> str:
    """
    Story-like reason text for non-technical users.
    """
    depth_ratio = float(row["depth_ratio_k_over_d"])
    occupancy = float(row["occupancy"])
    lod = float(row["avg_l_over_d"])
    distance = float(row["distance_m"])

    if depth_ratio <= 4.55e-3:
        depth_text = "dimplelar daha sığ kaldığı için top havayı daha temiz yarıyor"
    elif depth_ratio <= 6.82e-3:
        depth_text = "dimple derinliği dengeli kaldığı için top ne çok agresif ne de çok pasif davranıyor"
    else:
        depth_text = "dimplelar derin olduğu için topun hava ile etkileşimi daha güçlü oluyor"

    if occupancy >= 0.80:
        occ_text = "yüzeyin dimple ile kaplı kısmı yüksek olduğu için uçuş daha stabil ilerliyor"
    elif occupancy >= 0.63:
        occ_text = "kaplama oranı orta seviyede olduğu için performans dengeli kalıyor"
    else:
        occ_text = "kaplama oranı düşük olduğu için top bazı durumlarda daha erken hız kaybedebiliyor"

    if lod >= 0.50:
        lod_text = "kaldırma/sürükleme dengesi güçlü olduğu için mesafeyi iyi koruyor"
    elif lod >= 0.42:
        lod_text = "kaldırma/sürükleme dengesi kabul edilebilir olduğu için mesafe tatmin edici çıkıyor"
    else:
        lod_text = "kaldırma/sürükleme dengesi sınırlı kaldığı için mesafe avantajı azalıyor"

    return (
        f"Bu tasarımda {depth_text}; ayrıca {occ_text}. "
        f"Sonuçta {lod_text} ve tahmini mesafe yaklaşık {distance:.1f} m seviyesine geliyor."
    )


def faq_data() -> list[tuple[str, str]]:
    return [
        (
            "Bu uygulama ne yapiyor?",
            "Makalelerdeki parametre araliklarinda birden fazla golf topu tasarimini dener ve secilen siralama olcutune gore en iyi sonucu uste getirir.",
        ),
        (
            "Testi nasil yapiyor?",
            "Her tasarim icin simulasyon calistirir. Ruzgar tüneli tipinde (spin=0) ana degerlendirme Cd uzerinden yapilir. Spinli durumda ucus denklemi de kullanilir.",
        ),
        (
            "Kullanilan formuller nereden geliyor?",
            "Temel denklemler literaturdeki standart aerodinamik denklemlerden gelir: Re, Sp, Fd, Fl, Cd, Cl. Parametre trendleri de paylastigin makalelerdeki bulgulara gore kurulmustur.",
        ),
        (
            "Hiz araligi neden boyle?",
            "Senaryoya gore degisir: Ruzgar Tuneli modunda 20-140 km/h, Insan Vurusu modunda 45-85 m/s araligi kullanilir.",
        ),
        (
            "Neden 1. siradaki tasarim en iyi?",
            "Cunku secilen kosulda tum adaylar ayni sartlarda karsilastirilir ve secilen olcute gore en iyi olan 1. siraya yazilir. (No-spin testte Cd, spinli testte bileşik skor)",
        ),
        (
            "Bu sonuc kesin nihai mi?",
            "Bu model hizli tarama ve aday secimi icindir. Nihai onay icin secilen adaylarin CFD veya deney ile dogrulanmasi gerekir.",
        ),
        (
            "Tek degiskenli test ne demek?",
            "Sadece tek parametreyi degistirip digerlerini sabit tutmak. Boylece hangi etkinin nereden geldigi daha net gorulur.",
        ),
    ]


def run_cached_search(speed_mps: float, launch_angle_deg: float, spin_rpm: float, mode: str):
    launch = LaunchCondition(
        speed_mps=float(speed_mps),
        launch_angle_deg=float(launch_angle_deg),
        spin_rpm=float(spin_rpm),
    )
    ranked, _ = run_search_with_launch(launch, mode=mode)

    rows = []
    for rank, (design, result, score) in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "score": score,
                "distance_m": result.carry_distance_m,
                "flight_time_s": result.flight_time_s,
                "max_height_m": result.max_height_m,
                "avg_cd": result.avg_cd,
                "avg_cl": result.avg_cl,
                "avg_l_over_d": result.avg_l_over_d,
                "depth_ratio_k_over_d": design.depth_ratio,
                "occupancy": design.occupancy,
                "volume_ratio": design.volume_ratio,
                "dimple_count": design.dimple_count,
                "dimple_diameter_mm": design.dimple_diameter_mm,
            }
        )
    return pd.DataFrame(rows)


def run_cached_robust_search(speed_mps: float, launch_angle_deg: float, spin_rpm: float, mode: str):
    base_launch = LaunchCondition(
        speed_mps=float(speed_mps),
        launch_angle_deg=float(launch_angle_deg),
        spin_rpm=float(spin_rpm),
    )
    ranked, _, scenarios = run_robust_search(mode=mode, base_launch=base_launch)

    rows = []
    for rank, item in enumerate(ranked, start=1):
        design = item.design
        nominal = item.nominal_result
        rows.append(
            {
                "rank": rank,
                "robust_score": item.robust_score,
                "mean_score": item.mean_score,
                "worst_score": item.worst_score,
                "score_std": item.score_std,
                "mean_distance_m": item.mean_distance_m,
                "min_distance_m": item.min_distance_m,
                "nominal_distance_m": nominal.carry_distance_m,
                "nominal_avg_cd": nominal.avg_cd,
                "nominal_avg_cl": nominal.avg_cl,
                "nominal_avg_l_over_d": nominal.avg_l_over_d,
                "depth_ratio_k_over_d": design.depth_ratio,
                "occupancy": design.occupancy,
                "volume_ratio": design.volume_ratio,
                "dimple_count": design.dimple_count,
                "dimple_diameter_mm": design.dimple_diameter_mm,
            }
        )
    return pd.DataFrame(rows), len(scenarios)


def ranking_explainer(spin_rpm: int) -> str:
    if spin_rpm == 0:
        return "Siralama olcutu: **Ortalama Cd (kucuk daha iyi)**. (Ruzgar tuneli tipi, no-spin)"
    return "Siralama olcutu: **Mesafe + L/D + Cd bileşik skoru**. (Ucuş tipi, spinli)"


def format_unique(values: list[float], fmt: str) -> str:
    return ", ".join(format(v, fmt) for v in values)


def mode_test_summary(df: pd.DataFrame) -> tuple[list[str], list[str], dict[str, str]]:
    depth_vals = sorted(df["depth_ratio_k_over_d"].unique().tolist())
    occ_vals = sorted(df["occupancy"].unique().tolist())
    vdr_vals = sorted(df["volume_ratio"].unique().tolist())
    nd_vals = sorted(df["dimple_count"].unique().tolist())
    dd_vals = sorted(df["dimple_diameter_mm"].unique().tolist())

    specs = {
        "Derinlik orani (k/d)": format_unique(depth_vals, ".5f"),
        "Yuzey kaplama orani (%)": format_unique([v * 100 for v in occ_vals], ".1f"),
        "Dimple hacim orani (VDR)": format_unique(vdr_vals, ".4f"),
        "Toplam dimple sayisi": ", ".join(str(int(v)) for v in nd_vals),
        "Dimple capi (mm)": format_unique(dd_vals, ".2f"),
    }

    changing = [k for k, vals in [
        ("Derinlik orani (k/d)", depth_vals),
        ("Yuzey kaplama orani (%)", occ_vals),
        ("Dimple hacim orani (VDR)", vdr_vals),
        ("Toplam dimple sayisi", nd_vals),
        ("Dimple capi (mm)", dd_vals),
    ] if len(vals) > 1]
    fixed = [k for k in specs.keys() if k not in changing]
    return changing, fixed, specs


def build_cd_re_curve(
    design: DimpleDesign, spin_rpm: int, speed_min_mps: float, speed_max_mps: float
) -> pd.DataFrame:
    rows = []
    num_points = 25
    for i in range(num_points):
        speed_mps = speed_min_mps + i * (speed_max_mps - speed_min_mps) / max(num_points - 1, 1)
        cd, _ = model_cd_cl(speed_mps, float(spin_rpm), design)
        rows.append(
            {
                "re": reynolds_number(speed_mps),
                "cd": cd,
            }
        )
    return pd.DataFrame(rows)


def build_cd_re_comparison(
    designs: list[DimpleDesign],
    labels: list[str],
    spin_rpm: int,
    speed_min_mps: float,
    speed_max_mps: float,
) -> pd.DataFrame:
    """
    Build a wide dataframe for multi-design Cd-Re comparison.
    Index: Reynolds number
    Columns: one column per design label
    """
    merged_df: pd.DataFrame | None = None
    for idx, design in enumerate(designs, start=1):
        curve = build_cd_re_curve(design, spin_rpm, speed_min_mps, speed_max_mps)
        label = labels[idx - 1]
        col_df = curve[["re", "cd"]].rename(columns={"cd": label})
        if merged_df is None:
            merged_df = col_df
        else:
            merged_df = merged_df.merge(col_df, on="re", how="inner")
    assert merged_df is not None
    return merged_df.set_index("re")


def build_flight_comparison(
    designs: list[DimpleDesign], labels: list[str], launch: LaunchCondition
) -> pd.DataFrame:
    """
    Build multi-design flight trajectory comparison dataframe.
    Index: distance (m)
    Columns: one per design label, value=height (m)
    """
    all_paths: list[list[tuple[float, float]]] = []
    max_distance = 0.0
    for design in designs:
        _, path = simulate_flight_with_path(launch, design)
        path = sorted(path, key=lambda p: p[0])
        all_paths.append(path)
        if path:
            max_distance = max(max_distance, path[-1][0])

    if max_distance <= 0.0:
        return pd.DataFrame()

    # Common x-grid prevents sparse/empty-looking plots.
    num_points = 260
    x_grid = [i * max_distance / (num_points - 1) for i in range(num_points)]
    result_df = pd.DataFrame({"distance_m": x_grid})

    for idx, (design, path) in enumerate(zip(designs, all_paths), start=1):
        label = labels[idx - 1]

        px = [p[0] for p in path]
        py = [p[1] for p in path]

        y_interp: list[float] = []
        for x in x_grid:
            if x <= px[0]:
                y_interp.append(py[0])
                continue
            if x >= px[-1]:
                y_interp.append(py[-1])
                continue

            j = bisect_left(px, x)
            x0, y0 = px[j - 1], py[j - 1]
            x1, y1 = px[j], py[j]
            if x1 == x0:
                y_interp.append(y1)
            else:
                t = (x - x0) / (x1 - x0)
                y_interp.append(y0 + t * (y1 - y0))

        result_df[label] = y_interp

    return result_df.set_index("distance_m")


def build_design_legend(designs: list[DimpleDesign]) -> pd.DataFrame:
    rows = []
    for idx, d in enumerate(designs, start=1):
        rows.append(
            {
                "Etiket": f"S{idx}",
                "k/d": d.depth_ratio,
                "Yuzey kaplama (%)": d.occupancy * 100.0,
                "VDR": d.volume_ratio,
                "Dimple sayisi": d.dimple_count,
                "Dimple capi (mm)": d.dimple_diameter_mm,
            }
        )
    return pd.DataFrame(rows)


ensure_session_defaults()

with st.sidebar:
    st.header("1) Giris ayarlari")
    if "context_widget" not in st.session_state:
        st.session_state["context_widget"] = st.session_state["context"]
    context = st.selectbox(
        "Test senaryosu",
        options=list(CONTEXTS.keys()),
        format_func=lambda key: CONTEXTS[key]["label"],
        index=list(CONTEXTS.keys()).index(st.session_state["context_widget"]),
        key="context_widget",
    )
    st.session_state["context"] = context
    context_cfg = CONTEXTS[context]

    input_mode = st.radio(
        "Ayar yontemi",
        options=["Hazır Ayar", "Özel Ayar"],
        horizontal=True,
        index=0 if st.session_state["input_mode"] == "Hazır Ayar" else 1,
        key="input_mode",
    )

    if input_mode == "Hazır Ayar":
        preset_options = [k for k, v in PRESETS.items() if v["context"] == context]
        if st.session_state.get("preset_key") not in preset_options:
            st.session_state["preset_key"] = preset_options[0]
        preset_key = st.selectbox(
            "Hazir ayar sec",
            options=preset_options,
            format_func=lambda key: PRESETS[key]["label"],
            index=preset_options.index(st.session_state["preset_key"]),
            key="preset_key",
            help="Makalelerdeki test mantigina gore hazir ayarlar.",
        )
        st.caption("Hazır ayar seçtiğiniz anda otomatik uygulanır.")
        if st.session_state.get("applied_preset_key") != preset_key:
            p = PRESETS[preset_key]
            st.session_state["mode"] = p["mode"]
            st.session_state["speed_mps"] = p["speed_mps"]
            st.session_state["launch_angle_deg"] = p["launch_angle_deg"]
            st.session_state["spin_rpm"] = p["spin_rpm"]
            st.session_state["top_n"] = p["top_n"]
            st.session_state["view_mode"] = p["view_mode"]
            st.session_state["applied_preset_key"] = preset_key
            st.session_state["auto_recalc"] = True
            st.session_state["recalc_message"] = f"Hazir ayar otomatik uygulandi: {PRESETS[preset_key]['label']}"
            st.rerun()
    else:
        st.caption("Özel ayarda kaydırıcıları değiştirince sonuçlar otomatik güncellenir.")

    mode = st.selectbox(
        "Test tipi (acik secim)",
        options=list(MODE_LABELS.keys()),
        format_func=lambda key: MODE_LABELS[key],
        index=list(MODE_LABELS.keys()).index(st.session_state["mode"]),
        help="Makaleye uygun test turu. Varsayilan tek degisken.",
        key="mode",
    )
    current_speed_mps = float(st.session_state["speed_mps"])
    current_speed_mps = max(context_cfg["speed_min_mps"], min(context_cfg["speed_max_mps"], current_speed_mps))
    if context_cfg["speed_unit"] == "km/h":
        speed_input = st.slider(
            "Ruzgar tuneli hizi (km/h)",
            min_value=int(round(context_cfg["speed_min_mps"] * 3.6)),
            max_value=int(round(context_cfg["speed_max_mps"] * 3.6)),
            value=int(round(current_speed_mps * 3.6)),
            step=1,
        )
        speed_mps = speed_input / 3.6
    else:
        speed_input = st.slider(
            "Top cikis hizi (m/s)",
            min_value=float(context_cfg["speed_min_mps"]),
            max_value=float(context_cfg["speed_max_mps"]),
            value=float(round(current_speed_mps, 1)),
            step=0.5,
        )
        speed_mps = float(speed_input)
    st.session_state["speed_mps"] = speed_mps

    launch_angle_deg = st.slider(
        "Atis acisi (derece)",
        min_value=6.0,
        max_value=18.0,
        value=float(st.session_state["launch_angle_deg"]),
        step=0.1,
        key="launch_angle_deg",
    )
    spin_rpm = st.slider(
        "Spin (rpm)",
        min_value=int(context_cfg["spin_min"]),
        max_value=int(context_cfg["spin_max"]),
        value=int(st.session_state["spin_rpm"]),
        step=50,
        key="spin_rpm",
    )
    top_n = st.slider(
        "Listede kac sonuc goreyim?",
        min_value=3,
        max_value=25,
        value=int(st.session_state["top_n"]),
        step=1,
        key="top_n",
    )
    view_mode = st.radio(
        "Gorunum",
        options=["Basit", "Teknik"],
        horizontal=True,
        index=0 if st.session_state["view_mode"] == "Basit" else 1,
        key="view_mode",
    )
    st.caption(context_cfg["description"])

speed_mps = float(st.session_state["speed_mps"])

st.info(
    f"Senaryo: **{CONTEXTS[context]['label']}**  |  "
    f"Test tipi: **{MODE_LABELS[mode]}**  |  "
    f"Hiz: **{speed_mps:.1f} m/s ({speed_mps*3.6:.0f} km/h)**  |  "
    f"Spin: **{int(spin_rpm)} rpm**"
)

with st.expander("Yardim / Metodoloji / SSS", expanded=False):
    st.markdown(
        "**Hizli kullanim:** Hazir ayar sec -> Sistem otomatik uygular -> Sonuclari incele.\n\n"
        "**Secim onerisi:**\n"
        "- Ruzgar Tuneli Modu: no-spin aerodinamik karsilastirma (Cd odakli)\n"
        "- Insan Vurusu Modu: spinli ucus karsilastirmasi (mesafe ve L/D etkili)\n"
        "- Tek degisken analizi: `Derinlik etkisi` veya `Kaplama etkisi`\n\n"
        "**Model:**\n"
        "- `Re = rho*U*d/mu`, `Sp = pi*d*N/U`\n"
        "- `Fd = 0.5*rho*U^2*A*Cd`, `Fl = 0.5*rho*U^2*A*Cl`\n"
        "- `spin=0` modunda siralama `Cd` ile, spinli modda bileşik skor ile yapilir."
    )
    st.markdown("---")
    for q, a in faq_data():
        st.markdown(f"**Soru:** {q}")
        st.write(f"**Cevap:** {a}")
        st.markdown("---")


if st.session_state.get("auto_recalc", False):
    msg = st.session_state.get("recalc_message", "Ayarlar degisti")
    with st.spinner(f"{msg}. Sonuclar hesaplanıyor..."):
        time.sleep(1.2)
        df = run_cached_search(
            speed_mps=speed_mps,
            launch_angle_deg=launch_angle_deg,
            spin_rpm=spin_rpm,
            mode=mode,
        )
    st.success("Hesaplama tamamlandi. Sonuclar guncellendi.")
    st.session_state["auto_recalc"] = False
    st.session_state["recalc_message"] = ""
else:
    df = run_cached_search(
        speed_mps=speed_mps,
        launch_angle_deg=launch_angle_deg,
        spin_rpm=spin_rpm,
        mode=mode,
    )
top_df = df.head(top_n).copy()
best = top_df.iloc[0]
is_no_spin = int(spin_rpm) == 0
changing_params, fixed_params, tested_specs = mode_test_summary(df)

st.subheader("Bu testte sistem ne yapiyor?")
st.markdown(
    "1. Seçtiğin test tipine göre makaleden gelen aday tasarımlar hazırlanır.\n"
    "2. Her aday için `Cd/Cl` hesaplanır ve uçuş simülasyonu çalıştırılır.\n"
    "3. Sonuçlar karşılaştırılır.\n"
    "4. En iyi satır, seçilen sıralama ölçütüne göre 1. sıraya yazılır."
)

c1, c2 = st.columns(2)
c1.metric("Denenen tasarim sayisi", f"{len(df)}")
c2.metric("Degisen parametre sayisi", f"{len(changing_params)}")

st.markdown("**Bu testte degisen parametreler:** " + (", ".join(changing_params) if changing_params else "Yok"))
st.markdown("**Bu testte sabit kalan parametreler:** " + (", ".join(fixed_params) if fixed_params else "Yok"))

with st.expander("Bu testte denenen tum degerler"):
    for k, v in tested_specs.items():
        st.markdown(f"- **{k}:** {v}")

mean_distance = float(df["distance_m"].mean())
gain_vs_mean = float(best["distance_m"] - mean_distance)

col1, col2, col3 = st.columns(3)
col1.metric("En iyi mesafe", f"{best['distance_m']:.2f} m")
col2.metric("Ortalama sonuca gore fark", f"{gain_vs_mean:+.2f} m")
col3.metric("Denenen toplam tasarim", f"{len(df)}")
st.caption(ranking_explainer(int(spin_rpm)))

st.success(
    "Kisa ozet: Sistem tum adaylari ayni kosulda dener ve secilen siralama olcutune gore en iyi sonucu en uste getirir."
)

st.subheader("2) Sonuclar (en iyiden baslayarak)")
st.caption("Tabloda üretim için gerekli temel tasarım bilgileri bulunur.")

simple_df = top_df.copy()
simple_df["occupancy_pct"] = simple_df["occupancy"] * 100.0
simple_df["dimple_depth_mm"] = simple_df["depth_ratio_k_over_d"] * 42.67
simple_df["neden_iyi"] = simple_df.apply(result_comment, axis=1)

if view_mode == "Basit":
    show_df = simple_df[
        [
            "rank",
            "distance_m",
            "flight_time_s",
            "max_height_m",
            "dimple_depth_mm",
            "depth_ratio_k_over_d",
            "occupancy_pct",
            "volume_ratio",
            "dimple_count",
            "dimple_diameter_mm",
            "neden_iyi",
        ]
    ].rename(
        columns={
            "rank": "Sira",
            "distance_m": "Tahmini tasima mesafesi (m)",
            "flight_time_s": "Ucusta kalma suresi (s)",
            "max_height_m": "Maksimum yukseklik (m)",
            "dimple_depth_mm": "Dimple derinligi (mm)",
            "depth_ratio_k_over_d": "Derinlik orani (k/d)",
            "occupancy_pct": "Yuzey kaplama orani (%)",
            "volume_ratio": "Dimple hacim orani (VDR)",
            "dimple_count": "Toplam dimple sayisi (adet)",
            "dimple_diameter_mm": "Dimple capi (mm)",
            "neden_iyi": "Neden iyi?",
        }
    )
    st.dataframe(
        show_df.style.format(
            {
                "Tahmini tasima mesafesi (m)": "{:.3f}",
                "Ucusta kalma suresi (s)": "{:.2f}",
                "Maksimum yukseklik (m)": "{:.2f}",
                "Dimple derinligi (mm)": "{:.3f}",
                "Derinlik orani (k/d)": "{:.5f}",
                "Yuzey kaplama orani (%)": "{:.1f}",
                "Dimple hacim orani (VDR)": "{:.4f}",
                "Dimple capi (mm)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
else:
    tech_df = top_df.rename(
        columns={
            "rank": "Sira",
            "score": "Skor",
            "distance_m": "Tahmini tasima mesafesi (m)",
            "flight_time_s": "Ucusta kalma suresi (s)",
            "max_height_m": "Maksimum yukseklik (m)",
            "avg_cd": "Ortalama Cd",
            "avg_cl": "Ortalama Cl",
            "avg_l_over_d": "Ortalama L/D",
            "depth_ratio_k_over_d": "Derinlik orani (k/d)",
            "occupancy": "Yuzey kaplama orani (0-1)",
            "volume_ratio": "Dimple hacim orani (VDR)",
            "dimple_count": "Toplam dimple sayisi (adet)",
            "dimple_diameter_mm": "Dimple capi (mm)",
        }
    ).copy()
    tech_df["Dimple derinligi (mm)"] = tech_df["Derinlik orani (k/d)"] * 42.67
    tech_df["Yuzey kaplama orani (%)"] = tech_df["Yuzey kaplama orani (0-1)"] * 100.0
    if is_no_spin:
        # In no-spin wind tunnel mode, ranking is based on Cd.
        # Hide synthetic score to avoid confusion from negative values.
        tech_df = tech_df.drop(columns=["Skor"])

    tech_format = {
        "Skor": "{:.2f}",
        "Tahmini tasima mesafesi (m)": "{:.3f}",
        "Ucusta kalma suresi (s)": "{:.2f}",
        "Maksimum yukseklik (m)": "{:.2f}",
        "Ortalama Cd": "{:.3f}",
        "Ortalama Cl": "{:.3f}",
        "Ortalama L/D": "{:.3f}",
        "Derinlik orani (k/d)": "{:.5f}",
        "Yuzey kaplama orani (0-1)": "{:.3f}",
        "Yuzey kaplama orani (%)": "{:.1f}",
        "Dimple hacim orani (VDR)": "{:.4f}",
        "Dimple capi (mm)": "{:.2f}",
        "Dimple derinligi (mm)": "{:.3f}",
    }
    if is_no_spin and "Skor" in tech_format:
        tech_format.pop("Skor")

    st.dataframe(
        tech_df.style.format(tech_format),
        width="stretch",
        hide_index=True,
    )


st.subheader("2B) Global senaryo scoreboard (robust)")
if int(spin_rpm) == 0:
    st.info(
        "Global robust scoreboard spinli uretim senaryosu icindir. "
        "Bu nedenle spin=0 durumunda gosterilmez."
    )
else:
    robust_df, scenario_count = run_cached_robust_search(
        speed_mps=speed_mps,
        launch_angle_deg=launch_angle_deg,
        spin_rpm=spin_rpm,
        mode=mode,
    )
    robust_top_df = robust_df.head(top_n).copy()
    robust_best = robust_top_df.iloc[0]

    r1, r2, r3 = st.columns(3)
    r1.metric("Global en iyi robust skor", f"{robust_best['robust_score']:.2f}")
    r2.metric("Global ortalama tasima", f"{robust_best['mean_distance_m']:.2f} m")
    r3.metric("Global minimum tasima", f"{robust_best['min_distance_m']:.2f} m")

    st.caption(
        f"Bu tablo, secilen temel launch etrafinda {scenario_count} farkli hiz/aci/spin kombinasyonunun "
        "birlesik sonucunu verir. Uretimde daha guvenli aday secimi icin kullanilir."
    )

    robust_show_df = robust_top_df.rename(
        columns={
            "rank": "Sira",
            "robust_score": "Robust skor",
            "mean_score": "Ortalama skor",
            "worst_score": "En kotu skor",
            "score_std": "Skor sapmasi",
            "mean_distance_m": "Ortalama tasima (m)",
            "min_distance_m": "Minimum tasima (m)",
            "nominal_distance_m": "Nominal tasima (m)",
            "nominal_avg_cd": "Nominal Cd",
            "nominal_avg_cl": "Nominal Cl",
            "nominal_avg_l_over_d": "Nominal L/D",
            "depth_ratio_k_over_d": "Derinlik orani (k/d)",
            "occupancy": "Yuzey kaplama orani (0-1)",
            "volume_ratio": "Dimple hacim orani (VDR)",
            "dimple_count": "Toplam dimple sayisi (adet)",
            "dimple_diameter_mm": "Dimple capi (mm)",
        }
    )

    st.dataframe(
        robust_show_df.style.format(
            {
                "Robust skor": "{:.2f}",
                "Ortalama skor": "{:.2f}",
                "En kotu skor": "{:.2f}",
                "Skor sapmasi": "{:.2f}",
                "Ortalama tasima (m)": "{:.2f}",
                "Minimum tasima (m)": "{:.2f}",
                "Nominal tasima (m)": "{:.2f}",
                "Nominal Cd": "{:.3f}",
                "Nominal Cl": "{:.3f}",
                "Nominal L/D": "{:.3f}",
                "Derinlik orani (k/d)": "{:.5f}",
                "Yuzey kaplama orani (0-1)": "{:.3f}",
                "Dimple hacim orani (VDR)": "{:.4f}",
                "Dimple capi (mm)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Global robust tabloda yer alan tasarimlar icin Cd-Re karsilastirmasi")
    robust_compare_designs = [
        DimpleDesign(
            depth_ratio=float(row["depth_ratio_k_over_d"]),
            occupancy=float(row["occupancy"]),
            volume_ratio=float(row["volume_ratio"]),
            dimple_count=int(row["dimple_count"]),
            dimple_diameter_mm=float(row["dimple_diameter_mm"]),
        )
        for _, row in robust_top_df.iterrows()
    ]
    robust_compare_labels = [f"R{i}" for i in range(1, len(robust_compare_designs) + 1)]
    robust_legend_df = build_design_legend(robust_compare_designs).copy()
    robust_legend_df["Etiket"] = robust_compare_labels

    robust_cd_re_comp_df = build_cd_re_comparison(
        robust_compare_designs,
        robust_compare_labels,
        int(spin_rpm),
        float(context_cfg["speed_min_mps"]),
        float(context_cfg["speed_max_mps"]),
    )
    robust_cd_re_long = (
        robust_cd_re_comp_df.reset_index()
        .melt(id_vars="re", var_name="tasarim", value_name="cd")
        .dropna()
    )
    robust_cd_re_chart = (
        alt.Chart(robust_cd_re_long)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("re:Q", title="Reynolds sayisi (Re)", axis=alt.Axis(format=".2e")),
            y=alt.Y("cd:Q", title="Surukleme katsayisi (Cd)"),
            color=alt.Color("tasarim:N", title="Robust tasarim"),
            tooltip=[
                alt.Tooltip("tasarim:N", title="Tasarim"),
                alt.Tooltip("re:Q", title="Re", format=".2e"),
                alt.Tooltip("cd:Q", title="Cd", format=".4f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(robust_cd_re_chart, use_container_width=True)
    st.caption(
        "R1, R2, ... etiketleri robust scoreboard sirasini temsil eder. "
        "Egri ne kadar asagida ise ilgili hiz bandinda Cd o kadar dusuktur."
    )
    with st.expander("Robust Cd-Re grafik etiketleri (R1, R2, ...)"):
        st.dataframe(
            robust_legend_df.style.format(
                {
                    "k/d": "{:.5f}",
                    "Yuzey kaplama (%)": "{:.1f}",
                    "VDR": "{:.4f}",
                    "Dimple capi (mm)": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


st.subheader("3) Bir tasarim sec ve ucusunu gor")
compare_designs = [
    DimpleDesign(
        depth_ratio=float(row["depth_ratio_k_over_d"]),
        occupancy=float(row["occupancy"]),
        volume_ratio=float(row["volume_ratio"]),
        dimple_count=int(row["dimple_count"]),
        dimple_diameter_mm=float(row["dimple_diameter_mm"]),
    )
    for _, row in top_df.iterrows()
]
compare_labels = [f"S{i}" for i in range(1, len(compare_designs) + 1)]
legend_df = build_design_legend(compare_designs)
selected_launch = LaunchCondition(
    speed_mps=float(speed_mps),
    launch_angle_deg=float(launch_angle_deg),
    spin_rpm=float(spin_rpm),
)

st.markdown("#### Sonuç tablosundaki tasarımlar için uçuş eğrisi karşılaştırması")
flight_comp_df = build_flight_comparison(compare_designs, compare_labels, selected_launch)
flight_long = (
    flight_comp_df.reset_index()
    .melt(id_vars="distance_m", var_name="tasarim", value_name="height_m")
    .dropna()
)
flight_chart = (
    alt.Chart(flight_long)
    .mark_line(point=True, strokeWidth=2)
    .encode(
        x=alt.X("distance_m:Q", title="Mesafe (m)"),
        y=alt.Y("height_m:Q", title="Yukseklik (m)"),
        color=alt.Color("tasarim:N", title="Tasarim"),
        tooltip=[
            alt.Tooltip("tasarim:N", title="Tasarim"),
            alt.Tooltip("distance_m:Q", title="Mesafe (m)", format=".2f"),
            alt.Tooltip("height_m:Q", title="Yukseklik (m)", format=".3f"),
        ],
    )
    .properties(height=340)
    .interactive()
)
st.altair_chart(flight_chart, use_container_width=True)
st.caption(
    "Her çizgi bir tasarımı temsil eder. Daha uzun mesafe ve kontrol edilen yükseklik, "
    "uçuş performansı açısından daha olumlu yorumlanır."
)

with st.expander("Grafik etiketleri (S1, S2, ...)"):
    st.dataframe(
        legend_df.style.format(
            {
                "k/d": "{:.5f}",
                "Yuzey kaplama (%)": "{:.1f}",
                "VDR": "{:.4f}",
                "Dimple capi (mm)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

selected_rank = st.selectbox("Detay icin sira sec", options=top_df["rank"].tolist(), index=0)
selected_row = df.loc[df["rank"] == selected_rank].iloc[0]

selected_design = DimpleDesign(
    depth_ratio=float(selected_row["depth_ratio_k_over_d"]),
    occupancy=float(selected_row["occupancy"]),
    volume_ratio=float(selected_row["volume_ratio"]),
    dimple_count=int(selected_row["dimple_count"]),
    dimple_diameter_mm=float(selected_row["dimple_diameter_mm"]),
)
selected_result, _ = simulate_flight_with_path(selected_launch, selected_design)

st.markdown("#### Sonuç tablosundaki tasarımlar için Cd-Re karşılaştırması")
cd_re_comp_df = build_cd_re_comparison(
    compare_designs,
    compare_labels,
    int(spin_rpm),
    float(context_cfg["speed_min_mps"]),
    float(context_cfg["speed_max_mps"]),
)
cd_re_long = (
    cd_re_comp_df.reset_index()
    .melt(id_vars="re", var_name="tasarim", value_name="cd")
    .dropna()
)
cd_re_chart = (
    alt.Chart(cd_re_long)
    .mark_line(point=True, strokeWidth=2)
    .encode(
        x=alt.X("re:Q", title="Reynolds sayisi (Re)", axis=alt.Axis(format=".2e")),
        y=alt.Y("cd:Q", title="Surukleme katsayisi (Cd)"),
        color=alt.Color("tasarim:N", title="Tasarim"),
        tooltip=[
            alt.Tooltip("tasarim:N", title="Tasarim"),
            alt.Tooltip("re:Q", title="Re", format=".2e"),
            alt.Tooltip("cd:Q", title="Cd", format=".4f"),
        ],
    )
    .properties(height=340)
    .interactive()
)
st.altair_chart(cd_re_chart, use_container_width=True)
st.caption(
    "Grafik yorumu: Cd ne kadar düşükse hava direnci o kadar düşüktür. "
    "Grafikte tablodaki her tasarım ayrı çizgi olarak gösterilir; böylece tasarımları doğrudan karşılaştırabilirsin."
)

why_text = (
    f"Bu tasarimda k/d={selected_design.depth_ratio:.5f}, "
    f"occupancy={to_percent(selected_design.occupancy)}, "
    f"VDR={selected_design.volume_ratio:.4f}. "
    "Model bu geometriyi, secilen hiz-aci-spin kosulunda diger adaylarla karsilastiriyor ve "
    + (
        "no-spin modda en dusuk Cd degerine sahip olan tasarimi ust siraya koyuyor."
        if is_no_spin
        else "spinli modda en iyi genel skor (mesafe + L/D - Cd etkisi) veren tasarimi ust siraya koyuyor."
    )
)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Mesafe", f"{selected_result.carry_distance_m:.2f} m")
col_b.metric("Ucusta kalma suresi", f"{selected_result.flight_time_s:.2f} s")
col_c.metric("Maks yukseklik", f"{selected_result.max_height_m:.2f} m")

st.markdown("**Neden bu sonuc cikti?**")
st.write(why_text)

with st.expander("Terimler sozlugu (teknik bilmeyenler icin)"):
    st.markdown(
        "- **k/d:** Dimple derinligi / top capi. Buyudukce dimple daha derin olur.\n"
        "- **Occupancy:** Top yuzeyinin dimple ile kapli yuzdesi.\n"
        "- **VDR:** Dimple hacim orani.\n"
        "- **Cd:** Hava direnci gostergesi (kucuk olmasi genelde iyi).\n"
        "- **Cl:** Kaldirma etkisi gostergesi.\n"
        "- **L/D:** Kaldirma / surukleme orani. Buyuk olmasi genelde ucusu destekler."
    )
with st.expander("Teknik detaylar (istersen)"):
    st.markdown(
        f"- Secilen tasarim ortalama Cd: {selected_result.avg_cd:.3f}\n"
        f"- Secilen tasarim ortalama Cl: {selected_result.avg_cl:.3f}\n"
        f"- Secilen tasarim ortalama L/D: {selected_result.avg_l_over_d:.3f}\n"
        "- Simulasyon, makalelerdeki Re, Sp, Cd, Cl ve kuvvet denklemlerinin sade bir surrogate modelini kullanir."
    )

st.caption(
    "Not: Bu sayfa karar destegi icindir. Nihai urun secimi icin secilen adaylarin deney/CFD ile dogrulanmasi onerilir."
)
