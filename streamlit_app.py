from __future__ import annotations

from bisect import bisect_left
import time
import altair as alt
import pandas as pd
import streamlit as st

from golf_simulator import (
    BALL_DIAMETER,
    BALL_DIAMETER_MM,
    DimpleDesign,
    LaunchCondition,
    MIN_LAND_WIDTH_MM,
    estimate_hex_packing_land_width_mm,
    export_solidworks_csv,
    export_solidworks_macro,
    export_design_json,
    generate_fibonacci_dimple_centers,
    theoretical_max_dimple_count,
    validate_dimple_placement,
    model_cd_cl,
    reynolds_number,
    run_robust_search,
    run_search_with_launch,
    simulate_flight_with_path,
)


st.set_page_config(page_title="Golf Ball Simulator", layout="wide")
st.title("Golf Ball Simulator - Simple Overview")
st.caption(
    "Goal: quickly see which dimple setup performs better. "
    "Scenario selection: Wind Tunnel or Human Shot."
)


MODE_LABELS = {
    "paper_depth_only": "Depth effect test (single variable)",
    "paper_occupancy_only": "Coverage ratio effect test (single variable)",
    "paper_volume_only": "VDR trend test (literature set)",
    "paper_literature_grid": "All literature designs",
}

MODE_LONG_INFO = {
    "paper_depth_only": "What changes: only dimple depth. What stays fixed: same dimple family (O=81.2%, ND=476).",
    "paper_occupancy_only": "What changes: only coverage family (occupancy groups). What stays fixed: depth (k / ball diameter)=4.55e-3.",
    "paper_volume_only": "What changes: all 15 literature designs are evaluated together to observe the VDR trend.",
    "paper_literature_grid": "What changes: 50 literature-aligned designs (15 measured points + deterministic interpolations).",
}

CONTEXTS = {
    "wind_tunnel": {
        "label": "Wind Tunnel Mode",
        "description": "Wind tunnel conditions: speed 20-140 km/h, focused on no-spin.",
        "speed_unit": "km/h",
        "speed_min_mps": 20.0 / 3.6,
        "speed_max_mps": 140.0 / 3.6,
        "speed_default_mps": 100.0 / 3.6,
        "spin_min": 0,
        "spin_max": 2000,
        "spin_default": 0,
    },
    "human_flight": {
        "label": "Human Shot Mode (PGA reference)",
        "description": "Flight conditions: speed/spin around PGA average launch values.",
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
    "wind_recommended": {
        "label": "Wind Tunnel / Recommended (100 km/h, no-spin)",
        "context": "wind_tunnel",
        "mode": "paper_depth_only",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Simple",
    },
    "wind_min_speed": {
        "label": "Wind Tunnel / Minimum speed (20 km/h)",
        "context": "wind_tunnel",
        "mode": "paper_depth_only",
        "speed_mps": 20.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Simple",
    },
    "wind_max_speed": {
        "label": "Wind Tunnel / Maximum speed (140 km/h)",
        "context": "wind_tunnel",
        "mode": "paper_depth_only",
        "speed_mps": 140.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Simple",
    },
    "wind_occupancy": {
        "label": "Wind Tunnel / Occupancy comparison",
        "context": "wind_tunnel",
        "mode": "paper_occupancy_only",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Simple",
    },
    "wind_vdr": {
        "label": "Wind Tunnel / VDR trend",
        "context": "wind_tunnel",
        "mode": "paper_volume_only",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 8,
        "view_mode": "Simple",
    },
    "wind_full_grid": {
        "label": "Wind Tunnel / All literature designs",
        "context": "wind_tunnel",
        "mode": "paper_literature_grid",
        "speed_mps": 100.0 / 3.6,
        "launch_angle_deg": 11.2,
        "spin_rpm": 0,
        "top_n": 10,
        "view_mode": "Technical",
    },
    "human_pga_avg": {
        "label": "Human Shot / PGA average (74 m/s, 2685 rpm)",
        "context": "human_flight",
        "mode": "paper_literature_grid",
        "speed_mps": 74.0,
        "launch_angle_deg": 11.2,
        "spin_rpm": 2685,
        "top_n": 10,
        "view_mode": "Technical",
    },
    "human_low_spin": {
        "label": "Human Shot / Low spin (74 m/s, 2200 rpm)",
        "context": "human_flight",
        "mode": "paper_literature_grid",
        "speed_mps": 74.0,
        "launch_angle_deg": 11.2,
        "spin_rpm": 2200,
        "top_n": 10,
        "view_mode": "Technical",
    },
    "human_high_spin": {
        "label": "Human Shot / High spin (74 m/s, 3200 rpm)",
        "context": "human_flight",
        "mode": "paper_literature_grid",
        "speed_mps": 74.0,
        "launch_angle_deg": 11.2,
        "spin_rpm": 3200,
        "top_n": 10,
        "view_mode": "Technical",
    },
}


def ensure_session_defaults() -> None:
    p = PRESETS["wind_recommended"]
    st.session_state.setdefault("context", p["context"])
    st.session_state.setdefault("input_mode", "Preset")
    st.session_state.setdefault("preset_key", "wind_recommended")
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
    return f"{value * 100:.1f}%"


def result_comment(row: pd.Series) -> str:
    """
    Story-like reason text for non-technical users.
    """
    depth_ratio = float(row["depth_ratio_k_over_d"])
    occupancy = float(row["occupancy"])
    lod = float(row["avg_l_over_d"])
    distance = float(row["distance_m"])

    if depth_ratio <= 4.55e-3:
        depth_text = "the dimples stay shallow, so the ball cuts through air more cleanly"
    elif depth_ratio <= 6.82e-3:
        depth_text = "the dimple depth is balanced, so ball behavior is neither too aggressive nor too passive"
    else:
        depth_text = "the dimples are deep, so the ball interacts with airflow more strongly"

    if occupancy >= 0.80:
        occ_text = "a larger dimple-covered surface improves flight stability"
    elif occupancy >= 0.63:
        occ_text = "a medium coverage ratio keeps performance balanced"
    else:
        occ_text = "a low coverage ratio may cause earlier speed loss in some conditions"

    if lod >= 0.50:
        lod_text = "a strong lift/drag balance helps preserve distance"
    elif lod >= 0.42:
        lod_text = "an acceptable lift/drag balance delivers satisfactory distance"
    else:
        lod_text = "a limited lift/drag balance reduces distance advantage"

    return (
        f"In this design, {depth_text}; additionally, {occ_text}. "
        f"As a result, {lod_text}, and the estimated carry distance is around {distance:.1f} m."
    )


def faq_data() -> list[tuple[str, str]]:
    return [
        (
            "What does this app do?",
            "It evaluates multiple golf ball designs within literature-based parameter ranges and ranks the best result by the selected criterion.",
        ),
        (
            "How is the test performed?",
            "It runs a simulation for each design. In wind-tunnel style testing (spin=0), the main evaluation is based on Cd. In spin cases, flight equations are also used.",
        ),
        (
            "Where do the formulas come from?",
            "Core equations come from standard aerodynamic formulations in the literature: Re, Sp, Fd, Fl, Cd, Cl. Parameter trends are based on findings in the referenced papers.",
        ),
        (
            "Why is the speed range set this way?",
            "It depends on the scenario: 20-140 km/h in Wind Tunnel mode, and 45-85 m/s in Human Shot mode.",
        ),
        (
            "Why is the rank-1 design the best?",
            "Because all candidates are compared under the same selected condition, and the best one by the chosen metric is placed at rank 1. (Cd in no-spin tests, composite score in spin tests)",
        ),
        (
            "Is this result final?",
            "This model is for rapid screening and candidate selection. Final confirmation requires CFD or experimental validation.",
        ),
        (
            "What does a single-variable test mean?",
            "It means changing only one parameter while keeping others fixed, making each effect easier to isolate.",
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
                "estimated_land_width_mm": estimate_hex_packing_land_width_mm(design),
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
                "estimated_land_width_mm": estimate_hex_packing_land_width_mm(design),
            }
        )
    return pd.DataFrame(rows), len(scenarios)


def ranking_explainer(spin_rpm: int) -> str:
    if spin_rpm == 0:
        return "Ranking criterion: **Average Cd (lower is better)**. (Wind-tunnel style, no-spin)"
    return "Ranking criterion: **Composite score of Distance + L/D + Cd**. (Flight style, with spin)"


def format_unique(values: list[float], fmt: str) -> str:
    return ", ".join(format(v, fmt) for v in values)


def mode_test_summary(df: pd.DataFrame) -> tuple[list[str], list[str], dict[str, str]]:
    depth_vals = sorted(df["depth_ratio_k_over_d"].unique().tolist())
    occ_vals = sorted(df["occupancy"].unique().tolist())
    vdr_vals = sorted(df["volume_ratio"].unique().tolist())
    nd_vals = sorted(df["dimple_count"].unique().tolist())
    dd_vals = sorted(df["dimple_diameter_mm"].unique().tolist())

    specs = {
        "Depth ratio (k / ball diameter)": format_unique(depth_vals, ".5f"),
        "Surface coverage ratio (%)": format_unique([v * 100 for v in occ_vals], ".1f"),
        "Dimple volume ratio (VDR)": format_unique(vdr_vals, ".4f"),
        "Total dimple count": ", ".join(str(int(v)) for v in nd_vals),
        "Dimple diameter (mm)": format_unique(dd_vals, ".2f"),
    }

    changing = [k for k, vals in [
        ("Depth ratio (k / ball diameter)", depth_vals),
        ("Surface coverage ratio (%)", occ_vals),
        ("Dimple volume ratio (VDR)", vdr_vals),
        ("Total dimple count", nd_vals),
        ("Dimple diameter (mm)", dd_vals),
    ] if len(vals) > 1]
    fixed = [k for k in specs.keys() if k not in changing]
    return changing, fixed, specs


def build_results_formula_rows(is_no_spin: bool, view_mode: str) -> list[dict[str, str]]:
    rank_formula = (
        "Sort by 1 / Average Cd (descending), so lower drag gets higher rank."
        if is_no_spin
        else "Sort by Score (descending), where Score = carry + 25 * (L/D) - 30 * Cd."
    )
    rows = [
        {
            "Column": "Rank",
            "Formula / Calculation": rank_formula,
            "Meaning": "Position after comparing all tested candidates under the same launch condition.",
        },
        {
            "Column": "Estimated carry distance (m)",
            "Formula / Calculation": (
                "state=(x,y,vx,vy). Just before landing: (x1,y1) with y1>0. "
                "Next step: (x2,y2) with y2<0. "
                "Touchdown x at y=0 is found by linear interpolation: "
                "x_land = x1 + ((0-y1)/(y2-y1))*(x2-x1)."
            ),
            "Meaning": "Estimated horizontal landing point (carry) at the exact ground-crossing moment.",
        },
        {
            "Column": "Time in flight (s)",
            "Formula / Calculation": "Simulation advances with dt=0.002 s. Each loop: t += dt. Stop when y < 0.",
            "Meaning": "Total airborne duration from launch until ground contact.",
        },
        {
            "Column": "Maximum height (m)",
            "Formula / Calculation": "At each step, keep max_height = max(max_height, y).",
            "Meaning": "Highest y value reached during flight.",
        },
        {
            "Column": "Dimple depth (mm)",
            "Formula / Calculation": f"dimple_depth_mm = (k / ball_diameter) * {BALL_DIAMETER_MM:.2f}",
            "Meaning": "Absolute depth converted from depth ratio referenced to ball diameter.",
        },
        {
            "Column": "Depth ratio (k / ball diameter)",
            "Formula / Calculation": "Input geometry parameter from literature-supported design sets.",
            "Meaning": "Relative dimple depth (depth divided by ball diameter).",
        },
        {
            "Column": "Depth ratio (k / dimple diameter)",
            "Formula / Calculation": "depth_ratio_k_over_dimple = dimple_depth_mm / dimple_diameter_mm",
            "Meaning": "Manufacturing-focused depth ratio using dimple diameter as denominator.",
        },
        {
            "Column": "Surface coverage ratio (%)",
            "Formula / Calculation": "surface_coverage_pct = occupancy * 100",
            "Meaning": "Share of ball surface covered by dimples.",
        },
        {
            "Column": "Dimple volume ratio (VDR)",
            "Formula / Calculation": "Input geometry parameter from literature VDR matrix.",
            "Meaning": "Dimensionless dimple-volume indicator used by the Cd/Cl surrogate.",
        },
        {
            "Column": "Total dimple count",
            "Formula / Calculation": "Input geometry parameter (integer), not solved by flight equations.",
            "Meaning": "Number of dimples on the ball design.",
        },
        {
            "Column": "Dimple diameter (mm)",
            "Formula / Calculation": "From literature groups: dimple_diameter_mm = (Cmean/d) * ball_diameter_mm.",
            "Meaning": "Representative mean dimple diameter for that occupancy family.",
        },
        {
            "Column": "Estimated land width (mm)",
            "Formula / Calculation": (
                "Approx. hex-packing gap: area_per = ball_surface_area / dimple_count; "
                "center_spacing = sqrt((2*area_per)/sqrt(3)); land = center_spacing - dimple_diameter."
            ),
            "Meaning": "Estimated average gap between neighboring dimples used for manufacturability screening.",
        },
        {
            "Column": "Why is it good?",
            "Formula / Calculation": "Rule-based text from thresholds on depth ratio, occupancy, L/D and reported distance.",
            "Meaning": "Plain-language explanation generated for non-technical users.",
        },
    ]

    if view_mode != "Simple":
        rows.extend(
            [
                {
                    "Column": "Score",
                    "Formula / Calculation": "score = carry_distance + 25 * (L/D) - 30 * Cd",
                    "Meaning": "Composite optimization score used for spin-enabled ranking.",
                },
                {
                    "Column": "Average Cd",
                    "Formula / Calculation": "avg_cd = sum(Cd_t) / samples",
                    "Meaning": "Mean drag coefficient sampled along the trajectory.",
                },
                {
                    "Column": "Average Cl",
                    "Formula / Calculation": "avg_cl = sum(Cl_t) / samples",
                    "Meaning": "Mean lift coefficient sampled along the trajectory.",
                },
                {
                    "Column": "Average L/D",
                    "Formula / Calculation": "avg_l_over_d = sum(Cl_t / max(Cd_t, 1e-6)) / samples",
                    "Meaning": "Mean aerodynamic efficiency across the full flight.",
                },
                {
                    "Column": "Surface coverage ratio (0-1)",
                    "Formula / Calculation": "occupancy in native fractional scale (0 to 1).",
                    "Meaning": "Same quantity as percentage column, without x100 conversion.",
                },
            ]
        )

    return rows


def build_robust_formula_rows() -> list[dict[str, str]]:
    return [
        {
            "Column": "Rank",
            "Formula / Calculation": "Sort by (Robust score, Worst score, Average carry, -Nominal Cd, Nominal L/D), descending.",
            "Meaning": "Best robust performer gets rank 1 with deterministic tie-breaks.",
        },
        {
            "Column": "Robust score",
            "Formula / Calculation": "robust = 0.60 * mean_score + 0.35 * worst_score - 0.05 * score_std",
            "Meaning": "Balances average performance, worst-case safety, and consistency.",
        },
        {
            "Column": "Average score",
            "Formula / Calculation": "mean_score = mean(scenario_scores)",
            "Meaning": "Average score across all robustness scenarios.",
        },
        {
            "Column": "Worst score",
            "Formula / Calculation": "worst_score = min(scenario_scores)",
            "Meaning": "Conservative lower-bound performance across scenarios.",
        },
        {
            "Column": "Score std dev",
            "Formula / Calculation": "score_std = pstdev(scenario_scores)",
            "Meaning": "Population standard deviation of scenario scores.",
        },
        {
            "Column": "Average carry (m)",
            "Formula / Calculation": "mean_distance_m = mean(scenario_carry_distances)",
            "Meaning": "Average carry over all robustness scenarios.",
        },
        {
            "Column": "Minimum carry (m)",
            "Formula / Calculation": "min_distance_m = min(scenario_carry_distances)",
            "Meaning": "Worst carry observed in robustness scenarios.",
        },
        {
            "Column": "Nominal carry (m)",
            "Formula / Calculation": "carry_distance from simulate_flight(base_launch, design)",
            "Meaning": "Carry under the selected baseline launch only.",
        },
        {
            "Column": "Nominal Cd",
            "Formula / Calculation": "avg_cd from nominal simulation",
            "Meaning": "Baseline drag indicator for the chosen launch point.",
        },
        {
            "Column": "Nominal Cl",
            "Formula / Calculation": "avg_cl from nominal simulation",
            "Meaning": "Baseline lift indicator for the chosen launch point.",
        },
        {
            "Column": "Nominal L/D",
            "Formula / Calculation": "avg_l_over_d from nominal simulation",
            "Meaning": "Baseline aerodynamic efficiency for the chosen launch point.",
        },
        {
            "Column": "Depth ratio (k / ball diameter)",
            "Formula / Calculation": "Input geometry parameter from literature-supported design sets.",
            "Meaning": "Relative dimple depth.",
        },
        {
            "Column": "Depth ratio (k / dimple diameter)",
            "Formula / Calculation": "depth_ratio_k_over_dimple = dimple_depth_mm / dimple_diameter_mm",
            "Meaning": "Secondary depth ratio often used by design/manufacturing teams.",
        },
        {
            "Column": "Surface coverage ratio (0-1)",
            "Formula / Calculation": "Input occupancy as fraction in [0, 1].",
            "Meaning": "Fractional dimple-covered surface area.",
        },
        {
            "Column": "Dimple volume ratio (VDR)",
            "Formula / Calculation": "Input geometry parameter from literature VDR matrix.",
            "Meaning": "Dimensionless dimple-volume indicator.",
        },
        {
            "Column": "Total dimple count",
            "Formula / Calculation": "Input geometry parameter (integer).",
            "Meaning": "Total number of dimples in the design.",
        },
        {
            "Column": "Dimple diameter (mm)",
            "Formula / Calculation": "From literature groups: dimple_diameter_mm = (Cmean/d) * ball_diameter_mm.",
            "Meaning": "Representative mean dimple diameter for the design family.",
        },
        {
            "Column": "Estimated land width (mm)",
            "Formula / Calculation": (
                "Approx. hex-packing gap: area_per = ball_surface_area / dimple_count; "
                "center_spacing = sqrt((2*area_per)/sqrt(3)); land = center_spacing - dimple_diameter."
            ),
            "Meaning": f"Used for manufacturability filtering (must be >= {MIN_LAND_WIDTH_MM:.2f} mm).",
        },
    ]


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
                "Label": f"S{idx}",
                "k / ball diameter": d.depth_ratio,
                "Surface coverage (%)": d.occupancy * 100.0,
                "VDR": d.volume_ratio,
                "Dimple count": d.dimple_count,
                "Dimple diameter (mm)": d.dimple_diameter_mm,
            }
        )
    return pd.DataFrame(rows)


ensure_session_defaults()

with st.sidebar:
    st.header("1) Input settings")
    if "context_widget" not in st.session_state:
        st.session_state["context_widget"] = st.session_state["context"]
    context = st.selectbox(
        "Test scenario",
        options=list(CONTEXTS.keys()),
        format_func=lambda key: CONTEXTS[key]["label"],
        index=list(CONTEXTS.keys()).index(st.session_state["context_widget"]),
        key="context_widget",
    )
    st.session_state["context"] = context
    context_cfg = CONTEXTS[context]

    input_mode = st.radio(
        "Configuration mode",
        options=["Preset", "Custom"],
        horizontal=True,
        index=0 if st.session_state["input_mode"] == "Preset" else 1,
        key="input_mode",
    )

    if input_mode == "Preset":
        preset_options = [k for k, v in PRESETS.items() if v["context"] == context]
        if st.session_state.get("preset_key") not in preset_options:
            st.session_state["preset_key"] = preset_options[0]
        preset_key = st.selectbox(
            "Select preset",
            options=preset_options,
            format_func=lambda key: PRESETS[key]["label"],
            index=preset_options.index(st.session_state["preset_key"]),
            key="preset_key",
            help="Preset configurations based on test logic from literature.",
        )
        st.caption("The selected preset is applied automatically.")
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
            st.session_state["recalc_message"] = f"Preset applied automatically: {PRESETS[preset_key]['label']}"
            st.rerun()
    else:
        st.caption("In custom mode, results update automatically when sliders change.")

    mode = st.selectbox(
        "Test type (explicit selection)",
        options=list(MODE_LABELS.keys()),
        format_func=lambda key: MODE_LABELS[key],
        index=list(MODE_LABELS.keys()).index(st.session_state["mode"]),
        help="Literature-aligned test type. Default is single-variable.",
        key="mode",
    )
    current_speed_mps = float(st.session_state["speed_mps"])
    current_speed_mps = max(context_cfg["speed_min_mps"], min(context_cfg["speed_max_mps"], current_speed_mps))
    if context_cfg["speed_unit"] == "km/h":
        speed_input = st.slider(
            "Wind tunnel speed (km/h)",
            min_value=int(round(context_cfg["speed_min_mps"] * 3.6)),
            max_value=int(round(context_cfg["speed_max_mps"] * 3.6)),
            value=int(round(current_speed_mps * 3.6)),
            step=1,
        )
        speed_mps = speed_input / 3.6
    else:
        speed_input = st.slider(
            "Ball launch speed (m/s)",
            min_value=float(context_cfg["speed_min_mps"]),
            max_value=float(context_cfg["speed_max_mps"]),
            value=float(round(current_speed_mps, 1)),
            step=0.5,
        )
        speed_mps = float(speed_input)
    st.session_state["speed_mps"] = speed_mps

    launch_angle_deg = st.slider(
        "Launch angle (degrees)",
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
        "How many results should be listed?",
        min_value=3,
        max_value=25,
        value=int(st.session_state["top_n"]),
        step=1,
        key="top_n",
    )
    view_mode = st.radio(
        "View",
        options=["Simple", "Technical"],
        horizontal=True,
        index=0 if st.session_state["view_mode"] == "Simple" else 1,
        key="view_mode",
    )
    st.caption(context_cfg["description"])

speed_mps = float(st.session_state["speed_mps"])

st.info(
    f"Scenario: **{CONTEXTS[context]['label']}**  |  "
    f"Test type: **{MODE_LABELS[mode]}**  |  "
    f"Speed: **{speed_mps:.1f} m/s ({speed_mps*3.6:.0f} km/h)**  |  "
    f"Spin: **{int(spin_rpm)} rpm**"
)

with st.expander("Help / Methodology / FAQ", expanded=False):
    st.markdown(
        "**Quick usage:** Select a preset -> The system applies it automatically -> Review the results.\n\n"
        "**Selection tips:**\n"
        "- Wind Tunnel Mode: no-spin aerodynamic comparison (Cd-focused)\n"
        "- Human Shot Mode: spin flight comparison (distance and L/D-driven)\n"
        "- Single-variable analysis: `Depth effect` or `Coverage effect`\n\n"
        "**Model:**\n"
        "- `Re = rho*U*d/mu`, `Sp = pi*d*N/U`\n"
        "- `Fd = 0.5*rho*U^2*A*Cd`, `Fl = 0.5*rho*U^2*A*Cl`\n"
        "- In `spin=0` mode, ranking uses `Cd`; in spin mode, it uses a composite score."
    )
    st.markdown("---")
    for q, a in faq_data():
        st.markdown(f"**Question:** {q}")
        st.write(f"**Answer:** {a}")
        st.markdown("---")


if st.session_state.get("auto_recalc", False):
    msg = st.session_state.get("recalc_message", "Settings changed")
    with st.spinner(f"{msg}. Calculating results..."):
        time.sleep(1.2)
        df = run_cached_search(
            speed_mps=speed_mps,
            launch_angle_deg=launch_angle_deg,
            spin_rpm=spin_rpm,
            mode=mode,
        )
    st.success("Calculation completed. Results are updated.")
    st.session_state["auto_recalc"] = False
    st.session_state["recalc_message"] = ""
else:
    df = run_cached_search(
        speed_mps=speed_mps,
        launch_angle_deg=launch_angle_deg,
        spin_rpm=spin_rpm,
        mode=mode,
    )
if df.empty:
    st.error(
        "No manufacturable design found for this test mode. "
        "All candidate designs exceed the physical packing limit or fail the minimum land-width check. "
        "Please switch to a different test type (e.g. **Literature grid**) or adjust input parameters."
    )
    st.stop()

top_df = df.head(top_n).copy()
best = top_df.iloc[0]
is_no_spin = int(spin_rpm) == 0
changing_params, fixed_params, tested_specs = mode_test_summary(df)

st.subheader("What does the system do in this test?")
st.markdown(
    "1. Candidate designs are prepared according to the selected test type based on literature.\n"
    "2. `Cd/Cl` is computed for each candidate and a flight simulation is run.\n"
    "3. Results are compared.\n"
    "4. The best row is placed at rank 1 according to the selected ranking criterion."
)

c1, c2 = st.columns(2)
c1.metric("Tested design count", f"{len(df)}")
c2.metric("Number of varying parameters", f"{len(changing_params)}")

st.markdown("**Parameters that vary in this test:** " + (", ".join(changing_params) if changing_params else "None"))
st.markdown("**Parameters fixed in this test:** " + (", ".join(fixed_params) if fixed_params else "None"))

with st.expander("All values tested in this run"):
    for k, v in tested_specs.items():
        st.markdown(f"- **{k}:** {v}")

mean_distance = float(df["distance_m"].mean())
gain_vs_mean = float(best["distance_m"] - mean_distance)

col1, col2, col3 = st.columns(3)
col1.metric("Best distance", f"{best['distance_m']:.2f} m")
col2.metric("Difference vs mean result", f"{gain_vs_mean:+.2f} m")
col3.metric("Total tested designs", f"{len(df)}")
st.caption(ranking_explainer(int(spin_rpm)))

st.success(
    "Quick summary: The system tests all candidates under the same condition and places the best one at the top based on the chosen ranking criterion."
)

st.subheader("2) Results (starting from best)")
st.caption("The table includes core design information required for production decisions.")
st.caption(
    f"Manufacturability filter applied: estimated land width >= {MIN_LAND_WIDTH_MM:.2f} mm "
    "and dimple count <= theoretical packing limit. Non-buildable designs are excluded."
)

simple_df = top_df.copy()
simple_df["occupancy_pct"] = simple_df["occupancy"] * 100.0
simple_df["dimple_depth_mm"] = simple_df["depth_ratio_k_over_d"] * BALL_DIAMETER_MM
simple_df["depth_ratio_k_over_dimple"] = simple_df["dimple_depth_mm"] / simple_df["dimple_diameter_mm"]
simple_df["why_good"] = simple_df.apply(result_comment, axis=1)

if view_mode == "Simple":
    show_df = simple_df[
        [
            "rank",
            "distance_m",
            "flight_time_s",
            "max_height_m",
            "dimple_depth_mm",
            "depth_ratio_k_over_d",
            "depth_ratio_k_over_dimple",
            "occupancy_pct",
            "volume_ratio",
            "dimple_count",
            "dimple_diameter_mm",
            "estimated_land_width_mm",
            "why_good",
        ]
    ].rename(
        columns={
            "rank": "Rank",
            "distance_m": "Estimated carry distance (m)",
            "flight_time_s": "Time in flight (s)",
            "max_height_m": "Maximum height (m)",
            "dimple_depth_mm": "Dimple depth (mm)",
            "depth_ratio_k_over_d": "Depth ratio (k / ball diameter)",
            "depth_ratio_k_over_dimple": "Depth ratio (k / dimple diameter)",
            "occupancy_pct": "Surface coverage ratio (%)",
            "volume_ratio": "Dimple volume ratio (VDR)",
            "dimple_count": "Total dimple count",
            "dimple_diameter_mm": "Dimple diameter (mm)",
            "estimated_land_width_mm": "Estimated land width (mm)",
            "why_good": "Why is it good?",
        }
    )
    st.dataframe(
        show_df.style.format(
            {
                "Estimated carry distance (m)": "{:.3f}",
                "Time in flight (s)": "{:.2f}",
                "Maximum height (m)": "{:.2f}",
                "Dimple depth (mm)": "{:.3f}",
                "Depth ratio (k / ball diameter)": "{:.5f}",
                "Depth ratio (k / dimple diameter)": "{:.5f}",
                "Surface coverage ratio (%)": "{:.1f}",
                "Dimple volume ratio (VDR)": "{:.4f}",
                "Dimple diameter (mm)": "{:.2f}",
                "Estimated land width (mm)": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
else:
    tech_df = top_df.rename(
        columns={
            "rank": "Rank",
            "score": "Score",
            "distance_m": "Estimated carry distance (m)",
            "flight_time_s": "Time in flight (s)",
            "max_height_m": "Maximum height (m)",
            "avg_cd": "Average Cd",
            "avg_cl": "Average Cl",
            "avg_l_over_d": "Average L/D",
            "depth_ratio_k_over_d": "Depth ratio (k / ball diameter)",
            "occupancy": "Surface coverage ratio (0-1)",
            "volume_ratio": "Dimple volume ratio (VDR)",
            "dimple_count": "Total dimple count",
            "dimple_diameter_mm": "Dimple diameter (mm)",
            "estimated_land_width_mm": "Estimated land width (mm)",
        }
    ).copy()
    tech_df["Dimple depth (mm)"] = tech_df["Depth ratio (k / ball diameter)"] * BALL_DIAMETER_MM
    tech_df["Depth ratio (k / dimple diameter)"] = (
        tech_df["Dimple depth (mm)"] / tech_df["Dimple diameter (mm)"]
    )
    tech_df["Surface coverage ratio (%)"] = tech_df["Surface coverage ratio (0-1)"] * 100.0
    if is_no_spin:
        # In no-spin wind tunnel mode, ranking is based on Cd.
        # Hide synthetic score to avoid confusion from negative values.
        tech_df = tech_df.drop(columns=["Score"])

    tech_format = {
        "Score": "{:.2f}",
        "Estimated carry distance (m)": "{:.3f}",
        "Time in flight (s)": "{:.2f}",
        "Maximum height (m)": "{:.2f}",
        "Average Cd": "{:.3f}",
        "Average Cl": "{:.3f}",
        "Average L/D": "{:.3f}",
        "Depth ratio (k / ball diameter)": "{:.5f}",
        "Depth ratio (k / dimple diameter)": "{:.5f}",
        "Surface coverage ratio (0-1)": "{:.3f}",
        "Surface coverage ratio (%)": "{:.1f}",
        "Dimple volume ratio (VDR)": "{:.4f}",
        "Dimple diameter (mm)": "{:.2f}",
        "Dimple depth (mm)": "{:.3f}",
        "Estimated land width (mm)": "{:.3f}",
    }
    if is_no_spin and "Score" in tech_format:
        tech_format.pop("Score")

    st.dataframe(
        tech_df.style.format(tech_format),
        width="stretch",
        hide_index=True,
    )

with st.expander("Results table columns: formulas and meanings"):
    st.caption(
        "Flight metrics are produced by RK4 integration (dt=0.002 s) of aerodynamic forces "
        "Fd = 0.5 * rho * U^2 * A * Cd and Fl = 0.5 * rho * U^2 * A * Cl."
    )
    st.markdown(
        "- `state = (x, y, vx, vy)`: `x` is horizontal distance, `y` is height.\n"
        "- `(x1, y1)`: last point above ground (`y1 > 0`).\n"
        "- `(x2, y2)`: next point below ground (`y2 < 0`).\n"
        "- Since landing happens between these two points, the exact touchdown `x` is interpolated."
    )
    st.dataframe(
        pd.DataFrame(build_results_formula_rows(is_no_spin=is_no_spin, view_mode=view_mode)),
        width="stretch",
        hide_index=True,
    )


st.subheader("2B) Global scenario scoreboard (robust)")
if int(spin_rpm) == 0:
    st.info(
        "The global robust scoreboard is intended for spin-enabled production scenarios. "
        "Therefore, it is not shown when spin=0."
    )
else:
    robust_df, scenario_count = run_cached_robust_search(
        speed_mps=speed_mps,
        launch_angle_deg=launch_angle_deg,
        spin_rpm=spin_rpm,
        mode=mode,
    )
    if robust_df.empty:
        st.warning(
            "No manufacturable design found for robust analysis in this mode. "
            "Try a different test type."
        )
    else:
        robust_top_df = robust_df.head(top_n).copy()
        robust_best = robust_top_df.iloc[0]

        r1, r2, r3 = st.columns(3)
        r1.metric("Global best robust score", f"{robust_best['robust_score']:.2f}")
        r2.metric("Global average carry", f"{robust_best['mean_distance_m']:.2f} m")
        r3.metric("Global minimum carry", f"{robust_best['min_distance_m']:.2f} m")

        st.caption(
            f"This table aggregates outcomes from {scenario_count} different speed/angle/spin combinations "
            "around the selected baseline launch. It is used to choose safer candidates for production."
        )

        robust_show_df = robust_top_df.rename(
            columns={
                "rank": "Rank",
                "robust_score": "Robust score",
                "mean_score": "Average score",
                "worst_score": "Worst score",
                "score_std": "Score std dev",
                "mean_distance_m": "Average carry (m)",
                "min_distance_m": "Minimum carry (m)",
                "nominal_distance_m": "Nominal carry (m)",
                "nominal_avg_cd": "Nominal Cd",
                "nominal_avg_cl": "Nominal Cl",
                "nominal_avg_l_over_d": "Nominal L/D",
                "depth_ratio_k_over_d": "Depth ratio (k / ball diameter)",
                "occupancy": "Surface coverage ratio (0-1)",
                "volume_ratio": "Dimple volume ratio (VDR)",
                "dimple_count": "Total dimple count",
                "dimple_diameter_mm": "Dimple diameter (mm)",
                "estimated_land_width_mm": "Estimated land width (mm)",
            }
        )
        robust_show_df["Dimple depth (mm)"] = (
            robust_show_df["Depth ratio (k / ball diameter)"] * BALL_DIAMETER_MM
        )
        robust_show_df["Depth ratio (k / dimple diameter)"] = (
            robust_show_df["Dimple depth (mm)"] / robust_show_df["Dimple diameter (mm)"]
        )

        st.dataframe(
            robust_show_df.style.format(
                {
                    "Robust score": "{:.2f}",
                    "Average score": "{:.2f}",
                    "Worst score": "{:.2f}",
                    "Score std dev": "{:.2f}",
                    "Average carry (m)": "{:.2f}",
                    "Minimum carry (m)": "{:.2f}",
                    "Nominal carry (m)": "{:.2f}",
                    "Nominal Cd": "{:.3f}",
                    "Nominal Cl": "{:.3f}",
                    "Nominal L/D": "{:.3f}",
                    "Depth ratio (k / ball diameter)": "{:.5f}",
                    "Depth ratio (k / dimple diameter)": "{:.5f}",
                    "Surface coverage ratio (0-1)": "{:.3f}",
                    "Dimple volume ratio (VDR)": "{:.4f}",
                    "Dimple diameter (mm)": "{:.2f}",
                    "Dimple depth (mm)": "{:.3f}",
                    "Estimated land width (mm)": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        with st.expander("Global robust table columns: formulas and meanings"):
            st.caption(
                "Scenario set uses 27 launch combinations around baseline: "
                "speed = base * {0.92, 1.00, 1.08}, angle = base + {-1, 0, +1} deg, "
                "spin = clip(base * {0.90, 1.00, 1.10}, 1500, 4500). "
                "Each scenario score is rank_value(result, launch); in this table (spin-enabled), "
                "that score is carry + 25*(L/D) - 30*Cd."
            )
            st.dataframe(
                pd.DataFrame(build_robust_formula_rows()),
                width="stretch",
                hide_index=True,
            )

        st.markdown("#### Cd-Re comparison for designs in the global robust table")
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
        robust_legend_df["Label"] = robust_compare_labels

        robust_cd_re_comp_df = build_cd_re_comparison(
            robust_compare_designs,
            robust_compare_labels,
            int(spin_rpm),
            float(context_cfg["speed_min_mps"]),
            float(context_cfg["speed_max_mps"]),
        )
        robust_cd_re_long = (
            robust_cd_re_comp_df.reset_index()
            .melt(id_vars="re", var_name="design", value_name="cd")
            .dropna()
        )
        robust_cd_re_chart = (
            alt.Chart(robust_cd_re_long)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("re:Q", title="Reynolds number (Re)", axis=alt.Axis(format=".2e")),
                y=alt.Y("cd:Q", title="Drag coefficient (Cd)"),
                color=alt.Color("design:N", title="Robust design"),
                tooltip=[
                    alt.Tooltip("design:N", title="Design"),
                    alt.Tooltip("re:Q", title="Re", format=".2e"),
                    alt.Tooltip("cd:Q", title="Cd", format=".4f"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(robust_cd_re_chart, use_container_width=True)
        st.caption(
            "R1, R2, ... labels represent the robust scoreboard ranking. "
            "The lower the curve, the lower the Cd in that speed band."
        )
        with st.expander("Robust Cd-Re chart labels (R1, R2, ...)"):
            st.dataframe(
                robust_legend_df.style.format(
                    {
                        "k / ball diameter": "{:.5f}",
                        "Surface coverage (%)": "{:.1f}",
                        "VDR": "{:.4f}",
                        "Dimple diameter (mm)": "{:.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


st.subheader("3) Select a design and view its flight")
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

st.markdown("#### Flight trajectory comparison for designs in the results table")
flight_comp_df = build_flight_comparison(compare_designs, compare_labels, selected_launch)
flight_long = (
    flight_comp_df.reset_index()
    .melt(id_vars="distance_m", var_name="design", value_name="height_m")
    .dropna()
)
flight_chart = (
    alt.Chart(flight_long)
    .mark_line(point=True, strokeWidth=2)
    .encode(
        x=alt.X("distance_m:Q", title="Distance (m)"),
        y=alt.Y("height_m:Q", title="Height (m)"),
        color=alt.Color("design:N", title="Design"),
        tooltip=[
            alt.Tooltip("design:N", title="Design"),
            alt.Tooltip("distance_m:Q", title="Distance (m)", format=".2f"),
            alt.Tooltip("height_m:Q", title="Height (m)", format=".3f"),
        ],
    )
    .properties(height=340)
    .interactive()
)
st.altair_chart(flight_chart, use_container_width=True)
st.caption(
    "Each line represents one design. Longer distance and controlled height "
    "are generally interpreted as better flight performance."
)

with st.expander("Chart labels (S1, S2, ...)"):
    st.dataframe(
        legend_df.style.format(
            {
                "k / ball diameter": "{:.5f}",
                "Surface coverage (%)": "{:.1f}",
                "VDR": "{:.4f}",
                "Dimple diameter (mm)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

selected_rank = st.selectbox("Select rank for details", options=top_df["rank"].tolist(), index=0)
selected_row = df.loc[df["rank"] == selected_rank].iloc[0]

selected_design = DimpleDesign(
    depth_ratio=float(selected_row["depth_ratio_k_over_d"]),
    occupancy=float(selected_row["occupancy"]),
    volume_ratio=float(selected_row["volume_ratio"]),
    dimple_count=int(selected_row["dimple_count"]),
    dimple_diameter_mm=float(selected_row["dimple_diameter_mm"]),
)
selected_result, _ = simulate_flight_with_path(selected_launch, selected_design)

st.markdown("#### Cd-Re comparison for designs in the results table")
cd_re_comp_df = build_cd_re_comparison(
    compare_designs,
    compare_labels,
    int(spin_rpm),
    float(context_cfg["speed_min_mps"]),
    float(context_cfg["speed_max_mps"]),
)
cd_re_long = (
    cd_re_comp_df.reset_index()
    .melt(id_vars="re", var_name="design", value_name="cd")
    .dropna()
)
cd_re_chart = (
    alt.Chart(cd_re_long)
    .mark_line(point=True, strokeWidth=2)
    .encode(
        x=alt.X("re:Q", title="Reynolds number (Re)", axis=alt.Axis(format=".2e")),
        y=alt.Y("cd:Q", title="Drag coefficient (Cd)"),
        color=alt.Color("design:N", title="Design"),
        tooltip=[
            alt.Tooltip("design:N", title="Design"),
            alt.Tooltip("re:Q", title="Re", format=".2e"),
            alt.Tooltip("cd:Q", title="Cd", format=".4f"),
        ],
    )
    .properties(height=340)
    .interactive()
)
st.altair_chart(cd_re_chart, use_container_width=True)
st.caption(
    "Chart interpretation: lower Cd means lower aerodynamic drag. "
    "Each design in the table is shown as a separate line, allowing direct comparison."
)

why_text = (
    f"In this design, k/ball_d={selected_design.depth_ratio:.5f}, "
    f"occupancy={to_percent(selected_design.occupancy)}, "
    f"VDR={selected_design.volume_ratio:.4f}. "
    "The model compares this geometry against other candidates under the selected speed-angle-spin condition and "
    + (
        "in no-spin mode ranks the design with the lowest Cd at the top."
        if is_no_spin
        else "in spin mode ranks the design with the best overall score (distance + L/D - Cd effect) at the top."
    )
)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Distance", f"{selected_result.carry_distance_m:.2f} m")
col_b.metric("Time in flight", f"{selected_result.flight_time_s:.2f} s")
col_c.metric("Max height", f"{selected_result.max_height_m:.2f} m")

st.markdown("**Why did this result occur?**")
st.write(why_text)

with st.expander("Glossary (for non-technical users)"):
    st.markdown(
        "- **k / ball diameter:** Dimple depth divided by ball diameter. Larger values mean deeper dimples.\n"
        "- **Occupancy:** Percentage of the ball surface covered by dimples.\n"
        "- **VDR:** Dimple volume ratio.\n"
        "- **Cd:** Aerodynamic drag indicator (lower is generally better).\n"
        "- **Cl:** Lift effect indicator.\n"
        "- **L/D:** Lift / drag ratio. Higher values generally support better flight."
    )
with st.expander("Technical details"):
    st.markdown(
        f"- Selected design average Cd: {selected_result.avg_cd:.3f}\n"
        f"- Selected design average Cl: {selected_result.avg_cl:.3f}\n"
        f"- Selected design average L/D: {selected_result.avg_l_over_d:.3f}\n"
        "- The simulation uses a simplified surrogate model of Re, Sp, Cd, Cl, and force equations from the literature."
    )

st.caption(
    "Note: This page is intended for decision support. Final product selection should be validated with experiments/CFD."
)

# ---------------------------------------------------------------
# 4) Export for SolidWorks
# ---------------------------------------------------------------
st.subheader("4) Export for SolidWorks")
st.markdown(
    "Generate a **complete SolidWorks import pack** for the selected design.  \n"
    "This uses Fibonacci (golden-angle) lattice to distribute dimple centers uniformly "
    "across the sphere — much more efficient than latitude-ring / circular patterns."
)

import csv
import io
import json as _json
from pathlib import Path as _Path

export_depth_mm = selected_design.depth_ratio * BALL_DIAMETER_MM
export_depth_over_dimple = export_depth_mm / selected_design.dimple_diameter_mm

theo_max = theoretical_max_dimple_count(selected_design.dimple_diameter_mm)
utilization_pct = (selected_design.dimple_count / max(theo_max, 1)) * 100.0

e1, e2, e3 = st.columns(3)
e1.metric("Dimple depth (mm)", f"{export_depth_mm:.4f}")
e1.metric("Dimple diameter (mm)", f"{selected_design.dimple_diameter_mm:.2f}")
e2.metric("Total dimple count", f"{selected_design.dimple_count}")
e2.metric("Estimated land width (mm)", f"{estimate_hex_packing_land_width_mm(selected_design):.3f}")
e3.metric("Theoretical max count", f"{theo_max}")
e3.metric("Packing utilization", f"{utilization_pct:.1f}%")

if selected_design.dimple_count > theo_max:
    st.error(
        f"This design requests **{selected_design.dimple_count}** dimples but the realistic maximum "
        f"for **{selected_design.dimple_diameter_mm:.2f} mm** diameter on a **{BALL_DIAMETER_MM:.2f} mm** ball "
        f"is approximately **{theo_max}**. This design is **physically non-buildable** regardless of placement method. "
        "Please select a lower-ranked design or one with smaller dimple diameter."
    )
    st.stop()
elif utilization_pct > 90:
    st.warning(
        f"Packing utilization is **{utilization_pct:.1f}%** (very tight). "
        "Minor placement violations may occur. Verify in SolidWorks."
    )

with st.spinner("Generating optimized Fibonacci dimple placement (this may take a few seconds)..."):
    sw_points = generate_fibonacci_dimple_centers(
        selected_design.dimple_count,
        dimple_diameter_mm=selected_design.dimple_diameter_mm,
        optimize=True,
    )
    sw_validation = validate_dimple_placement(
        sw_points, selected_design.dimple_diameter_mm
    )

if sw_validation["passed"]:
    st.success(
        f"Placement validation **PASSED**. "
        f"No overlaps, no tight gaps. "
        f"Min gap = {sw_validation['min_gap_mm']:.3f} mm "
        f"(required >= {sw_validation['required_min_gap_mm']:.2f} mm). "
        f"Checked {sw_validation['total_pairs']} pairs."
    )
elif sw_validation["buildable"]:
    st.info(
        f"Placement is **BUILDABLE** (no physical overlaps). "
        f"Min gap = {sw_validation['min_gap_mm']:.3f} mm. "
        f"{sw_validation['tight_gaps']} pair(s) have gap < {sw_validation['required_min_gap_mm']:.2f} mm "
        f"but no dimple edges actually overlap. "
        "This is acceptable for SolidWorks — tight land width only."
    )
else:
    st.error(
        f"Placement has **{sw_validation['overlaps']} overlap(s)** "
        f"(dimple edges physically intersect). "
        f"Min gap = {sw_validation['min_gap_mm']:.3f} mm. "
        "This design cannot be built. Select a lower-ranked design."
    )

# --- CSV download ---
csv_buf = io.StringIO()
csv_writer = csv.writer(csv_buf)
csv_writer.writerow(["index", "x_mm", "y_mm", "z_mm", "nx", "ny", "nz"])
for idx_p, pt in enumerate(sw_points, start=1):
    csv_writer.writerow([
        idx_p,
        round(pt["x"], 6), round(pt["y"], 6), round(pt["z"], 6),
        round(pt["nx"], 6), round(pt["ny"], 6), round(pt["nz"], 6),
    ])

# --- JSON download ---
json_data = {
    "ball_diameter_mm": BALL_DIAMETER_MM,
    "ball_radius_mm": BALL_DIAMETER_MM / 2.0,
    "dimple_diameter_mm": selected_design.dimple_diameter_mm,
    "dimple_depth_mm": round(export_depth_mm, 4),
    "depth_ratio_k_over_ball_d": selected_design.depth_ratio,
    "depth_ratio_k_over_dimple_d": round(export_depth_over_dimple, 6),
    "dimple_count": selected_design.dimple_count,
    "occupancy": selected_design.occupancy,
    "volume_ratio": selected_design.volume_ratio,
    "estimated_land_width_mm": round(estimate_hex_packing_land_width_mm(selected_design), 4),
    "min_land_width_mm": MIN_LAND_WIDTH_MM,
    "placement_method": "Fibonacci golden-angle lattice",
    "validation": sw_validation,
}

# --- Macro download ---
from golf_simulator import SW_MACRO_TEMPLATE
macro_content = SW_MACRO_TEMPLATE.format(
    ball_diameter_mm=BALL_DIAMETER_MM,
    dimple_diameter_mm=selected_design.dimple_diameter_mm,
    dimple_depth_mm=export_depth_mm,
    dimple_count=selected_design.dimple_count,
    csv_filename="dimple_centers.csv",
).strip() + "\n"

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.download_button(
        label="Download dimple_centers.csv",
        data=csv_buf.getvalue(),
        file_name="dimple_centers.csv",
        mime="text/csv",
    )
with d2:
    st.download_button(
        label="Download design_params.json",
        data=_json.dumps(json_data, indent=2, ensure_ascii=False),
        file_name="design_params.json",
        mime="application/json",
    )
with d3:
    st.download_button(
        label="Download SW macro (.bas)",
        data=macro_content,
        file_name="import_dimples.bas",
        mime="text/plain",
    )
with d4:
    st.download_button(
        label="Download validation.json",
        data=_json.dumps(sw_validation, indent=2, ensure_ascii=False),
        file_name="validation.json",
        mime="application/json",
    )

with st.expander("SolidWorks step-by-step instructions"):
    st.markdown(f"""
**Step 1 — Open base ball**
- New Part, units MMGS.
- Sketch semicircle radius **{BALL_DIAMETER_MM / 2:.3f} mm**, revolve 360° to create sphere.

**Step 2 — Import dimple centers**
- Download `dimple_centers.csv` above.
- Place the CSV in the same folder as the part file.
- Run `import_dimples.bas` macro (Tools > Macro > Run).
- The macro creates 3D sketch points at each dimple center.

**Step 3 — Create master dimple**
- Create one spherical-cap cut:
  - Mouth diameter = **{selected_design.dimple_diameter_mm:.2f} mm**
  - Depth = **{export_depth_mm:.4f} mm**

**Step 4 — Replicate to all points**
- Use the imported center points to apply the master cut at each location.
- Each cut should be oriented along the outward normal (nx, ny, nz from CSV).

**Step 5 — Validate**
- Confirm count = **{selected_design.dimple_count}**.
- Run Interference Detection.
- Verify minimum gap matches expected land width (**{estimate_hex_packing_land_width_mm(selected_design):.3f} mm**).

**Important:** Do NOT change optimized values. If CAD fails, use the next feasible ranked design.
""")

st.caption(
    "The Fibonacci lattice ensures near-uniform global coverage without polar clustering, "
    "unlike latitude-ring or circular-pattern approaches."
)
