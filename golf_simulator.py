"""
Simple golf ball aerodynamics simulator and parameter search.

This script:
1) Simulates ball flight with drag/lift forces (RK4 integrator).
2) Uses literature-inspired relationships for Cd/Cl.
3) Sweeps dimple geometry parameters and reports best candidates.

Run:
    python golf_simulator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path
import csv
from statistics import mean, pstdev


# -----------------------------
# Physical constants and setup
# -----------------------------

RHO_AIR = 1.225  # kg/m^3
MU_AIR = 1.81e-5  # Pa.s
G = 9.81  # m/s^2

BALL_DIAMETER = 0.04267  # m
BALL_RADIUS = BALL_DIAMETER / 2.0
BALL_AREA = pi * BALL_RADIUS**2  # projected area
BALL_MASS = 0.04593  # kg
BALL_DIAMETER_MM = BALL_DIAMETER * 1000.0
BALL_SURFACE_AREA_MM2 = 4.0 * pi * (BALL_DIAMETER_MM / 2.0) ** 2

# Approximate manufacturability guard:
# required minimum land width (gap) between neighboring dimples.
MIN_LAND_WIDTH_MM = 0.25

# Wind-tunnel-focused speed band from the studied papers:
# 20-120 km/h and 40-140 km/h -> union 20-140 km/h
WIND_TUNNEL_SPEED_MIN_MPS = 20.0 / 3.6
WIND_TUNNEL_SPEED_MAX_MPS = 140.0 / 3.6
WIND_TUNNEL_SPEED_REF_MPS = 100.0 / 3.6


@dataclass(frozen=True)
class LaunchCondition:
    speed_mps: float
    launch_angle_deg: float
    spin_rpm: float


@dataclass(frozen=True)
class DimpleDesign:
    depth_ratio: float  # k / ball_diameter
    occupancy: float  # 0.0 - 1.0
    volume_ratio: float  # dimensionless, e.g. 0.011 = 11e-3
    dimple_count: int
    dimple_diameter_mm: float


@dataclass
class SimulationResult:
    carry_distance_m: float
    flight_time_s: float
    max_height_m: float
    avg_cd: float
    avg_cl: float
    avg_l_over_d: float


@dataclass
class RobustDesignResult:
    design: DimpleDesign
    nominal_result: SimulationResult
    robust_score: float
    mean_score: float
    worst_score: float
    score_std: float
    mean_distance_m: float
    min_distance_m: float


@dataclass(frozen=True)
class SearchConfig:
    mode: str
    description: str


PAPER_DEPTH_RATIOS = [4.55e-3, 6.82e-3, 9.09e-3]

# Literature-aligned design family (based on occupancy groups used in 2023 study).
# Each occupancy group has fixed dimple-count and representative mean dimple diameter ratio.
# cm_over_d values are converted to mm using BALL_DIAMETER.
PAPER_OCCUPANCY_GROUPS = [
    {"occupancy": 0.526, "dimple_count": 314, "cm_over_d": 81.8e-3},
    {"occupancy": 0.631, "dimple_count": 314, "cm_over_d": 89.5e-3},
    {"occupancy": 0.666, "dimple_count": 398, "cm_over_d": 81.8e-3},
    {"occupancy": 0.812, "dimple_count": 476, "cm_over_d": 81.8e-3},
    {"occupancy": 0.831, "dimple_count": 314, "cm_over_d": 102.0e-3},
]

# VDR matrix from literature (rows: depth ratio, cols: occupancy groups above), x10^-3 values.
PAPER_VDR_MATRIX = {
    4.55e-3: [7.17e-3, 8.59e-3, 9.09e-3, 11.1e-3, 11.3e-3],
    6.82e-3: [10.8e-3, 12.9e-3, 13.7e-3, 16.6e-3, 17.0e-3],
    9.09e-3: [14.5e-3, 17.3e-3, 18.3e-3, 22.3e-3, 22.7e-3],
}

# Reference setup used when one variable is scanned and others are fixed.
REF_DESIGN = DimpleDesign(
    depth_ratio=4.55e-3,
    occupancy=0.812,
    volume_ratio=0.0111,
    dimple_count=314,
    dimple_diameter_mm=3.5,
)


SEARCH_MODES: dict[str, SearchConfig] = {
    "paper_depth_only": SearchConfig(
        mode="paper_depth_only",
        description="Depth effect test: same dimple family, only depth changes (literature-aligned).",
    ),
    "paper_occupancy_only": SearchConfig(
        mode="paper_occupancy_only",
        description="Occupancy effect test: depth fixed, occupancy family changes with literature values.",
    ),
    "paper_volume_only": SearchConfig(
        mode="paper_volume_only",
        description="VDR trend test: uses all literature designs and analyzes trend by volume ratio (not isolated).",
    ),
    "paper_literature_grid": SearchConfig(
        mode="paper_literature_grid",
        description=(
            "Extended literature-aligned set: 50 designs "
            "(15 measured + deterministic interpolated dummy points)."
        ),
    ),
}

# Launch presets:
# - WIND_TUNNEL_REFERENCE_LAUNCH preserves the zero-spin paper reference.
# - DEFAULT_PRODUCTION_LAUNCH is used for design ranking toward manufacturable gameplay use.
WIND_TUNNEL_REFERENCE_LAUNCH = LaunchCondition(
    speed_mps=WIND_TUNNEL_SPEED_REF_MPS,
    launch_angle_deg=11.2,
    spin_rpm=0.0,
)

# Production launch calibration knobs (edit these to match your real target ball speed).
PRODUCTION_TARGET_BALL_SPEED_MPS = 74.0  # ~266.4 km/h (PGA-like baseline)
PRODUCTION_LAUNCH_ANGLE_DEG = 11.2
PRODUCTION_SPIN_REF_RPM = 2685.0
PRODUCTION_SPIN_REF_SPEED_MPS = 74.0
PRODUCTION_SPIN_MIN_RPM = 1500.0
PRODUCTION_SPIN_MAX_RPM = 4500.0
ROBUST_SPEED_FACTORS = [0.92, 1.00, 1.08]
ROBUST_ANGLE_OFFSETS_DEG = [-1.0, 0.0, 1.0]
ROBUST_SPIN_FACTORS = [0.90, 1.00, 1.10]


def calibrate_production_launch(
    target_ball_speed_mps: float = PRODUCTION_TARGET_BALL_SPEED_MPS,
    launch_angle_deg: float = PRODUCTION_LAUNCH_ANGLE_DEG,
    spin_rpm: float | None = None,
) -> LaunchCondition:
    """
    Build production launch conditions from a target ball speed.
    If spin is not provided, spin is speed-scaled from a PGA-like reference.
    """
    if spin_rpm is None:
        spin_rpm = PRODUCTION_SPIN_REF_RPM * (
            target_ball_speed_mps / max(PRODUCTION_SPIN_REF_SPEED_MPS, 0.1)
        )
    spin_rpm = max(PRODUCTION_SPIN_MIN_RPM, min(PRODUCTION_SPIN_MAX_RPM, float(spin_rpm)))
    return LaunchCondition(
        speed_mps=float(target_ball_speed_mps),
        launch_angle_deg=float(launch_angle_deg),
        spin_rpm=spin_rpm,
    )


DEFAULT_PRODUCTION_LAUNCH = calibrate_production_launch()


def interpolate_series(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear interpolation on a sorted axis with edge clamping."""
    if len(xs) != len(ys):
        raise ValueError("Axis and value sizes do not match.")
    if not xs:
        raise ValueError("Interpolation axis is empty.")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        x0 = xs[i]
        x1 = xs[i + 1]
        if x0 <= x <= x1:
            y0 = ys[i]
            y1 = ys[i + 1]
            if abs(x1 - x0) < 1e-12:
                return y0
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return ys[-1]


def literature_group_value_at_occupancy(occupancy: float, key: str) -> float:
    """Interpolate occupancy-group properties from literature family points."""
    occ_axis = [float(g["occupancy"]) for g in PAPER_OCCUPANCY_GROUPS]
    val_axis = [float(g[key]) for g in PAPER_OCCUPANCY_GROUPS]
    return interpolate_series(occupancy, occ_axis, val_axis)


def literature_vdr_at(depth_ratio: float, occupancy: float) -> float:
    """
    Bilinear interpolation over the 3x5 literature VDR table.
    This keeps dummy points trend-aligned with measured paper data.
    """
    depth_axis = list(PAPER_DEPTH_RATIOS)
    occ_axis = [float(g["occupancy"]) for g in PAPER_OCCUPANCY_GROUPS]
    row_values = [
        interpolate_series(occupancy, occ_axis, list(PAPER_VDR_MATRIX[dr])) for dr in depth_axis
    ]
    return interpolate_series(depth_ratio, depth_axis, row_values)


# Dense but deterministic paper_literature_grid axes:
# 5 depth levels x 10 occupancy levels = 50 tested designs.
PAPER_GRID_DEPTH_LEVELS = [4.55e-3, 5.68e-3, 6.82e-3, 7.95e-3, 9.09e-3]
PAPER_GRID_OCCUPANCY_LEVELS = [0.526, 0.561, 0.596, 0.631, 0.666, 0.703, 0.739, 0.775, 0.812, 0.831]


def spin_parameter(speed: float, spin_rpm: float) -> float:
    """Sp = pi * d * N / U, where N is rotation speed in rps."""
    n_rps = spin_rpm / 60.0
    return pi * BALL_DIAMETER * n_rps / max(speed, 0.1)


def reynolds_number(speed: float) -> float:
    return RHO_AIR * speed * BALL_DIAMETER / MU_AIR


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def estimate_hex_packing_land_width_mm(design: DimpleDesign) -> float:
    """
    Estimate average gap between dimples by assuming near-hexagonal packing
    over the sphere's surface. This is a practical manufacturability proxy.
    """
    if design.dimple_count <= 0:
        return -design.dimple_diameter_mm
    area_per_dimple = BALL_SURFACE_AREA_MM2 / float(design.dimple_count)
    center_spacing_mm = sqrt((2.0 * area_per_dimple) / sqrt(3.0))
    return center_spacing_mm - design.dimple_diameter_mm


def is_design_manufacturable(
    design: DimpleDesign,
    min_land_width_mm: float = MIN_LAND_WIDTH_MM,
) -> bool:
    return estimate_hex_packing_land_width_mm(design) >= min_land_width_mm


def apply_manufacturability_filter(
    designs: list[DimpleDesign],
    min_land_width_mm: float = MIN_LAND_WIDTH_MM,
) -> list[DimpleDesign]:
    """
    Keep only manufacturable candidates. If a given mode has zero passing designs,
    fall back to the original set so UI flows never crash.
    """
    filtered = [d for d in designs if is_design_manufacturable(d, min_land_width_mm)]
    return filtered if filtered else designs


def model_cd_cl(speed: float, spin_rpm: float, design: DimpleDesign) -> tuple[float, float]:
    """
    Literature-inspired surrogate model for Cd and Cl.

    Notes:
    - It is intentionally simple and transparent.
    - Coefficients are tuned to reflect trends reported in the papers:
      * deeper dimples -> earlier transition but higher transcritical Cd
      * shallow depth + high occupancy can improve L/D
      * larger volume ratio can increase Cd
      * Cl grows with spin parameter
    """
    re = reynolds_number(speed)
    sp = spin_parameter(speed, spin_rpm)
    eps = design.depth_ratio  # uses k / ball_diameter scale from the literature set

    # Base drag from roughness correlations (2016 study trend).
    # cd_min ~= 4.9278*eps + 0.0621
    cd_min = 4.9278 * eps + 0.0621

    # Critical Re trend from roughness correlation:
    # Re_crit(x1e4) ~= -100.16*eps + 6.4554
    re_crit = max(2.0e4, (-100.16 * eps + 6.4554) * 1.0e4)

    # Transition shape: before and after Re_crit
    if re < re_crit:
        # pre-critical: higher drag
        cd_re = cd_min + 0.20 * (1.0 - re / re_crit)
    else:
        # trans/supercritical rise
        cd_re = cd_min + 0.06 * ((re - re_crit) / re_crit)

    # Volume ratio effect (higher volume tends to increase drag)
    vol_term = 1.8 * (design.volume_ratio - 0.011)

    # Occupancy-depth coupling:
    # shallow depth + occupancy >= 0.8 tends to improve L/D (lower effective Cd)
    occupancy_bonus = 0.0
    if design.depth_ratio <= 4.55e-3 and design.occupancy >= 0.80:
        occupancy_bonus = -0.02 * (design.occupancy - 0.80) / 0.03

    # Spin slightly increases drag
    spin_drag = 0.035 * sp

    cd = cd_re + vol_term + occupancy_bonus + spin_drag
    cd = clamp(cd, 0.05, 0.55)

    # Lift model:
    # Wind tunnel non-spinning condition should yield near-zero lift.
    # Geometry terms modulate spin-generated lift instead of creating lift at zero spin.
    cl_spin = 0.58 * sp
    cl_geom_factor = (
        1.0
        + 0.30 * (design.occupancy - 0.75)
        - 35.0 * (design.depth_ratio - 4.55e-3)
        + 5.0 * (design.volume_ratio - 0.011)
    )
    cl_geom_factor = clamp(cl_geom_factor, 0.55, 1.45)
    cl = cl_spin * cl_geom_factor
    cl = clamp(cl, 0.0, 0.32)

    return cd, cl


def derivatives(state: tuple[float, float, float, float], spin_rpm: float, design: DimpleDesign):
    """
    state = (x, y, vx, vy)
    Returns time derivatives (dx, dy, dvx, dvy).
    """
    _, _, vx, vy = state
    speed = sqrt(vx * vx + vy * vy)
    if speed < 1e-6:
        return (vx, vy, 0.0, -G), 0.0, 0.0

    cd, cl = model_cd_cl(speed, spin_rpm, design)

    q = 0.5 * RHO_AIR * speed * speed
    drag = q * BALL_AREA * cd
    lift = q * BALL_AREA * cl

    # Unit vectors
    ux = vx / speed
    uy = vy / speed

    # Drag opposite to velocity
    fdx = -drag * ux
    fdy = -drag * uy

    # Lift perpendicular to velocity (upward side)
    flx = -lift * uy
    fly = lift * ux

    ax = (fdx + flx) / BALL_MASS
    ay = (fdy + fly) / BALL_MASS - G

    return (vx, vy, ax, ay), cd, cl


def rk4_step(state, dt: float, spin_rpm: float, design: DimpleDesign):
    k1, _, _ = derivatives(state, spin_rpm, design)
    s2 = tuple(state[i] + 0.5 * dt * k1[i] for i in range(4))
    k2, _, _ = derivatives(s2, spin_rpm, design)
    s3 = tuple(state[i] + 0.5 * dt * k2[i] for i in range(4))
    k3, _, _ = derivatives(s3, spin_rpm, design)
    s4 = tuple(state[i] + dt * k3[i] for i in range(4))
    k4, _, _ = derivatives(s4, spin_rpm, design)

    return tuple(
        state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(4)
    )


def simulate_flight(launch: LaunchCondition, design: DimpleDesign) -> SimulationResult:
    theta = launch.launch_angle_deg * pi / 180.0
    vx0 = launch.speed_mps * cos(theta)
    vy0 = launch.speed_mps * sin(theta)

    state = (0.0, 0.0, vx0, vy0)
    dt = 0.002
    max_t = 12.0

    t = 0.0
    max_h = 0.0
    cd_sum = 0.0
    cl_sum = 0.0
    lod_sum = 0.0
    samples = 0

    prev_state = state
    while t < max_t:
        deriv, cd, cl = derivatives(state, launch.spin_rpm, design)
        _ = deriv  # unused, kept for readability
        cd_sum += cd
        cl_sum += cl
        lod_sum += cl / max(cd, 1e-6)
        samples += 1

        prev_state = state
        state = rk4_step(state, dt, launch.spin_rpm, design)
        t += dt
        max_h = max(max_h, state[1])

        if state[1] < 0.0 and t > 0.05:
            break

    # Linear interpolation for better landing x at y=0
    x1, y1, *_ = prev_state
    x2, y2, *_ = state
    if y2 == y1:
        x_land = x2
    else:
        r = (0.0 - y1) / (y2 - y1)
        x_land = x1 + r * (x2 - x1)

    return SimulationResult(
        carry_distance_m=max(x_land, 0.0),
        flight_time_s=t,
        max_height_m=max_h,
        avg_cd=cd_sum / max(samples, 1),
        avg_cl=cl_sum / max(samples, 1),
        avg_l_over_d=lod_sum / max(samples, 1),
    )


def simulate_flight_with_path(
    launch: LaunchCondition, design: DimpleDesign
) -> tuple[SimulationResult, list[tuple[float, float]]]:
    """
    Simulate flight and return both summary metrics and trajectory path points.
    Each path point is (x, y) in meters.
    """
    theta = launch.launch_angle_deg * pi / 180.0
    vx0 = launch.speed_mps * cos(theta)
    vy0 = launch.speed_mps * sin(theta)

    state = (0.0, 0.0, vx0, vy0)
    dt = 0.002
    max_t = 12.0

    t = 0.0
    max_h = 0.0
    cd_sum = 0.0
    cl_sum = 0.0
    lod_sum = 0.0
    samples = 0
    path = [(0.0, 0.0)]

    prev_state = state
    while t < max_t:
        _, cd, cl = derivatives(state, launch.spin_rpm, design)
        cd_sum += cd
        cl_sum += cl
        lod_sum += cl / max(cd, 1e-6)
        samples += 1

        prev_state = state
        state = rk4_step(state, dt, launch.spin_rpm, design)
        t += dt
        max_h = max(max_h, state[1])
        path.append((state[0], max(state[1], 0.0)))

        if state[1] < 0.0 and t > 0.05:
            break

    x1, y1, *_ = prev_state
    x2, y2, *_ = state
    if y2 == y1:
        x_land = x2
    else:
        r = (0.0 - y1) / (y2 - y1)
        x_land = x1 + r * (x2 - x1)

    # Ensure the final point lies on the ground.
    if path:
        path[-1] = (max(x_land, 0.0), 0.0)

    result = SimulationResult(
        carry_distance_m=max(x_land, 0.0),
        flight_time_s=t,
        max_height_m=max_h,
        avg_cd=cd_sum / max(samples, 1),
        avg_cl=cl_sum / max(samples, 1),
        avg_l_over_d=lod_sum / max(samples, 1),
    )
    return result, path


def score_result(result: SimulationResult) -> float:
    """
    Weighted score for optimization:
      + distance (maximize)
      + lift/drag (maximize)
      - average Cd (minimize)
    """
    return (
        1.00 * result.carry_distance_m
        + 25.0 * result.avg_l_over_d
        - 30.0 * result.avg_cd
    )


def rank_value(result: SimulationResult, launch: LaunchCondition) -> float:
    """
    Ranking logic:
    - If spin is ~0 (wind-tunnel-like), prioritize lower drag (Cd).
    - Otherwise, use distance-based composite score.
    """
    if abs(launch.spin_rpm) < 1e-9:
        # Use inverse-Cd so "higher is better" remains true in the table.
        return 1.0 / max(result.avg_cd, 1e-9)
    return score_result(result)


def build_design_space(mode: str = "paper_literature_grid") -> list[DimpleDesign]:
    """
    Build design space strictly from literature-supported points.
    Modes are intentionally limited to paper-aligned test styles.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    ref = REF_DESIGN

    if mode == "paper_depth_only":
        # Fix occupancy family to O=81.2% (ND=476, Cmean/d=81.8e-3), vary only depth.
        grp = next(g for g in PAPER_OCCUPANCY_GROUPS if abs(g["occupancy"] - 0.812) < 1e-9)
        i = PAPER_OCCUPANCY_GROUPS.index(grp)
        dimple_diam_mm = grp["cm_over_d"] * BALL_DIAMETER * 1000.0
        return [
            DimpleDesign(
                depth_ratio=dr,
                occupancy=grp["occupancy"],
                volume_ratio=PAPER_VDR_MATRIX[dr][i],
                dimple_count=grp["dimple_count"],
                dimple_diameter_mm=dimple_diam_mm,
            )
            for dr in PAPER_DEPTH_RATIOS
        ]

    if mode == "paper_occupancy_only":
        # Fix depth to D/d=4.55e-3, vary occupancy family.
        depth_fixed = 4.55e-3
        return [
            DimpleDesign(
                depth_ratio=depth_fixed,
                occupancy=grp["occupancy"],
                volume_ratio=PAPER_VDR_MATRIX[depth_fixed][i],
                dimple_count=grp["dimple_count"],
                dimple_diameter_mm=grp["cm_over_d"] * BALL_DIAMETER * 1000.0,
            )
            for i, grp in enumerate(PAPER_OCCUPANCY_GROUPS)
        ]

    if mode == "paper_volume_only":
        # VDR trend is assessed from the full literature set.
        # Keep this mode as an alias to the same 50-design set for clear trend inspection.
        mode = "paper_literature_grid"

    # paper_literature_grid
    # Build 50 deterministic, literature-aligned designs:
    # - original measured ranges are preserved
    # - extra dummy points are interpolation-based (never random)
    designs = []
    for dr in PAPER_GRID_DEPTH_LEVELS:
        for occupancy in PAPER_GRID_OCCUPANCY_LEVELS:
            cm_over_d = literature_group_value_at_occupancy(occupancy, "cm_over_d")
            mean_dimple_diam_mm = cm_over_d * BALL_DIAMETER * 1000.0
            dimple_count = int(round(literature_group_value_at_occupancy(occupancy, "dimple_count")))
            designs.append(
                DimpleDesign(
                    depth_ratio=dr,
                    occupancy=occupancy,
                    volume_ratio=literature_vdr_at(dr, occupancy),
                    dimple_count=dimple_count,
                    dimple_diameter_mm=mean_dimple_diam_mm,
                )
            )
    if len(designs) != 50:
        raise RuntimeError(f"Expected 50 paper_literature_grid designs, got {len(designs)}")
    return designs


def run_search() -> tuple[list[tuple[DimpleDesign, SimulationResult, float]], LaunchCondition]:
    # Default search targets production-oriented ranking (non-zero spin flight behavior).
    return run_search_with_launch(DEFAULT_PRODUCTION_LAUNCH, mode="paper_literature_grid")


def run_search_with_launch(
    launch: LaunchCondition,
    mode: str = "paper_literature_grid",
) -> tuple[list[tuple[DimpleDesign, SimulationResult, float]], LaunchCondition]:
    all_designs = build_design_space(mode=mode)
    design_space = apply_manufacturability_filter(all_designs)

    ranked: list[tuple[DimpleDesign, SimulationResult, float]] = []
    for design in design_space:
        result = simulate_flight(launch, design)
        s = rank_value(result, launch)
        ranked.append((design, result, s))

    # Best -> worst with deterministic tie-breakers.
    ranked.sort(
        key=lambda x: (
            x[2],  # ranking score (higher is better)
            x[1].carry_distance_m,  # farther carry is better
            -x[1].avg_cd,  # lower drag is better
            x[1].avg_l_over_d,  # higher aerodynamic efficiency is better
        ),
        reverse=True,
    )
    return ranked, launch


def build_robust_launch_set(base_launch: LaunchCondition) -> list[LaunchCondition]:
    """
    Build a small production envelope around the nominal launch.
    These scenarios emulate user-to-user strike variation.
    """
    scenarios: list[LaunchCondition] = []
    for speed_factor in ROBUST_SPEED_FACTORS:
        for angle_offset in ROBUST_ANGLE_OFFSETS_DEG:
            for spin_factor in ROBUST_SPIN_FACTORS:
                spin = base_launch.spin_rpm * spin_factor
                spin = max(PRODUCTION_SPIN_MIN_RPM, min(PRODUCTION_SPIN_MAX_RPM, spin))
                scenarios.append(
                    LaunchCondition(
                        speed_mps=base_launch.speed_mps * speed_factor,
                        launch_angle_deg=base_launch.launch_angle_deg + angle_offset,
                        spin_rpm=spin,
                    )
                )
    return scenarios


def robust_score_from_scenarios(scores: list[float]) -> tuple[float, float, float, float]:
    """
    Combine scenario performance into one robust score.
    Prioritize high average + strong worst-case + lower variability.
    """
    if not scores:
        return 0.0, 0.0, 0.0, 0.0
    mean_s = mean(scores)
    worst_s = min(scores)
    std_s = pstdev(scores) if len(scores) > 1 else 0.0
    robust_s = 0.60 * mean_s + 0.35 * worst_s - 0.05 * std_s
    return robust_s, mean_s, worst_s, std_s


def run_robust_search(
    mode: str = "paper_literature_grid",
    base_launch: LaunchCondition = DEFAULT_PRODUCTION_LAUNCH,
) -> tuple[list[RobustDesignResult], LaunchCondition, list[LaunchCondition]]:
    all_designs = build_design_space(mode=mode)
    design_space = apply_manufacturability_filter(all_designs)
    scenarios = build_robust_launch_set(base_launch)

    ranked: list[RobustDesignResult] = []
    for design in design_space:
        nominal_result = simulate_flight(base_launch, design)
        scenario_scores: list[float] = []
        scenario_distances: list[float] = []

        for launch in scenarios:
            result = simulate_flight(launch, design)
            scenario_scores.append(rank_value(result, launch))
            scenario_distances.append(result.carry_distance_m)

        robust_s, mean_s, worst_s, std_s = robust_score_from_scenarios(scenario_scores)
        ranked.append(
            RobustDesignResult(
                design=design,
                nominal_result=nominal_result,
                robust_score=robust_s,
                mean_score=mean_s,
                worst_score=worst_s,
                score_std=std_s,
                mean_distance_m=mean(scenario_distances) if scenario_distances else 0.0,
                min_distance_m=min(scenario_distances) if scenario_distances else 0.0,
            )
        )

    ranked.sort(
        key=lambda x: (
            x.robust_score,  # higher robust score is better
            x.worst_score,  # safer worst-case is better
            x.mean_distance_m,  # farther average carry is better
            -x.nominal_result.avg_cd,  # lower drag is better
            x.nominal_result.avg_l_over_d,  # higher L/D is better
        ),
        reverse=True,
    )
    return ranked, base_launch, scenarios


def write_csv(path: Path, ranked: list[tuple[DimpleDesign, SimulationResult, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "score",
                "carry_distance_m",
                "flight_time_s",
                "max_height_m",
                "avg_cd",
                "avg_cl",
                "avg_l_over_d",
                "depth_ratio_k_over_ball_d",
                "occupancy",
                "volume_ratio",
                "dimple_count",
                "dimple_diameter_mm",
                "estimated_land_width_mm",
            ]
        )

        for idx, (d, r, s) in enumerate(ranked, start=1):
            writer.writerow(
                [
                    idx,
                    round(s, 4),
                    round(r.carry_distance_m, 4),
                    round(r.flight_time_s, 4),
                    round(r.max_height_m, 4),
                    round(r.avg_cd, 5),
                    round(r.avg_cl, 5),
                    round(r.avg_l_over_d, 5),
                    d.depth_ratio,
                    d.occupancy,
                    d.volume_ratio,
                    d.dimple_count,
                    d.dimple_diameter_mm,
                    round(estimate_hex_packing_land_width_mm(d), 4),
                ]
            )


def write_summary(path: Path, ranked: list[tuple[DimpleDesign, SimulationResult, float]], launch: LaunchCondition) -> None:
    top = ranked[:5]
    best_d, best_r, best_s = top[0]

    lines = []
    lines.append("# Golf Ball Simulation Summary")
    lines.append("")
    lines.append("## Launch Conditions")
    lines.append(f"- Speed: {launch.speed_mps:.1f} m/s")
    lines.append(f"- Launch angle: {launch.launch_angle_deg:.1f} deg")
    lines.append(f"- Spin: {launch.spin_rpm:.0f} rpm")
    lines.append("")
    lines.append("## Best Design (Rank 1)")
    lines.append(f"- Score: {best_s:.3f}")
    lines.append(f"- Carry distance: {best_r.carry_distance_m:.2f} m")
    lines.append(f"- Average Cd: {best_r.avg_cd:.3f}")
    lines.append(f"- Average Cl: {best_r.avg_cl:.3f}")
    lines.append(f"- Average L/D: {best_r.avg_l_over_d:.3f}")
    lines.append(f"- Depth ratio (k / ball diameter): {best_d.depth_ratio:.5f}")
    lines.append(f"- Occupancy: {best_d.occupancy:.3f}")
    lines.append(f"- Volume ratio: {best_d.volume_ratio:.4f}")
    lines.append(f"- Dimple count: {best_d.dimple_count}")
    lines.append(f"- Dimple diameter: {best_d.dimple_diameter_mm:.2f} mm")
    lines.append(f"- Estimated land width: {estimate_hex_packing_land_width_mm(best_d):.3f} mm")
    lines.append("")
    lines.append("## Top 5 Designs")
    for i, (d, r, s) in enumerate(top, start=1):
        lines.append(
            f"- #{i}: score={s:.2f}, distance={r.carry_distance_m:.2f} m, "
            f"Cd={r.avg_cd:.3f}, Cl={r.avg_cl:.3f}, L/D={r.avg_l_over_d:.3f}, "
            f"k/ball_d={d.depth_ratio:.5f}, O={d.occupancy:.3f}, VDR={d.volume_ratio:.4f}, "
            f"ND={d.dimple_count}, dimple_diam={d.dimple_diameter_mm:.2f} mm, "
            f"land={estimate_hex_packing_land_width_mm(d):.3f} mm"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_robust_csv(path: Path, ranked: list[RobustDesignResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "robust_score",
                "mean_score",
                "worst_score",
                "score_std",
                "mean_distance_m",
                "min_distance_m",
                "nominal_carry_distance_m",
                "nominal_avg_cd",
                "nominal_avg_cl",
                "nominal_avg_l_over_d",
                "depth_ratio_k_over_ball_d",
                "occupancy",
                "volume_ratio",
                "dimple_count",
                "dimple_diameter_mm",
                "estimated_land_width_mm",
            ]
        )

        for idx, item in enumerate(ranked, start=1):
            d = item.design
            n = item.nominal_result
            writer.writerow(
                [
                    idx,
                    round(item.robust_score, 4),
                    round(item.mean_score, 4),
                    round(item.worst_score, 4),
                    round(item.score_std, 4),
                    round(item.mean_distance_m, 4),
                    round(item.min_distance_m, 4),
                    round(n.carry_distance_m, 4),
                    round(n.avg_cd, 5),
                    round(n.avg_cl, 5),
                    round(n.avg_l_over_d, 5),
                    d.depth_ratio,
                    d.occupancy,
                    d.volume_ratio,
                    d.dimple_count,
                    d.dimple_diameter_mm,
                    round(estimate_hex_packing_land_width_mm(d), 4),
                ]
            )


def write_robust_summary(
    path: Path,
    ranked: list[RobustDesignResult],
    base_launch: LaunchCondition,
    scenario_count: int,
) -> None:
    top = ranked[:5]
    best = top[0]
    d = best.design
    n = best.nominal_result

    lines = []
    lines.append("# Robust Golf Ball Design Summary")
    lines.append("")
    lines.append("## Robust Search Envelope")
    lines.append(f"- Base speed: {base_launch.speed_mps:.1f} m/s")
    lines.append(f"- Base launch angle: {base_launch.launch_angle_deg:.1f} deg")
    lines.append(f"- Base spin: {base_launch.spin_rpm:.0f} rpm")
    lines.append(f"- Scenario count: {scenario_count}")
    lines.append(
        f"- Speed factors: {', '.join(f'{x:.2f}' for x in ROBUST_SPEED_FACTORS)} | "
        f"Angle offsets: {', '.join(f'{x:.1f}' for x in ROBUST_ANGLE_OFFSETS_DEG)} deg | "
        f"Spin factors: {', '.join(f'{x:.2f}' for x in ROBUST_SPIN_FACTORS)}"
    )
    lines.append("")
    lines.append("## Best Robust Design (Rank 1)")
    lines.append(f"- Robust score: {best.robust_score:.3f}")
    lines.append(f"- Mean scenario score: {best.mean_score:.3f}")
    lines.append(f"- Worst scenario score: {best.worst_score:.3f}")
    lines.append(f"- Score std: {best.score_std:.3f}")
    lines.append(f"- Mean carry: {best.mean_distance_m:.2f} m")
    lines.append(f"- Minimum carry: {best.min_distance_m:.2f} m")
    lines.append(f"- Nominal carry: {n.carry_distance_m:.2f} m")
    lines.append(f"- Nominal Cd: {n.avg_cd:.3f}, Cl: {n.avg_cl:.3f}, L/D: {n.avg_l_over_d:.3f}")
    lines.append(
        f"- Geometry: k/ball_d={d.depth_ratio:.5f}, O={d.occupancy:.3f}, VDR={d.volume_ratio:.4f}, "
        f"ND={d.dimple_count}, dimple_diam={d.dimple_diameter_mm:.2f} mm, "
        f"land={estimate_hex_packing_land_width_mm(d):.3f} mm"
    )
    lines.append("")
    lines.append("## Top 5 Robust Designs")
    for i, item in enumerate(top, start=1):
        d = item.design
        lines.append(
            f"- #{i}: robust={item.robust_score:.2f}, mean={item.mean_score:.2f}, "
            f"worst={item.worst_score:.2f}, std={item.score_std:.2f}, "
            f"mean_carry={item.mean_distance_m:.2f} m, min_carry={item.min_distance_m:.2f} m, "
            f"k/ball_d={d.depth_ratio:.5f}, O={d.occupancy:.3f}, VDR={d.volume_ratio:.4f}, "
            f"ND={d.dimple_count}, land={estimate_hex_packing_land_width_mm(d):.3f} mm"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ranked, launch = run_search()
    robust_ranked, robust_launch, robust_scenarios = run_robust_search()
    out_csv = Path("results.csv")
    out_md = Path("best_design_summary.md")
    robust_out_csv = Path("robust_results.csv")
    robust_out_md = Path("robust_design_summary.md")

    write_csv(out_csv, ranked)
    write_summary(out_md, ranked, launch)
    write_robust_csv(robust_out_csv, robust_ranked)
    write_robust_summary(robust_out_md, robust_ranked, robust_launch, len(robust_scenarios))

    best_d, best_r, best_s = ranked[0]
    robust_best = robust_ranked[0]
    robust_best_d = robust_best.design
    print("Simulation completed.")
    print(f"Tested designs: {len(ranked)}")
    print(f"Best score: {best_s:.3f}")
    print(f"Best carry distance: {best_r.carry_distance_m:.2f} m")
    print(
        "Best parameters: "
        f"k/ball_d={best_d.depth_ratio:.5f}, "
        f"occupancy={best_d.occupancy:.3f}, "
        f"volume_ratio={best_d.volume_ratio:.4f}, "
        f"dimple_count={best_d.dimple_count}, "
        f"dimple_diameter={best_d.dimple_diameter_mm:.2f} mm, "
        f"land_width={estimate_hex_packing_land_width_mm(best_d):.3f} mm"
    )
    print(
        "Best robust parameters: "
        f"k/ball_d={robust_best_d.depth_ratio:.5f}, "
        f"occupancy={robust_best_d.occupancy:.3f}, "
        f"volume_ratio={robust_best_d.volume_ratio:.4f}, "
        f"dimple_count={robust_best_d.dimple_count}, "
        f"dimple_diameter={robust_best_d.dimple_diameter_mm:.2f} mm, "
        f"land_width={estimate_hex_packing_land_width_mm(robust_best_d):.3f} mm"
    )
    print(f"Best robust score: {robust_best.robust_score:.3f}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Saved: {robust_out_csv}")
    print(f"Saved: {robust_out_md}")


if __name__ == "__main__":
    main()
