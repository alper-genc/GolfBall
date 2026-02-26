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
from math import acos, atan2, cos, floor, pi, sin, sqrt
from pathlib import Path
import csv
import json
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


def theoretical_max_dimple_count(
    dimple_diameter_mm: float,
    min_land_width_mm: float = MIN_LAND_WIDTH_MM,
    ball_surface_area_mm2: float = BALL_SURFACE_AREA_MM2,
    packing_efficiency: float = 0.87,
) -> int:
    """
    Estimate max dimple count using hex-packing formula with a realistic
    packing efficiency factor (~0.87 accounts for real lattice imperfections).
    """
    center_dist = dimple_diameter_mm + min_land_width_mm
    area_per_hex = sqrt(3.0) / 2.0 * center_dist * center_dist
    ideal_max = ball_surface_area_mm2 / area_per_hex
    return int(ideal_max * packing_efficiency)


def geometric_occupancy(dimple_count: int, dimple_diameter_mm: float) -> float:
    """Actual surface coverage fraction from dimple count and diameter."""
    return dimple_count * pi * (dimple_diameter_mm / 2.0) ** 2 / BALL_SURFACE_AREA_MM2


def max_feasible_dimple_diameter(
    dimple_count: int,
    min_land_width_mm: float = MIN_LAND_WIDTH_MM,
    ball_surface_area_mm2: float = BALL_SURFACE_AREA_MM2,
    packing_efficiency: float = 0.87,
) -> float:
    """
    Inverse of theoretical_max_dimple_count: given a target dimple count,
    return the largest dimple diameter that physically fits on the ball.
    """
    if dimple_count <= 0:
        return 0.0
    area_per_hex = ball_surface_area_mm2 * packing_efficiency / dimple_count
    center_dist = sqrt(2.0 * area_per_hex / sqrt(3.0))
    return max(center_dist - min_land_width_mm, 0.1)


def is_design_manufacturable(
    design: DimpleDesign,
    min_land_width_mm: float = MIN_LAND_WIDTH_MM,
) -> bool:
    if estimate_hex_packing_land_width_mm(design) < min_land_width_mm:
        return False
    max_n = theoretical_max_dimple_count(design.dimple_diameter_mm, min_land_width_mm)
    return design.dimple_count <= max_n


def _fibonacci_seed_points(n: int) -> list[list[float]]:
    """Shifted Fibonacci lattice on the unit sphere (avoids polar clustering)."""
    golden_angle = pi * (3.0 - sqrt(5.0))
    pts: list[list[float]] = []
    for i in range(n):
        y = 1.0 - (2.0 * (i + 0.5)) / n
        r = sqrt(max(1.0 - y * y, 0.0))
        theta = golden_angle * i
        pts.append([cos(theta) * r, y, sin(theta) * r])
    return pts


def _repulsion_optimize(
    pts: list[list[float]],
    min_center_dist: float,
    iterations: int = 200,
    step_scale: float = 0.03,
) -> list[list[float]]:
    """
    Iteratively push points apart on the unit sphere using Coulomb-like
    repulsion. All pair interactions (not just close ones) contribute, but
    close pairs get stronger forces. After force accumulation, each point
    is moved along the tangent plane and re-projected onto the sphere.
    """
    n = len(pts)

    for _iteration in range(iterations):
        forces = [[0.0, 0.0, 0.0] for _ in range(n)]
        max_force = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                dz = pts[i][2] - pts[j][2]
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq < 1e-20:
                    dist_sq = 1e-20
                inv_dist_sq = 1.0 / dist_sq
                inv_dist = sqrt(inv_dist_sq)

                fx = inv_dist_sq * inv_dist * dx
                fy = inv_dist_sq * inv_dist * dy
                fz = inv_dist_sq * inv_dist * dz

                forces[i][0] += fx
                forces[i][1] += fy
                forces[i][2] += fz
                forces[j][0] -= fx
                forces[j][1] -= fy
                forces[j][2] -= fz

        for i in range(n):
            px, py, pz = pts[i]
            dot = forces[i][0] * px + forces[i][1] * py + forces[i][2] * pz
            forces[i][0] -= dot * px
            forces[i][1] -= dot * py
            forces[i][2] -= dot * pz

            fm = sqrt(
                forces[i][0] ** 2 + forces[i][1] ** 2 + forces[i][2] ** 2
            )
            if fm > max_force:
                max_force = fm

        if max_force < 1e-12:
            break

        step = step_scale / max_force
        for i in range(n):
            pts[i][0] += step * forces[i][0]
            pts[i][1] += step * forces[i][1]
            pts[i][2] += step * forces[i][2]

            mag = sqrt(pts[i][0] ** 2 + pts[i][1] ** 2 + pts[i][2] ** 2)
            if mag > 1e-10:
                pts[i][0] /= mag
                pts[i][1] /= mag
                pts[i][2] /= mag

    return pts


def _local_push_apart(
    pts: list[list[float]],
    min_center_dist: float,
    iterations: int = 500,
) -> list[list[float]]:
    """
    After global Coulomb optimization, specifically push apart any pairs
    that are still closer than min_center_dist. Faster convergence for
    the last-mile overlap fixes.
    """
    n = len(pts)
    for _it in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                dz = pts[i][2] - pts[j][2]
                dist = sqrt(dx * dx + dy * dy + dz * dz)
                if dist >= min_center_dist:
                    continue
                if dist < 1e-10:
                    dist = 1e-10
                    dx, dy, dz = 1e-5, 0.0, 0.0

                overlap = (min_center_dist - dist) * 0.52
                ux, uy, uz = dx / dist, dy / dist, dz / dist

                pts[i][0] += overlap * ux
                pts[i][1] += overlap * uy
                pts[i][2] += overlap * uz
                pts[j][0] -= overlap * ux
                pts[j][1] -= overlap * uy
                pts[j][2] -= overlap * uz

                for k in (i, j):
                    mag = sqrt(pts[k][0] ** 2 + pts[k][1] ** 2 + pts[k][2] ** 2)
                    if mag > 1e-10:
                        pts[k][0] /= mag
                        pts[k][1] /= mag
                        pts[k][2] /= mag

                moved = True

        if not moved:
            break

    return pts


def generate_fibonacci_dimple_centers(
    dimple_count: int,
    ball_radius_mm: float = BALL_DIAMETER_MM / 2.0,
    dimple_diameter_mm: float | None = None,
    optimize: bool = True,
) -> list[dict[str, float]]:
    """
    Generate near-uniform dimple center points on a sphere.

    1. Start with shifted Fibonacci (golden-angle) lattice.
    2. Optionally refine with electrostatic repulsion optimization
       to maximize minimum nearest-neighbor distance.
    3. Scale to actual ball radius and return positions + normals.
    """
    unit_pts = _fibonacci_seed_points(dimple_count)

    if optimize and dimple_diameter_mm is not None and dimple_count > 1:
        min_center_dist_unit = (dimple_diameter_mm + MIN_LAND_WIDTH_MM) / ball_radius_mm
        unit_pts = _repulsion_optimize(unit_pts, min_center_dist_unit)
        unit_pts = _local_push_apart(unit_pts, min_center_dist_unit)

    points: list[dict[str, float]] = []
    for p in unit_pts:
        nx, ny, nz = p[0], p[1], p[2]
        points.append({
            "x": nx * ball_radius_mm,
            "y": ny * ball_radius_mm,
            "z": nz * ball_radius_mm,
            "nx": nx,
            "ny": ny,
            "nz": nz,
        })

    return points


def validate_dimple_placement(
    points: list[dict[str, float]],
    dimple_diameter_mm: float,
    min_land_width_mm: float = MIN_LAND_WIDTH_MM,
) -> dict[str, object]:
    """
    Check all generated dimple center pairs for spacing.
    Distinguishes between:
      - overlaps: center distance < dimple diameter (physical impossibility)
      - tight gaps: dimple diameter <= center distance < dimple diameter + min land width
    """
    n = len(points)
    min_gap_found = float("inf")
    overlaps = 0
    tight_gaps = 0
    min_center_dist = dimple_diameter_mm + min_land_width_mm

    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i]["x"] - points[j]["x"]
            dy = points[i]["y"] - points[j]["y"]
            dz = points[i]["z"] - points[j]["z"]
            dist = sqrt(dx * dx + dy * dy + dz * dz)
            gap = dist - dimple_diameter_mm
            if gap < min_gap_found:
                min_gap_found = gap
            if dist < dimple_diameter_mm:
                overlaps += 1
            elif dist < min_center_dist:
                tight_gaps += 1

    no_overlap = overlaps == 0
    return {
        "passed": no_overlap and tight_gaps == 0,
        "buildable": no_overlap,
        "total_pairs": n * (n - 1) // 2,
        "overlaps": overlaps,
        "tight_gaps": tight_gaps,
        "min_gap_mm": round(min_gap_found, 4),
        "required_min_gap_mm": min_land_width_mm,
    }


def export_solidworks_csv(path: Path, points: list[dict[str, float]]) -> None:
    """Write dimple centers to CSV for SolidWorks import."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x_mm", "y_mm", "z_mm", "nx", "ny", "nz"])
        for idx, p in enumerate(points, start=1):
            writer.writerow([
                idx,
                round(p["x"], 6),
                round(p["y"], 6),
                round(p["z"], 6),
                round(p["nx"], 6),
                round(p["ny"], 6),
                round(p["nz"], 6),
            ])


def export_design_json(path: Path, design: DimpleDesign) -> None:
    """Write design parameters to JSON for reference."""
    depth_mm = design.depth_ratio * BALL_DIAMETER_MM
    data = {
        "ball_diameter_mm": BALL_DIAMETER_MM,
        "ball_radius_mm": BALL_DIAMETER_MM / 2.0,
        "dimple_diameter_mm": design.dimple_diameter_mm,
        "dimple_depth_mm": round(depth_mm, 4),
        "depth_ratio_k_over_ball_d": design.depth_ratio,
        "depth_ratio_k_over_dimple_d": round(depth_mm / design.dimple_diameter_mm, 6),
        "dimple_count": design.dimple_count,
        "occupancy": design.occupancy,
        "volume_ratio": design.volume_ratio,
        "estimated_land_width_mm": round(
            estimate_hex_packing_land_width_mm(design), 4
        ),
        "min_land_width_mm": MIN_LAND_WIDTH_MM,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


SW_MACRO_TEMPLATE = r'''
' ============================================================
' SolidWorks VBA Macro: Import dimple centers and cut dimples
' Generated by Golf Ball Simulator
' ============================================================
' USAGE:
'   1. Open the base ball part (sphere, D={ball_diameter_mm:.2f} mm).
'   2. Tools > Macro > Run, select this .bas file.
'   3. The macro reads "{csv_filename}" from the same folder,
'      creates reference points, and cuts spherical-cap dimples.
'
' PARAMETERS (from optimization):
'   Ball diameter  = {ball_diameter_mm:.2f} mm
'   Dimple diameter = {dimple_diameter_mm:.2f} mm
'   Dimple depth   = {dimple_depth_mm:.4f} mm
'   Dimple count   = {dimple_count}
' ============================================================

Option Explicit

Sub Main()

    Dim swApp As SldWorks.SldWorks
    Dim swModel As SldWorks.ModelDoc2
    Dim swPart As SldWorks.PartDoc
    Dim swSkMgr As SldWorks.SketchManager
    Dim swFeatMgr As SldWorks.FeatureManager

    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc

    If swModel Is Nothing Then
        MsgBox "Please open the golf ball part first.", vbExclamation
        Exit Sub
    End If

    Set swPart = swModel
    Set swSkMgr = swModel.SketchManager
    Set swFeatMgr = swModel.FeatureManager

    ' --- Configuration ---
    Dim csvPath As String
    csvPath = swModel.GetPathName()
    csvPath = Left(csvPath, InStrRev(csvPath, "\")) & "{csv_filename}"

    Dim dimpleDiameterM As Double
    dimpleDiameterM = {dimple_diameter_mm:.6f} / 1000#   ' convert mm to meters

    Dim dimpleDepthM As Double
    dimpleDepthM = {dimple_depth_mm:.6f} / 1000#          ' convert mm to meters

    Dim dimpleRadiusM As Double
    dimpleRadiusM = dimpleDiameterM / 2#

    Dim ballRadiusM As Double
    ballRadiusM = {ball_diameter_mm:.6f} / 2# / 1000#

    ' --- Read CSV ---
    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")

    If Not fso.FileExists(csvPath) Then
        MsgBox "CSV not found: " & csvPath, vbExclamation
        Exit Sub
    End If

    Dim ts As Object
    Set ts = fso.OpenTextFile(csvPath, 1)
    ts.SkipLine  ' header

    Dim lineData As String
    Dim parts() As String
    Dim cx As Double, cy As Double, cz As Double
    Dim nx As Double, ny As Double, nz As Double
    Dim count As Long
    count = 0

    Do While Not ts.AtEndOfStream
        lineData = ts.ReadLine
        parts = Split(lineData, ",")
        If UBound(parts) >= 6 Then
            cx = CDbl(parts(1)) / 1000#
            cy = CDbl(parts(2)) / 1000#
            cz = CDbl(parts(3)) / 1000#
            nx = CDbl(parts(4))
            ny = CDbl(parts(5))
            nz = CDbl(parts(6))

            ' Create a 3D sketch point at dimple center
            swModel.Insert3DSketch
            swSkMgr.CreatePoint cx, cy, cz
            swModel.InsertSketch2 True

            count = count + 1
        End If
    Loop
    ts.Close

    MsgBox "Done. Created " & count & " dimple center points." & vbCrLf & _
           "Next step: select each point and apply a spherical-cap cut " & _
           "(diameter=" & Format(dimpleDiameterM * 1000, "0.00") & " mm, " & _
           "depth=" & Format(dimpleDepthM * 1000, "0.000") & " mm).", vbInformation

End Sub
'''


def export_solidworks_macro(
    path: Path, design: DimpleDesign, csv_filename: str = "dimple_centers.csv"
) -> None:
    """Write a SolidWorks VBA macro template."""
    depth_mm = design.depth_ratio * BALL_DIAMETER_MM
    content = SW_MACRO_TEMPLATE.format(
        ball_diameter_mm=BALL_DIAMETER_MM,
        dimple_diameter_mm=design.dimple_diameter_mm,
        dimple_depth_mm=depth_mm,
        dimple_count=design.dimple_count,
        csv_filename=csv_filename,
    )
    path.write_text(content.strip() + "\n", encoding="utf-8")


def export_solidworks_pack(
    output_dir: Path, design: DimpleDesign
) -> dict[str, str]:
    """
    Generate a complete SolidWorks import pack:
      - dimple_centers.csv  (x,y,z,nx,ny,nz for each dimple)
      - design_params.json  (all key parameters)
      - import_dimples.bas  (VBA macro template)
      - validation.json     (placement check results)
    Returns dict of filename -> description.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    points = generate_fibonacci_dimple_centers(
        design.dimple_count,
        dimple_diameter_mm=design.dimple_diameter_mm,
        optimize=True,
    )
    validation = validate_dimple_placement(
        points, design.dimple_diameter_mm
    )

    csv_path = output_dir / "dimple_centers.csv"
    json_path = output_dir / "design_params.json"
    macro_path = output_dir / "import_dimples.bas"
    valid_path = output_dir / "validation.json"

    export_solidworks_csv(csv_path, points)
    export_design_json(json_path, design)
    export_solidworks_macro(macro_path, design, csv_filename="dimple_centers.csv")
    valid_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "dimple_centers.csv": f"{design.dimple_count} dimple center coordinates (x,y,z + normals)",
        "design_params.json": "All design parameters for reference",
        "import_dimples.bas": "SolidWorks VBA macro template for importing points",
        "validation.json": (
            f"Placement: {'PASSED' if validation['passed'] else 'BUILDABLE (tight gaps)' if validation['buildable'] else 'FAILED (overlaps)'}, "
            f"min gap={validation['min_gap_mm']} mm, overlaps={validation['overlaps']}, tight={validation['tight_gaps']}"
        ),
    }


def apply_manufacturability_filter(
    designs: list[DimpleDesign],
    min_land_width_mm: float = MIN_LAND_WIDTH_MM,
) -> list[DimpleDesign]:
    """
    Keep only manufacturable candidates.
    Returns an empty list when no design is physically buildable.
    """
    return [d for d in designs if is_design_manufacturable(d, min_land_width_mm)]


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
    cl_spin = 1.6 * sp
    cl_geom_factor = (
        1.0
        + 0.30 * (design.occupancy - 0.75)
        - 35.0 * (design.depth_ratio - 4.55e-3)
        + 5.0 * (design.volume_ratio - 0.011)
    )
    cl_geom_factor = clamp(cl_geom_factor, 0.55, 1.45)
    cl = cl_spin * cl_geom_factor
    cl = clamp(cl, 0.0, 0.40)

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

    if mode == "paper_depth_only":
        grp = next(g for g in PAPER_OCCUPANCY_GROUPS if abs(g["occupancy"] - 0.812) < 1e-9)
        i = PAPER_OCCUPANCY_GROUPS.index(grp)
        dimple_diam_mm = grp["cm_over_d"] * BALL_DIAMETER * 1000.0
        n = grp["dimple_count"]
        occ = grp["occupancy"]
        if n > theoretical_max_dimple_count(dimple_diam_mm):
            dimple_diam_mm = floor(max_feasible_dimple_diameter(n) * 100) / 100
            occ = geometric_occupancy(n, dimple_diam_mm)
        return [
            DimpleDesign(
                depth_ratio=dr,
                occupancy=occ,
                volume_ratio=PAPER_VDR_MATRIX[dr][i],
                dimple_count=n,
                dimple_diameter_mm=dimple_diam_mm,
            )
            for dr in PAPER_DEPTH_RATIOS
        ]

    if mode == "paper_occupancy_only":
        depth_fixed = 4.55e-3
        designs_occ = []
        for i, grp in enumerate(PAPER_OCCUPANCY_GROUPS):
            diam = grp["cm_over_d"] * BALL_DIAMETER * 1000.0
            n = grp["dimple_count"]
            occ = grp["occupancy"]
            if n > theoretical_max_dimple_count(diam):
                diam = floor(max_feasible_dimple_diameter(n) * 100) / 100
                occ = geometric_occupancy(n, diam)
            designs_occ.append(
                DimpleDesign(
                    depth_ratio=depth_fixed,
                    occupancy=occ,
                    volume_ratio=PAPER_VDR_MATRIX[depth_fixed][i],
                    dimple_count=n,
                    dimple_diameter_mm=diam,
                )
            )
        return designs_occ

    if mode == "paper_volume_only":
        # VDR trend is assessed from the full literature set.
        # Keep this mode as an alias to the same 50-design set for clear trend inspection.
        mode = "paper_literature_grid"

    # paper_literature_grid
    # Build 50 deterministic, literature-aligned designs:
    # - original measured ranges are preserved
    # - extra dummy points are interpolation-based (never random)
    # - if interpolated (N, diameter) exceeds the physical packing limit,
    #   the diameter is shrunk to the largest feasible value so that
    #   high-occupancy designs remain in the candidate pool.
    designs = []
    for dr in PAPER_GRID_DEPTH_LEVELS:
        for occupancy in PAPER_GRID_OCCUPANCY_LEVELS:
            cm_over_d = literature_group_value_at_occupancy(occupancy, "cm_over_d")
            mean_dimple_diam_mm = cm_over_d * BALL_DIAMETER * 1000.0
            dimple_count = int(round(literature_group_value_at_occupancy(occupancy, "dimple_count")))

            actual_occ = occupancy
            if dimple_count > theoretical_max_dimple_count(mean_dimple_diam_mm):
                mean_dimple_diam_mm = floor(
                    max_feasible_dimple_diameter(dimple_count) * 100
                ) / 100
                actual_occ = geometric_occupancy(dimple_count, mean_dimple_diam_mm)

            designs.append(
                DimpleDesign(
                    depth_ratio=dr,
                    occupancy=actual_occ,
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
