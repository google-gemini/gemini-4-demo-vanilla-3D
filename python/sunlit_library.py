#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sunlit Concrete & Walnut Library
================================

Physically-Based Architectural Interior Ray-Tracing & Radiosity Engine
----------------------------------------------------------------------
- 35mm Tilt-Shift two-point perspective camera (strictly plumb verticals, shifted
  optical center) tracing 1,600 x 1,200 primary rays in 64-bit floating point.
- Multi-layered volumetric coastal sky with high-altitude wind-sheared cirrus,
  3D ray-marched cumulus/stratocumulus cloud banks (Beer-Lambert self-shadowing +
  Mie multiple-scattering powder effect), layered coastal headlands, limestone
  promontory with maquis scrub, bush-hammered exterior terrace coping, and
  wind-rippled coastal water.
- True 3D extruded blackened-steel window mullions, transoms, and EPDM rubber
  glazing gaskets.
- Authentic cast-in-place board-formed mineral grey concrete with stochastic 3D
  bug-holes (pinhole air voids), proud cement-slurry board-seam fins, form-tie cones
  with mortar plugs and mineral weeping tracks, and architectural shadow-gap reveals.
- Vein-cut honed Roman travertine slabs with sedimentary bedding, crystalline calcite
  veinlets, and high-frequency 2mm-8mm micro-vugs; wide-plank oiled European oak
  with medullary rays, sound pin knots with grain deflection, and per-plank micro-tilt;
  and a raised 3D hand-loomed Belgian wool bouclé area rug with bound linen border.
- Sculpted aniline cognac saddle-leather lounge chair with 3-channel stitched tufting,
  radiating sitting creases, pull-up patina, integrated cylindrical headrest bolster,
  and wrapped leather armrests on a cantilevered steel sled frame.
- Lived-in walnut library with vertical & leaning book runs, horizontal folio stacks,
  raised bindery hubs, gold-foil stamping, travertine/brass bookends, wheel-thrown
  ribbed stoneware vases, recessed shelf grazing light, and a back credenza vignette
  with a leaning framed architectural bas-relief artwork.
- Foreground coffee table still life with connected eucalyptus stems & veined leaves
  in a wheel-thrown stoneware vase, ribbed glass carafe, faceted crystal tumbler on
  brass coaster, multi-wavelength dispersive nephroid caustics, open monograph with
  silk ribbon, stoneware espresso cup, saucer & brass demitasse spoon, and stacked
  architecture monographs.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ==============================================================================
# 1. MATHEMATICAL, OPTICAL & MICROFACET BRDF UTILITY FUNCTIONS
# ==============================================================================

EPSILON: float = 1e-6
PI: float = math.pi


def saturate(x: np.ndarray | float) -> np.ndarray | float:
    """Clamp array or scalar to [0.0, 1.0]."""
    return np.clip(x, 0.0, 1.0)


def smoothstep(edge0: np.ndarray | float, edge1: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    """Hermite C1 smooth interpolation on [edge0, edge1]."""
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: np.ndarray | float, edge1: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    """Quintic C2-continuous interpolation on [edge0, edge1]."""
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def normalize(v: np.ndarray) -> np.ndarray:
    """Fast vector normalization along the last axis."""
    norm = np.sqrt(np.maximum(np.sum(v * v, axis=-1, keepdims=True), 1e-24))
    return v / norm


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product along the last axis."""
    return np.sum(a * b, axis=-1)


def dot_keep(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product along the last axis with keepdims=True."""
    return np.sum(a * b, axis=-1, keepdims=True)


def reflect(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect incident vector d around unit surface normal n."""
    return incident - 2.0 * dot_keep(incident, normal) * normal


def refract(incident: np.ndarray, normal: np.ndarray, eta: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Snell's Law vector refraction for unit incident vector `incident` and unit `normal`.
    `eta` = n_i / n_t. Returns (refracted_dir, valid_mask).
    """
    cos_i = -dot_keep(incident, normal)
    if isinstance(eta, np.ndarray) and eta.ndim < cos_i.ndim:
        eta = eta[..., None]
    sin2_t = (eta * eta) * np.maximum(0.0, 1.0 - cos_i * cos_i)
    valid = (sin2_t <= 1.0)[..., 0]
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
    refr_dir = eta * incident + (eta * cos_i - cos_t) * normal
    refr_dir = normalize(refr_dir)
    return refr_dir, valid


def schlick_fresnel(cos_theta: np.ndarray, f0: np.ndarray | float) -> np.ndarray:
    """Schlick approximation to Fresnel reflectance."""
    c = np.clip(1.0 - np.clip(cos_theta, 0.0, 1.0), 0.0, 1.0)
    c2 = c * c
    c5 = c2 * c2 * c
    if isinstance(f0, np.ndarray) and f0.ndim > cos_theta.ndim:
        c5 = c5[..., None]
    return f0 + (1.0 - f0) * c5


def dielectric_fresnel_exact(cos_i: np.ndarray, eta_i: float, eta_t: float) -> np.ndarray:
    """Exact unpolarized dielectric Fresnel reflectance for glass and water."""
    ci = np.clip(np.abs(cos_i), 0.0, 1.0)
    sin2_t = (eta_i / eta_t) ** 2 * np.maximum(0.0, 1.0 - ci * ci)
    total_internal = sin2_t >= 1.0
    ct = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
    rs = (eta_i * ci - eta_t * ct) / np.maximum(eta_i * ci + eta_t * ct, 1e-8)
    rp = (eta_t * ci - eta_i * ct) / np.maximum(eta_t * ci + eta_i * ct, 1e-8)
    f = 0.5 * (rs * rs + rp * rp)
    return np.where(total_internal, 1.0, f)


def ggx_isotropic_d(n_dot_h: np.ndarray, roughness: np.ndarray | float) -> np.ndarray:
    """Trowbridge-Reitz (GGX) isotropic normal distribution function D(h)."""
    alpha = np.maximum(roughness * roughness, 0.0012)
    a2 = alpha * alpha
    c2 = np.clip(n_dot_h, 0.0, 1.0) ** 2
    denom = c2 * (a2 - 1.0) + 1.0
    return a2 / np.maximum(PI * denom * denom, 1e-10)


def ggx_anisotropic_d(
    n_dot_h: np.ndarray,
    t_dot_h: np.ndarray,
    b_dot_h: np.ndarray,
    ax: np.ndarray | float,
    ay: np.ndarray | float,
) -> np.ndarray:
    """
    Anisotropic Trowbridge-Reitz (GGX) normal distribution function D(h) for
    brushed metals, quarter-sawn timber fibers, and silk ribbons.
    """
    ax_s = np.maximum(ax, 0.002)
    ay_s = np.maximum(ay, 0.002)
    term = (t_dot_h / ax_s) ** 2 + (b_dot_h / ay_s) ** 2 + np.clip(n_dot_h, 0.0, 1.0) ** 2
    return 1.0 / np.maximum(PI * ax_s * ay_s * term * term, 1e-10)


def smith_ggx_g(n_dot_v: np.ndarray, n_dot_l: np.ndarray, roughness: np.ndarray | float) -> np.ndarray:
    """Height-correlated Smith GGX geometric shadowing-masking factor G / (4 * N.V * N.L)."""
    alpha = np.maximum(roughness * roughness, 0.0012)
    a2 = alpha * alpha
    nv = np.clip(n_dot_v, 1e-4, 1.0)
    nl = np.clip(n_dot_l, 1e-4, 1.0)
    gv = nl * np.sqrt(nv * nv * (1.0 - a2) + a2)
    gl = nv * np.sqrt(nl * nl * (1.0 - a2) + a2)
    return 0.5 / np.maximum(gv + gl, 1e-8)


def oren_nayar_diffuse(
    normal: np.ndarray,
    view_dir: np.ndarray,
    light_dir: np.ndarray,
    n_dot_l: np.ndarray,
    n_dot_v: np.ndarray,
    roughness: np.ndarray,
) -> np.ndarray:
    """
    Full Oren-Nayar rough micro-cavity diffuse reflectance factor (returns multiplier
    on Lambertian (albedo / PI) * N.L) for porous stone, concrete, paper, and wool.
    """
    l_vec = light_dir[None, :] if light_dir.ndim == 1 else light_dir
    sigma2 = (roughness * 0.65) ** 2
    a_on = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33))
    b_on = 0.45 * (sigma2 / (sigma2 + 0.09))

    v_perp = view_dir - normal * n_dot_v[..., None]
    l_perp = l_vec - normal * n_dot_l[..., None]
    v_perp_norm = np.sqrt(np.maximum(dot(v_perp, v_perp), 1e-10))
    l_perp_norm = np.sqrt(np.maximum(dot(l_perp, l_perp), 1e-10))
    cos_phi_diff = np.maximum(0.0, dot(v_perp, l_perp) / (v_perp_norm * l_perp_norm))

    sin_theta_i = np.sqrt(np.maximum(0.0, 1.0 - n_dot_l * n_dot_l))
    sin_theta_r = np.sqrt(np.maximum(0.0, 1.0 - n_dot_v * n_dot_v))
    sin_alpha = np.maximum(sin_theta_i, sin_theta_r)
    tan_beta = np.minimum(sin_theta_i / np.maximum(n_dot_l, 1e-4), sin_theta_r / np.maximum(n_dot_v, 1e-4))

    return a_on + b_on * cos_phi_diff * sin_alpha * tan_beta


def cook_torrance_brdf(
    normal: np.ndarray,
    view_dir: np.ndarray,
    light_dir: np.ndarray,
    albedo: np.ndarray,
    roughness: np.ndarray,
    f0: np.ndarray,
    metallic: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Cook-Torrance GGX microfacet BRDF + Oren-Nayar rough diffuse * (N . L),
    split into (diffuse, specular) radiance transfer factors.
    """
    l_vec = light_dir[None, :] if light_dir.ndim == 1 else light_dir
    h_vec = normalize(view_dir + l_vec)

    n_dot_l = np.clip(dot(normal, l_vec), 0.0, 1.0)
    n_dot_v = np.clip(dot(normal, view_dir), 1e-4, 1.0)
    n_dot_h = np.clip(dot(normal, h_vec), 0.0, 1.0)
    v_dot_h = np.clip(dot(view_dir, h_vec), 0.0, 1.0)

    d_term = ggx_isotropic_d(n_dot_h, roughness)
    g_term = smith_ggx_g(n_dot_v, n_dot_l, roughness)
    f_term = schlick_fresnel(v_dot_h, f0)

    if isinstance(metallic, np.ndarray) and metallic.ndim < f_term.ndim:
        m_exp = metallic[..., None]
    else:
        m_exp = metallic

    spec = (d_term * g_term * n_dot_l)[..., None] * f_term
    kd = (1.0 - f_term) * (1.0 - m_exp)
    on_mult = oren_nayar_diffuse(normal, view_dir, l_vec, n_dot_l, n_dot_v, roughness)
    diff = kd * (albedo / PI) * (n_dot_l * on_mult)[..., None]
    return diff, spec


# ==============================================================================
# 2. HIGH-SPEED PRECOMPUTED C2-CONTINUOUS MATERIAL & CELLULAR NOISE ENGINE
# ==============================================================================


def hash2d(ix: np.ndarray, iy: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fast vectorized integer lattice hash returning float64 in [0, 1]."""
    n = (ix * 374761393 + iy * 668265263 + seed * 1442695041) & 0x7FFFFFFF
    n = (n ^ (n >> 13)) * 1274126177
    n = (n ^ (n >> 16)) & 0x7FFFFFFF
    return n.astype(np.float64) / 2147483647.0


def _build_periodic_lattice(grid_n: int = 1024, seed: int = 20301) -> np.ndarray:
    """
    Precompute a periodic 1024 x 1024 float64 random lattice so continuous
    Hermite-interpolated 2D noise evaluations execute at high speed without
    nearest-neighbor quantization artifacts at any frequency.
    """
    gx, gy = np.meshgrid(
        np.arange(grid_n, dtype=np.int64),
        np.arange(grid_n, dtype=np.int64),
        indexing="xy",
    )
    return np.ascontiguousarray(hash2d(gx, gy, seed=seed), dtype=np.float64)


_RAND_1024: np.ndarray = _build_periodic_lattice(1024, seed=20301)


def value_noise_2d(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Continuous Hermite-interpolated 2D value noise over a periodic 1024x1024 lattice."""
    ox = (seed * 173 + 37) & 1023
    oy = (seed * 293 + 91) & 1023
    x_s = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y_s = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    xf = np.floor(x_s)
    yf = np.floor(y_s)
    fx = x_s - xf
    fy = y_s - yf
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    ix0 = (xf.astype(np.int64) + ox) & 1023
    iy0 = (yf.astype(np.int64) + oy) & 1023
    ix1 = (ix0 + 1) & 1023
    iy1 = (iy0 + 1) & 1023

    c00 = _RAND_1024[iy0, ix0]
    c10 = _RAND_1024[iy0, ix1]
    c01 = _RAND_1024[iy1, ix0]
    c11 = _RAND_1024[iy1, ix1]

    v0 = c00 + ux * (c10 - c00)
    v1 = c01 + ux * (c11 - c01)
    return v0 + uy * (v1 - v0)


def value_noise_2d_grad(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous quintic-interpolated 2D value noise returning (val, dval_dx, dval_dy)
    for analytic micro-normal bump mapping on leather, concrete, stone, and bouclé yarn.
    """
    ox = (seed * 173 + 37) & 1023
    oy = (seed * 293 + 91) & 1023
    x_s = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y_s = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    xf = np.floor(x_s)
    yf = np.floor(y_s)
    fx = x_s - xf
    fy = y_s - yf

    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    dux = 30.0 * fx * fx * (fx * (fx - 2.0) + 1.0)
    duy = 30.0 * fy * fy * (fy * (fy - 2.0) + 1.0)

    ix0 = (xf.astype(np.int64) + ox) & 1023
    iy0 = (yf.astype(np.int64) + oy) & 1023
    ix1 = (ix0 + 1) & 1023
    iy1 = (iy0 + 1) & 1023

    c00 = _RAND_1024[iy0, ix0]
    c10 = _RAND_1024[iy0, ix1]
    c01 = _RAND_1024[iy1, ix0]
    c11 = _RAND_1024[iy1, ix1]

    k0 = c00
    k1 = c10 - c00
    k2 = c01 - c00
    k3 = c00 - c10 - c01 + c11

    val = k0 + k1 * ux + k2 * uy + k3 * ux * uy
    dv_dx = (k1 + k3 * uy) * dux
    dv_dy = (k2 + k3 * ux) * duy
    return val, dv_dx, dv_dy


def fbm2d(
    x: np.ndarray,
    y: np.ndarray,
    octaves: int = 3,
    lacunarity: float = 2.03,
    gain: float = 0.5,
    seed: int = 10,
) -> np.ndarray:
    """Multi-octave fractional Brownian motion in 2D."""
    total = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    cs, sn = math.cos(0.52), math.sin(0.52)
    xr, yr = x, y
    for i in range(octaves):
        total += amp * value_noise_2d(xr * freq, yr * freq, seed + i * 31)
        norm += amp
        amp *= gain
        freq *= lacunarity
        xr, yr = cs * xr - sn * yr, sn * xr + cs * yr
    return total / max(norm, 1e-8)


def billow_fbm2d(
    x: np.ndarray,
    y: np.ndarray,
    octaves: int = 4,
    lacunarity: float = 2.07,
    gain: float = 0.52,
    seed: int = 101,
) -> np.ndarray:
    """
    Multi-octave cumuliform billow noise (`1.0 - |2*n - 1|`) that synthesizes
    rounded cauliflower cloud domes and crisp inter-lobe crevices.
    """
    total = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    cs, sn = math.cos(0.48), math.sin(0.48)
    xr, yr = x, y
    for i in range(octaves):
        n_raw = value_noise_2d(xr * freq, yr * freq, seed + i * 37)
        billow = 1.0 - np.abs(2.0 * n_raw - 1.0)
        billow_shaped = billow * billow * (1.5 - 0.5 * billow)
        total += amp * (0.62 * billow_shaped + 0.38 * n_raw)
        norm += amp
        amp *= gain
        freq *= lacunarity
        xr, yr = cs * xr - sn * yr, sn * xr + cs * yr
    return total / max(norm, 1e-8)


def cellular_bugholes_2d(
    u: np.ndarray,
    v: np.ndarray,
    density_prob: float = 0.38,
    seed: int = 555,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stochastic cellular pinhole / bug-hole generator for cast-in-place architectural
    concrete and porous stone. Returns (pit_mask, rim_nx, rim_ny) where `pit_mask` in [0, 1]
    marks the concave void and `(rim_nx, rim_ny)` is the 3D indentation normal vector
    so each pinhole catches grazing light on its bottom/side rim.
    """
    u_s = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v_s = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.floor(u_s).astype(np.int64)
    iv = np.floor(v_s).astype(np.int64)
    fu = u_s - iu
    fv = v_s - iv

    cell_prob = hash2d(iu, iv, seed=seed)
    cx = 0.22 + 0.56 * hash2d(iu, iv, seed=seed + 13)
    cy = 0.22 + 0.56 * hash2d(iu, iv, seed=seed + 29)
    radius = 0.11 + 0.24 * (hash2d(iu, iv, seed=seed + 47) ** 1.6)

    dx = fu - cx
    dy = fv - cy
    r = np.hypot(dx, dy)
    active = cell_prob < density_prob

    norm_r = r / np.maximum(radius, 1e-4)
    pit_depth = np.where(active & (norm_r < 1.0), np.maximum(1.0 - norm_r * norm_r, 0.0) ** 1.5, 0.0)
    rim_slope = np.where(
        active & (norm_r < 1.15) & (norm_r > 0.15),
        np.exp(-((norm_r - 0.82) / 0.24) ** 2),
        0.0,
    )
    inv_r = 1.0 / np.maximum(r, 1e-4)
    rim_nx = -dx * inv_r * rim_slope
    rim_ny = -dy * inv_r * rim_slope
    return pit_depth, rim_nx, rim_ny


def gaussian_blur_rgb(arr: np.ndarray, radius: float) -> np.ndarray:
    """Separable 1D Gaussian blur in float64 using pure NumPy."""
    if radius <= 0.25:
        return arr
    sigma = float(radius)
    half = max(1, math.ceil(sigma * 2.6))
    xs = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= np.sum(kernel)

    if arr.ndim == 2:
        p0 = np.pad(arr, ((half, half), (0, 0)), mode="edge")
        out0 = np.zeros_like(arr, dtype=np.float64)
        for k, w in enumerate(kernel):
            out0 += w * p0[k : k + arr.shape[0], :]
        p1 = np.pad(out0, ((0, 0), (half, half)), mode="edge")
        out1 = np.zeros_like(arr, dtype=np.float64)
        for k, w in enumerate(kernel):
            out1 += w * p1[:, k : k + arr.shape[1]]
        return out1

    p0 = np.pad(arr, ((half, half), (0, 0), (0, 0)), mode="edge")
    out0 = np.zeros_like(arr, dtype=np.float64)
    for k, w in enumerate(kernel):
        out0 += w * p0[k : k + arr.shape[0], :, :]
    p1 = np.pad(out0, ((0, 0), (half, half), (0, 0)), mode="edge")
    out1 = np.zeros_like(arr, dtype=np.float64)
    for k, w in enumerate(kernel):
        out1 += w * p1[:, k : k + arr.shape[1], :]
    return out1


# ==============================================================================
# 3. ARCHITECTURAL ROOM, CALIBRATED SKY & MATERIAL SPECIFICATION
# ==============================================================================
# World Coordinates (meters):
#   X in [-3.10, +3.20] : Left Walnut Library (-3.10) to Right Window Curtain Wall (+3.20)
#   Y in [ 0.00, +4.25] : Wide-Plank European Oak / Travertine Floor (0.00) to Concrete Ceiling (+4.25)
#   Z in [-1.80, +7.60] : Camera Foreground (-1.52) to Back Board-Formed Concrete & Corner Window (+7.60)

ROOM_X_LEFT: float = -3.10
ROOM_X_RIGHT: float = 3.20
ROOM_Y_FLOOR: float = 0.00
ROOM_Y_CEIL: float = 4.25
ROOM_Z_BACK: float = 7.60
BACK_CONCRETE_X_MAX: float = 1.08

# Camera parameters: 35mm architectural tilt-shift lens (strictly plumb verticals)
CAM_POS = np.array([-0.18, 1.38, -1.52], dtype=np.float64)
CAM_YAW_RAD: float = math.radians(2.35)
CAM_FOV_SCALE: float = 0.615
CAM_V_SHIFT: float = -0.045

# Solar & Calibrated Coastal Sky Lighting Vectors
# Calibrated so ACES tone-mapping preserves a rich cerulean-cobalt zenith,
# azure mid-sky, golden-apricot horizon, and bright sculpted cumulus clouds.
SUN_DIR = normalize(np.array([0.755, 0.295, 0.585], dtype=np.float64))
SUN_RADIANCE = np.array([15.8, 11.4, 6.7], dtype=np.float64)
SKY_ZENITH_RAD = np.array([0.085, 0.31, 1.14], dtype=np.float64)
SKY_MID_RAD = np.array([0.19, 0.49, 1.34], dtype=np.float64)
SKY_HORIZON_RAD = np.array([0.88, 0.78, 0.72], dtype=np.float64)

# Warm architectural brass reading sconce mounted on the left walnut library vertical stile (Z = 2.55)
SCONCE_POS = np.array([-2.68, 2.26, 2.55], dtype=np.float64)
SCONCE_RADIANCE = np.array([5.85, 3.10, 1.02], dtype=np.float64)

RIGHT_MULLION_Z = np.array([-0.20, 1.65, 3.55, 5.50, 7.52], dtype=np.float64)
RIGHT_TRANSOM_Y = np.array([0.09, 1.08, 2.98, 3.88], dtype=np.float64)
BACK_MULLION_X = np.array([1.14, 2.18, 3.14], dtype=np.float64)

BOOKCASE_SHELF_Y = np.array([0.08, 0.66, 1.22, 1.78, 2.34, 2.90, 3.46, 4.02], dtype=np.float64)
BOOKCASE_STILE_Z = np.array([-0.80, 0.90, 2.55, 4.25, 5.92, 7.58], dtype=np.float64)

# Material IDs
MAT_FLOOR_OAK: int = 1
MAT_FLOOR_TRAVERTINE: int = 2
MAT_BACK_CONCRETE: int = 3
MAT_CEILING_CONCRETE: int = 4
MAT_WALNUT_CASEWORK: int = 5
MAT_BOOK_SPINE: int = 6
MAT_BOOK_PAGES: int = 7
MAT_CERAMIC_VESSEL: int = 8
MAT_BRASS_SCONCE: int = 9
MAT_SCONCE_BULB: int = 10
MAT_STEEL_MULLION: int = 11
MAT_WINDOW_GLASS: int = 12
MAT_COFFEE_TABLE_TOP: int = 13
MAT_COFFEE_TABLE_BASE: int = 14
MAT_OPEN_BOOK_PAGE: int = 15
MAT_OPEN_BOOK_COVER: int = 16
MAT_LEATHER_CHAIR: int = 17
MAT_CHAIR_STEEL: int = 18
MAT_GLASS_CARAFE: int = 19
MAT_WATER_TUMBLER: int = 20
MAT_EUCALYPTUS_LEAF: int = 21
MAT_EUCALYPTUS_STEM: int = 22
MAT_BRASS_INLAY: int = 24
MAT_BOUCLE_RUG: int = 25
MAT_RUG_LINEN_BORDER: int = 26
MAT_LEATHER_STRAP: int = 27
MAT_ART_FRAME: int = 28
MAT_ART_CANVAS: int = 29
MAT_FLUTED_TRAVERTINE_VASE: int = 30
MAT_BOOKEND_STONE: int = 31
MAT_TABLE_CLOSED_BOOK: int = 32
MAT_ESPRESSO_CUP: int = 33
MAT_ESPRESSO_CREMA: int = 34
MAT_BOOKMARK_RIBBON: int = 36
MAT_GLAZING_GASKET: int = 37



# ==============================================================================
# 4. TILT-SHIFT CAMERA & BOUNDING-SPHERE RAY CULLING
# ==============================================================================


def build_camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (forward, right, up) basis with zero pitch for plumb verticals."""
    sn, cs = math.sin(CAM_YAW_RAD), math.cos(CAM_YAW_RAD)
    forward = np.array([sn, 0.0, cs], dtype=np.float64)
    right = np.array([cs, 0.0, -sn], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return forward, right, up


def generate_camera_rays(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate primary 35mm tilt-shift camera ray directions (H, W, 3) and screen (u, v).
    In widescreen 16:9 mode (--16x9 / ASPECT_16X9=1 / width/height > 1.45), keeps exact
    vertical half-extent v_half = 0.75 and expands horizontal frustum to u_half = v_half * (width / height).
    """
    forward, right, up = build_camera_basis()
    is_16x9 = (
        ("--16x9" in sys.argv)
        or (os.environ.get("ASPECT_16X9", "0") == "1")
        or (float(width) / float(max(height, 1)) > 1.45)
    )
    if is_16x9:
        v_half = 0.75
        u_half = v_half * (float(width) / float(max(height, 1)))
    else:
        u_half = 1.0
        v_half = float(height) / float(width)
    u_1d = np.linspace(-u_half, u_half, width, dtype=np.float64)
    v_1d = np.linspace(v_half, -v_half, height, dtype=np.float64) + CAM_V_SHIFT
    uu, vv = np.meshgrid(u_1d, v_1d)

    dirs = (
        forward[None, None, :]
        + (uu * CAM_FOV_SCALE)[..., None] * right[None, None, :]
        + (vv * CAM_FOV_SCALE)[..., None] * up[None, None, :]
    )
    dirs = normalize(dirs)
    return dirs, uu, vv


def rays_hitting_sphere(
    ro: np.ndarray,
    rd: np.ndarray,
    center: np.ndarray,
    radius: float,
    t_max: np.ndarray,
) -> np.ndarray:
    """
    Ultra-fast O(N) vectorized bounding-sphere test returning a boolean mask of rays
    that pass within `radius` of `center` before `t_max`.
    """
    oc = center[None, :] - ro
    t_proj = dot(oc, rd)
    oc_len2 = dot(oc, oc)
    d2 = oc_len2 - t_proj * t_proj
    r2 = radius * radius
    return (d2 <= r2) & (t_proj + radius > 1e-4) & (t_proj - radius < t_max)


# ==============================================================================
# 5. ANALYTIC 3D RAY-PRIMITIVE INTERSECTION KERNELS
# ==============================================================================


def intersect_aabb(
    ro: np.ndarray,
    rd: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact slab intersection of ray (ro + t * rd) against axis-aligned box [bmin, bmax].
    """
    inv_d = 1.0 / np.where(np.abs(rd) < 1e-12, np.copysign(1e-12, rd), rd)
    t0 = (bmin - ro) * inv_d
    t1 = (bmax - ro) * inv_d
    tmin_v = np.minimum(t0, t1)
    tmax_v = np.maximum(t0, t1)

    t_near = np.max(tmin_v, axis=-1)
    t_far = np.min(tmax_v, axis=-1)

    is_x = np.abs(t_near - tmin_v[..., 0]) < 1e-7
    is_y = (~is_x) & (np.abs(t_near - tmin_v[..., 1]) < 1e-7)
    is_z = (~is_x) & (~is_y)

    sgn = -np.sign(rd)
    nx = np.where(is_x, sgn[..., 0], 0.0)
    ny = np.where(is_y, sgn[..., 1], 0.0)
    nz = np.where(is_z, sgn[..., 2], 0.0)
    normal = np.stack([nx, ny, nz], axis=-1)
    return t_near, t_far, normal


def intersect_rounded_box(
    ro: np.ndarray,
    rd: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    bevel_r: float = 0.008,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intersect ray with an axis-aligned box and compute smooth beveled edge normals.
    """
    bmin = center - half_extents
    bmax = center + half_extents
    t_near, t_far, flat_n = intersect_aabb(ro, rd, bmin, bmax)
    hit = (t_far >= np.maximum(t_near, 1e-4)) & (t_near > 1e-4)

    p_local = (ro + t_near[..., None] * rd) - center
    inner = np.maximum(half_extents - bevel_r, 1e-4)
    q = np.abs(p_local) - inner
    outside = np.maximum(q, 0.0) * np.sign(p_local)
    outside_norm = np.sqrt(np.maximum(np.sum(outside * outside, axis=-1, keepdims=True), 1e-16))
    smooth_n = np.where(outside_norm > 1e-5, outside / outside_norm, flat_n)
    normal = normalize(flat_n + 0.85 * smooth_n)
    return t_near, hit, normal


def intersect_obb_y(
    ro: np.ndarray,
    rd: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    yaw_rad: float,
    bevel_r: float = 0.012,
    tilt_x_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Intersect ray with an oriented 3D bounding box rotated by `yaw_rad` around Y
    and optionally `tilt_x_rad` around local X.
    """
    rel = ro - center
    cy, sy = math.cos(-yaw_rad), math.sin(-yaw_rad)

    rx = cy * rel[..., 0] + sy * rel[..., 2]
    ry = rel[..., 1]
    rz = -sy * rel[..., 0] + cy * rel[..., 2]

    dx = cy * rd[..., 0] + sy * rd[..., 2]
    dy = rd[..., 1]
    dz = -sy * rd[..., 0] + cy * rd[..., 2]

    if abs(tilt_x_rad) > 1e-6:
        cx, sx = math.cos(-tilt_x_rad), math.sin(-tilt_x_rad)
        ry2 = cx * ry - sx * rz
        rz2 = sx * ry + cx * rz
        dy2 = cx * dy - sx * dz
        dz2 = sx * dy + cx * dz
        ry, rz, dy, dz = ry2, rz2, dy2, dz2

    ro_loc = np.stack([rx, ry, rz], axis=-1)
    rd_loc = np.stack([dx, dy, dz], axis=-1)

    t_near, hit, n_loc = intersect_rounded_box(
        ro_loc, rd_loc, np.zeros(3, dtype=np.float64), half_extents, bevel_r=bevel_r
    )
    p_loc = ro_loc + t_near[..., None] * rd_loc

    nx, ny, nz = n_loc[..., 0], n_loc[..., 1], n_loc[..., 2]
    if abs(tilt_x_rad) > 1e-6:
        cx, sx = math.cos(tilt_x_rad), math.sin(tilt_x_rad)
        ny2 = cx * ny - sx * nz
        nz2 = sx * ny + cx * nz
        ny, nz = ny2, nz2

    cy_w, sy_w = math.cos(yaw_rad), math.sin(yaw_rad)
    wx = cy_w * nx + sy_w * nz
    wy = ny
    wz = -sy_w * nx + cy_w * nz
    n_world = normalize(np.stack([wx, wy, wz], axis=-1))
    return t_near, hit, n_world, p_loc


def intersect_cylinder_segment(
    ro: np.ndarray,
    rd: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    radius: float,
    capped: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact ray intersection with a finite 3D cylinder from `p_a` to `p_b` of radius `radius`.
    """
    ba = p_b - p_a
    ba_len2 = float(np.dot(ba, ba))
    ba_len = math.sqrt(max(ba_len2, 1e-12))
    axis = ba / ba_len

    oc = ro - p_a
    rd_dot_a = dot(rd, axis)
    oc_dot_a = dot(oc, axis)

    d_perp = rd - rd_dot_a[..., None] * axis
    o_perp = oc - oc_dot_a[..., None] * axis

    a_coef = dot(d_perp, d_perp)
    b_coef = dot(o_perp, d_perp)
    c_coef = dot(o_perp, o_perp) - radius * radius

    disc = b_coef * b_coef - a_coef * c_coef
    valid_disc = (disc >= 0.0) & (a_coef > 1e-10)
    sqrt_disc = np.sqrt(np.maximum(0.0, disc))

    t_side = (-b_coef - sqrt_disc) / np.maximum(a_coef, 1e-10)
    y_side = oc_dot_a + t_side * rd_dot_a
    hit_side = valid_disc & (t_side > 1e-4) & (y_side >= 0.0) & (y_side <= ba_len)

    p_hit = oc + t_side[..., None] * rd
    n_side = normalize(p_hit - y_side[..., None] * axis)

    if not capped:
        return t_side, hit_side, n_side

    inv_rd_a = 1.0 / np.where(np.abs(rd_dot_a) < 1e-10, np.copysign(1e-10, rd_dot_a), rd_dot_a)
    t_cap0 = -oc_dot_a * inv_rd_a
    p_cap0 = o_perp + t_cap0[..., None] * d_perp
    hit_cap0 = (t_cap0 > 1e-4) & (dot(p_cap0, p_cap0) <= radius * radius)

    t_cap1 = (ba_len - oc_dot_a) * inv_rd_a
    p_cap1 = o_perp + t_cap1[..., None] * d_perp
    hit_cap1 = (t_cap1 > 1e-4) & (dot(p_cap1, p_cap1) <= radius * radius)

    t_best = np.where(hit_side, t_side, 1e9)
    n_best = n_side.copy()

    use_c0 = hit_cap0 & (t_cap0 < t_best)
    t_best = np.where(use_c0, t_cap0, t_best)
    n_best = np.where(use_c0[..., None], -axis[None, :], n_best)

    use_c1 = hit_cap1 & (t_cap1 < t_best)
    t_best = np.where(use_c1, t_cap1, t_best)
    n_best = np.where(use_c1[..., None], axis[None, :], n_best)

    hit_any = hit_side | hit_cap0 | hit_cap1
    return t_best, hit_any, n_best


def intersect_ellipsoid(
    ro: np.ndarray,
    rd: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    yaw_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact ray intersection with an oriented 3D ellipsoid."""
    rel = ro - center
    cy, sy = math.cos(-yaw_rad), math.sin(-yaw_rad)
    rx = cy * rel[..., 0] + sy * rel[..., 2]
    ry = rel[..., 1]
    rz = -sy * rel[..., 0] + cy * rel[..., 2]

    dx = cy * rd[..., 0] + sy * rd[..., 2]
    dy = rd[..., 1]
    dz = -sy * rd[..., 0] + cy * rd[..., 2]

    inv_r = 1.0 / radii
    ox, oy, oz = rx * inv_r[0], ry * inv_r[1], rz * inv_r[2]
    vx, vy, vz = dx * inv_r[0], dy * inv_r[1], dz * inv_r[2]

    a = vx * vx + vy * vy + vz * vz
    b = ox * vx + oy * vy + oz * vz
    c = ox * ox + oy * oy + oz * oz - 1.0

    disc = b * b - a * c
    hit = disc >= 0.0
    sqrt_d = np.sqrt(np.maximum(0.0, disc))
    t = (-b - sqrt_d) / np.maximum(a, 1e-10)
    hit = hit & (t > 1e-4)

    px = rx + t * dx
    py = ry + t * dy
    pz = rz + t * dz
    nx_l = px * (inv_r[0] ** 2)
    ny_l = py * (inv_r[1] ** 2)
    nz_l = pz * (inv_r[2] ** 2)

    cy_w, sy_w = math.cos(yaw_rad), math.sin(yaw_rad)
    wx = cy_w * nx_l + sy_w * nz_l
    wy = ny_l
    wz = -sy_w * nx_l + cy_w * nz_l
    normal = normalize(np.stack([wx, wy, wz], axis=-1))
    return t, hit, normal


# ==============================================================================
# 6. LEFT WALL ARCHITECTURAL WALNUT LIBRARY, VERTICAL & LEANING RUNS,
#    HORIZONTAL FOLIO STACKS, TRAVERTINE BOOKENDS & RIBBED STONEWARE VASES
# ==============================================================================


class WalnutLibraryCatalog:
    """
    Deterministic procedural generator and vectorized 3D ray-intersector for the
    floor-to-ceiling American walnut library wall at X in [-3.10, -2.72].
    Includes:
      - 340+ vertical hardcover books with per-volume cloth/leather grain, raised
        horizontal bindery ribs/hubs, gold-foil/blind-debossed title blocks, and
        library classification labels.
      - Tilted/leaning books at the ends of vertical book runs.
      - Honed Roman travertine & brushed-brass architectural bookends.
      - 7 horizontal stacks of oversized art monographs lying flat on the shelves.
      - Wheel-thrown ribbed stoneware ceramic vessels with necks, lips, and
        iron-speckled celadon / shino ash / terracotta glazes.
    """

    def __init__(self, seed: int = 20301) -> None:
        rng = np.random.default_rng(seed + 77)
        self.wall_back_x: float = ROOM_X_LEFT
        self.case_front_x: float = -2.72
        self.shelf_thick: float = 0.042
        self.stile_thick: float = 0.052

        self.shelf_y = np.array([0.08, 0.66, 1.22, 1.78, 2.34, 2.90, 3.46, 4.02], dtype=np.float64)
        self.stile_z = np.array([-0.80, 0.90, 2.55, 4.25, 5.92, 7.58], dtype=np.float64)

        palette = np.array(
            [
                [0.44, 0.37, 0.29],  # Raw Belgian linen
                [0.46, 0.40, 0.32],  # Warm sand vellum
                [0.18, 0.11, 0.08],  # Deep espresso calfskin
                [0.46, 0.16, 0.10],  # Venetian burnt sienna cloth
                [0.09, 0.14, 0.22],  # Prussian indigo buckram
                [0.24, 0.28, 0.21],  # Muted sage olive linen
                [0.50, 0.31, 0.14],  # Warm ochre mustard cloth
                [0.12, 0.12, 0.13],  # Charcoal graphite monograph
                [0.48, 0.20, 0.14],  # Terracotta clay binding
                [0.32, 0.26, 0.22],  # Umber taupe cloth
                [0.49, 0.43, 0.35],  # Natural parchment folio
                [0.19, 0.23, 0.28],  # Slate blue-grey cloth
                [0.34, 0.12, 0.13],  # Deep oxblood morocco leather
                [0.14, 0.21, 0.18],  # British racing green buckram
            ],
            dtype=np.float64,
        )

        self.z_min: float = -0.80
        self.z_max: float = 7.58
        self.n_bins: int = 4190
        self.n_tiers: int = len(self.shelf_y) - 1

        self.book_present = np.zeros((self.n_tiers, self.n_bins), dtype=bool)
        self.book_height = np.zeros((self.n_tiers, self.n_bins), dtype=np.float64)
        self.book_front_x = np.full((self.n_tiers, self.n_bins), -2.78, dtype=np.float64)
        self.book_z0 = np.zeros((self.n_tiers, self.n_bins), dtype=np.float64)
        self.book_z1 = np.zeros((self.n_tiers, self.n_bins), dtype=np.float64)
        self.book_u_spine = np.zeros((self.n_tiers, self.n_bins), dtype=np.float64)
        self.book_color = np.zeros((self.n_tiers, self.n_bins, 3), dtype=np.float64)
        self.book_band_y0 = np.zeros((self.n_tiers, self.n_bins), dtype=np.float64)
        self.book_band_y1 = np.zeros((self.n_tiers, self.n_bins), dtype=np.float64)
        self.book_band_col = np.zeros((self.n_tiers, self.n_bins, 3), dtype=np.float64)
        self.book_rough = np.full((self.n_tiers, self.n_bins), 0.55, dtype=np.float64)
        self.book_has_hubs = np.zeros((self.n_tiers, self.n_bins), dtype=bool)
        self.book_has_label = np.zeros((self.n_tiers, self.n_bins), dtype=bool)
        self.book_lean_tilt = np.zeros((self.n_tiers, self.n_bins), dtype=np.float64)

        self.vessels: list[dict[str, float | int | np.ndarray]] = []
        self.horizontal_stacks: list[dict[str, float | np.ndarray | list[dict[str, float | np.ndarray]]]] = []
        self.leaning_books: list[dict[str, float | np.ndarray]] = []
        self.bookends: list[dict[str, float | np.ndarray]] = []

        for tier in range(self.n_tiers):
            tier_h = float(self.shelf_y[tier + 1] - self.shelf_y[tier] - self.shelf_thick)
            shelf_surface_y = float(self.shelf_y[tier] + self.shelf_thick)

            for bay in range(len(self.stile_z) - 1):
                z_start = float(self.stile_z[bay] + self.stile_thick * 0.58)
                z_end = float(self.stile_z[bay + 1] - self.stile_thick * 0.58)

                # Reserve curated bays for wheel-thrown stoneware vessels & horizontal folio stacks
                if bay == 2 and tier == 3:
                    # Hero ribbed celadon stoneware vase near the brass reading sconce
                    self.vessels.append(
                        {
                            "center": np.array([-2.865, shelf_surface_y, 3.04], dtype=np.float64),
                            "height": 0.31,
                            "radius": 0.096,
                            "neck_r": 0.042,
                            "lip_r": 0.052,
                            "color": np.array([0.46, 0.53, 0.47], dtype=np.float64),
                            "rough": 0.24,
                            "glaze": 0,
                        }
                    )
                    # Horizontal stack of 4 oversized art monographs next to the hero vase
                    stack_z = 3.44
                    stack_w = 0.36
                    layers = []
                    y_acc = shelf_surface_y
                    for k_lay in range(4):
                        lh = float(rng.uniform(0.032, 0.046))
                        lw = stack_w - k_lay * 0.014
                        lcol = palette[int(rng.integers(0, len(palette)))].copy()
                        layers.append(
                            {
                                "y0": y_acc,
                                "y1": y_acc + lh,
                                "z0": stack_z - 0.5 * lw,
                                "z1": stack_z + 0.5 * lw,
                                "front_x": -2.755 + k_lay * 0.005,
                                "color": lcol,
                                "rough": float(rng.uniform(0.38, 0.58)),
                            }
                        )
                        y_acc += lh
                    self.horizontal_stacks.append({"tier": tier, "bay": bay, "layers": layers})
                    z_start = 3.68

                elif (bay == 1 and tier == 2) or (bay == 3 and tier == 4) or (bay == 1 and tier == 0):
                    vessel_z = z_end - 0.28
                    glaze_idx = (bay + tier) % 3
                    v_cols = [
                        np.array([0.48, 0.54, 0.46], dtype=np.float64),  # Celadon green-grey
                        np.array([0.64, 0.58, 0.49], dtype=np.float64),  # Warm shino ash
                        np.array([0.52, 0.33, 0.24], dtype=np.float64),  # Smoked terracotta
                    ]
                    vh = float(rng.uniform(0.23, 0.32))
                    vr = float(rng.uniform(0.078, 0.102))
                    self.vessels.append(
                        {
                            "center": np.array([-2.86, shelf_surface_y, vessel_z], dtype=np.float64),
                            "height": vh,
                            "radius": vr,
                            "neck_r": vr * 0.44,
                            "lip_r": vr * 0.54,
                            "color": v_cols[glaze_idx],
                            "rough": float(rng.uniform(0.22, 0.38)),
                            "glaze": glaze_idx,
                        }
                    )
                    # Add a travertine bookend holding the vertical book run before the vessel
                    be_z = z_end - 0.58
                    self.bookends.append(
                        {
                            "center": np.array([-2.83, shelf_surface_y + 0.085, be_z], dtype=np.float64),
                            "ext": np.array([0.065, 0.085, 0.032], dtype=np.float64),
                        }
                    )
                    z_end = be_z - 0.038

                elif (bay == 1 and tier == 1) or (bay == 2 and tier == 2) or (bay == 1 and tier == 4) or (bay == 3 and tier == 2):
                    # Horizontal stack of 3-4 oversized architectural monographs at the left or right of the bay
                    at_left = (tier + bay) % 2 == 0
                    stack_w = 0.34
                    if at_left:
                        stack_z = z_start + 0.5 * stack_w + 0.02
                        z_start = stack_z + 0.5 * stack_w + 0.035
                    else:
                        stack_z = z_end - 0.5 * stack_w - 0.02
                        z_end = stack_z - 0.5 * stack_w - 0.035

                    n_lay = 4 if tier % 2 == 0 else 3
                    layers = []
                    y_acc = shelf_surface_y
                    for k_lay in range(n_lay):
                        lh = float(rng.uniform(0.030, 0.045))
                        lw = stack_w - k_lay * 0.016
                        lcol = palette[int(rng.integers(0, len(palette)))].copy()
                        layers.append(
                            {
                                "y0": y_acc,
                                "y1": y_acc + lh,
                                "z0": stack_z - 0.5 * lw,
                                "z1": stack_z + 0.5 * lw,
                                "front_x": -2.758 + k_lay * 0.006,
                                "color": lcol,
                                "rough": float(rng.uniform(0.36, 0.58)),
                            }
                        )
                        y_acc += lh
                    self.horizontal_stacks.append({"tier": tier, "bay": bay, "layers": layers})

                z_cur = z_start
                cluster_books_left = 0
                cluster_h_base = 0.76 * tier_h
                cluster_fx_base = -2.775
                cluster_col_base = palette[0].copy()
                while z_cur < z_end - 0.032:
                    # Occasional breathing gap with a leaning book + travertine bookend
                    if rng.random() < 0.092 and (z_end - z_cur) > 0.32 and (z_cur - z_start) > 0.20:
                        lean_h = float(rng.uniform(0.64 * tier_h, 0.88 * tier_h))
                        lean_thick = float(rng.uniform(0.030, 0.044))
                        lean_col = palette[int(rng.integers(0, len(palette)))].copy()
                        lean_fx = float(rng.uniform(-2.845, -2.815))
                        self.leaning_books.append(
                            {
                                "center": np.array(
                                    [lean_fx, shelf_surface_y + 0.48 * lean_h, z_cur + 0.055], dtype=np.float64
                                ),
                                "ext": np.array([0.078, 0.48 * lean_h, 0.5 * lean_thick], dtype=np.float64),
                                "tilt_x": float(rng.uniform(-0.26, -0.17)),  # Leans back against the vertical run
                                "color": lean_col,
                                "rough": float(rng.uniform(0.38, 0.56)),
                            }
                        )
                        z_cur += float(rng.uniform(0.14, 0.21))
                        cluster_books_left = 0
                        continue

                    # Occasional small natural gap (1.6cm - 3.6cm) between book clusters revealing 3D side boards
                    if rng.random() < 0.075 and (z_cur - z_start) > 0.12 and (z_end - z_cur) > 0.18:
                        z_cur += float(rng.uniform(0.016, 0.036))
                        cluster_books_left = 0
                        continue

                    # Architectural library rhythm: mix slim octavos, standard quartos, tall art folios, and 2-3 vol sets
                    if cluster_books_left > 0:
                        cluster_books_left -= 1
                        thickness = float(rng.uniform(0.026, 0.044))
                        b_height = float(np.clip(cluster_h_base + rng.uniform(-0.008, 0.008), 0.52 * tier_h, 0.94 * tier_h))
                        front_x = float(np.clip(cluster_fx_base + rng.uniform(-0.006, 0.006), -2.818, -2.752))
                        col = np.clip(cluster_col_base * rng.uniform(0.92, 1.08, size=3), 0.05, 0.56)
                    else:
                        fmt_roll = rng.random()
                        if fmt_roll < 0.24:
                            # Tall, deep-spined architectural folio / monograph
                            thickness = float(rng.uniform(0.036, 0.064))
                            b_height = float(rng.uniform(0.83 * tier_h, 0.94 * tier_h))
                            front_x = float(rng.uniform(-2.774, -2.752))
                        elif fmt_roll < 0.54:
                            # Compact recessed octavo / literary edition
                            thickness = float(rng.uniform(0.018, 0.034))
                            b_height = float(rng.uniform(0.54 * tier_h, 0.69 * tier_h))
                            front_x = float(rng.uniform(-2.818, -2.782))
                        else:
                            # Standard quarto / reference volume with varied pull-out depth
                            thickness = float(rng.uniform(0.024, 0.050))
                            b_height = float(rng.uniform(0.68 * tier_h, 0.84 * tier_h))
                            front_x = float(rng.uniform(-2.796, -2.762))

                        col = palette[int(rng.integers(0, len(palette)))].copy()
                        col *= rng.uniform(0.82, 1.16, size=3)
                        col = np.clip(col, 0.05, 0.56)

                        if rng.random() < 0.22:
                            cluster_books_left = int(rng.integers(1, 3))
                            cluster_h_base = b_height
                            cluster_fx_base = front_x
                            cluster_col_base = col.copy()

                    if z_cur + thickness > z_end:
                        thickness = z_end - z_cur
                    if thickness < 0.017:
                        break

                    dz_bin = (self.z_max - self.z_min) / self.n_bins
                    b0 = int(
                        np.clip((z_cur - self.z_min) / dz_bin, 0, self.n_bins - 1)
                    )
                    b1 = int(
                        np.clip(
                            (z_cur + thickness - self.z_min) / dz_bin,
                            0,
                            self.n_bins,
                        )
                    )
                    if b1 > b0:
                        nb = b1 - b0
                        z_bin_0 = self.z_min + b0 * dz_bin
                        z_bin_1 = self.z_min + b1 * dz_bin
                        u_sp = np.linspace(-1.0, 1.0, nb, dtype=np.float64)
                        self.book_present[tier, b0:b1] = True
                        self.book_height[tier, b0:b1] = b_height
                        self.book_front_x[tier, b0:b1] = front_x
                        self.book_z0[tier, b0:b1] = z_bin_0
                        self.book_z1[tier, b0:b1] = z_bin_1
                        self.book_u_spine[tier, b0:b1] = u_sp
                        self.book_color[tier, b0:b1, :] = col[None, :]
                        self.book_rough[tier, b0:b1] = float(rng.uniform(0.32, 0.74))
                        self.book_has_hubs[tier, b0:b1] = bool(rng.random() < 0.38)
                        self.book_has_label[tier, b0:b1] = bool(rng.random() < 0.22)

                        by0 = float(rng.uniform(0.55, 0.82)) * b_height
                        by1 = by0 + float(rng.uniform(0.016, 0.048))
                        bcol = (
                            np.array([0.52, 0.39, 0.19], dtype=np.float64)
                            if rng.random() < 0.65
                            else np.array([0.54, 0.50, 0.43], dtype=np.float64)
                        )
                        self.book_band_y0[tier, b0:b1] = by0
                        self.book_band_y1[tier, b0:b1] = by1
                        self.book_band_col[tier, b0:b1, :] = bcol[None, :]

                    z_cur = self.z_min + b1 * dz_bin

    def query_book_bins(self, tier_idx: np.ndarray, z_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map (tier_idx, z_coord) arrays into book table indices."""
        t_clamped = np.clip(tier_idx, 0, self.n_tiers - 1)
        b_idx = np.clip(
            ((z_coord - self.z_min) / (self.z_max - self.z_min) * self.n_bins).astype(np.int64),
            0,
            self.n_bins - 1,
        )
        return t_clamped, b_idx


# ==============================================================================
# 7. VOLUMETRIC CUMULUS & CIRRUS CLOUD SKY, COASTAL HEADLANDS & COASTAL WATER
# ==============================================================================


def evaluate_realistic_cloud_sky(d: np.ndarray) -> np.ndarray:
    """
    Evaluate physically-based multi-layered coastal sky & volumetric cloud radiance
    for ray direction array `d` (N, 3):
      1. Rayleigh-Mie clear-sky gradient: rich cerulean-cobalt zenith -> azure mid-sky ->
         warm golden-apricot horizon haze + solar aureole.
      2. Upper-tropospheric wind-sheared cirrus & cirrostratus ice-crystal filaments.
      3. Towering 3D-lit coastal cumulus & stratocumulus cloud banks with domain-warped
         cauliflower billow lobes, 3-step directional solar optical depth (`tau_sun`),
         Beer-Lambert self-shadowed slate-indigo bases, and Mie multiple-scattering
         silver-gold powder rims (`1 - exp(-3.8 * tau)`).
    """
    elev = np.clip(d[:, 1], 0.0, 1.0)
    sun_cos = np.clip(dot(d, SUN_DIR), -1.0, 1.0)
    sun_pos = np.maximum(0.0, sun_cos)

    # 1. Calibrated Rayleigh + Mie Clear-Sky Gradient (preserves rich cerulean-azure blue under ACES)
    t_low = smoothstep(0.0, 0.085, elev)
    t_high = smoothstep(0.08, 0.42, elev)
    sky_clear = (
        SKY_HORIZON_RAD[None, :] * (1.0 - t_low[:, None])
        + SKY_MID_RAD[None, :] * (t_low * (1.0 - t_high))[:, None]
        + SKY_ZENITH_RAD[None, :] * t_high[:, None]
    )

    # Tight golden-apricot sea-horizon band & focused solar Mie aureole (preserves deep blue mid/upper sky)
    horizon_band = np.exp(-elev * 22.0) * (0.30 + 0.50 * (sun_pos ** 3))
    sky_clear += horizon_band[:, None] * np.array([0.58, 0.36, 0.16], dtype=np.float64)[None, :]

    mie_aureole = (sun_pos ** 42) * 0.75 + (sun_pos ** 120) * 4.5
    sun_disc = smoothstep(0.9992, 0.99985, sun_cos) * 36.0
    sky_clear += (mie_aureole + sun_disc)[:, None] * np.array([1.25, 0.86, 0.44], dtype=np.float64)[None, :]

    # 2. 3D Planar Atmospheric Perspective Projection onto Horizontal Cloud Ceiling
    inv_y = 1.0 / np.maximum(elev + 0.065, 0.065)
    u_sky = d[:, 0] * inv_y
    v_sky = d[:, 2] * inv_y

    def eval_cloud_deck_density(u_p: np.ndarray, v_p: np.ndarray, el_p: np.ndarray) -> np.ndarray:
        """
        Evaluate a 3-deck coastal cloud density field on the 3D perspective-projected
        cloud ceiling (u_p, v_p, el_p) using 2-stage domain-warped FBM, billowy
        cauliflower turbulence, and fractal wisp erosion (zero hand-placed ellipses):
          - Deck A: Sweeping Coastal Stratocumulus / Cumulus Humilis (el_p in [0.025, 0.28])
          - Deck B: Mid-Level Broken Altocumulus Floccus (el_p in [0.13, 0.46])
          - Deck C: High-Altitude Windswept Cirrus Uncinus Mares' Tails (el_p > 0.16)
        """
        # Two-stage domain warping so cloud banks are organically sheared & shredded by coastal wind
        w1_u = fbm2d(u_p * 0.55 + 1.7, v_p * 0.42 + 3.1, octaves=4, seed=701) - 0.5
        w1_v = fbm2d(u_p * 0.42 - 4.2, v_p * 0.55 + 8.4, octaves=4, seed=709) - 0.5
        uu = u_p * 0.72 + 1.15 * w1_u
        vv = v_p * 0.52 + 1.15 * w1_v

        w2_u = fbm2d(uu * 1.85 - 2.3, vv * 1.65 + 1.9, octaves=3, seed=713) - 0.5
        w2_v = fbm2d(uu * 1.65 + 5.1, vv * 1.85 - 3.7, octaves=3, seed=719) - 0.5
        uu2 = uu + 0.42 * w2_u
        vv2 = vv + 0.42 * w2_v

        # Shared multi-scale cauliflower billow & fractal wisp erosion
        billow = 1.0 - np.abs(2.0 * fbm2d(uu2 * 2.1, vv2 * 1.9, octaves=4, seed=733) - 1.0)
        billow_sub = billow_fbm2d(uu2 * 3.6 + 1.4, vv2 * 3.2 - 0.9, octaves=3, seed=739)
        fine = fbm2d(uu2 * 6.4, vv2 * 6.0, octaves=3, seed=747)
        micro = value_noise_2d(uu2 * 16.0, vv2 * 16.0, seed=759)

        # Deck A: Sweeping Coastal Stratocumulus / Cumulus Humilis Deck (elev in [0.025, 0.28])
        macro_band = fbm2d(uu * 0.65 + 0.4, vv * 0.48 - 0.7, octaves=4, seed=721)
        street_wave = 0.08 * np.sin(u_p * 1.25 - v_p * 0.52 + 2.2 * w1_u)
        macro_top = np.clip(0.56 * macro_band + 0.44 * billow + street_wave, 0.0, 1.0)

        base_cut = smoothstep(0.028, 0.056, el_p)
        top_cut = 1.0 - smoothstep(0.10 + 0.14 * macro_top, 0.165 + 0.16 * macro_top, el_p)

        raw_deck_a = (
            0.46 * macro_band
            + 0.24 * billow
            + 0.14 * billow_sub
            + 0.11 * fine
            + 0.05 * micro
            + street_wave
        )
        deck_a = smoothstep(0.50, 0.73, raw_deck_a) * base_cut * top_cut

        # Deck B: Mid-Level Broken Altocumulus Floccus (elev in [0.13, 0.46])
        ac_macro = fbm2d(uu * 1.45 + 5.0, vv * 1.25 - 3.0, octaves=4, seed=773)
        ac_env = smoothstep(0.13, 0.19, el_p) * (1.0 - smoothstep(0.31, 0.46, el_p))
        raw_deck_b = 0.50 * ac_macro + 0.32 * billow_sub + 0.13 * fine + 0.05 * micro
        deck_b = 0.80 * smoothstep(0.53, 0.74, raw_deck_b) * ac_env

        # Deck C: High-Altitude Windswept Cirrus Uncinus Mares' Tails (elev > 0.16)
        ci_streaks = fbm2d(uu * 0.95 + vv * 0.60, vv * 4.80 - uu * 1.40, octaves=4, seed=787)
        ci_fine = fbm2d(uu2 * 2.6 + vv2 * 1.4, vv2 * 10.5 - uu2 * 2.8, octaves=3, seed=793)
        ci_env = smoothstep(0.16, 0.24, el_p) * (1.0 - smoothstep(0.40, 0.58, el_p))
        raw_deck_c = 0.66 * ci_streaks + 0.24 * ci_fine + 0.10 * fine
        deck_c = 0.52 * smoothstep(0.56, 0.77, raw_deck_c) * ci_env

        return np.clip(
            deck_a
            + deck_b * (1.0 - 0.65 * deck_a)
            + deck_c * (1.0 - 0.75 * np.maximum(deck_a, deck_b)),
            0.0,
            1.0,
        )

    dens_0 = eval_cloud_deck_density(u_sky, v_sky, elev)
    has_cloud = dens_0 > 1e-3

    out_sky = sky_clear.copy()
    if np.any(has_cloud):
        el_c = elev[has_cloud]
        u_c = u_sky[has_cloud]
        v_c = v_sky[has_cloud]
        dens_here = dens_0[has_cloud]
        sun_c = sun_pos[has_cloud]
        bg_c = sky_clear[has_cloud]

        # Directional Solar Finite-Difference Self-Shadowing (Beer-Lambert-Powder)
        # Step along the 3D planar projection of SUN_DIR = [0.755, 0.295, 0.585]
        sun_du = 0.22
        sun_dv = 0.17
        dens_sun1 = eval_cloud_deck_density(u_c + sun_du, v_c + sun_dv, el_c + 0.018)
        dens_sun2 = eval_cloud_deck_density(u_c + 2.4 * sun_du, v_c + 2.4 * sun_dv, el_c + 0.038)

        tau_sun = (
            1.8 * dens_sun1
            + 2.4 * dens_sun2
            + 0.7 * np.maximum(0.0, 0.14 - el_c) * 6.0 * dens_here
        )
        trans_sun = np.exp(-tau_sun)
        powder = 1.0 - np.exp(-tau_sun * 2.2)
        silver_lining = (
            np.exp(-tau_sun * 0.75)
            * (1.0 - np.exp(-dens_here * 4.5))
            * (0.38 + 0.62 * (sun_c ** 3))
        )

        direct_weight = np.clip(
            trans_sun * (0.54 + 0.58 * powder) + 0.46 * silver_lining,
            0.0,
            1.38,
        )

        # Height fraction for ambient sky-dome top bounce vs shadowed marine cloud base
        h_norm = np.clip((el_c - 0.032) / 0.18, 0.0, 1.0)

        # Photographic coastal cloud palette:
        #   - Sunlit cloud tops: warm cream-gold [2.15, 1.92, 1.68]
        #   - Silver-amber Mie forward-scattered translucent rim: [2.62, 2.20, 1.72]
        #   - Shadowed cloud undersides: cool slate-blue/lavender [0.44, 0.54, 0.70]
        #   - Sky-dome ambient top bounce: [0.52, 0.68, 0.94]
        col_sunlit = np.array([2.15, 1.92, 1.68], dtype=np.float64)
        col_rim = np.array([2.62, 2.20, 1.72], dtype=np.float64)
        col_base_shadow = np.array([0.32, 0.44, 0.64], dtype=np.float64)
        col_sky_ambient = np.array([0.46, 0.64, 0.90], dtype=np.float64)

        ambient_cloud = (
            col_base_shadow[None, :] * (1.0 - 0.55 * h_norm[:, None])
            + col_sky_ambient[None, :] * (0.55 * h_norm[:, None])
        )
        cloud_rad = (
            ambient_cloud * (1.0 - 0.62 * direct_weight[:, None])
            + col_sunlit[None, :] * direct_weight[:, None]
            + col_rim[None, :] * (silver_lining * 0.46)[:, None]
        )

        # Soft feathered vaporous alpha transitions into the cerulean sky
        atm_fade = smoothstep(0.022, 0.052, el_c)
        cloud_alpha = np.clip(
            smoothstep(0.02, 0.64, dens_here) * (0.58 + 0.42 * atm_fade) * 0.94,
            0.0,
            0.95,
        )
        out_sky[has_cloud] = bg_c * (1.0 - cloud_alpha[:, None]) + cloud_rad * cloud_alpha[:, None]

    return out_sky


def evaluate_exterior_courtyard(ro: np.ndarray, rd: np.ndarray) -> np.ndarray:
    """
    Evaluate linear HDR radiance for rays exiting through the floor-to-ceiling
    windows (X > 3.20 or Z > 7.60 in the corner glass return).
    Includes the volumetric cumulus/cirrus sky, distant layered coastal headlands,
    sunlit Mediterranean limestone promontory with craggy strata & maquis scrub,
    bush-hammered concrete exterior terrace coping border, and wind-rippled water
    reflections.
    """
    shape = rd.shape[:-1]
    rd_flat = rd.reshape(-1, 3)
    ro_flat = np.broadcast_to(ro, rd.shape).reshape(-1, 3)

    water_y = -0.04
    t_water = (water_y - ro_flat[:, 1]) / np.where(np.abs(rd_flat[:, 1]) < 1e-8, -1e-8, rd_flat[:, 1])
    hits_water = (rd_flat[:, 1] < -1e-4) & (t_water > 0.0)

    def sample_courtyard_backdrop(d_vec: np.ndarray) -> np.ndarray:
        rad = evaluate_realistic_cloud_sky(d_vec)
        phi = np.arctan2(d_vec[:, 0], np.maximum(d_vec[:, 2], 1e-5))
        elev = d_vec[:, 1]

        # 1. Distant Layered Coastal Mountain Headlands across the Sea Horizon
        ridge_macro = fbm2d(phi * 14.0 + 0.7, elev * 28.0, octaves=4, seed=301)
        ridge_crag = fbm2d(phi * 48.0 - 1.3, elev * 95.0, octaves=3, seed=307)
        ridge_fine = value_noise_2d(phi * 135.0, elev * 220.0, seed=311)

        far_headland_h = (
            0.030
            + 0.018 * np.sin(phi * 3.1 + 0.45)
            + 0.010 * np.cos(phi * 7.8 - 0.6)
            + 0.011 * (ridge_macro - 0.5)
            + 0.0045 * (ridge_crag - 0.5)
        )
        is_far_headland = (elev < far_headland_h) & (elev >= -0.002)
        # Sunlit vs shadowed mountain gullies on distant range
        far_gully_sun = smoothstep(0.38, 0.65, ridge_crag + 0.35 * np.cos(phi * 18.0))
        far_shadow_col = np.array([0.22, 0.32, 0.48], dtype=np.float64)
        far_sun_col = np.array([0.42, 0.44, 0.52], dtype=np.float64)
        far_col = (
            far_shadow_col[None, :] * (1.0 - 0.48 * far_gully_sun[:, None])
            + far_sun_col[None, :] * (0.48 * far_gully_sun[:, None])
        ) * (0.90 + 0.16 * ridge_macro[:, None])
        rad = np.where(is_far_headland[:, None], far_col, rad)

        # 2. Closer Craggy Limestone Coastal Promontory & Mediterranean Cypress/Olive Maquis
        near_env = (
            0.025 * np.exp(-(((phi - 0.23) / 0.14) ** 2))
            + 0.027 * smoothstep(0.44, 0.68, phi) * (0.72 + 0.28 * np.sin(phi * 5.2 - 0.3))
        )
        near_headland_h = (
            near_env * (0.78 + 0.36 * ridge_macro + 0.16 * (ridge_crag - 0.5) + 0.06 * (ridge_fine - 0.5))
        ) * smoothstep(0.12, 0.16, phi)
        is_near_headland = (elev < near_headland_h) & (elev >= -0.002)

        cliff_strata = 0.5 + 0.5 * np.sin(elev * 420.0 + ridge_macro * 6.5 + ridge_crag * 2.5)
        maquis_scrub = smoothstep(0.44, 0.66, ridge_crag * 0.7 + ridge_fine * 0.3) * smoothstep(0.008, 0.019, elev)
        facet_sun = smoothstep(0.35, 0.68, ridge_macro + 0.45 * np.cos(phi * 24.0 - 0.4))

        near_shadow = np.array([0.15, 0.21, 0.24], dtype=np.float64)
        near_limestone = np.array([0.54, 0.46, 0.35], dtype=np.float64)
        near_olive = np.array([0.20, 0.27, 0.16], dtype=np.float64)
        near_col = (
            near_shadow[None, :] * (1.0 - 0.65 * facet_sun[:, None])
            + near_limestone[None, :] * (0.65 * facet_sun[:, None])
        ) * (0.84 + 0.24 * cliff_strata[:, None])
        near_col = near_col * (1.0 - 0.62 * maquis_scrub[:, None]) + near_olive[None, :] * (0.62 * maquis_scrub[:, None])
        return np.where(is_near_headland[:, None], near_col, rad)

    direct_backdrop = sample_courtyard_backdrop(rd_flat)

    if np.any(hits_water):
        idx_w = np.where(hits_water)[0]
        p_w = ro_flat[idx_w] + t_water[idx_w, None] * rd_flat[idx_w]
        d_w = rd_flat[idx_w]
        dist_w = t_water[idx_w]

        in_patio = (p_w[:, 0] < ROOM_X_RIGHT + 0.72) & (p_w[:, 2] < ROOM_Z_BACK + 0.72)

        wx, wz = p_w[:, 0], p_w[:, 2]
        atten_w = 1.0 / (1.0 + 0.045 * dist_w)
        atten_cap = 1.0 / (1.0 + 0.11 * dist_w)

        # Multi-train directional capillary-gravity wave spectrum + analytical noise gradients
        w_n1, dw1_dx, dw1_dz = value_noise_2d_grad(wx * 1.85 + wz * 0.65, wz * 2.95 - wx * 0.45, seed=411)
        w_n2, dw2_dx, dw2_dz = value_noise_2d_grad(wx * 5.20 - wz * 1.45, wz * 7.60 + wx * 1.05, seed=419)
        w_n3, dw3_dx, dw3_dz = value_noise_2d_grad(wx * 13.5 + wz * 3.8, wz * 17.5 - wx * 2.9, seed=431)

        # Wind-driven swell & capillary wave trains (non-commensurate directions & Gerstner-like harmonic sharpening)
        ph1 = wx * 4.4 + wz * 6.8 + 1.8 * (w_n1 - 0.5)
        ph2 = wx * 7.8 - wz * 5.4 + 1.5 * (w_n2 - 0.5)
        ph3 = wx * 14.5 + wz * 11.2 + 1.2 * (w_n2 - 0.5)
        ph4 = wx * 24.0 - wz * 16.5 + 1.0 * (w_n3 - 0.5)
        ph5 = wx * 38.0 + wz * 26.0

        s1 = np.cos(ph1) + 0.35 * np.cos(2.0 * ph1)
        s2 = np.cos(ph2) + 0.30 * np.cos(2.0 * ph2)
        s3 = np.cos(ph3)
        s4 = np.cos(ph4)
        s5 = np.cos(ph5)

        nx_swell = (0.018 * s1 * 4.4 / 8.1 + 0.014 * s2 * 7.8 / 9.5 + 0.016 * dw1_dx) * atten_w
        nz_swell = (0.018 * s1 * 6.8 / 8.1 - 0.014 * s2 * 5.4 / 9.5 + 0.018 * dw1_dz) * atten_w

        nx_cap = (
            0.015 * s3 * 0.79
            + 0.012 * s4 * 0.82
            + 0.008 * s5 * 0.82
            + 0.016 * dw2_dx
            + 0.011 * dw3_dx
        ) * atten_cap
        nz_cap = (
            0.015 * s3 * 0.61
            - 0.012 * s4 * 0.57
            + 0.008 * s5 * 0.56
            + 0.018 * dw2_dz
            + 0.012 * dw3_dz
        ) * atten_cap

        # Wind-streak slick modulation (calmer capillary patches alternating with ruffled wind lanes)
        wind_lane = 0.45 + 0.55 * smoothstep(
            0.30, 0.72, value_noise_2d(wx * 0.42 + wz * 0.18, wz * 1.15 - wx * 0.25, seed=443)
        )
        nx_w = nx_swell + nx_cap * wind_lane
        nz_w = nz_swell + nz_cap * wind_lane
        n_water = normalize(np.stack([nx_w, np.ones_like(nx_w), nz_w], axis=-1))

        refl_d = reflect(d_w, n_water)
        refl_d[:, 1] = np.abs(refl_d[:, 1]) + 0.004
        refl_d = normalize(refl_d)

        refl_backdrop = sample_courtyard_backdrop(refl_d)
        cos_theta = np.clip(-dot(d_w, n_water), 0.0, 1.0)
        f_water = dielectric_fresnel_exact(cos_theta, 1.0, 1.333)

        # Shore-to-deep Mediterranean turquoise-teal to sapphire-cobalt depth gradient & subsurface caustics
        off_x = np.maximum(0.0, wx - (ROOM_X_RIGHT + 0.72))
        off_z = np.maximum(0.0, wz - (ROOM_Z_BACK + 0.72))
        shore_dist = np.maximum(off_x, off_z)
        shallow_w = np.exp(-0.22 * shore_dist)
        mid_w = np.exp(-0.045 * shore_dist) * (1.0 - shallow_w)

        caustic_net = np.power(
            np.clip(0.5 + 0.28 * np.sin(ph1 * 1.7 + ph2) + 0.22 * np.cos(ph3 * 1.3 - ph2), 0.0, 1.0),
            3.5,
        )
        col_shallow_turquoise = np.array([0.048, 0.165, 0.185], dtype=np.float64)
        col_mid_cerulean = np.array([0.024, 0.088, 0.142], dtype=np.float64)
        col_deep_sapphire = np.array([0.012, 0.042, 0.088], dtype=np.float64)

        basin_col = (
            col_shallow_turquoise[None, :] * shallow_w[:, None]
            + col_mid_cerulean[None, :] * mid_w[:, None]
            + col_deep_sapphire[None, :] * np.maximum(0.0, 1.0 - shallow_w - mid_w)[:, None]
            + (caustic_net * shallow_w * atten_cap * 0.085)[:, None] * np.array([0.18, 0.42, 0.44], dtype=np.float64)[None, :]
        )
        # Wave-crest subsurface forward-scattered turquoise glow
        wave_crest = np.clip(0.5 * (np.sin(ph1) + np.sin(ph2)), 0.0, 1.0) * atten_w
        basin_col += (wave_crest * 0.028)[:, None] * np.array([0.06, 0.24, 0.28], dtype=np.float64)[None, :]

        water_rad = (1.0 - f_water[:, None]) * basin_col + f_water[:, None] * refl_backdrop

        # Dual-lobe solar wave-facet glints (crisp capillary sparkles + broader swell sheen)
        h_sun = normalize(SUN_DIR[None, :] - d_w)
        ndh_w = np.clip(dot(n_water, h_sun), 0.0, 1.0)
        glint_broad = ggx_isotropic_d(ndh_w, 0.075) * 0.045
        glint_crisp = ggx_isotropic_d(ndh_w, 0.024) * 0.065 * wind_lane
        water_rad += (glint_broad + glint_crisp)[:, None] * SUN_RADIANCE[None, :]

        # Bush-hammered concrete exterior terrace coping border along the glass wall
        patio_noise = value_noise_2d(wx * 12.0, wz * 12.0, seed=405)
        patio_joint = smoothstep(0.0, 0.005, np.abs((wz / 1.20) - np.round(wz / 1.20)) * 1.20)
        patio_col = (
            np.array([0.36, 0.35, 0.33], dtype=np.float64)
            * (0.88 + 0.22 * patio_noise[:, None])
            * (0.45 + 0.55 * patio_joint[:, None])
        )
        patio_rad = patio_col * (SUN_RADIANCE[None, :] * 0.19 + SKY_MID_RAD[None, :] * 0.52)

        surf_rad = np.where(in_patio[:, None], patio_rad, water_rad)
        direct_backdrop[idx_w] = surf_rad

    return direct_backdrop.reshape((*shape, 3))


# ==============================================================================
# 8. SOLAR WINDOW TRANSMISSION & SOFT AREA-LIGHT PENUMBRA SHADOW ENGINE
# ==============================================================================


def evaluate_window_sun_transmission(p_world: np.ndarray) -> np.ndarray:
    """
    For any array of 3D points `p_world` (..., 3) inside the room, trace a shadow
    ray along `SUN_DIR` to the right curtain wall (X = +3.20) or back curtain
    wall (Z = +7.60), and compute the exact soft solar-disc penumbra transmission
    including 3D window mullions, furniture, leaning art frame, and tabletop still life.
    """
    px = p_world[..., 0]
    py = p_world[..., 1]
    pz = p_world[..., 2]

    tx = (ROOM_X_RIGHT - px) / SUN_DIR[0]
    tz = (ROOM_Z_BACK - pz) / SUN_DIR[2]
    t_exit = np.minimum(tx, tz)
    valid_exit = t_exit > 0.0

    wx = px + t_exit * SUN_DIR[0]
    wy = py + t_exit * SUN_DIR[1]
    wz = pz + t_exit * SUN_DIR[2]

    pen_r = 0.016 + 0.022 * np.clip(t_exit, 0.0, 10.0)
    pen_y_top = pen_r * (1.45 + 0.45 * smoothstep(4.2, 7.0, t_exit))

    on_right_wall = tx <= tz
    in_y_open = smoothstep(0.090 - pen_r, 0.090 + pen_r, wy) * (
        1.0 - smoothstep(3.86 - pen_y_top, 3.86 + pen_y_top, wy)
    )
    in_z_open = smoothstep(-0.65 - pen_r, -0.65 + pen_r, wz) * (
        1.0 - smoothstep(ROOM_Z_BACK, ROOM_Z_BACK + pen_r, wz)
    )
    in_x_open = smoothstep(1.14 - pen_r, 1.14 + pen_r, wx) * (
        1.0 - smoothstep(ROOM_X_RIGHT, ROOM_X_RIGHT + pen_r, wx)
    )

    # Exterior courtyard overhang / canopy softly attenuates direct beam on the far-left bookcase wall (px < -2.15)
    far_wall_atten = 0.08 + 0.92 * smoothstep(-2.72, -2.15, px)
    aperture_vis = in_y_open * np.where(on_right_wall, in_z_open, in_x_open) * far_wall_atten

    mullion_half_w = 0.044
    transom_half_h = 0.032

    # Cast right-wall vertical mullion shadows strictly from the visible in-frame mullions (Z = 3.55, 5.50, 7.52)
    # and attenuate narrow mullion contrast across the 6m room width so shadows connect cleanly to the window base
    dist_mull_z = np.minimum(np.minimum(np.abs(wz - 3.55), np.abs(wz - 5.50)), np.abs(wz - 7.52))
    mull_lo = np.maximum(0.004, mullion_half_w - pen_r * 0.35)
    mull_hi = mullion_half_w + pen_r * 0.85
    raw_mull_z = smoothstep(mull_lo, mull_hi, dist_mull_z)
    mull_near_w = 1.0 - 0.72 * smoothstep(4.2, 6.8, t_exit)
    vis_mull_z = 1.0 - mull_near_w * (1.0 - raw_mull_z)

    dist_mull_x = np.abs(wx - 2.18)
    raw_mull_x = smoothstep(mull_lo, mull_hi, dist_mull_x)
    vis_mull_x = 1.0 - mull_near_w * (1.0 - raw_mull_x)

    vis_vert_mull = np.where(on_right_wall, vis_mull_z, vis_mull_x)

    # Horizontal glazing transoms only cast near-field occlusion right beside the glass wall (t_exit < 1.1m)
    dist_trans_y = np.minimum(np.abs(wy - 1.08), np.abs(wy - 2.98))
    trans_lo = np.maximum(0.004, transom_half_h - pen_r * 0.35)
    trans_hi = transom_half_h + pen_r * 0.85
    raw_trans = smoothstep(trans_lo, trans_hi, dist_trans_y)
    trans_near_w = 1.0 - smoothstep(0.35, 1.10, t_exit)
    vis_trans = 1.0 - trans_near_w * (1.0 - raw_trans)

    # Single unified corner structural column at (ROOM_X_RIGHT, ROOM_Z_BACK) = (3.20, 7.60)
    corner_dist = np.hypot(wx - ROOM_X_RIGHT, wz - ROOM_Z_BACK)
    corner_lo = np.maximum(0.010, 0.078 - pen_r * 0.35)
    corner_hi = 0.078 + pen_r * 0.85
    raw_corner = smoothstep(corner_lo, corner_hi, corner_dist)
    vis_corner = 1.0 - mull_near_w * (1.0 - raw_corner)

    sun_vis = aperture_vis * vis_vert_mull * vis_trans * vis_corner
    sun_vis = np.where(valid_exit, sun_vis, 0.0)

    # Interior 3D Furniture Shadow Occluders (exact continuous 3D swept ray-box distance, zero discrete slicing)
    sun_sl_x = SUN_DIR[0] / SUN_DIR[1]
    sun_sl_z = SUN_DIR[2] / SUN_DIR[1]
    inv_sun_y = 1.0 / SUN_DIR[1]

    def swept_box_shadow(
        mask_active: np.ndarray,
        cx: float,
        cz: float,
        ex: float,
        ez: float,
        y_lo: float,
        y_hi: float,
        pen_base: float = 0.012,
        pen_slope: float = 0.024,
        max_opacity: float = 0.94,
        yaw_rad: float = 0.0,
    ) -> np.ndarray:
        out = np.zeros_like(py)
        valid = mask_active & (py < y_hi - 0.002)
        if not np.any(valid):
            return out
        px_v = px[valid]
        py_v = py[valid]
        pz_v = pz[valid]
        rx = px_v - cx
        rz = pz_v - cz
        if yaw_rad != 0.0:
            c_y = math.cos(yaw_rad)
            s_y = math.sin(yaw_rad)
            ox = c_y * rx - s_y * rz
            oz = s_y * rx + c_y * rz
            vx = c_y * sun_sl_x - s_y * sun_sl_z
            vz = s_y * sun_sl_x + c_y * sun_sl_z
        else:
            ox = rx
            oz = rz
            vx = sun_sl_x
            vz = sun_sl_z

        dh0 = np.maximum(y_lo - py_v, 0.002)
        dh1 = np.maximum(y_hi - py_v, dh0 + 1e-5)

        safe_vx = vx if abs(vx) > 1e-8 else 1e-8
        safe_vz = vz if abs(vz) > 1e-8 else 1e-8
        v_norm2 = vx * vx + vz * vz + 1e-12

        cands = [
            dh0,
            dh1,
            np.clip((-ex - ox) / safe_vx, dh0, dh1),
            np.clip((ex - ox) / safe_vx, dh0, dh1),
            np.clip((-ez - oz) / safe_vz, dh0, dh1),
            np.clip((ez - oz) / safe_vz, dh0, dh1),
        ]
        for sgn_x in (-1.0, 1.0):
            for sgn_z in (-1.0, 1.0):
                dh_c = ((sgn_x * ex - ox) * vx + (sgn_z * ez - oz) * vz) / v_norm2
                cands.append(np.clip(dh_c, dh0, dh1))

        d_min = np.full_like(py_v, 1e9)
        dh_best = dh0.copy()
        for dh_k in cands:
            dx = np.maximum(np.abs(ox + vx * dh_k) - ex, 0.0)
            dz = np.maximum(np.abs(oz + vz * dh_k) - ez, 0.0)
            dk = np.hypot(dx, dz)
            take_k = dk < d_min
            d_min = np.where(take_k, dk, d_min)
            dh_best = np.where(take_k, dh_k, dh_best)

        pen = pen_base + pen_slope * (dh_best * inv_sun_y)
        out[valid] = (1.0 - smoothstep(0.0, pen, d_min)) * max_opacity
        return out

    mask_tab_rec = (py < 0.392) & (px > -1.35) & (px < 1.05) & (pz > 0.72) & (pz < 2.65)
    table_shadow = swept_box_shadow(
        mask_tab_rec, 0.30, 1.77, 0.68, 0.69, 0.330, 0.390, pen_base=0.014, pen_slope=0.024, max_opacity=0.94
    )

    # Exact 3D coffee-table walnut plinth legs & center stretcher (zero height-slice staircase)
    plinth_shadow = np.maximum(
        swept_box_shadow(
            mask_tab_rec, 0.30, 1.28, 0.52, 0.042, 0.000, 0.335, pen_base=0.010, pen_slope=0.020, max_opacity=0.94
        ),
        np.maximum(
            swept_box_shadow(
                mask_tab_rec, 0.30, 2.26, 0.52, 0.042, 0.000, 0.335, pen_base=0.010, pen_slope=0.020, max_opacity=0.94
            ),
            swept_box_shadow(
                mask_tab_rec, 0.30, 1.77, 0.045, 0.48, 0.075, 0.205, pen_base=0.010, pen_slope=0.020, max_opacity=0.92
            ),
        ),
    )

    # Continuous swept lounge chair shadow (slung seat + inclined backrest + bolster + armrests + steel runners)
    cy_c, sy_c = math.cos(-0.56), math.sin(-0.56)
    on_rug_ch = (py < 0.35) & (px > -2.55) & (px < -0.25) & (pz > 1.20) & (pz < 3.25)
    chair_seat_sh = np.zeros_like(py)
    chair_leg_sh = np.zeros_like(py)
    if np.any(on_rug_ch):
        def chair_box_sh(lx: float, lz: float, ex: float, ez: float, y_lo: float, y_hi: float, op: float = 0.92) -> np.ndarray:
            wx = -1.18 + cy_c * lx + sy_c * lz
            wz = 2.28 - sy_c * lx + cy_c * lz
            return swept_box_shadow(
                on_rug_ch, wx, wz, ex, ez, y_lo, y_hi, pen_base=0.018, pen_slope=0.026, max_opacity=op, yaw_rad=-0.56
            )

        chair_seat_sh = np.maximum(
            chair_box_sh(0.0, -0.02, 0.41, 0.395, 0.23, 0.38, 0.92),
            np.maximum(
                chair_box_sh(0.0, 0.312, 0.40, 0.085, 0.29, 0.84, 0.92),
                np.maximum(
                    chair_box_sh(-0.425, -0.01, 0.028, 0.245, 0.44, 0.49, 0.88),
                    chair_box_sh(0.425, -0.01, 0.028, 0.245, 0.44, 0.49, 0.88),
                ),
            ),
        )

        d_seg_min = np.full_like(py, 1e6)

        def chair_proj_xz(lx: float, ly: float, lz: float) -> tuple[np.ndarray, np.ndarray]:
            wx = -1.18 + cy_c * lx + sy_c * lz
            wz = 2.28 - sy_c * lx + cy_c * lz
            dh = np.maximum(ly - py, 0.0)
            return wx - dh * sun_sl_x, wz - dh * sun_sl_z

        for side_s in (-0.425, 0.425):
            rf_x, rf_z = chair_proj_xz(side_s, 0.017, -0.37)
            rb_x, rb_z = chair_proj_xz(side_s, 0.017, 0.38)
            ft_x, ft_z = chair_proj_xz(side_s, 0.36, -0.32)
            bt_x, bt_z = chair_proj_xz(side_s, 0.38, 0.32)
            for ax, az, bx, bz in (
                (rf_x, rf_z, rb_x, rb_z),
                (rf_x, rf_z, ft_x, ft_z),
                (rb_x, rb_z, bt_x, bt_z),
            ):
                abx = bx - ax
                abz = bz - az
                apx = px - ax
                apz = pz - az
                t_seg = np.clip((apx * abx + apz * abz) / np.maximum(abx * abx + abz * abz, 1e-8), 0.0, 1.0)
                d_seg = np.hypot(apx - t_seg * abx, apz - t_seg * abz)
                d_seg_min = np.minimum(d_seg_min, d_seg)

        chair_leg_sh = np.where(on_rug_ch, (1.0 - smoothstep(0.010, 0.028, d_seg_min)) * 0.90, 0.0)

    chair_shadow = np.maximum(chair_seat_sh, chair_leg_sh)

    # Exact continuous 3D swept shadow for the American walnut & Roman travertine credenza cabinet + recessed plinth
    on_floor_bn = (py < 0.528) & (pz > 6.05)
    bench_shadow = np.maximum(
        swept_box_shadow(
            on_floor_bn, -0.85, 7.22, 1.70, 0.34, 0.085, 0.495, pen_base=0.014, pen_slope=0.024, max_opacity=0.92
        ),
        np.maximum(
            swept_box_shadow(
                on_floor_bn, -0.85, 7.22, 1.72, 0.345, 0.495, 0.526, pen_base=0.015, pen_slope=0.024, max_opacity=0.92
            ),
            swept_box_shadow(
                on_floor_bn, -0.85, 7.25, 1.62, 0.28, 0.000, 0.085, pen_base=0.010, pen_slope=0.020, max_opacity=0.93
            ),
        ),
    )
    for foot_x in (-2.42, -1.37, -0.33, 0.72):
        bench_shadow = np.maximum(
            bench_shadow,
            swept_box_shadow(
                on_floor_bn, foot_x, 6.94, 0.022, 0.022, 0.000, 0.088, pen_base=0.008, pen_slope=0.016, max_opacity=0.90
            ),
        )

    # Swept solar shadows for the back credenza vessels, folio stack & artwork onto credenza top and back wall
    on_cred_rec = ((py > 0.515) & (py < 0.526) & (pz > 6.85)) | ((pz > 7.56) & (py > 0.52) & (py < 1.38))
    if np.any(on_cred_rec):
        sun_sl_x = SUN_DIR[0] / SUN_DIR[1]
        sun_sl_z = SUN_DIR[2] / SUN_DIR[1]
        # Fluted ceramic vessel at [0.12, 0.52..0.85, 7.16] (R=0.086)
        dh_v1 = np.clip(0.84 - py, 0.005, 0.32)
        sv1_x = -dh_v1 * sun_sl_x
        sv1_z = -dh_v1 * sun_sl_z
        rv1_x = px - 0.12
        rv1_z = pz - 7.16
        tv1 = np.clip((rv1_x * sv1_x + rv1_z * sv1_z) / np.maximum(sv1_x * sv1_x + sv1_z * sv1_z, 1e-8), 0.0, 1.0)
        dv1 = np.hypot(rv1_x - tv1 * sv1_x, rv1_z - tv1 * sv1_z)
        sh_v1 = np.where(
            ~((py > 0.528) & (np.hypot(rv1_x, rv1_z) < 0.095)),
            (1.0 - smoothstep(0.074, 0.096, dv1)) * 0.86,
            0.0,
        )
        # Companion low patinated bronze bowl at [0.34, 0.52..0.65, 7.24] (R=0.064)
        dh_v2 = np.clip(0.64 - py, 0.005, 0.13)
        sv2_x = -dh_v2 * sun_sl_x
        sv2_z = -dh_v2 * sun_sl_z
        rv2_x = px - 0.34
        rv2_z = pz - 7.24
        tv2 = np.clip((rv2_x * sv2_x + rv2_z * sv2_z) / np.maximum(sv2_x * sv2_x + sv2_z * sv2_z, 1e-8), 0.0, 1.0)
        dv2 = np.hypot(rv2_x - tv2 * sv2_x, rv2_z - tv2 * sv2_z)
        sh_v2 = np.where(
            ~((py > 0.528) & (np.hypot(rv2_x, rv2_z) < 0.072)),
            (1.0 - smoothstep(0.052, 0.072, dv2)) * 0.84,
            0.0,
        )
        bench_shadow = np.maximum(bench_shadow, np.where(on_cred_rec, np.maximum(sh_v1, sh_v2), 0.0))

    # Full 3D cross-object & tabletop solar occlusion across all foreground coffee-table still-life geometry
    in_still_life = (py > 0.386) & (py < 0.625) & (px > -0.68) & (px < 1.00) & (pz > 1.05) & (pz < 2.48)
    still_life_shadow = np.zeros_like(py)
    if np.any(in_still_life):
        sun_sl_x = SUN_DIR[0] / SUN_DIR[1]
        sun_sl_z = SUN_DIR[2] / SUN_DIR[1]

        def _ray_ellipsoid_shadow(
            cx: float,
            cy: float,
            cz: float,
            rx: float,
            ry: float,
            rz: float,
            pen_base: float = 0.004,
            pen_slope: float = 0.022,
            self_pad: float = 1.028,
        ) -> np.ndarray:
            ox = (px - cx) / rx
            oy = (py - cy) / ry
            oz = (pz - cz) / rz
            vx = SUN_DIR[0] / rx
            vy = SUN_DIR[1] / ry
            vz = SUN_DIR[2] / rz
            v2 = vx * vx + vy * vy + vz * vz
            t_pos = np.maximum(-(ox * vx + oy * vy + oz * vz) / v2, 0.0)
            qx = ox + t_pos * vx
            qy = oy + t_pos * vy
            qz = oz + t_pos * vz
            r_self = np.sqrt(ox * ox + oy * oy + oz * oz)
            r_eff = (rx + ry + rz) / 3.0
            d_surf = (np.sqrt(qx * qx + qy * qy + qz * qz) - 1.0) * r_eff
            pen = pen_base + pen_slope * t_pos
            return np.where(r_self > self_pad, 1.0 - smoothstep(-pen, pen, d_surf), 0.0)

        def _ray_cyl_y_shadow(
            cx: float,
            cz: float,
            y0: float,
            y1: float,
            radius: float,
            pen_base: float = 0.004,
            pen_slope: float = 0.022,
            exclude_outer_skin: np.ndarray | None = None,
        ) -> np.ndarray:
            ocx = px - cx
            ocz = pz - cz
            a_xz = SUN_DIR[0] * SUN_DIR[0] + SUN_DIR[2] * SUN_DIR[2]
            t_xz = np.maximum(-(ocx * SUN_DIR[0] + ocz * SUN_DIR[2]) / a_xz, 0.0)
            # Clamp ray evaluation height to the finite vertical span [y0, y1] of the cylinder
            t_y0 = np.maximum((y0 - py) / SUN_DIR[1], 0.0)
            t_y1 = np.maximum((y1 - py) / SUN_DIR[1], 0.0)
            t_eval = np.clip(t_xz, t_y0, t_y1)
            dx_c = ocx + t_eval * SUN_DIR[0]
            dz_c = ocz + t_eval * SUN_DIR[2]
            dy_c = np.maximum(np.maximum(y0 - (py + t_eval * SUN_DIR[1]), (py + t_eval * SUN_DIR[1]) - y1), 0.0)
            d_rad = np.hypot(dx_c, dz_c) - radius
            d_3d = np.hypot(np.maximum(d_rad, 0.0), dy_c) + np.minimum(d_rad, 0.0)
            pen = pen_base + pen_slope * t_eval
            valid = t_y1 > 0.002
            if exclude_outer_skin is not None:
                valid = valid & (~exclude_outer_skin)
            return np.where(valid, 1.0 - smoothstep(-pen, pen, d_3d), 0.0)

        # 1. Exact 3D Stoneware Cappuccino Cup ([0.24, 0.400..0.482, 1.78], R=0.053),
        #    Loop Handle ([0.182, 0.442, 1.770]), & Saucer ([0.24, 0.390..0.408, 1.78], R=0.082)
        #    Casts across tabletop, saucer dish, brass spoon, crema, handle, and open art monograph.
        cup_rx = px - 0.24
        cup_rz = pz - 1.78
        is_cup_outer_wall = (
            (py > 0.399)
            & (py <= 0.484)
            & (np.hypot(cup_rx, cup_rz) > 0.0495)
            & ((cup_rx * SUN_DIR[0] + cup_rz * SUN_DIR[2]) > -0.008)
        )
        sh_cup_body = _ray_cyl_y_shadow(
            0.24, 1.78, 0.400, 0.482, 0.053, pen_base=0.003, pen_slope=0.024, exclude_outer_skin=is_cup_outer_wall
        ) * 0.90
        sh_cup_handle = _ray_ellipsoid_shadow(
            0.182, 0.442, 1.770, 0.021, 0.027, 0.0085, pen_base=0.003, pen_slope=0.024, self_pad=1.08
        ) * 0.86

        # Stoneware saucer rim ([0.24, 0.390..0.408, 1.78], R=0.082) onto tabletop & open book edge
        is_saucer_self = (py >= 0.392) & (py <= 0.410) & (np.hypot(cup_rx, cup_rz) < 0.084)
        sh_saucer = _ray_cyl_y_shadow(
            0.24, 1.78, 0.390, 0.407, 0.081, pen_base=0.003, pen_slope=0.022, exclude_outer_skin=is_saucer_self
        ) * 0.85

        # 2. Swept Rotated OBB Shadow for Stacked Closed Monographs (cb1: [-0.12, 0.405, 2.04], yaw=0.22; cb2: [-0.11, 0.433, 2.03], yaw=0.09)
        cy_cb1, sy_cb1 = math.cos(0.22), math.sin(0.22)
        cy_cb2, sy_cb2 = math.cos(0.09), math.sin(0.09)
        d_cb_min = np.full_like(py, 1e6)
        for y_top_b, y_bot_b, bcx, bcz, cy_b, sy_b, hx_b, hz_b in (
            (0.420, 0.390, -0.12, 2.04, cy_cb1, sy_cb1, 0.162, 0.210),
            (0.446, 0.420, -0.11, 2.03, cy_cb2, sy_cb2, 0.148, 0.194),
        ):
            for frac_b in (0.15, 0.40, 0.68, 0.96):
                y_sl = y_bot_b + frac_b * (y_top_b - y_bot_b)
                dh_b = np.maximum(y_sl - py, 0.0)
                bx = px + dh_b * sun_sl_x - bcx
                bz = pz + dh_b * sun_sl_z - bcz
                lx = cy_b * bx - sy_b * bz
                lz = sy_b * bx + cy_b * bz
                d_b = np.where(
                    py < y_sl - 0.0025,
                    np.hypot(np.maximum(np.abs(lx) - hx_b, 0.0), np.maximum(np.abs(lz) - hz_b, 0.0)),
                    1e6,
                )
                d_cb_min = np.minimum(d_cb_min, d_b)
        sh_cbook = (1.0 - smoothstep(0.0, 0.011, d_cb_min)) * 0.90

        # 3. Exact 3D Swept OBB & Arched-Leaf Shadow for Open Art Monograph (book_c=[0.04, 0.396, 1.50], yaw=-0.19)
        cy_ob, sy_ob = math.cos(-0.19), math.sin(-0.19)
        # Receiver local coords relative to book center [0.04, 1.50] for self-arch elevation test
        rec_dx = px - 0.04
        rec_dz = pz - 1.50
        rec_lox = cy_ob * rec_dx - sy_ob * rec_dz
        rec_loz = sy_ob * rec_dx + cy_ob * rec_dz
        u_rec = np.clip(rec_lox / 0.222, -1.2, 1.2)
        on_open_page_surf = (
            (py >= 0.415)
            & (py <= 0.432)
            & (np.abs(rec_lox) <= 0.225)
            & (np.abs(rec_loz) <= 0.161)
        )
        rec_arch_y = 0.419 + 0.0085 * np.sin(np.pi * np.clip(np.abs(u_rec), 0.0, 1.0))
        eff_rec_y = np.where(on_open_page_surf, rec_arch_y, py)

        d_ob_min = np.full_like(py, 1e6)
        # (a) Hardcover base slab (cov_ext=[0.235, 0.006, 0.168], y in [0.390, 0.402]) onto tabletop
        for y_cov_s in (0.394, 0.4015):
            dh_c = np.maximum(y_cov_s - eff_rec_y, 0.0)
            ox_c = px + dh_c * sun_sl_x - 0.04
            oz_c = pz + dh_c * sun_sl_z - 1.50
            lox_c = cy_ob * ox_c - sy_ob * oz_c
            loz_c = sy_ob * ox_c + cy_ob * oz_c
            d_c = np.where(
                eff_rec_y < y_cov_s - 0.002,
                np.hypot(np.maximum(np.abs(lox_c) - 0.235, 0.0), np.maximum(np.abs(loz_c) - 0.168, 0.0)),
                1e6,
            )
            d_ob_min = np.minimum(d_ob_min, d_c)

        # (b) Thick cream page block & arched open leaves (page_ext=[0.222, 0.011, 0.158], y up to 0.4275)
        #     Casts onto tabletop (py ~ 0.390), dark hardcover lip (py ~ 0.402), and central spine gutter.
        for y_pg_s in (0.404, 0.412, 0.419, 0.424, 0.4272):
            dh_p = np.maximum(y_pg_s - eff_rec_y, 0.0)
            ox_p = px + dh_p * sun_sl_x - 0.04
            oz_p = pz + dh_p * sun_sl_z - 1.50
            lox_p = cy_ob * ox_p - sy_ob * oz_p
            loz_p = sy_ob * ox_p + cy_ob * oz_p
            u_ray = np.clip(lox_p / 0.222, -1.2, 1.2)
            leaf_top_y = np.where(
                np.abs(lox_p) <= 0.222,
                0.419 + 0.0085 * np.sin(np.pi * np.clip(np.abs(u_ray), 0.0, 1.0)),
                0.419,
            )
            d_p = np.where(
                (eff_rec_y < y_pg_s - 0.0018) & (y_pg_s <= leaf_top_y),
                np.hypot(np.maximum(np.abs(lox_p) - 0.222, 0.0), np.maximum(np.abs(loz_p) - 0.158, 0.0)),
                1e6,
            )
            d_ob_min = np.minimum(d_ob_min, d_p)

        sh_obook_raw = 1.0 - smoothstep(0.0, 0.0095, d_ob_min)
        sh_obook = np.where(on_open_page_surf, sh_obook_raw * 0.46, sh_obook_raw * 0.90)

        # 4. Full 3D Stoneware Olive Vase ([0.44, 0.390, 1.98]: belly ellipsoid + cylindrical neck + lip + leaves)
        #    Evaluated across all receivers in `in_still_life` (cup, rim, crema, handle, saucer, spoon, book, table).
        sh_ovase_belly = _ray_ellipsoid_shadow(
            0.44, 0.488, 1.98, 0.086, 0.098, 0.086, pen_base=0.004, pen_slope=0.022, self_pad=1.028
        )
        is_ovase_neck_self = (
            (py > 0.548) & (py < 0.625) & (np.hypot(px - 0.44, pz - 1.98) < 0.052)
        )
        sh_ovase_neck = _ray_cyl_y_shadow(
            0.44, 1.98, 0.552, 0.612, 0.040, pen_base=0.004, pen_slope=0.024, exclude_outer_skin=is_ovase_neck_self
        )
        sh_ovase_lip = _ray_ellipsoid_shadow(
            0.44, 0.612, 1.98, 0.048, 0.010, 0.048, pen_base=0.004, pen_slope=0.024, self_pad=1.05
        )
        sh_ovase = np.maximum(sh_ovase_belly, np.maximum(sh_ovase_neck, sh_ovase_lip)) * 0.90

        still_life_shadow = np.where(
            in_still_life,
            np.maximum(
                np.maximum(np.maximum(sh_cup_body, sh_cup_handle), sh_saucer),
                np.maximum(sh_ovase, np.maximum(sh_cbook, sh_obook)),
            ),
            0.0,
        )

    # Soft shelf-header & vertical stile alcove occlusion inside recessed bookcase bays (px < -2.745)
    in_book_recess = (px < -2.745) & (py > 0.08) & (py < 4.06) & (pz > -0.82) & (pz < 7.60)
    shelf_alcove_shadow = np.zeros_like(py)
    if np.any(in_book_recess):
        recess_dx = np.clip(-2.72 - px, 0.0, 0.36)
        dy_below_shelf = np.full_like(py, 10.0)
        for sy_ceil in BOOKCASE_SHELF_Y[1:]:
            in_tier = (py <= sy_ceil) & (py >= sy_ceil - 0.58)
            dy_below_shelf = np.where(in_tier, sy_ceil - py, dy_below_shelf)
        overhang_margin = dy_below_shelf - recess_dx * (SUN_DIR[1] / SUN_DIR[0])
        sh_overhang = (1.0 - smoothstep(-0.025, 0.095, overhang_margin)) * 0.42

        z_at_stile_front = pz + recess_dx * (SUN_DIR[2] / SUN_DIR[0])
        d_stile = np.full_like(pz, 10.0)
        for sz_bar in BOOKCASE_STILE_Z:
            d_stile = np.minimum(d_stile, np.abs(z_at_stile_front - sz_bar))
        sh_stile = (1.0 - smoothstep(0.015, 0.095, d_stile)) * 0.42
        sh_recess_base = 0.25 * (1.0 - smoothstep(-2.94, -2.72, px)) + 0.55 * np.maximum(sh_overhang, sh_stile)
        shelf_alcove_shadow = np.where(in_book_recess, np.clip(sh_recess_base, 0.0, 0.72), 0.0)

    furn_trans = (
        (1.0 - table_shadow)
        * (1.0 - plinth_shadow)
        * (1.0 - chair_shadow)
        * (1.0 - bench_shadow)
        * (1.0 - still_life_shadow)
        * (1.0 - shelf_alcove_shadow)
    )
    return sun_vis * furn_trans



# ==============================================================================
# 9. PROCEDURAL ARCHITECTURAL MATERIAL SHADERS (TACTILE IMPERFECTIONS & PATINA)
# ==============================================================================


def material_board_formed_concrete(
    p_world: np.ndarray,
    normal: np.ndarray,
    is_ceiling: bool = False,
    fast_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tadao Ando / Peter Zumthor cast-in-place board-formed architectural mineral concrete:
      - Cool-warm mineral grey limestone/slate cement matrix with fine river-sand & basalt speckle.
      - Charred/wire-brushed Japanese cedar shuttering grain Transfer.
      - Stochastic 3D bug-holes (3mm-14mm air-pocket pinholes) clustered near pour seams with
        concave 3D rim normals that catch grazing sunlight.
      - Horizontal shuttering board cupping and proud cement-slurry squeeze-out fins.
      - Form-tie cones with recessed lead/mortar plugs, concentric ring marks, 3D view-ray
        parallax depth, and subtle rust-mineral water-run weeping tracks below select tie-holes.
      - Recessed 2.5cm architectural shadow-gap reveal at floor and ceiling junctions.
    """
    is_horiz = np.abs(normal[..., 1]) > 0.5
    is_x_face = (np.abs(normal[..., 0]) > 0.5) & (~is_horiz)
    is_z_face = (~is_horiz) & (~is_x_face)

    u_coord = np.where(
        is_horiz,
        p_world[..., 0],
        np.where(is_x_face, p_world[..., 2], p_world[..., 0] + 3.10),
    )
    v_coord = np.where(
        is_horiz,
        p_world[..., 2],
        p_world[..., 1],
    )
    if is_ceiling:
        panel_w = 1.55
        panel_h = 1.85
        board_h = 0.205
    else:
        panel_w = 1.393
        panel_h = 1.0625
        board_h = 0.2125

    board_idx = np.floor(v_coord / board_h).astype(np.int64)
    panel_u_idx = np.floor(u_coord / panel_w).astype(np.int64)
    panel_v_idx = np.floor(v_coord / panel_h).astype(np.int64)

    board_hash = hash2d(board_idx, panel_u_idx, seed=501)
    lift_hash = hash2d(panel_v_idx, panel_u_idx, seed=503)
    board_tint = (0.93 + 0.12 * board_hash) * (0.96 + 0.07 * lift_hash)

    u_local_board = u_coord * 2.3 + board_hash * 13.7
    v_local_board = (v_coord / board_h - board_idx) * 6.2
    n_oct = 1 if fast_mode else 3
    grain_f = fbm2d(u_local_board * 1.45, v_local_board * 5.4, octaves=n_oct, seed=509)
    cathedral = np.sin(v_local_board * 4.8 + grain_f * 6.8) * 0.5 + 0.5

    # Authentic mineral grey concrete palette (cool slate-limestone + warm sand aggregate)
    base_concrete = np.array([0.425, 0.422, 0.412], dtype=np.float64)
    cool_slate = np.array([0.385, 0.395, 0.405], dtype=np.float64)
    warm_sand = np.array([0.465, 0.442, 0.408], dtype=np.float64)

    albedo = (
        base_concrete[None, :] * (0.58 + 0.42 * grain_f[..., None])
        + (warm_sand - base_concrete)[None, :] * (cathedral * 0.32)[..., None]
        + (cool_slate - base_concrete)[None, :] * ((1.0 - grain_f) * 0.28)[..., None]
    ) * board_tint[..., None]

    v_in_board = (v_coord / board_h) - board_idx
    v_frac = np.abs((v_coord / board_h) - np.round(v_coord / board_h)) * board_h
    u_frac = np.abs((u_coord / panel_w) - np.round(u_coord / panel_w)) * panel_w
    pv_frac = np.abs((v_coord / panel_h) - np.round(v_coord / panel_h)) * panel_h

    board_seam = 1.0 - 0.32 * (1.0 - smoothstep(0.0, 0.0042, v_frac))
    panel_seam = 1.0 - 0.52 * (
        1.0 - np.minimum(smoothstep(0.0, 0.0060, u_frac), smoothstep(0.0, 0.0060, pv_frac))
    )
    albedo *= (board_seam * panel_seam)[..., None]

    n_out = normal.copy()
    # Shuttering board cylindrical cup/warp + proud cement-slurry squeeze-out fins along board seams
    board_cup_slope = (v_in_board - 0.5) * 0.055 * (0.6 + 0.8 * board_hash)
    seam_slope_v = np.sin((v_coord / board_h) * 2.0 * PI) * np.exp(-((v_frac / 0.0052) ** 2)) * 0.18
    grain_slope = (cathedral - 0.5) * 0.042

    if not fast_mode:
        # Fine river-sand & basalt aggregate speckle + lime efflorescence clouds
        fine_pores, dp_du, dp_dv = value_noise_2d_grad(u_coord * 46.0, v_coord * 46.0, seed=521)
        micro_speck = value_noise_2d(u_coord * 135.0, v_coord * 135.0, seed=527)
        lime_bloom = fbm2d(u_coord * 1.65, v_coord * 1.35, octaves=2, seed=533)
        albedo *= 0.92 + 0.12 * fine_pores[..., None] + 0.05 * (micro_speck - 0.5)[..., None]
        albedo += np.array([0.028, 0.028, 0.026], dtype=np.float64)[None, :] * smoothstep(0.58, 0.82, lime_bloom)[..., None]

        # Proud cement-slurry fin along select horizontal board joints
        fin_mask = np.exp(-((v_frac / 0.0022) ** 2)) * smoothstep(0.45, 0.78, fine_pores)
        albedo += fin_mask[..., None] * np.array([0.045, 0.044, 0.040], dtype=np.float64)[None, :]

        # Stochastic 3D Bug-Holes (air-pocket pinholes clustered near pour lines)
        seam_cluster = 0.65 + 0.55 * np.exp(-((v_frac / 0.035) ** 2))
        bug1_d, bug1_nx, bug1_ny = cellular_bugholes_2d(u_coord * 28.0, v_coord * 30.0, density_prob=0.34, seed=541)
        bug2_d, bug2_nx, bug2_ny = cellular_bugholes_2d(u_coord * 64.0, v_coord * 68.0, density_prob=0.42, seed=557)
        bug_depth = np.clip((bug1_d + 0.65 * bug2_d) * seam_cluster, 0.0, 1.0)
        bug_nx = (bug1_nx + 0.65 * bug2_nx) * 0.28 + dp_du * 0.015
        bug_ny = (bug1_ny + 0.65 * bug2_ny) * 0.28 + dp_dv * 0.015
        albedo *= (1.0 - 0.58 * bug_depth)[..., None]
    else:
        bug_depth = np.zeros_like(u_coord)
        bug_nx = np.zeros_like(u_coord)
        bug_ny = np.zeros_like(u_coord)

    du_total = bug_nx.copy()
    dv_total = seam_slope_v + board_cup_slope + grain_slope + bug_ny

    if not fast_mode:
        has_tie_holes = (~is_horiz) | (is_ceiling & is_horiz & (p_world[..., 1] > 4.23))
        pu = u_coord - panel_u_idx * panel_w
        pv = v_coord - panel_v_idx * panel_h
        col_centers = np.array([0.22 * panel_w, 0.50 * panel_w, 0.78 * panel_w], dtype=np.float64)
        row_centers = np.array([0.26 * panel_h, 0.74 * panel_h], dtype=np.float64)

        idx_u = np.argmin(np.abs(pu[..., None] - col_centers), axis=-1)
        idx_v = np.argmin(np.abs(pv[..., None] - row_centers), axis=-1)
        vec_u = pu - col_centers[idx_u]
        vec_v = pv - row_centers[idx_v]
        r_hole_raw = np.hypot(vec_u, vec_v)

        cone_outer_r = 0.025
        cone_inner_r = 0.0105

        # 3D view-ray parallax shift inside the recessed conical form-tie socket
        rd_view = normalize(p_world - CAM_POS[None, :])
        view_u = np.where(is_horiz, rd_view[..., 0], np.where(is_x_face, rd_view[..., 2], rd_view[..., 0]))
        view_v = np.where(is_horiz, rd_view[..., 2], rd_view[..., 1])
        cone_depth_est = np.clip((cone_outer_r - r_hole_raw) / cone_outer_r, 0.0, 1.0) * 0.014
        vec_u_par = vec_u + view_u * cone_depth_est
        vec_v_par = vec_v + view_v * cone_depth_est
        r_hole = np.hypot(vec_u_par, vec_v_par)

        in_cone = has_tie_holes & (r_hole < cone_outer_r)
        in_plug = has_tie_holes & (r_hole < cone_inner_r)

        # Concentric ring tool marks inside the form-tie cone + recessed lead/mortar plug
        cone_ring = 0.88 + 0.12 * np.sin(r_hole * 850.0)
        cone_ao = np.where(
            in_plug,
            0.16,
            np.where(
                in_cone,
                (0.38 + 0.62 * ((r_hole - cone_inner_r) / (cone_outer_r - cone_inner_r))) * cone_ring,
                1.0,
            ),
        )
        albedo *= cone_ao[..., None]

        bevel_strength = np.where(in_cone & (~in_plug), 0.76, 0.0)
        inv_r = 1.0 / np.maximum(r_hole, 1e-5)
        du_total += -vec_u_par * inv_r * bevel_strength
        dv_total += -vec_v_par * inv_r * bevel_strength

        if not is_ceiling:
            # Subtle mineral water-run / rust-tinted weeping tracks below select form-tie cones on vertical walls
            tie_hash = hash2d(panel_u_idx * 3 + idx_u, panel_v_idx * 2 + idx_v, seed=571)
            below_cone = (~is_horiz) & (vec_v < -0.010) & (vec_v > -0.24) & (np.abs(vec_u) < 0.018)
            weep_len = 0.08 + 0.15 * tie_hash
            weep_profile = (
                np.exp(-((vec_u / 0.0075) ** 2))
                * np.exp(-((-vec_v - 0.01) / np.maximum(weep_len, 0.02)))
                * (tie_hash > 0.32)
                * below_cone
            )
            rust_stain_col = np.array([0.31, 0.26, 0.21], dtype=np.float64)
            albedo = albedo * (1.0 - 0.38 * weep_profile[..., None]) + rust_stain_col[None, :] * (
                0.38 * weep_profile[..., None]
            )

            # Architectural 2.5cm dark recessed shadow-gap reveal at floor (Y < 0.024) and ceiling (Y > 4.226)
            reveal_gap = np.where(
                ~is_horiz,
                smoothstep(0.018, 0.026, v_coord) * (1.0 - smoothstep(4.222, 4.230, v_coord)),
                1.0,
            )
            albedo *= (0.14 + 0.86 * reveal_gap)[..., None]

    n_out[..., 0] += np.where(is_horiz | is_z_face, du_total, 0.0)
    n_out[..., 1] += np.where(~is_horiz, dv_total, 0.0)
    n_out[..., 2] += np.where(is_horiz, dv_total, np.where(is_x_face, du_total, 0.0))

    n_out = normalize(n_out)
    roughness = np.clip(0.32 + 0.13 * (1.0 - grain_f) - 0.06 * cathedral + 0.22 * bug_depth, 0.20, 0.58)
    return albedo, roughness, n_out


def material_oiled_oak_and_travertine_floor(
    p_world: np.ndarray,
    is_travertine_zone: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate surface albedo, isotropic roughness, anisotropic roughness (ax, ay),
    and micro-perturbed normal for:
      A. Wide-plank oiled European oak floor (heartwood rings, quarter-sawn silver
         medullary rays, dark resinous pin knots with concentric grain deflection,
         open earlywood pores, V-groove bevels, and per-plank micro-tilt).
      B. Vein-cut honed Roman travertine hearth slabs (`travertino romano al contro`:
         90cm x 120cm slabs with bevelled mortar joints, multi-scale sedimentary
         bedding strata, crystalline calcite veinlets, and 2mm-8mm elongated micro-vugs
         with 3D rim catchlights).
    """
    fx = p_world[..., 0]
    fz = p_world[..., 2]

    # --- A. Wide-Plank Oiled European Oak (23cm wide planks running along Z) ---
    plank_w = 0.23
    plank_idx = np.floor((fx + 4.0) / plank_w).astype(np.int64)
    u_plank = (fx + 4.0) / plank_w - plank_idx

    plank_offset = hash2d(plank_idx, np.zeros_like(plank_idx), seed=601) * 2.4
    plank_z_idx = np.floor((fz + plank_offset) / 2.4).astype(np.int64)
    v_plank = (fz + plank_offset) / 2.4 - plank_z_idx

    id_hash = hash2d(plank_idx, plank_z_idx, seed=607)
    ring_center_u = 0.22 + 0.56 * hash2d(plank_idx, plank_z_idx, seed=613)

    # Select-grade architectural oak: occasional tiny sound pin-knots with gentle grain deflection
    knot_u = 0.22 + 0.56 * hash2d(plank_idx, plank_z_idx, seed=641)
    knot_v = 0.20 + 0.60 * hash2d(plank_idx, plank_z_idx, seed=647)
    has_knot = hash2d(plank_idx, plank_z_idx, seed=653) < 0.16
    du_k = (u_plank - knot_u) * plank_w
    dv_k = (v_plank - knot_v) * 2.4
    r_knot = np.hypot(du_k, dv_k * 0.45)
    knot_core = np.where(has_knot, np.exp(-((r_knot / 0.0085) ** 2)), 0.0)
    knot_halo = np.where(has_knot, np.exp(-((r_knot / 0.038) ** 2)), 0.0)

    grain_warp = fbm2d(fx * 5.5 + id_hash * 7.0, fz * 1.4, octaves=2, seed=619)
    ring_phase = (
        ((u_plank - ring_center_u) * 14.0) ** 2
        + fz * 2.6
        + grain_warp * 5.8
        + knot_halo * 6.5
    )
    rings = 0.5 + 0.5 * np.sin(ring_phase)
    rings_sharp = rings ** 1.85

    fine_fiber, df_dx, _ = value_noise_2d_grad(fx * 92.0, fz * 16.0, seed=631)
    earlywood_pores = smoothstep(0.72, 0.90, value_noise_2d(fx * 210.0, fz * 32.0, seed=659)) * rings_sharp

    # Quarter-sawn medullary rays (delicate silver-gold ribbon flecks running across the plank)
    ray_wave = value_noise_2d(fx * 18.0 + id_hash * 5.0, fz * 58.0, seed=661)
    medullary_ray = smoothstep(0.79, 0.93, ray_wave) * (0.4 + 0.6 * np.sin(u_plank * PI)) * (id_hash > 0.35)

    oak_light = np.array([0.39, 0.255, 0.145], dtype=np.float64)
    oak_dark = np.array([0.195, 0.115, 0.062], dtype=np.float64)
    oak_amber = np.array([0.45, 0.29, 0.155], dtype=np.float64)
    oak_ray_col = np.array([0.49, 0.34, 0.195], dtype=np.float64)
    knot_col = np.array([0.165, 0.095, 0.048], dtype=np.float64)

    plank_tone = 0.83 + 0.32 * id_hash
    oak_albedo = (
        oak_light[None, :] * (1.0 - 0.52 * rings_sharp[..., None])
        + oak_dark[None, :] * (0.52 * rings_sharp[..., None])
    ) * plank_tone[..., None]
    oak_albedo = (
        oak_albedo * (0.90 + 0.18 * fine_fiber[..., None] - 0.14 * earlywood_pores[..., None])
        + oak_amber[None, :] * (0.08 * grain_warp[..., None])
        + oak_ray_col[None, :] * (0.14 * medullary_ray[..., None])
    )
    oak_albedo = oak_albedo * (1.0 - 0.45 * knot_core[..., None]) + knot_col[None, :] * (0.45 * knot_core[..., None])

    edge_u = np.minimum(u_plank, 1.0 - u_plank) * plank_w
    edge_v = np.minimum(v_plank, 1.0 - v_plank) * 2.4
    groove_u = smoothstep(0.0, 0.0042, edge_u)
    groove_v = smoothstep(0.0, 0.0052, edge_v)
    groove_mask = groove_u * groove_v
    oak_albedo *= (0.24 + 0.76 * groove_mask)[..., None]

    traffic_scuff = value_noise_2d(fx * 3.5, fz * 3.5, seed=673)
    oak_rough = np.clip(
        0.105
        + 0.062 * rings_sharp
        + 0.045 * earlywood_pores
        + 0.030 * (1.0 - id_hash)
        + 0.025 * traffic_scuff
        - 0.025 * medullary_ray
        + 0.26 * (1.0 - groove_mask),
        0.080,
        0.44,
    )

    # Plank-to-plank micro-tilt (±0.28 deg) + V-groove bevel + fiber & earlywood pore normals
    plank_tilt_nx = (hash2d(plank_idx, plank_z_idx, seed=677) - 0.5) * 0.0095
    plank_tilt_nz = (hash2d(plank_idx, plank_z_idx, seed=683) - 0.5) * 0.0065
    bevel_nx = np.where(u_plank < 0.03, -1.0, np.where(u_plank > 0.97, 1.0, 0.0)) * (1.0 - groove_u) * 0.24
    fiber_nx = df_dx * 0.012 - earlywood_pores * 0.018

    # --- B. Vein-Cut Honed Roman Travertine Architectural Hearth Border ---
    slab_ix = np.floor(fx / 0.90).astype(np.int64)
    slab_iz = np.floor(fz / 1.20).astype(np.int64)
    slab_hash = hash2d(slab_ix, slab_iz, seed=703)
    slab_shift = slab_hash * 4.7

    # Multi-scale sedimentary bedding strata (`travertino romano al contro`)
    trav_warp = fbm2d(fx * 1.85 + slab_shift, fz * 0.85, octaves=3, seed=701)
    trav_strata1 = 0.5 + 0.5 * np.sin((fx + slab_shift) * 9.5 + trav_warp * 6.8 + fz * 1.3)
    trav_strata2 = 0.5 + 0.5 * np.cos((fx - slab_shift) * 24.0 + trav_warp * 9.2 - fz * 2.1)
    trav_vein = 0.62 * trav_strata1 + 0.38 * trav_strata2

    # Delicate crystalline calcite veinlets (bright mineral hairlines)
    calcite_wave = np.abs(np.sin(fx * 18.5 + trav_warp * 8.0 + fz * 3.4))
    calcite_veinlet = np.exp(-((calcite_wave / 0.045) ** 2)) * (0.4 + 0.6 * trav_strata2)

    # High-frequency elongated sedimentary micro-vugs (2mm-8mm cavities along bedding planes)
    vug_d1, vug_nx1, vug_nz1 = cellular_bugholes_2d(
        fx * 165.0 + trav_warp * 4.0, fz * 48.0, density_prob=0.36, seed=719
    )
    vug_d2, vug_nx2, vug_nz2 = cellular_bugholes_2d(
        fx * 88.0 - trav_warp * 3.0, fz * 26.0, density_prob=0.28, seed=727
    )
    vug_band_mask = smoothstep(0.28, 0.68, 1.0 - trav_strata1)
    vug_pit = np.clip((vug_d1 + 0.70 * vug_d2) * vug_band_mask, 0.0, 1.0)

    trav_ivory = np.array([0.60, 0.535, 0.44], dtype=np.float64)
    trav_ochre = np.array([0.47, 0.395, 0.31], dtype=np.float64)
    trav_sienna = np.array([0.53, 0.435, 0.335], dtype=np.float64)
    calcite_col = np.array([0.72, 0.66, 0.56], dtype=np.float64)

    slab_tone = 0.94 + 0.11 * slab_hash
    trav_albedo = (
        trav_ivory[None, :] * (0.52 + 0.48 * trav_vein[..., None])
        + trav_ochre[None, :] * (0.36 * (1.0 - trav_vein[..., None]))
        + trav_sienna[None, :] * (0.12 * trav_strata2[..., None])
        + calcite_col[None, :] * (0.14 * calcite_veinlet[..., None])
    ) * (0.93 + 0.12 * trav_warp[..., None]) * slab_tone[..., None]
    trav_albedo *= (1.0 - 0.44 * vug_pit)[..., None]

    # 90cm x 120cm slab mortar joints with 3D bevel normals
    u_slab = (fx / 0.90) - np.round(fx / 0.90)
    v_slab = (fz / 1.20) - np.round(fz / 1.20)
    joint_x = np.abs(u_slab) * 0.90
    joint_z = np.abs(v_slab) * 1.20
    stone_joint = smoothstep(0.0, 0.0032, np.minimum(joint_x, joint_z))
    trav_albedo *= (0.36 + 0.64 * stone_joint)[..., None]

    joint_bevel_nx = np.sign(u_slab) * np.exp(-((joint_x / 0.0035) ** 2)) * 0.18
    joint_bevel_nz = np.sign(v_slab) * np.exp(-((joint_z / 0.0035) ** 2)) * 0.18
    trav_nx = (trav_warp - 0.5) * 0.010 + (vug_nx1 + 0.7 * vug_nx2) * vug_band_mask * 0.16 + joint_bevel_nx
    trav_nz = (vug_nz1 + 0.7 * vug_nz2) * vug_band_mask * 0.12 + joint_bevel_nz

    trav_rough = np.clip(
        0.088 + 0.048 * (1.0 - trav_vein) + 0.26 * vug_pit + 0.22 * (1.0 - stone_joint),
        0.075,
        0.42,
    )

    albedo = np.where(is_travertine_zone[..., None], trav_albedo, oak_albedo)
    roughness = np.where(is_travertine_zone, trav_rough, oak_rough)
    nx_pert = np.where(is_travertine_zone, trav_nx, bevel_nx + fiber_nx + plank_tilt_nx)
    nz_pert = np.where(is_travertine_zone, trav_nz, plank_tilt_nz)
    n_out = normalize(np.stack([nx_pert, np.ones_like(nx_pert), nz_pert], axis=-1))

    aniso = np.stack([roughness * 0.72, roughness * 1.68], axis=-1)
    return albedo, roughness, aniso, n_out


def material_woven_boucle_rug(
    p_world: np.ndarray,
    normal: np.ndarray,
    is_border: np.ndarray,
    fast_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hand-loomed Belgian unbleached ecru/oatmeal Berber-bouclé wool area rug (`MAT_BOUCLE_RUG`)
    with bound natural flax-linen tapestry edge trim (`MAT_RUG_LINEN_BORDER`):
      - High-frequency staggered warp/weft looped wool yarn domes (215 x 165 loops/m).
      - Natural heather yarn fleck variation (ecru, oatmeal, warm flax, dark umber wool).
      - 3D yarn-loop micro-normals & compression darkening under chair sleds and table legs.
    """
    fx = p_world[..., 0]
    fz = p_world[..., 2]

    if fast_mode:
        macro_w = value_noise_2d(fx * 6.0, fz * 6.0, seed=751)
        alb = np.where(
            is_border[..., None],
            np.array([0.46, 0.41, 0.34], dtype=np.float64)[None, :],
            np.array([0.58, 0.53, 0.45], dtype=np.float64)[None, :] * (0.92 + 0.14 * macro_w[..., None]),
        )
        return alb, np.full_like(fx, 0.82), normal

    # Distance-based Nyquist anti-aliasing weight so fine loops never moiré with pixel rays
    dist_fade = 1.0 / (1.0 + 0.36 * np.maximum(fz - 1.0, 0.0) ** 2)

    # Fine-scale (4mm-6mm) C2 quintic wool-bouclé micro-tufts & analytical gradients (no macro blotches)
    tuft1, dt1_dx, dt1_dz = value_noise_2d_grad(fx * 165.0 + fz * 18.0, fz * 155.0 - fx * 15.0, seed=761)
    tuft2, dt2_dx, dt2_dz = value_noise_2d_grad(fx * 275.0 - fz * 28.0, fz * 260.0 + fx * 24.0, seed=769)
    fine_fleck = value_noise_2d(fx * 340.0 + 1.7, fz * 320.0 - 2.3, seed=773)

    # Subtle hand-loomed warp/weft yarn course alignment (damped with distance to prevent aliasing)
    weft_course = 0.5 + 0.5 * np.sin(fz * 145.0 + (tuft1 - 0.5) * 1.4)
    warp_course = 0.5 + 0.5 * np.sin(fx * 160.0 + (tuft2 - 0.5) * 1.2)
    loom_weave = (0.58 * weft_course + 0.42 * warp_course - 0.5) * dist_fade

    # Rounded bouclé wool micro-loop domes
    b1 = 1.0 - np.abs(2.0 * tuft1 - 1.0)
    b2 = 1.0 - np.abs(2.0 * tuft2 - 1.0)
    loop_dome = np.clip(0.50 * b1 + 0.50 * b2 * dist_fade + 0.16 * loom_weave, 0.0, 1.0)

    macro_patina = fbm2d(fx * 1.6, fz * 1.6, octaves=2, seed=793)

    wool_ecru = np.array([0.61, 0.56, 0.48], dtype=np.float64)
    wool_oatmeal = np.array([0.55, 0.495, 0.415], dtype=np.float64)
    wool_flax_fleck = np.array([0.47, 0.415, 0.34], dtype=np.float64)

    yarn_mix = 0.55 * tuft1 + 0.45 * fine_fleck
    is_dark_fleck = (fine_fleck > 0.88) & (tuft2 > 0.60)
    yarn_col = (
        wool_ecru[None, :] * (0.50 + 0.50 * yarn_mix[..., None])
        + wool_oatmeal[None, :] * (0.50 * (1.0 - yarn_mix[..., None]))
    )
    fleck_w = (is_dark_fleck * dist_fade)[..., None]
    yarn_col = yarn_col * (1.0 - 0.22 * fleck_w) + wool_flax_fleck[None, :] * (0.22 * fleck_w)

    # Physical wool-pile compression trough & puff collar where the lounge chair steel sled runners press into the rug
    cy_ch_r, sy_ch_r = math.cos(-0.56), math.sin(-0.56)
    lcx_r = cy_ch_r * (fx - (-1.18)) - sy_ch_r * (fz - 2.28)
    lcz_r = sy_ch_r * (fx - (-1.18)) + cy_ch_r * (fz - 2.28)
    signed_dx_run = (np.abs(lcx_r) - 0.425) * np.sign(lcx_r)
    along_run = 1.0 - smoothstep(0.365, 0.415, np.abs(lcz_r - 0.005))
    d_run_abs = np.abs(np.abs(lcx_r) - 0.425)
    trough_darken = 1.0 - 0.38 * along_run * np.exp(-((d_run_abs / 0.020) ** 2))
    # Derivative of wool indentation profile: tilts normal inward toward the runner slot and outward on the puff collar
    pile_slope_local_x = (
        along_run
        * (signed_dx_run / 0.018)
        * np.exp(-((d_run_abs / 0.022) ** 2))
        * 0.28
    )
    nx_trough = cy_ch_r * pile_slope_local_x
    nz_trough = -sy_ch_r * pile_slope_local_x

    boucle_alb = (
        yarn_col
        * (0.91 + 0.12 * loop_dome[..., None] * dist_fade[..., None])
        * (0.97 + 0.05 * macro_patina[..., None])
        * trough_darken[..., None]
    )

    # Bound natural flax-linen border tape with diagonal twill weave
    twill_wave = (0.5 + 0.5 * np.sin((fx + fz) * 140.0)) * dist_fade
    linen_base = np.array([0.46, 0.405, 0.335], dtype=np.float64)
    border_alb = linen_base[None, :] * (0.92 + 0.12 * twill_wave[..., None])

    albedo = np.where(is_border[..., None], border_alb, boucle_alb)

    # Subtle 3D yarn-loop micro-normal perturbation + runner wool-pile compression normal
    nx_wool = (dt1_dx * 0.026 + dt2_dx * 0.022) * dist_fade + nx_trough
    nz_wool = (dt1_dz * 0.026 + dt2_dz * 0.022) * dist_fade + nz_trough
    nx_loop = np.where(is_border, (twill_wave - 0.25) * 0.04, nx_wool)
    nz_loop = np.where(is_border, -(twill_wave - 0.25) * 0.04, nz_wool)
    n_out = normalize(normal + np.stack([nx_loop, np.zeros_like(nx_loop), nz_loop], axis=-1))
    roughness = np.where(is_border, 0.68, np.clip(0.78 + 0.10 * (1.0 - loop_dome), 0.72, 0.90))
    return albedo, roughness, n_out


def material_american_walnut(
    p_world: np.ndarray, normal: np.ndarray, fast_mode: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quarter-sawn satin American walnut casework & coffee table plinths:
    - Smooth, cohesive warm satin walnut base [0.195, 0.108, 0.058] with fine ±5-7%
      micro-fiber variation (zero high-contrast periodic Z-sine bands).
    - Physically oriented grain: horizontal along board length (Z on bookcase shelves,
      X on credenza rails) and vertical along Y on stiles, back panels, and doors.
    """
    px, py, pz = p_world[..., 0], p_world[..., 1], p_world[..., 2]
    shelf_centers = BOOKCASE_SHELF_Y + 0.021
    dist_shelf_y = np.min(np.abs(py[..., None] - shelf_centers[None, :]), axis=-1)

    on_bookcase_shelf = (px < -2.65) & (
        (dist_shelf_y <= 0.035) | (py < 0.125) | (py > 4.00) | (np.abs(normal[..., 1]) > 0.5)
    )
    on_horiz_cred = (px >= -2.65) & (
        (np.abs(normal[..., 1]) > 0.5) | (py > 0.472) | (py < 0.104)
    )

    u_along = np.where(
        on_bookcase_shelf,
        pz * 0.35,
        np.where(on_horiz_cred, px * 0.35, py * 0.35),
    )
    v_across = np.where(
        on_bookcase_shelf,
        py * 45.0 + px * 14.0,
        np.where(
            on_horiz_cred,
            py * 45.0 + pz * 14.0,
            np.where(px < -2.65, pz * 38.0 + px * 16.0, px * 38.0 + pz * 16.0),
        ),
    )

    w_macro = (
        value_noise_2d(u_along * 0.8, v_across * 0.06, seed=809)
        if fast_mode
        else fbm2d(u_along * 0.8, v_across * 0.06, octaves=2, seed=809)
    )
    fine_fiber = (
        np.full_like(u_along, 0.5)
        if fast_mode
        else value_noise_2d(u_along * 2.2, v_across * 1.0, seed=821)
    )
    micro_pore = (
        np.full_like(u_along, 0.5)
        if fast_mode
        else value_noise_2d(u_along * 6.5, v_across * 2.8, seed=839)
    )

    grain_mod = (
        1.0
        + 0.055 * (w_macro - 0.5)
        + 0.045 * (fine_fiber - 0.5)
        + 0.025 * (micro_pore - 0.5)
    )
    walnut_base = np.array([0.195, 0.108, 0.058], dtype=np.float64)
    walnut_warm_tint = np.array([0.014, 0.007, 0.003], dtype=np.float64)
    albedo = (
        walnut_base[None, :] * grain_mod[..., None]
        + walnut_warm_tint[None, :] * (fine_fiber - 0.5)[..., None]
    )
    roughness = np.clip(
        0.22 - 0.015 * (fine_fiber - 0.5) - 0.010 * (micro_pore - 0.5),
        0.19,
        0.25,
    )
    n_out = (
        normal
        if fast_mode
        else normalize(
            normal
            + np.stack(
                [
                    (micro_pore - 0.5) * 0.008,
                    (fine_fiber - 0.5) * 0.008,
                    np.zeros_like(fine_fiber),
                ],
                axis=-1,
            )
        )
    )
    return albedo, roughness, n_out


def material_cognac_saddle_leather(
    p_world: np.ndarray,
    normal: np.ndarray,
    uv_local: np.ndarray,
    fast_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-grain vegetable-tanned aniline cognac saddle leather:
      - 3-channel stitched cushion tufting with recessed crease AO.
      - Authentic seam-gather tension folds & organic aniline hide creases.
      - Pull-up leather patina: warm amber-honey burnished crowns, rich cognac midtones,
        and deep chestnut-espresso crease valleys.
      - Subtle waxed-tan perimeter & channel saddle-stitching.
    """
    u_c = uv_local[:, 0]
    v_c = uv_local[:, 1]

    pore, dp_dx, dp_dz = value_noise_2d_grad(
        p_world[:, 0] * 125.0, p_world[:, 2] * 125.0 + p_world[:, 1] * 95.0, seed=901
    )
    patina = fbm2d(p_world[:, 0] * 7.5, p_world[:, 2] * 7.5 + p_world[:, 1] * 5.8, octaves=3, seed=907)
    crease_n, dc_dx, dc_dz = value_noise_2d_grad(
        u_c * 9.5 + patina * 1.8, v_c * 14.0 - patina * 1.5, seed=913
    )

    # 3-channel horizontal/vertical stitched tufting grooves + perimeter welt piping
    channel_dist = np.minimum(np.abs(v_c - (-0.32)), np.abs(v_c - 0.32))
    channel_groove = np.exp(-((channel_dist / 0.036) ** 2))
    edge_d = np.maximum(np.abs(u_c), np.abs(v_c))
    welt_groove = np.exp(-(((edge_d - 0.91) / 0.022) ** 2))
    welt_crown = np.exp(-(((edge_d - 0.955) / 0.024) ** 2))

    # Natural seam-gather tension folds emerging perpendicular to the tufting channels
    gather_env = np.exp(-(((channel_dist - 0.038) / 0.068) ** 2)) * (1.0 - smoothstep(0.76, 0.92, np.abs(u_c)))
    gather_wave = np.sin(u_c * 34.0 + patina * 5.2 + crease_n * 2.5) * gather_env
    wrinkle_valley = np.maximum(0.0, -gather_wave) * 0.65 + np.maximum(0.0, 0.48 - crease_n) * 0.45

    # Subtle waxed-tan linen saddle-stitch dashes right inside the channel seam and perimeter welt
    stitch_along_u = (np.sin(u_c * 165.0) > 0.35) & (np.abs(v_c) > 0.906) & (np.abs(v_c) < 0.918)
    stitch_along_v = (np.sin(v_c * 165.0) > 0.35) & (np.abs(u_c) > 0.906) & (np.abs(u_c) < 0.918)
    stitch_chan = (np.sin(u_c * 165.0) > 0.38) & (channel_dist < 0.0055)
    is_stitch = (stitch_along_u | stitch_along_v | stitch_chan) & (not fast_mode)

    # Pull-up aniline cognac leather palette
    cog_espresso = np.array([0.15, 0.048, 0.016], dtype=np.float64)
    cog_cognac = np.array([0.33, 0.118, 0.038], dtype=np.float64)
    cog_pullup = np.array([0.50, 0.205, 0.068], dtype=np.float64)
    stitch_tan = np.array([0.46, 0.25, 0.11], dtype=np.float64)

    panel_crown = (1.0 - smoothstep(0.0, 0.26, np.abs(np.abs(v_c) - 0.62))) + (1.0 - smoothstep(0.0, 0.24, np.abs(v_c)))
    crown_stretch = np.clip(0.55 * panel_crown * (1.0 - channel_groove) + 0.42 * welt_crown, 0.0, 1.0)
    valley_darken = np.clip(0.58 * channel_groove + 0.48 * welt_groove + 0.32 * wrinkle_valley, 0.0, 0.85)

    l_alb = (
        cog_cognac[None, :] * (0.62 + 0.38 * patina[:, None])
        + (cog_pullup - cog_cognac)[None, :] * (crown_stretch * (0.38 + 0.32 * patina))[:, None]
    )
    l_alb = l_alb * (1.0 - valley_darken[:, None]) + cog_espresso[None, :] * (valley_darken * 0.68)[:, None]
    l_alb *= (0.92 + 0.14 * pore)[:, None]
    l_alb = np.where(is_stitch[:, None], stitch_tan[None, :], l_alb)

    # 3D normal perturbation from tufting channels, seam-gather folds, and aniline hide pores
    chan_slope = np.sign(v_c - np.where(v_c >= 0.0, 0.32, -0.32)) * channel_groove * 0.28
    gather_slope = np.cos(u_c * 34.0 + patina * 5.2 + crease_n * 2.5) * gather_env * 0.09
    n_out = normalize(
        normal
        + np.stack(
            [
                dp_dx * 0.020 + dc_dx * 0.032 + gather_slope,
                chan_slope,
                dp_dz * 0.020 + dc_dz * 0.032 - chan_slope * 0.55,
            ],
            axis=-1,
        )
    )
    roughness = np.clip(
        0.21 + 0.10 * (1.0 - patina) - 0.05 * crown_stretch + 0.14 * valley_darken + 0.05 * (1.0 - pore),
        0.15,
        0.42,
    )
    return l_alb, roughness, n_out



# ==============================================================================
# 10. FULL 3D ROOM & FURNITURE RAY-TRACING ENGINE (WITH BOUNDING-SPHERE CULLING)
# ==============================================================================


def trace_room_and_furniture(
    ro: np.ndarray,
    rd: np.ndarray,
    catalog: WalnutLibraryCatalog,
    include_glass_vessels: bool = True,
    fast_secondary: bool = False,
    include_casework: bool = True,
    include_coffee_table: bool = True,
    include_chair_and_rug: bool = True,
) -> dict[str, np.ndarray]:
    """
    Trace primary or secondary rays `ro + t * rd` against the complete 3D
    architectural interior, 3D window mullions, raised bouclé wool rug, walnut
    library (vertical/leaning books, horizontal folio stacks, bookends, ribbed
    stoneware vases), back credenza vignette (leaning framed bas-relief artwork,
    folio stack, fluted travertine vessels), sculpted cognac leather chair
    (tufted cushions, seamless headrest bolster, wrapped armrests), and foreground
    coffee table still life (connected eucalyptus branch in stoneware vase, ribbed
    carafe, crystal tumbler on brass coaster, open monograph with silk ribbon,
    stoneware espresso cup, saucer & brass demitasse spoon, and stacked architecture
    monographs).
    """
    orig_shape = rd.shape[:-1]
    rd_flat = rd.reshape(-1, 3)
    ro_flat = np.broadcast_to(ro, rd.shape).reshape(-1, 3)
    n_rays = rd_flat.shape[0]

    t_best = np.full(n_rays, 1e9, dtype=np.float64)
    n_best = np.zeros((n_rays, 3), dtype=np.float64)
    mat_best = np.zeros(n_rays, dtype=np.int32)
    alb_best = np.zeros((n_rays, 3), dtype=np.float64)
    rough_best = np.full(n_rays, 0.5, dtype=np.float64)
    f0_best = np.full((n_rays, 3), 0.04, dtype=np.float64)
    metal_best = np.zeros(n_rays, dtype=np.float64)
    uv_best = np.zeros((n_rays, 2), dtype=np.float64)

    inv_dx = 1.0 / np.where(np.abs(rd_flat[:, 0]) < 1e-12, np.copysign(1e-12, rd_flat[:, 0]), rd_flat[:, 0])
    inv_dy = 1.0 / np.where(np.abs(rd_flat[:, 1]) < 1e-12, np.copysign(1e-12, rd_flat[:, 1]), rd_flat[:, 1])
    inv_dz = 1.0 / np.where(np.abs(rd_flat[:, 2]) < 1e-12, np.copysign(1e-12, rd_flat[:, 2]), rd_flat[:, 2])

    # ------------------------------------------------------------------
    # A. 5 Architectural Room Boundary Planes & 3D Window Mullions
    # ------------------------------------------------------------------
    t_floor = (ROOM_Y_FLOOR - ro_flat[:, 1]) * inv_dy
    hit_floor = (rd_flat[:, 1] < -1e-6) & (t_floor > 1e-4) & (t_floor < t_best)
    t_best = np.where(hit_floor, t_floor, t_best)
    n_best[hit_floor] = np.array([0.0, 1.0, 0.0])
    mat_best = np.where(hit_floor, MAT_FLOOR_OAK, mat_best)

    t_ceil = (ROOM_Y_CEIL - ro_flat[:, 1]) * inv_dy
    hit_ceil = (rd_flat[:, 1] > 1e-6) & (t_ceil > 1e-4) & (t_ceil < t_best)
    t_best = np.where(hit_ceil, t_ceil, t_best)
    n_best[hit_ceil] = np.array([0.0, -1.0, 0.0])
    mat_best = np.where(hit_ceil, MAT_CEILING_CONCRETE, mat_best)

    # 3D Coffered Architectural Concrete Ceiling Beams
    if not fast_secondary:
        mask_up = (rd_flat[:, 1] > 0.01) & (ro_flat[:, 1] < ROOM_Y_CEIL)
        idx_up = np.where(mask_up)[0]
        if idx_up.size > 0:
            ro_u = ro_flat[idx_up]
            rd_u = rd_flat[idx_up]
            tb_u = t_best[idx_up]
            nb_u = n_best[idx_up]
            mb_u = mat_best[idx_up]
            for beam_z in (1.65, 3.55, 5.50):
                beam_c = np.array([0.25, 4.09, beam_z], dtype=np.float64)
                beam_ext = np.array([2.95, 0.16, 0.095], dtype=np.float64)
                t_bm, hit_bm, n_bm = intersect_rounded_box(ro_u, rd_u, beam_c, beam_ext, bevel_r=0.008)
                take_bm = hit_bm & (t_bm < tb_u)
                tb_u = np.where(take_bm, t_bm, tb_u)
                nb_u = np.where(take_bm[:, None], n_bm, nb_u)
                mb_u = np.where(take_bm, MAT_CEILING_CONCRETE, mb_u)
            t_best[idx_up] = tb_u
            n_best[idx_up] = nb_u
            mat_best[idx_up] = mb_u

    # 3. Back Wall Plane (Z = 7.60) with EPDM Glazing Gaskets
    t_back = (ROOM_Z_BACK - ro_flat[:, 2]) * inv_dz
    hit_back = (rd_flat[:, 2] > 1e-6) & (t_back > 1e-4) & (t_back < t_best)
    if np.any(hit_back):
        idx_bk = np.where(hit_back)[0]
        bx = ro_flat[idx_bk, 0] + t_back[idx_bk] * rd_flat[idx_bk, 0]
        by = ro_flat[idx_bk, 1] + t_back[idx_bk] * rd_flat[idx_bk, 1]
        t_best[idx_bk] = t_back[idx_bk]
        n_best[idx_bk] = np.array([0.0, 0.0, -1.0])

        is_back_win = (bx > BACK_CONCRETE_X_MAX) & (bx < ROOM_X_RIGHT) & (by > 0.09) & (by < 3.88)
        dist_bmull = np.min(np.abs(bx[:, None] - BACK_MULLION_X), axis=-1)
        dist_btrans = np.min(np.abs(by[:, None] - RIGHT_TRANSOM_Y), axis=-1)
        is_back_mull = is_back_win & ((dist_bmull < 0.038) | (dist_btrans < 0.030))
        is_back_gasket = is_back_win & (~is_back_mull) & ((dist_bmull < 0.048) | (dist_btrans < 0.039))
        mat_best[idx_bk] = np.where(
            is_back_mull,
            MAT_STEEL_MULLION,
            np.where(is_back_gasket, MAT_GLAZING_GASKET, np.where(is_back_win, MAT_WINDOW_GLASS, MAT_BACK_CONCRETE)),
        )

    # 4. Right Window Curtain Wall Plane (X = +3.20) with EPDM Glazing Gaskets
    t_right = (ROOM_X_RIGHT - ro_flat[:, 0]) * inv_dx
    hit_right = (rd_flat[:, 0] > 1e-6) & (t_right > 1e-4) & (t_right < t_best)
    if np.any(hit_right):
        idx_rt = np.where(hit_right)[0]
        ry = ro_flat[idx_rt, 1] + t_right[idx_rt] * rd_flat[idx_rt, 1]
        rz = ro_flat[idx_rt, 2] + t_right[idx_rt] * rd_flat[idx_rt, 2]
        t_best[idx_rt] = t_right[idx_rt]
        n_best[idx_rt] = np.array([-1.0, 0.0, 0.0])

        is_right_open = (ry > 0.09) & (ry < 3.88) & (rz > -0.20) & (rz < 7.52)
        dist_rmull = np.min(np.abs(rz[:, None] - RIGHT_MULLION_Z), axis=-1)
        dist_rtrans = np.min(np.abs(ry[:, None] - RIGHT_TRANSOM_Y), axis=-1)
        is_right_mull = is_right_open & ((dist_rmull < 0.042) | (dist_rtrans < 0.032))
        is_right_gasket = is_right_open & (~is_right_mull) & ((dist_rmull < 0.052) | (dist_rtrans < 0.041))
        mat_best[idx_rt] = np.where(
            is_right_mull,
            MAT_STEEL_MULLION,
            np.where(
                is_right_gasket, MAT_GLAZING_GASKET, np.where(is_right_open, MAT_WINDOW_GLASS, MAT_BACK_CONCRETE)
            ),
        )

    # 4b. True 3D Extruded Blackened-Steel Window Mullions & Transoms
    if not fast_secondary:
        mask_win3d = (rd_flat[:, 0] > 0.02) | ((rd_flat[:, 2] > 0.05) & (ro_flat[:, 0] + t_best * rd_flat[:, 0] > 0.95))
        idx_w3 = np.where(mask_win3d)[0]
        if idx_w3.size > 0:
            ro_w3 = ro_flat[idx_w3]
            rd_w3 = rd_flat[idx_w3]
            tb_w3 = t_best[idx_w3]
            nb_w3 = n_best[idx_w3]
            mb_w3 = mat_best[idx_w3]

            # Right wall 3D vertical steel box mullions
            for mz in RIGHT_MULLION_Z:
                mc = np.array([ROOM_X_RIGHT - 0.038, 1.985, float(mz)], dtype=np.float64)
                mext = np.array([0.038, 1.90, 0.040], dtype=np.float64)
                t_m, hit_m, n_m = intersect_rounded_box(ro_w3, rd_w3, mc, mext, bevel_r=0.004)
                take_m = hit_m & (t_m < tb_w3)
                tb_w3 = np.where(take_m, t_m, tb_w3)
                nb_w3 = np.where(take_m[:, None], n_m, nb_w3)
                mb_w3 = np.where(take_m, MAT_STEEL_MULLION, mb_w3)

            # Right wall 3D horizontal steel transoms
            for ty in RIGHT_TRANSOM_Y:
                tc = np.array([ROOM_X_RIGHT - 0.034, float(ty), 3.66], dtype=np.float64)
                text = np.array([0.034, 0.030, 3.88], dtype=np.float64)
                t_tr, hit_tr, n_tr = intersect_rounded_box(ro_w3, rd_w3, tc, text, bevel_r=0.003)
                take_tr = hit_tr & (t_tr < tb_w3)
                tb_w3 = np.where(take_tr, t_tr, tb_w3)
                nb_w3 = np.where(take_tr[:, None], n_tr, nb_w3)
                mb_w3 = np.where(take_tr, MAT_STEEL_MULLION, mb_w3)

            # Back corner window 3D vertical steel mullions & horizontal transoms
            for mx in BACK_MULLION_X:
                mc_b = np.array([float(mx), 1.985, ROOM_Z_BACK - 0.038], dtype=np.float64)
                mext_b = np.array([0.036, 1.90, 0.038], dtype=np.float64)
                t_mb, hit_mb, n_mb = intersect_rounded_box(ro_w3, rd_w3, mc_b, mext_b, bevel_r=0.004)
                take_mb = hit_mb & (t_mb < tb_w3)
                tb_w3 = np.where(take_mb, t_mb, tb_w3)
                nb_w3 = np.where(take_mb[:, None], n_mb, nb_w3)
                mb_w3 = np.where(take_mb, MAT_STEEL_MULLION, mb_w3)

            for ty in RIGHT_TRANSOM_Y:
                tc_b = np.array([2.14, float(ty), ROOM_Z_BACK - 0.034], dtype=np.float64)
                text_b = np.array([1.06, 0.029, 0.034], dtype=np.float64)
                t_tb2, hit_tb2, n_tb2 = intersect_rounded_box(ro_w3, rd_w3, tc_b, text_b, bevel_r=0.003)
                take_tb2 = hit_tb2 & (t_tb2 < tb_w3)
                tb_w3 = np.where(take_tb2, t_tb2, tb_w3)
                nb_w3 = np.where(take_tb2[:, None], n_tb2, nb_w3)
                mb_w3 = np.where(take_tb2, MAT_STEEL_MULLION, mb_w3)

            t_best[idx_w3] = tb_w3
            n_best[idx_w3] = nb_w3
            mat_best[idx_w3] = mb_w3

    # 5. Left Walnut Library Backboard Plane (X = -3.10)
    t_left_back = (ROOM_X_LEFT - ro_flat[:, 0]) * inv_dx
    hit_left_back = (rd_flat[:, 0] < -1e-6) & (t_left_back > 1e-4) & (t_left_back < t_best)
    t_best = np.where(hit_left_back, t_left_back, t_best)
    n_best[hit_left_back] = np.array([1.0, 0.0, 0.0])
    mat_best = np.where(hit_left_back, MAT_WALNUT_CASEWORK, mat_best)

    # 6. 3D Raised Hand-Loomed Belgian Wool Bouclé Area Rug (Y in [0.0, 0.013])
    # Leaves a generous Honed Roman Travertine perimeter border (X in [-1.92, 1.58], Z in [0.42, 4.45])
    # and Wide-Plank Oiled European Oak visible all around the seating area.
    mask_rug_cand = (rd_flat[:, 1] < -1e-4) & (ro_flat[:, 1] > 0.01)
    idx_rg = np.where(mask_rug_cand)[0]
    if include_chair_and_rug and idx_rg.size > 0:
        ro_r = ro_flat[idx_rg]
        rd_r = rd_flat[idx_rg]
        tb_r = t_best[idx_rg]
        nb_r = n_best[idx_rg]
        mb_r = mat_best[idx_rg]

        rug_c = np.array([-0.42, 0.0065, 2.02], dtype=np.float64)
        rug_ext = np.array([1.46, 0.0065, 1.08], dtype=np.float64)
        t_rug, hit_rug, n_rug = intersect_rounded_box(ro_r, rd_r, rug_c, rug_ext, bevel_r=0.0045)
        take_rug = hit_rug & (t_rug < tb_r)
        if np.any(take_rug):
            p_rug = ro_r + t_rug[:, None] * rd_r
            dx_edge = rug_ext[0] - np.abs(p_rug[:, 0] - rug_c[0])
            dz_edge = rug_ext[2] - np.abs(p_rug[:, 2] - rug_c[2])
            on_linen_border = np.minimum(dx_edge, dz_edge) < 0.036
            tb_r = np.where(take_rug, t_rug, tb_r)
            nb_r = np.where(take_rug[:, None], n_rug, nb_r)
            mb_r = np.where(
                take_rug,
                np.where(on_linen_border, MAT_RUG_LINEN_BORDER, MAT_BOUCLE_RUG),
                mb_r,
            )
            t_best[idx_rg] = tb_r
            n_best[idx_rg] = nb_r
            mat_best[idx_rg] = mb_r

    # ------------------------------------------------------------------
    # B. Left Wall 3D Walnut Casework, Shelves, Stiles, Vertical & Leaning
    #    Books, Horizontal Folio Stacks, Bookends & Ribbed Stoneware Vases
    # ------------------------------------------------------------------
    t_case = (catalog.case_front_x - ro_flat[:, 0]) * inv_dx
    can_hit_left = (rd_flat[:, 0] < -1e-6) & (t_case > 1e-4) & (t_case < t_best + 0.4)
    idx_left = np.where(can_hit_left)[0]
    if include_casework and idx_left.size > 0:
        ro_l = ro_flat[idx_left]
        rd_l = rd_flat[idx_left]
        tc_l = t_case[idx_left]
        tb_l = t_best[idx_left]
        nb_l = n_best[idx_left]
        mb_l = mat_best[idx_left]
        ab_l = alb_best[idx_left]
        rb_l = rough_best[idx_left]

        cy_hit = ro_l[:, 1] + tc_l * rd_l[:, 1]
        cz_hit = ro_l[:, 2] + tc_l * rd_l[:, 2]
        in_case_bounds = (cy_hit >= 0.0) & (cy_hit <= ROOM_Y_CEIL) & (cz_hit >= -0.85) & (cz_hit <= ROOM_Z_BACK)

        dist_shelf_face = np.min(np.abs(cy_hit[:, None] - (catalog.shelf_y + 0.5 * catalog.shelf_thick)), axis=-1)
        on_shelf_front = (
            (dist_shelf_face <= 0.5 * catalog.shelf_thick)
            | (cy_hit < 0.08 + catalog.shelf_thick)
            | (cy_hit > 4.02)
        )
        dist_stile_face = np.min(np.abs(cz_hit[:, None] - catalog.stile_z), axis=-1)
        on_stile_front = dist_stile_face <= 0.5 * catalog.stile_thick

        hit_frame_front = in_case_bounds & (on_shelf_front | on_stile_front) & (tc_l < tb_l)
        tb_l = np.where(hit_frame_front, tc_l, tb_l)
        nb_l[hit_frame_front] = np.array([1.0, 0.0, 0.0])
        mb_l = np.where(hit_frame_front, MAT_WALNUT_CASEWORK, mb_l)

        in_bay_alcove = in_case_bounds & (~(on_shelf_front | on_stile_front))
        idx_alc = np.where(in_bay_alcove)[0]
        if idx_alc.size > 0:
            ro_a = ro_l[idx_alc]
            rd_a = rd_l[idx_alc]
            tc_a = tc_l[idx_alc]
            tb_a = tb_l[idx_alc]
            nb_a = nb_l[idx_alc]
            mb_a = mb_l[idx_alc]
            ab_a = ab_l[idx_alc]
            rb_a = rb_l[idx_alc]

            inv_dx_a = 1.0 / np.where(np.abs(rd_a[:, 0]) < 1e-12, np.copysign(1e-12, rd_a[:, 0]), rd_a[:, 0])
            inv_dy_a = 1.0 / np.where(np.abs(rd_a[:, 1]) < 1e-12, np.copysign(1e-12, rd_a[:, 1]), rd_a[:, 1])
            inv_dz_a = 1.0 / np.where(np.abs(rd_a[:, 2]) < 1e-12, np.copysign(1e-12, rd_a[:, 2]), rd_a[:, 2])

            if not fast_secondary:
                for sy in catalog.shelf_y:
                    y_top = float(sy + catalog.shelf_thick)
                    t_sh_top = (y_top - ro_a[:, 1]) * inv_dy_a
                    hx = ro_a[:, 0] + t_sh_top * rd_a[:, 0]
                    hz = ro_a[:, 2] + t_sh_top * rd_a[:, 2]
                    hit_sh_top = (
                        (rd_a[:, 1] < -1e-6)
                        & (t_sh_top > tc_a)
                        & (t_sh_top < tb_a)
                        & (hx >= ROOM_X_LEFT)
                        & (hx <= catalog.case_front_x)
                        & (hz >= -0.85)
                        & (hz <= ROOM_Z_BACK)
                    )
                    tb_a = np.where(hit_sh_top, t_sh_top, tb_a)
                    nb_a[hit_sh_top] = np.array([0.0, 1.0, 0.0])
                    mb_a = np.where(hit_sh_top, MAT_WALNUT_CASEWORK, mb_a)

                    t_sh_bot = (float(sy) - ro_a[:, 1]) * inv_dy_a
                    hx_b = ro_a[:, 0] + t_sh_bot * rd_a[:, 0]
                    hz_b = ro_a[:, 2] + t_sh_bot * rd_a[:, 2]
                    hit_sh_bot = (
                        (rd_a[:, 1] > 1e-6)
                        & (t_sh_bot > tc_a)
                        & (t_sh_bot < tb_a)
                        & (hx_b >= ROOM_X_LEFT)
                        & (hx_b <= catalog.case_front_x)
                        & (hz_b >= -0.85)
                        & (hz_b <= ROOM_Z_BACK)
                    )
                    tb_a = np.where(hit_sh_bot, t_sh_bot, tb_a)
                    nb_a[hit_sh_bot] = np.array([0.0, -1.0, 0.0])
                    mb_a = np.where(hit_sh_bot, MAT_WALNUT_CASEWORK, mb_a)

                for sz in catalog.stile_z:
                    for side_sgn in (-1.0, 1.0):
                        z_plane = float(sz + side_sgn * 0.5 * catalog.stile_thick)
                        t_st = (z_plane - ro_a[:, 2]) * inv_dz_a
                        hx_s = ro_a[:, 0] + t_st * rd_a[:, 0]
                        hy_s = ro_a[:, 1] + t_st * rd_a[:, 1]
                        hit_st = (
                            (rd_a[:, 2] * side_sgn < -1e-6)
                            & (t_st > tc_a)
                            & (t_st < tb_a)
                            & (hx_s >= ROOM_X_LEFT)
                            & (hx_s <= catalog.case_front_x)
                            & (hy_s >= 0.0)
                            & (hy_s <= ROOM_Y_CEIL)
                        )
                        tb_a = np.where(hit_st, t_st, tb_a)
                        nb_a[hit_st] = np.array([0.0, 0.0, side_sgn])
                        mb_a = np.where(hit_st, MAT_WALNUT_CASEWORK, mb_a)

            # Recessed Dark American Walnut Bookcase Backboard (X = -2.93)
            t_backboard = (-2.93 - ro_a[:, 0]) * inv_dx_a
            hy_bb = ro_a[:, 1] + t_backboard * rd_a[:, 1]
            hz_bb = ro_a[:, 2] + t_backboard * rd_a[:, 2]
            hit_bb = (
                (t_backboard > tc_a)
                & (t_backboard < tb_a)
                & (hy_bb >= 0.0)
                & (hy_bb <= ROOM_Y_CEIL)
                & (hz_bb >= -0.85)
                & (hz_bb <= ROOM_Z_BACK)
            )
            tb_a = np.where(hit_bb, t_backboard, tb_a)
            nb_a[hit_bb] = np.array([1.0, 0.0, 0.0])
            mb_a = np.where(hit_bb, MAT_WALNUT_CASEWORK, mb_a)

            # Intersect Vertical Hardcover Books via exact 3D AABB book bounds
            # [x_back=-2.93, bx_front] x [shelf_base_y, shelf_base_y + b_h] x [bz0, bz1]
            inv_d_a = np.stack([inv_dx_a, inv_dy_a, inv_dz_a], axis=-1)
            page_col = np.array([0.52, 0.47, 0.39], dtype=np.float64)
            label_col = np.array([0.58, 0.52, 0.43], dtype=np.float64)

            x_probes_full = (-2.758, -2.776, -2.796, -2.820, -2.852, -2.888)
            x_probes_fast = (-2.765, -2.795, -2.845)
            b_offsets_full = (0, 6, -6, 12, -12)
            b_offsets_fast = (0, 8)
            for x_probe in (x_probes_fast if fast_secondary else x_probes_full):
                t_probe = (x_probe - ro_a[:, 0]) * inv_dx_a
                py_probe = ro_a[:, 1] + t_probe * rd_a[:, 1]
                pz_probe = ro_a[:, 2] + t_probe * rd_a[:, 2]

                tier_idx = np.searchsorted(catalog.shelf_y, py_probe) - 1
                valid_tier = (tier_idx >= 0) & (tier_idx < catalog.n_tiers)
                t_c, b_base = catalog.query_book_bins(tier_idx, pz_probe)

                for b_offset in (b_offsets_fast if fast_secondary else b_offsets_full):
                    b_cand = np.clip(b_base + b_offset, 0, catalog.n_bins - 1)
                    present = valid_tier & catalog.book_present[t_c, b_cand]
                    if not np.any(present):
                        continue

                    shelf_base_y = catalog.shelf_y[t_c] + catalog.shelf_thick
                    b_h = catalog.book_height[t_c, b_cand]
                    bx_front = catalog.book_front_x[t_c, b_cand]
                    bz0 = catalog.book_z0[t_c, b_cand]
                    bz1 = catalog.book_z1[t_c, b_cand]

                    box_min = np.stack([np.full_like(bx_front, -2.93), shelf_base_y, bz0], axis=-1)
                    box_max = np.stack([bx_front, shelf_base_y + b_h, bz1], axis=-1)

                    t0s = (box_min - ro_a) * inv_d_a
                    t1s = (box_max - ro_a) * inv_d_a
                    tmin_v = np.minimum(t0s, t1s)
                    tmax_v = np.maximum(t0s, t1s)
                    t_enter = np.max(tmin_v, axis=-1)
                    t_exit = np.min(tmax_v, axis=-1)

                    hit_box = (
                        present
                        & (t_exit >= np.maximum(t_enter, 1e-4))
                        & (t_enter > np.maximum(tc_a, 1e-4))
                        & (t_enter < tb_a)
                    )
                    if not np.any(hit_box):
                        continue

                    p_hit = ro_a + t_enter[:, None] * rd_a
                    rel_y = np.clip(p_hit[:, 1] - shelf_base_y, 0.0, b_h)
                    u_sp = np.clip(
                        (p_hit[:, 2] - 0.5 * (bz0 + bz1)) / np.maximum(0.5 * (bz1 - bz0), 1e-4),
                        -1.0,
                        1.0,
                    )

                    hit_x_face = (tmin_v[:, 0] >= tmin_v[:, 1] - 1e-6) & (tmin_v[:, 0] >= tmin_v[:, 2] - 1e-6)
                    hit_y_face = (~hit_x_face) & (tmin_v[:, 1] >= tmin_v[:, 2] - 1e-6)

                    # 1. Spine face (+X) normal, French-groove shoulders, raised bindery hubs, foil bands & heel label
                    french_groove = 0.24 * np.sin((u_sp / 0.80) * PI) * np.exp(-((np.abs(u_sp) - 0.78) / 0.065) ** 2)
                    nz_sp = 0.54 * u_sp + 0.18 * (u_sp ** 3) + french_groove
                    ny_sp = np.where(rel_y > b_h - 0.014, 0.42, np.where(rel_y < 0.014, -0.42, 0.0))
                    has_hubs = catalog.book_has_hubs[t_c, b_cand]
                    y_norm_b = rel_y / np.maximum(b_h, 0.05)
                    hub_dist = np.minimum(
                        np.minimum(np.abs(y_norm_b - 0.18), np.abs(y_norm_b - 0.36)),
                        np.minimum(np.abs(y_norm_b - 0.62), np.abs(y_norm_b - 0.84)),
                    )
                    hub_ny = np.where(
                        has_hubs & (hub_dist < 0.020),
                        np.sin((y_norm_b * 55.0) * PI) * 0.34,
                        0.0,
                    )
                    ny_sp = ny_sp + hub_ny
                    nx_sp = np.sqrt(np.maximum(0.08, 1.0 - nz_sp * nz_sp - ny_sp * ny_sp))
                    n_spine = normalize(np.stack([nx_sp, ny_sp, nz_sp], axis=-1))

                    # Deep inter-book V-groove crevice shadow + 3D neighbor depth-step & shelf overhang AO
                    b_col = catalog.book_color[t_c, b_cand]
                    v_groove_ao = 1.0 - 0.64 * np.exp(-((1.0 - np.abs(u_sp)) / 0.115) ** 2)
                    hinge_crease = 1.0 - 0.28 * np.exp(-((np.abs(u_sp) - 0.78) / 0.055) ** 2)
                    depth_recess_ao = 0.58 + 0.42 * smoothstep(-2.820, -2.752, bx_front)

                    t_next = np.minimum(t_c + 1, catalog.n_tiers)
                    upper_shelf_y = catalog.shelf_y[t_next]
                    dist_to_upper_shelf = np.maximum(0.0, upper_shelf_y - p_hit[:, 1])
                    shelf_overhang_ao = 0.46 + 0.54 * smoothstep(0.01, 0.18, dist_to_upper_shelf)
                    bottom_contact_ao = 0.62 + 0.38 * smoothstep(0.0, 0.025, rel_y)

                    crease = v_groove_ao * hinge_crease * depth_recess_ao * shelf_overhang_ao * bottom_contact_ao

                    in_band = (
                        (rel_y >= catalog.book_band_y0[t_c, b_cand])
                        & (rel_y <= catalog.book_band_y1[t_c, b_cand])
                        & (np.abs(u_sp) < 0.68)
                    )
                    in_fillet = (
                        ((np.abs(y_norm_b - 0.88) < 0.007) | (np.abs(y_norm_b - 0.10) < 0.007))
                        & (np.abs(u_sp) < 0.72)
                        & has_hubs
                    )
                    in_label = (
                        catalog.book_has_label[t_c, b_cand]
                        & (rel_y >= 0.020)
                        & (rel_y <= 0.052)
                        & (np.abs(u_sp) < 0.56)
                    )
                    band_col = catalog.book_band_col[t_c, b_cand] * (0.68 + 0.32 * depth_recess_ao[:, None])
                    spine_alb = np.where(
                        in_label[:, None],
                        label_col[None, :] * (0.72 + 0.28 * depth_recess_ao[:, None]),
                        np.where(
                            (in_band | in_fillet)[:, None],
                            band_col,
                            b_col * crease[:, None],
                        ),
                    )
                    spine_rough = np.where(in_band | in_fillet, 0.16, catalog.book_rough[t_c, b_cand])

                    # 2. Top page block (+Y) with fine paper-leaf striations and 3. Side cover board (-Z / +Z)
                    n_y_face = np.stack(
                        [np.zeros_like(t_enter), -np.sign(rd_a[:, 1]), np.zeros_like(t_enter)],
                        axis=-1,
                    )
                    n_z_face = normalize(
                        np.stack(
                            [np.full_like(t_enter, 0.78), np.zeros_like(t_enter), -np.sign(rd_a[:, 2]) * 0.62],
                            axis=-1,
                        )
                    )
                    on_top_cover_rim = (np.abs(u_sp) > 0.84) | (p_hit[:, 0] > bx_front - 0.006)
                    page_striation = 0.86 + 0.14 * np.sin(p_hit[:, 2] * 920.0 + p_hit[:, 0] * 140.0)
                    top_depth_ao = 0.44 + 0.56 * smoothstep(-2.90, -2.760, p_hit[:, 0])
                    top_alb = np.where(
                        on_top_cover_rim[:, None],
                        b_col * (0.82 * top_depth_ao)[:, None],
                        page_col[None, :] * (page_striation * top_depth_ao)[:, None],
                    )
                    side_depth_ao = 0.34 + 0.60 * smoothstep(-2.90, -2.755, p_hit[:, 0])
                    side_alb = b_col * (side_depth_ao * shelf_overhang_ao)[:, None]

                    n_book_hit = np.where(
                        hit_x_face[:, None],
                        n_spine,
                        np.where(hit_y_face[:, None], n_y_face, n_z_face),
                    )
                    m_book_hit = np.where(
                        hit_y_face & (~on_top_cover_rim),
                        MAT_BOOK_PAGES,
                        MAT_BOOK_SPINE,
                    )
                    alb_book_hit = np.where(
                        hit_x_face[:, None],
                        spine_alb,
                        np.where(hit_y_face[:, None], top_alb, side_alb),
                    )
                    rgh_book_hit = np.where(
                        hit_x_face,
                        spine_rough,
                        np.where(hit_y_face, 0.58, catalog.book_rough[t_c, b_cand]),
                    )

                    tb_a = np.where(hit_box, t_enter, tb_a)
                    nb_a = np.where(hit_box[:, None], n_book_hit, nb_a)
                    mb_a = np.where(hit_box, m_book_hit, mb_a)
                    ab_a = np.where(hit_box[:, None], alb_book_hit, ab_a)
                    rb_a = np.where(hit_box, rgh_book_hit, rb_a)

            if not fast_secondary:
                # Horizontal Stacks of Oversized Art Monographs on Shelves
                for h_stack in catalog.horizontal_stacks:
                    for lay in h_stack["layers"]:
                        lc = np.array(
                            [
                                0.5 * (-2.92 + float(lay["front_x"])),
                                0.5 * (float(lay["y0"]) + float(lay["y1"])),
                                0.5 * (float(lay["z0"]) + float(lay["z1"])),
                            ],
                            dtype=np.float64,
                        )
                        lext = np.array(
                            [
                                0.5 * (float(lay["front_x"]) - (-2.92)),
                                0.5 * (float(lay["y1"]) - float(lay["y0"])),
                                0.5 * (float(lay["z1"]) - float(lay["z0"])),
                            ],
                            dtype=np.float64,
                        )
                        t_hs, hit_hs, n_hs = intersect_rounded_box(ro_a, rd_a, lc, lext, bevel_r=0.003)
                        take_hs = hit_hs & (t_hs < tb_a)
                        if np.any(take_hs):
                            p_hs = ro_a + t_hs[:, None] * rd_a
                            rel_hs_y = (p_hs[:, 1] - float(lay["y0"])) / np.maximum(2.0 * lext[1], 1e-4)
                            is_page_edge = (np.abs(n_hs[:, 2]) > 0.6) & (rel_hs_y > 0.14) & (rel_hs_y < 0.86)
                            lay_col = np.where(
                                is_page_edge[:, None],
                                np.array([[0.72, 0.67, 0.58]], dtype=np.float64),
                                lay["color"][None, :],
                            )
                            tb_a = np.where(take_hs, t_hs, tb_a)
                            nb_a = np.where(take_hs[:, None], n_hs, nb_a)
                            mb_a = np.where(take_hs, MAT_BOOK_SPINE, mb_a)
                            ab_a = np.where(take_hs[:, None], lay_col, ab_a)
                            rb_a = np.where(take_hs, float(lay["rough"]), rb_a)

                # Tilted / Leaning Hardcover Books at Ends of Vertical Runs
                for lb_obj in catalog.leaning_books:
                    t_lb, hit_lb, n_lb, _ = intersect_obb_y(
                        ro_a,
                        rd_a,
                        lb_obj["center"],
                        lb_obj["ext"],
                        yaw_rad=0.0,
                        bevel_r=0.003,
                        tilt_x_rad=float(lb_obj["tilt_x"]),
                    )
                    take_lb = hit_lb & (t_lb < tb_a)
                    if np.any(take_lb):
                        tb_a = np.where(take_lb, t_lb, tb_a)
                        nb_a = np.where(take_lb[:, None], n_lb, nb_a)
                        mb_a = np.where(take_lb, MAT_BOOK_SPINE, mb_a)
                        ab_a = np.where(take_lb[:, None], lb_obj["color"][None, :], ab_a)
                        rb_a = np.where(take_lb, float(lb_obj["rough"]), rb_a)

                # Honed Travertine & Brass Architectural Bookends
                for be_obj in catalog.bookends:
                    t_be, hit_be, n_be = intersect_rounded_box(
                        ro_a, rd_a, be_obj["center"], be_obj["ext"], bevel_r=0.004
                    )
                    take_be = hit_be & (t_be < tb_a)
                    if np.any(take_be):
                        tb_a = np.where(take_be, t_be, tb_a)
                        nb_a = np.where(take_be[:, None], n_be, nb_a)
                        mb_a = np.where(take_be, MAT_BOOKEND_STONE, mb_a)
                        ab_a = np.where(take_be[:, None], np.array([[0.58, 0.52, 0.43]]), ab_a)
                        rb_a = np.where(take_be, 0.24, rb_a)

                # Wheel-Thrown Ribbed Stoneware Ceramic Vessels (Belly + Neck + Flared Lip)
                for v_obj in catalog.vessels:
                    vc = v_obj["center"]
                    vh = float(v_obj["height"])
                    vr = float(v_obj["radius"])
                    neck_r = float(v_obj.get("neck_r", vr * 0.44))
                    lip_r = float(v_obj.get("lip_r", vr * 0.54))

                    ell_c = vc + np.array([0.0, 0.43 * vh, 0.0], dtype=np.float64)
                    ell_r = np.array([vr, 0.43 * vh, vr], dtype=np.float64)
                    t_vb, hit_vb, n_vb = intersect_ellipsoid(ro_a, rd_a, ell_c, ell_r)

                    nk_a = vc + np.array([0.0, 0.72 * vh, 0.0], dtype=np.float64)
                    nk_b = vc + np.array([0.0, 0.96 * vh, 0.0], dtype=np.float64)
                    t_vn, hit_vn, n_vn = intersect_cylinder_segment(ro_a, rd_a, nk_a, nk_b, radius=neck_r, capped=False)

                    lp_c = vc + np.array([0.0, 0.96 * vh, 0.0], dtype=np.float64)
                    lp_r = np.array([lip_r, 0.04 * vh, lip_r], dtype=np.float64)
                    t_vl, hit_vl, n_vl = intersect_ellipsoid(ro_a, rd_a, lp_c, lp_r)

                    t_v = np.where(hit_vb, t_vb, 1e9)
                    n_v = n_vb.copy()
                    u_vn = hit_vn & (t_vn < t_v)
                    t_v = np.where(u_vn, t_vn, t_v)
                    n_v = np.where(u_vn[:, None], n_vn, n_v)
                    u_vl = hit_vl & (t_vl < t_v)
                    t_v = np.where(u_vl, t_vl, t_v)
                    n_v = np.where(u_vl[:, None], n_vl, n_v)

                    take_v = (t_v < tb_a) & (hit_vb | hit_vn | hit_vl)
                    if np.any(take_v):
                        p_vhit = ro_a + t_v[:, None] * rd_a
                        rib_ny = np.sin(p_vhit[:, 1] * 220.0) * 0.14
                        n_v_rib = normalize(
                            n_v + np.stack([np.zeros_like(rib_ny), rib_ny, np.zeros_like(rib_ny)], axis=-1)
                        )
                        speck = value_noise_2d(p_vhit[:, 1] * 140.0, p_vhit[:, 2] * 140.0, seed=841)
                        v_col = v_obj["color"][None, :] * (0.88 + 0.22 * speck[:, None])
                        tb_a = np.where(take_v, t_v, tb_a)
                        nb_a = np.where(take_v[:, None], n_v_rib, nb_a)
                        mb_a = np.where(take_v, MAT_CERAMIC_VESSEL, mb_a)
                        ab_a = np.where(take_v[:, None], v_col, ab_a)
                        rb_a = np.where(take_v, float(v_obj["rough"]), rb_a)

            tb_l[idx_alc] = tb_a
            nb_l[idx_alc] = nb_a
            mb_l[idx_alc] = mb_a
            ab_l[idx_alc] = ab_a
            rb_l[idx_alc] = rb_a

        t_best[idx_left] = tb_l
        n_best[idx_left] = nb_l
        mat_best[idx_left] = mb_l
        alb_best[idx_left] = ab_l
        rough_best[idx_left] = rb_l

    # ------------------------------------------------------------------
    # C. Architectural Brushed-Brass Wall Sconces (Left Library Pier &
    #    Two Prominent Back Concrete Wall Gallery / Cylinder Sconces)
    # ------------------------------------------------------------------
    mask_sc = (
        rays_hitting_sphere(
            ro_flat, rd_flat, np.array([-2.54, 2.28, 2.55], dtype=np.float64), 0.45, t_best
        )
        | rays_hitting_sphere(
            ro_flat, rd_flat, np.array([-0.66, 1.94, 7.38], dtype=np.float64), 1.35, t_best
        )
    )
    idx_sc = np.where(mask_sc)[0]
    if include_casework and idx_sc.size > 0:
        ro_s = ro_flat[idx_sc]
        rd_s = rd_flat[idx_sc]
        tb_s = t_best[idx_sc]
        nb_s = n_best[idx_sc]
        mb_s = mat_best[idx_sc]

        # 1. Left Library Pier Articulated Brass Sconce (X = -2.72..-2.36, Y = 2.14..2.42, Z = 2.55)
        t_bp1, hit_bp1, n_bp1 = intersect_ellipsoid(
            ro_s,
            rd_s,
            np.array([-2.71, 2.32, 2.55], dtype=np.float64),
            np.array([0.015, 0.072, 0.072], dtype=np.float64),
        )
        take_bp1 = hit_bp1 & (t_bp1 < tb_s)
        tb_s = np.where(take_bp1, t_bp1, tb_s)
        nb_s = np.where(take_bp1[:, None], n_bp1, nb_s)
        mb_s = np.where(take_bp1, MAT_BRASS_SCONCE, mb_s)

        for arm_a, arm_b, arm_r in (
            (np.array([-2.72, 2.34, 2.55]), np.array([-2.38, 2.34, 2.55]), 0.018),
            (np.array([-2.71, 2.41, 2.55]), np.array([-2.46, 2.34, 2.55]), 0.011),
            (np.array([-2.40, 2.38, 2.55]), np.array([-2.35, 2.16, 2.55]), 0.076),
            # Knurled collar rings & top finial on Left Pier Sconce
            (np.array([-2.405, 2.41, 2.55]), np.array([-2.395, 2.375, 2.55]), 0.032),
            (np.array([-2.355, 2.185, 2.55]), np.array([-2.348, 2.155, 2.55]), 0.083),
            # 2. Back-Left Gallery Brass Picture-Light Sconce over Framed Artwork (X = -1.58, Y = 1.98, Z = 7.22..7.55)
            (np.array([-1.58, 1.98, 7.55]), np.array([-1.58, 1.98, 7.46]), 0.036),
            (np.array([-1.58, 1.98, 7.55]), np.array([-1.58, 1.98, 7.22]), 0.017),
            (np.array([-1.58, 2.04, 7.48]), np.array([-1.58, 1.98, 7.28]), 0.011),
            (np.array([-1.84, 1.96, 7.22]), np.array([-1.32, 1.96, 7.22]), 0.044),
            # Machined brass end-cap bezels & center knuckle collar on Gallery Picture Light
            (np.array([-1.855, 1.96, 7.22]), np.array([-1.815, 1.96, 7.22]), 0.051),
            (np.array([-1.345, 1.96, 7.22]), np.array([-1.305, 1.96, 7.22]), 0.051),
            (np.array([-1.635, 1.96, 7.22]), np.array([-1.525, 1.96, 7.22]), 0.052),
            (np.array([-1.58, 2.025, 7.235]), np.array([-1.58, 1.955, 7.235]), 0.056),
            # 3. Back-Right Architectural Brass Downlight Sconce over Right Credenza (X = 0.25, Y = 1.84..2.02, Z = 7.26..7.55)
            (np.array([0.25, 1.96, 7.55]), np.array([0.25, 1.96, 7.46]), 0.036),
            (np.array([0.25, 1.96, 7.55]), np.array([0.25, 1.96, 7.26]), 0.017),
            (np.array([0.25, 2.02, 7.48]), np.array([0.25, 1.97, 7.30]), 0.011),
            (np.array([0.25, 2.022, 7.26]), np.array([0.25, 1.842, 7.26]), 0.072),
            # Machined solid top cap bezel, mid-band & lower downlight baffle bezel on Back-Right Sconce
            (np.array([0.25, 2.030, 7.26]), np.array([0.25, 2.004, 7.26]), 0.078),
            (np.array([0.25, 1.944, 7.26]), np.array([0.25, 1.920, 7.26]), 0.077),
            (np.array([0.25, 1.868, 7.26]), np.array([0.25, 1.840, 7.26]), 0.079),
        ):
            t_arm, hit_arm, n_arm = intersect_cylinder_segment(ro_s, rd_s, arm_a, arm_b, radius=arm_r)
            take_arm = hit_arm & (t_arm < tb_s)
            tb_s = np.where(take_arm, t_arm, tb_s)
            nb_s = np.where(take_arm[:, None], n_arm, nb_s)
            mb_s = np.where(take_arm, MAT_BRASS_SCONCE, mb_s)

        # Stepped circular brushed-brass mounting backplates & articulated ball-joint knuckles
        for bp_c, bp_r in (
            (np.array([-1.58, 1.98, 7.545], dtype=np.float64), np.array([0.082, 0.082, 0.016], dtype=np.float64)),
            (np.array([-1.58, 1.98, 7.528], dtype=np.float64), np.array([0.052, 0.052, 0.018], dtype=np.float64)),
            (np.array([-1.58, 1.98, 7.310], dtype=np.float64), np.array([0.028, 0.028, 0.028], dtype=np.float64)),
            (np.array([0.25, 1.96, 7.545], dtype=np.float64), np.array([0.082, 0.082, 0.016], dtype=np.float64)),
            (np.array([0.25, 1.96, 7.528], dtype=np.float64), np.array([0.052, 0.052, 0.018], dtype=np.float64)),
            (np.array([0.25, 1.96, 7.340], dtype=np.float64), np.array([0.028, 0.028, 0.028], dtype=np.float64)),
        ):
            t_bp, hit_bp, n_bp = intersect_ellipsoid(ro_s, rd_s, bp_c, bp_r)
            take_bp = hit_bp & (t_bp < tb_s)
            tb_s = np.where(take_bp, t_bp, tb_s)
            nb_s = np.where(take_bp[:, None], n_bp, nb_s)
            mb_s = np.where(take_bp, MAT_BRASS_SCONCE, mb_s)

        # Recessed warm linear/disc diffuser apertures tucked flush inside the bottom brass rims
        t_blb2, hit_blb2, n_blb2 = intersect_cylinder_segment(
            ro_s,
            rd_s,
            np.array([-1.79, 1.925, 7.22], dtype=np.float64),
            np.array([-1.37, 1.925, 7.22], dtype=np.float64),
            radius=0.019,
        )
        take_blb2 = hit_blb2 & (t_blb2 < tb_s)
        tb_s = np.where(take_blb2, t_blb2, tb_s)
        nb_s = np.where(take_blb2[:, None], n_blb2, nb_s)
        mb_s = np.where(take_blb2, MAT_SCONCE_BULB, mb_s)

        for bulb_c, bulb_r in (
            (np.array([-2.35, 2.162, 2.55], dtype=np.float64), np.array([0.058, 0.015, 0.058], dtype=np.float64)),
            (np.array([0.25, 1.843, 7.26], dtype=np.float64), np.array([0.058, 0.013, 0.058], dtype=np.float64)),
        ):
            t_bulb, hit_bulb, n_bulb = intersect_ellipsoid(ro_s, rd_s, bulb_c, bulb_r)
            take_bulb = hit_bulb & (t_bulb < tb_s)
            tb_s = np.where(take_bulb, t_bulb, tb_s)
            nb_s = np.where(take_bulb[:, None], n_bulb, nb_s)
            mb_s = np.where(take_bulb, MAT_SCONCE_BULB, mb_s)

        t_best[idx_sc] = tb_s
        n_best[idx_sc] = nb_s
        mat_best[idx_sc] = mb_s

    # ------------------------------------------------------------------
    # D. Back American Walnut & Travertine Credenza Cupboard (with 6 Inset
    #    Doors, Shadow-Gap Reveals & Brushed-Brass Pulls), Leaning Framed
    #    Bas-Relief Artwork, Stacked Architecture Monographs & Vessels
    # ------------------------------------------------------------------
    mask_bn = rays_hitting_sphere(
        ro_flat, rd_flat, np.array([-0.85, 0.65, 7.22], dtype=np.float64), 2.15, t_best
    )
    idx_bn = np.where(mask_bn)[0]
    if include_casework and idx_bn.size > 0:
        ro_b = ro_flat[idx_bn]
        rd_b = rd_flat[idx_bn]
        tb_b = t_best[idx_bn]
        nb_b = n_best[idx_bn]
        mb_b = mat_best[idx_bn]
        ab_b = alb_best[idx_bn]
        rb_b = rough_best[idx_bn]
        uv_b = uv_best[idx_bn]

        # 1. Recessed Dark Bronze Plinth Base (Y in [0.0, 0.085]) + 4 Turned Brass Feet
        plinth_c = np.array([-0.85, 0.0425, 7.25], dtype=np.float64)
        plinth_ext = np.array([1.62, 0.0425, 0.28], dtype=np.float64)
        t_pl, hit_pl, n_pl = intersect_rounded_box(ro_b, rd_b, plinth_c, plinth_ext, bevel_r=0.004)
        take_pl = hit_pl & (t_pl < tb_b)
        tb_b = np.where(take_pl, t_pl, tb_b)
        nb_b = np.where(take_pl[:, None], n_pl, nb_b)
        mb_b = np.where(take_pl, MAT_ART_FRAME, mb_b)

        for foot_x in (-2.42, -1.37, -0.33, 0.72):
            ft_a = np.array([foot_x, 0.0, 6.94], dtype=np.float64)
            ft_b = np.array([foot_x, 0.088, 6.94], dtype=np.float64)
            t_ft, hit_ft, n_ft = intersect_cylinder_segment(ro_b, rd_b, ft_a, ft_b, radius=0.022)
            take_ft = hit_ft & (t_ft < tb_b)
            tb_b = np.where(take_ft, t_ft, tb_b)
            nb_b = np.where(take_ft[:, None], n_ft, nb_b)
            mb_b = np.where(take_ft, MAT_BRASS_SCONCE, mb_b)

        # 2. Enclosed Cabinet Carcass & Dark Shadow-Gap Reveal Frame (Y in [0.085, 0.495])
        carc_c = np.array([-0.85, 0.290, 7.23], dtype=np.float64)
        carc_ext = np.array([1.70, 0.205, 0.33], dtype=np.float64)
        t_carc, hit_carc, n_carc = intersect_rounded_box(ro_b, rd_b, carc_c, carc_ext, bevel_r=0.006)
        take_carc = hit_carc & (t_carc < tb_b)
        if np.any(take_carc):
            p_carc = ro_b + t_carc[:, None] * rd_b
            in_reveal = (
                (n_carc[:, 2] < -0.7)
                & (p_carc[:, 0] >= -2.51)
                & (p_carc[:, 0] <= 0.81)
                & (p_carc[:, 1] >= 0.100)
                & (p_carc[:, 1] <= 0.480)
            )
            tb_b = np.where(take_carc, t_carc, tb_b)
            nb_b = np.where(take_carc[:, None], n_carc, nb_b)
            mb_b = np.where(take_carc, np.where(in_reveal, MAT_ART_FRAME, MAT_WALNUT_CASEWORK), mb_b)

        # 3. 6 Inset American Walnut Cupboard Doors across X in [-2.51, 0.81]
        for k_door in range(6):
            door_cx = -2.51 + (k_door + 0.5) * 0.553333
            door_c = np.array([door_cx, 0.290, 6.892], dtype=np.float64)
            door_ext = np.array([0.267, 0.186, 0.012], dtype=np.float64)
            t_dr, hit_dr, n_dr = intersect_rounded_box(ro_b, rd_b, door_c, door_ext, bevel_r=0.004)
            take_dr = hit_dr & (t_dr < tb_b)
            tb_b = np.where(take_dr, t_dr, tb_b)
            nb_b = np.where(take_dr[:, None], n_dr, nb_b)
            mb_b = np.where(take_dr, MAT_WALNUT_CASEWORK, mb_b)

        # 4. 6 Vertical Brushed-Brass Door Handles / Pulls on the 3 Double-Door Pairs
        for pull_x in (-2.002, -1.912, -0.895, -0.805, 0.212, 0.302):
            pl_a = np.array([pull_x, 0.220, 6.866], dtype=np.float64)
            pl_b = np.array([pull_x, 0.365, 6.866], dtype=np.float64)
            t_plh, hit_plh, n_plh = intersect_cylinder_segment(ro_b, rd_b, pl_a, pl_b, radius=0.0095)
            take_plh = hit_plh & (t_plh < tb_b)
            tb_b = np.where(take_plh, t_plh, tb_b)
            nb_b = np.where(take_plh[:, None], n_plh, nb_b)
            mb_b = np.where(take_plh, MAT_BRASS_SCONCE, mb_b)

        # 5. Honed Roman Travertine Credenza Top Slab (Y in [0.495, 0.526])
        ctop_c = np.array([-0.85, 0.5105, 7.22], dtype=np.float64)
        ctop_ext = np.array([1.72, 0.0155, 0.345], dtype=np.float64)
        t_btop, hit_btop, n_btop = intersect_rounded_box(ro_b, rd_b, ctop_c, ctop_ext, bevel_r=0.005)
        take_btop = hit_btop & (t_btop < tb_b)
        tb_b = np.where(take_btop, t_btop, tb_b)
        nb_b = np.where(take_btop[:, None], n_btop, nb_b)
        mb_b = np.where(take_btop, MAT_COFFEE_TABLE_TOP, mb_b)

        if (not fast_secondary) and bool(np.any(catalog.book_present)):
            # 1. Large Leaning Framed Architectural Bas-Relief / Charcoal Canvas on Credenza Left
            art_c = np.array([-1.58, 1.086, 7.24], dtype=np.float64)
            art_ext = np.array([0.48, 0.56, 0.022], dtype=np.float64)
            t_art, hit_art, n_art, ploc_art = intersect_obb_y(
                ro_b, rd_b, art_c, art_ext, yaw_rad=0.04, bevel_r=0.004, tilt_x_rad=-0.11
            )
            take_art = hit_art & (t_art < tb_b)
            if np.any(take_art):
                u_art = ploc_art[:, 0] / art_ext[0]
                v_art = ploc_art[:, 1] / art_ext[1]
                on_frame = (np.abs(u_art) > 0.91) | (np.abs(v_art) > 0.92) | (ploc_art[:, 2] > -0.005)
                tb_b = np.where(take_art, t_art, tb_b)
                nb_b = np.where(take_art[:, None], n_art, nb_b)
                mb_b = np.where(take_art, np.where(on_frame, MAT_ART_FRAME, MAT_ART_CANVAS), mb_b)
                uv_b = np.where(take_art[:, None], np.stack([u_art, v_art], axis=-1), uv_b)

            # 2. Stack of 3 Oversized Taschen-Style Architectural Folios on Credenza Center
            bench_folios = [
                (np.array([-0.52, 0.543, 7.18]), np.array([0.22, 0.017, 0.16]), 0.08, np.array([0.14, 0.16, 0.20])),
                (np.array([-0.51, 0.576, 7.18]), np.array([0.205, 0.016, 0.15]), -0.05, np.array([0.52, 0.22, 0.15])),
                (np.array([-0.52, 0.607, 7.18]), np.array([0.19, 0.015, 0.14]), 0.12, np.array([0.66, 0.61, 0.52])),
            ]
            for bf_c, bf_ext, bf_yaw, bf_col in bench_folios:
                t_bf, hit_bf, n_bf, _ = intersect_obb_y(ro_b, rd_b, bf_c, bf_ext, yaw_rad=bf_yaw, bevel_r=0.003)
                take_bf = hit_bf & (t_bf < tb_b)
                if np.any(take_bf):
                    tb_b = np.where(take_bf, t_bf, tb_b)
                    nb_b = np.where(take_bf[:, None], n_bf, nb_b)
                    mb_b = np.where(take_bf, MAT_BOOK_SPINE, mb_b)
                    ab_b = np.where(take_bf[:, None], bf_col[None, :], ab_b)
                    rb_b = np.where(take_bf, 0.44, rb_b)

            # 3. Sculptural Fluted Travertine Vessel + Low Patinated Bronze Bowl on Credenza Right
            bv_c = np.array([0.22, 0.681, 7.16], dtype=np.float64)
            bv_r = np.array([0.115, 0.155, 0.115], dtype=np.float64)
            t_bv, hit_bv, n_bv = intersect_ellipsoid(ro_b, rd_b, bv_c, bv_r)
            take_bv = hit_bv & (t_bv < tb_b)
            if np.any(take_bv):
                p_bv = ro_b + t_bv[:, None] * rd_b
                theta_bv = np.arctan2(p_bv[:, 2] - 7.16, p_bv[:, 0] - 0.22)
                flute_nx = np.cos(theta_bv * 28.0) * 0.22
                flute_nz = -np.sin(theta_bv * 28.0) * 0.22
                n_bv_flute = normalize(
                    n_bv + np.stack([flute_nx, np.zeros_like(flute_nx), flute_nz], axis=-1)
                )
                tb_b = np.where(take_bv, t_bv, tb_b)
                nb_b = np.where(take_bv[:, None], n_bv_flute, nb_b)
                mb_b = np.where(take_bv, MAT_FLUTED_TRAVERTINE_VASE, mb_b)
                ab_b = np.where(take_bv[:, None], np.array([[0.60, 0.54, 0.45]]), ab_b)
                rb_b = np.where(take_bv, 0.26, rb_b)

            # Companion low bronze vessel beside the fluted travertine vase
            bv2_c = np.array([0.46, 0.586, 7.22], dtype=np.float64)
            bv2_r = np.array([0.088, 0.062, 0.088], dtype=np.float64)
            t_bv2, hit_bv2, n_bv2 = intersect_ellipsoid(ro_b, rd_b, bv2_c, bv2_r)
            take_bv2 = hit_bv2 & (t_bv2 < tb_b)
            if np.any(take_bv2):
                tb_b = np.where(take_bv2, t_bv2, tb_b)
                nb_b = np.where(take_bv2[:, None], n_bv2, nb_b)
                mb_b = np.where(take_bv2, MAT_BRASS_SCONCE, mb_b)
                ab_b = np.where(take_bv2[:, None], np.array([[0.32, 0.24, 0.16]]), ab_b)
                rb_b = np.where(take_bv2, 0.28, rb_b)

        t_best[idx_bn] = tb_b
        n_best[idx_bn] = nb_b
        mat_best[idx_bn] = mb_b
        alb_best[idx_bn] = ab_b
        rough_best[idx_bn] = rb_b
        uv_best[idx_bn] = uv_b

    # ------------------------------------------------------------------
    # E. Sculpted Cognac Saddle-Leather Lounge Chair, Integrated Headrest
    #    Bolster, Bridle-Leather Straps, Wrapped Armrests & Steel Sled Frame
    # ------------------------------------------------------------------
    mask_ch = rays_hitting_sphere(
        ro_flat, rd_flat, np.array([-1.18, 0.44, 2.28], dtype=np.float64), 0.86, t_best
    )
    idx_ch = np.where(mask_ch)[0]
    if include_chair_and_rug and idx_ch.size > 0:
        ro_c = ro_flat[idx_ch]
        rd_c = rd_flat[idx_ch]
        tb_c = t_best[idx_ch]
        nb_c = n_best[idx_ch]
        mb_c = mat_best[idx_ch]
        uv_c = uv_best[idx_ch]

        chair_yaw = -0.56
        chair_base = np.array([-1.18, 0.0, 2.28], dtype=np.float64)
        cy_c, sy_c = math.cos(chair_yaw), math.sin(chair_yaw)

        def chair_pt(lx: float, ly: float, lz: float) -> np.ndarray:
            return chair_base + np.array(
                [cy_c * lx + sy_c * lz, ly, -sy_c * lx + cy_c * lz],
                dtype=np.float64,
            )

        # 1. Sculpted 3-Channel Tufted Seat Cushion
        seat_c = chair_pt(0.0, 0.305, -0.02)
        seat_ext = np.array([0.41, 0.072, 0.395], dtype=np.float64)
        t_seat, hit_seat, n_seat, ploc_seat = intersect_obb_y(
            ro_c, rd_c, seat_c, seat_ext, yaw_rad=chair_yaw, bevel_r=0.036, tilt_x_rad=0.14
        )
        take_seat = hit_seat & (t_seat < tb_c)
        if np.any(take_seat):
            tuft = 0.14 * np.sin((ploc_seat[:, 0] / 0.41) * PI) * np.cos((ploc_seat[:, 2] / 0.395) * PI)
            n_seat_mod = normalize(n_seat + np.stack([tuft, np.zeros_like(tuft), tuft * 0.6], axis=-1))
            tb_c = np.where(take_seat, t_seat, tb_c)
            nb_c = np.where(take_seat[:, None], n_seat_mod, nb_c)
            mb_c = np.where(take_seat, MAT_LEATHER_CHAIR, mb_c)
            uv_c = np.where(take_seat[:, None], ploc_seat[:, [0, 2]] / np.array([[0.41, 0.395]]), uv_c)

        # 2. Angled 3-Channel Tufted Backrest Cushion (extended slightly higher so bolster sits flush)
        back_c = chair_pt(0.0, 0.565, 0.312)
        back_ext = np.array([0.40, 0.282, 0.072], dtype=np.float64)
        t_bkr, hit_bkr, n_bkr, ploc_bkr = intersect_obb_y(
            ro_c, rd_c, back_c, back_ext, yaw_rad=chair_yaw, bevel_r=0.036, tilt_x_rad=0.32
        )
        take_bkr = hit_bkr & (t_bkr < tb_c)
        if np.any(take_bkr):
            tuft_b = 0.15 * np.sin((ploc_bkr[:, 0] / 0.40) * PI) * np.sin((ploc_bkr[:, 1] / 0.282) * PI)
            n_bkr_mod = normalize(n_bkr + np.stack([tuft_b, tuft_b * 0.5, np.zeros_like(tuft_b)], axis=-1))
            tb_c = np.where(take_bkr, t_bkr, tb_c)
            nb_c = np.where(take_bkr[:, None], n_bkr_mod, nb_c)
            mb_c = np.where(take_bkr, MAT_LEATHER_CHAIR, mb_c)
            uv_c = np.where(take_bkr[:, None], ploc_bkr[:, [0, 1]] / np.array([[0.40, 0.282]]), uv_c)

        # 3. Seamlessly Integrated Cylindrical Headrest Bolster (snuggled against upper backrest face)
        bol_p0 = chair_pt(-0.355, 0.705, 0.338)
        bol_p1 = chair_pt(0.355, 0.705, 0.338)
        t_bol, hit_bol, n_bol = intersect_cylinder_segment(ro_c, rd_c, bol_p0, bol_p1, radius=0.068)
        take_bol = hit_bol & (t_bol < tb_c)
        if np.any(take_bol):
            p_bol = ro_c + t_bol[:, None] * rd_c - chair_base
            lx_bol = (cy_c * p_bol[:, 0] - sy_c * p_bol[:, 2]) / 0.355
            ly_bol = (p_bol[:, 1] - 0.705) / 0.068
            tb_c = np.where(take_bol, t_bol, tb_c)
            nb_c = np.where(take_bol[:, None], n_bol, nb_c)
            mb_c = np.where(take_bol, MAT_LEATHER_CHAIR, mb_c)
            uv_c = np.where(take_bol[:, None], np.stack([lx_bol, ly_bol], axis=-1), uv_c)

        # 4. Bridle-Leather Headrest Straps & Brushed-Brass Buckles holding the Bolster
        if not fast_secondary:
            for strap_lx in (-0.23, 0.23):
                st_c = chair_pt(strap_lx, 0.755, 0.365)
                st_ext = np.array([0.018, 0.095, 0.048], dtype=np.float64)
                t_st, hit_st, n_st, _ = intersect_obb_y(
                    ro_c, rd_c, st_c, st_ext, yaw_rad=chair_yaw, bevel_r=0.005, tilt_x_rad=0.32
                )
                take_st = hit_st & (t_st < tb_c)
                tb_c = np.where(take_st, t_st, tb_c)
                nb_c = np.where(take_st[:, None], n_st, nb_c)
                mb_c = np.where(take_st, MAT_LEATHER_STRAP, mb_c)

                # Brass buckle on each strap
                bk_c = chair_pt(strap_lx, 0.785, 0.325)
                bk_ext = np.array([0.022, 0.012, 0.012], dtype=np.float64)
                t_bkc, hit_bkc, n_bkc, _ = intersect_obb_y(
                    ro_c, rd_c, bk_c, bk_ext, yaw_rad=chair_yaw, bevel_r=0.003, tilt_x_rad=0.32
                )
                take_bkc = hit_bkc & (t_bkc < tb_c)
                tb_c = np.where(take_bkc, t_bkc, tb_c)
                nb_c = np.where(take_bkc[:, None], n_bkc, nb_c)
                mb_c = np.where(take_bkc, MAT_BRASS_SCONCE, mb_c)

            # 5. Cantilevered Satin Stainless-Steel Sled Frame & Wrapped Saddle-Leather Armrests
            for side_s in (-0.425, 0.425):
                r_front = chair_pt(side_s, 0.0165, -0.37)
                r_back = chair_pt(side_s, 0.0165, 0.38)
                f_top = chair_pt(side_s, 0.42, -0.31)
                b_top = chair_pt(side_s, 0.50, 0.31)

                for seg_a, seg_b in ((r_front, r_back), (r_front, f_top), (r_back, b_top), (f_top, b_top)):
                    t_fr, hit_fr, n_fr = intersect_cylinder_segment(ro_c, rd_c, seg_a, seg_b, radius=0.0155)
                    take_fr = hit_fr & (t_fr < tb_c)
                    tb_c = np.where(take_fr, t_fr, tb_c)
                    nb_c = np.where(take_fr[:, None], n_fr, nb_c)
                    mb_c = np.where(take_fr, MAT_CHAIR_STEEL, mb_c)

                # Flat-bottomed brushed-steel runner shoe bar pressed 4mm into the 13mm rug pile
                shoe_c = chair_pt(side_s, 0.014, 0.005)
                shoe_ext = np.array([0.0155, 0.0055, 0.378], dtype=np.float64)
                t_sh, hit_sh, n_sh, _ = intersect_obb_y(
                    ro_c, rd_c, shoe_c, shoe_ext, yaw_rad=chair_yaw, bevel_r=0.0025
                )
                take_sh = hit_sh & (t_sh < tb_c)
                tb_c = np.where(take_sh, t_sh, tb_c)
                nb_c = np.where(take_sh[:, None], n_sh, nb_c)
                mb_c = np.where(take_sh, MAT_CHAIR_STEEL, mb_c)

                # Saddle-leather wrapped armrest pad along the top steel bar
                arm_c = chair_pt(side_s, 0.464, -0.01)
                arm_ext = np.array([0.028, 0.018, 0.245], dtype=np.float64)
                t_arm_p, hit_arm_p, n_arm_p, ploc_arm = intersect_obb_y(
                    ro_c, rd_c, arm_c, arm_ext, yaw_rad=chair_yaw, bevel_r=0.008, tilt_x_rad=0.128
                )
                take_arm_p = hit_arm_p & (t_arm_p < tb_c)
                tb_c = np.where(take_arm_p, t_arm_p, tb_c)
                nb_c = np.where(take_arm_p[:, None], n_arm_p, nb_c)
                mb_c = np.where(take_arm_p, MAT_LEATHER_CHAIR, mb_c)
                uv_c = np.where(
                    take_arm_p[:, None],
                    ploc_arm[:, [0, 2]] / np.array([[0.028, 0.245]]),
                    uv_c,
                )

        t_best[idx_ch] = tb_c
        n_best[idx_ch] = nb_c
        mat_best[idx_ch] = mb_c
        uv_best[idx_ch] = uv_c

    # ------------------------------------------------------------------
    # F. Low Honed Roman Travertine & Walnut Architectural Coffee Table
    # ------------------------------------------------------------------
    mask_tb = rays_hitting_sphere(
        ro_flat, rd_flat, np.array([0.33, 0.24, 1.77], dtype=np.float64), 1.05, t_best
    )
    idx_tb = np.where(mask_tb)[0]
    if include_coffee_table and idx_tb.size > 0:
        ro_t = ro_flat[idx_tb]
        rd_t = rd_flat[idx_tb]
        tb_t = t_best[idx_tb]
        nb_t = n_best[idx_tb]
        mb_t = mat_best[idx_tb]

        table_top_c = np.array([0.30, 0.36, 1.77], dtype=np.float64)
        table_top_ext = np.array([0.68, 0.030, 0.69], dtype=np.float64)
        t_tt, hit_tt, n_tt = intersect_rounded_box(ro_t, rd_t, table_top_c, table_top_ext, bevel_r=0.006)
        take_tt = hit_tt & (t_tt < tb_t)
        tb_t = np.where(take_tt, t_tt, tb_t)
        nb_t = np.where(take_tt[:, None], n_tt, nb_t)
        mb_t = np.where(take_tt, MAT_COFFEE_TABLE_TOP, mb_t)

        for leg_z in (1.28, 2.26):
            leg_c = np.array([0.30, 0.165, leg_z], dtype=np.float64)
            leg_ext = np.array([0.52, 0.165, 0.042], dtype=np.float64)
            t_tl, hit_tl, n_tl = intersect_rounded_box(ro_t, rd_t, leg_c, leg_ext, bevel_r=0.005)
            take_tl = hit_tl & (t_tl < tb_t)
            tb_t = np.where(take_tl, t_tl, tb_t)
            nb_t = np.where(take_tl[:, None], n_tl, nb_t)
            mb_t = np.where(take_tl, MAT_COFFEE_TABLE_BASE, mb_t)

        str_c = np.array([0.30, 0.14, 1.77], dtype=np.float64)
        str_ext = np.array([0.045, 0.065, 0.48], dtype=np.float64)
        t_ts, hit_ts, n_ts = intersect_rounded_box(ro_t, rd_t, str_c, str_ext, bevel_r=0.004)
        take_ts = hit_ts & (t_ts < tb_t)
        tb_t = np.where(take_ts, t_ts, tb_t)
        nb_t = np.where(take_ts[:, None], n_ts, nb_t)
        mb_t = np.where(take_ts, MAT_COFFEE_TABLE_BASE, mb_t)

        t_best[idx_tb] = tb_t
        n_best[idx_tb] = nb_t
        mat_best[idx_tb] = mb_t

    # ------------------------------------------------------------------
    # G. Coffee Table Still Life Part 1: Open Art Monograph with Silk
    #    Bookmark Ribbon, Stacked Closed Monographs & Hand-Thrown
    #    Stoneware Espresso Cup & Saucer
    # ------------------------------------------------------------------
    mask_bk = rays_hitting_sphere(
        ro_flat, rd_flat, np.array([0.06, 0.43, 1.76], dtype=np.float64), 0.78, t_best
    )
    idx_bk2 = np.where(mask_bk)[0]
    if include_coffee_table and idx_bk2.size > 0:
        ro_k = ro_flat[idx_bk2]
        rd_k = rd_flat[idx_bk2]
        tb_k = t_best[idx_bk2]
        nb_k = n_best[idx_bk2]
        mb_k = mat_best[idx_bk2]
        ab_k = alb_best[idx_bk2]
        rb_k = rough_best[idx_bk2]
        uv_k = uv_best[idx_bk2]

        # 1. Open Hardcover Art Monograph
        book_yaw = -0.19
        book_c = np.array([0.04, 0.396, 1.50], dtype=np.float64)
        cov_ext = np.array([0.235, 0.006, 0.168], dtype=np.float64)
        t_cov, hit_cov, n_cov, _ = intersect_obb_y(ro_k, rd_k, book_c, cov_ext, yaw_rad=book_yaw, bevel_r=0.002)
        take_cov = hit_cov & (t_cov < tb_k)
        tb_k = np.where(take_cov, t_cov, tb_k)
        nb_k = np.where(take_cov[:, None], n_cov, nb_k)
        mb_k = np.where(take_cov, MAT_OPEN_BOOK_COVER, mb_k)

        page_c = book_c + np.array([0.0, 0.012, 0.0], dtype=np.float64)
        page_ext = np.array([0.222, 0.011, 0.158], dtype=np.float64)
        t_pg, hit_pg, n_pg, ploc_pg = intersect_obb_y(ro_k, rd_k, page_c, page_ext, yaw_rad=book_yaw, bevel_r=0.003)
        take_pg = hit_pg & (t_pg < tb_k)
        if np.any(take_pg):
            u_pg = ploc_pg[:, 0] / 0.222
            v_pg = ploc_pg[:, 2] / 0.158
            sgn_u = np.sign(u_pg)
            slope_u = sgn_u * (0.38 * np.exp(-np.abs(u_pg) * 4.5) - 0.18 * np.abs(u_pg))
            cy_b, sy_b = math.cos(book_yaw), math.sin(book_yaw)
            n_arch = normalize(
                n_pg + np.stack([-cy_b * slope_u, np.zeros_like(slope_u), sy_b * slope_u], axis=-1)
            )
            tb_k = np.where(take_pg, t_pg, tb_k)
            nb_k = np.where(take_pg[:, None], n_arch, nb_k)
            mb_k = np.where(take_pg, MAT_OPEN_BOOK_PAGE, mb_k)
            uv_k = np.where(take_pg[:, None], np.stack([u_pg, v_pg], axis=-1), uv_k)

        if not fast_secondary:
            # 2. Draped Burgundy-Crimson Silk Bookmark Ribbon emerging from the central gutter
            rib_a = np.array([0.04, 0.419, 1.46], dtype=np.float64)
            rib_b = np.array([0.068, 0.393, 1.28], dtype=np.float64)
            t_rib, hit_rib, n_rib = intersect_cylinder_segment(ro_k, rd_k, rib_a, rib_b, radius=0.0055)
            take_rib = hit_rib & (t_rib < tb_k)
            tb_k = np.where(take_rib, t_rib, tb_k)
            nb_k = np.where(take_rib[:, None], n_rib, nb_k)
            mb_k = np.where(take_rib, MAT_BOOKMARK_RIBBON, mb_k)

            # 3. Stack of Two Closed Hardcover Architecture Monographs squarely on Left-Rear of Table
            cb1_c = np.array([-0.12, 0.405, 2.04], dtype=np.float64)
            cb1_ext = np.array([0.162, 0.015, 0.210], dtype=np.float64)
            t_cb1, hit_cb1, n_cb1, ploc_cb1 = intersect_obb_y(ro_k, rd_k, cb1_c, cb1_ext, yaw_rad=0.22, bevel_r=0.003)
            take_cb1 = hit_cb1 & (t_cb1 < tb_k)
            if np.any(take_cb1):
                is_pg1 = (ploc_cb1[:, 0] > 0.142) & (np.abs(ploc_cb1[:, 1]) < 0.011)
                col1 = np.where(is_pg1[:, None], np.array([[0.74, 0.70, 0.61]]), np.array([[0.11, 0.16, 0.24]]))
                tb_k = np.where(take_cb1, t_cb1, tb_k)
                nb_k = np.where(take_cb1[:, None], n_cb1, nb_k)
                mb_k = np.where(take_cb1, MAT_TABLE_CLOSED_BOOK, mb_k)
                ab_k = np.where(take_cb1[:, None], col1, ab_k)
                rb_k = np.where(take_cb1, 0.46, rb_k)

            cb2_c = np.array([-0.11, 0.433, 2.03], dtype=np.float64)
            cb2_ext = np.array([0.148, 0.013, 0.194], dtype=np.float64)
            t_cb2, hit_cb2, n_cb2, ploc_cb2 = intersect_obb_y(ro_k, rd_k, cb2_c, cb2_ext, yaw_rad=0.09, bevel_r=0.003)
            take_cb2 = hit_cb2 & (t_cb2 < tb_k)
            if np.any(take_cb2):
                is_pg2 = (ploc_cb2[:, 0] > 0.128) & (np.abs(ploc_cb2[:, 1]) < 0.009)
                col2 = np.where(is_pg2[:, None], np.array([[0.75, 0.71, 0.62]]), np.array([[0.48, 0.20, 0.13]]))
                tb_k = np.where(take_cb2, t_cb2, tb_k)
                nb_k = np.where(take_cb2[:, None], n_cb2, nb_k)
                mb_k = np.where(take_cb2, MAT_TABLE_CLOSED_BOOK, mb_k)
                ab_k = np.where(take_cb2[:, None], col2, ab_k)
                rb_k = np.where(take_cb2, 0.42, rb_k)

            # 4. Prominent Hand-Thrown Warm Cream Stoneware Cappuccino Cup, Dished Saucer,
            #    Recessed Espresso Crema & Brushed Brass Demitasse Spoon
            esp_base = np.array([0.24, 0.390, 1.78], dtype=np.float64)
            # Dished stoneware saucer (17.6cm diameter, R=0.088) with concave well normal
            t_sau, hit_sau, n_sau = intersect_ellipsoid(
                ro_k,
                rd_k,
                esp_base + np.array([0.0, 0.010, 0.0], dtype=np.float64),
                np.array([0.088, 0.0105, 0.088], dtype=np.float64),
            )
            take_sau = hit_sau & (t_sau < tb_k)
            if np.any(take_sau):
                p_sau = ro_k + t_sau[:, None] * rd_k
                sau_rx = p_sau[:, 0] - 0.24
                sau_rz = p_sau[:, 2] - 1.78
                sau_r = np.hypot(sau_rx, sau_rz)
                # Concave dish slope inside the saucer rim (r < 0.076)
                in_dish = (n_sau[:, 1] > 0.35) & (sau_r < 0.076)
                dish_slope = np.where(in_dish, -0.34 * (sau_r / 0.076), 0.0)
                n_sau_dish = normalize(
                    n_sau
                    + np.stack(
                        [
                            dish_slope * sau_rx / np.maximum(sau_r, 1e-5),
                            np.zeros_like(sau_r),
                            dish_slope * sau_rz / np.maximum(sau_r, 1e-5),
                        ],
                        axis=-1,
                    )
                )
                tb_k = np.where(take_sau, t_sau, tb_k)
                nb_k = np.where(take_sau[:, None], n_sau_dish, nb_k)
                mb_k = np.where(take_sau, MAT_ESPRESSO_CUP, mb_k)

            # Stoneware cappuccino cup body (10.6cm diameter, R=0.053, height=0.082m)
            # with a 7mm thick ceramic rim and a recessed interior coffee meniscus (y=0.468)
            cup_a = esp_base + np.array([0.0, 0.010, 0.0], dtype=np.float64)
            cup_b = esp_base + np.array([0.0, 0.092, 0.0], dtype=np.float64)
            t_cup, hit_cup, n_cup = intersect_cylinder_segment(ro_k, rd_k, cup_a, cup_b, radius=0.053, capped=True)
            take_cup = hit_cup & (t_cup < tb_k)
            if np.any(take_cup):
                p_cup = ro_k + t_cup[:, None] * rd_k
                r_top = np.hypot(p_cup[:, 0] - 0.24, p_cup[:, 2] - 1.78)
                r_inner = 0.0455
                y_coffee = 0.390 + 0.078  # Recessed 14mm below the 0.482m top rim
                in_hollow = take_cup & (n_cup[:, 1] > 0.85) & (r_top < r_inner)

                # Trace rays entering the hollow cup opening down to the recessed coffee meniscus or inner ceramic wall
                t_men = (y_coffee - ro_k[:, 1]) / np.where(np.abs(rd_k[:, 1]) < 1e-8, -1e-8, rd_k[:, 1])
                p_men = ro_k + t_men[:, None] * rd_k
                men_rx = p_men[:, 0] - 0.24
                men_rz = p_men[:, 2] - 1.78
                men_r = np.hypot(men_rx, men_rz)
                hits_meniscus = in_hollow & (t_men > t_cup) & (men_r <= r_inner)

                # Inner cylindrical ceramic wall intersection (far root of x^2 + z^2 = r_inner^2)
                oc_x = ro_k[:, 0] - 0.24
                oc_z = ro_k[:, 2] - 1.78
                a_in = np.maximum(rd_k[:, 0] ** 2 + rd_k[:, 2] ** 2, 1e-12)
                b_in = oc_x * rd_k[:, 0] + oc_z * rd_k[:, 2]
                c_in = oc_x * oc_x + oc_z * oc_z - r_inner * r_inner
                disc_in = np.maximum(b_in * b_in - a_in * c_in, 0.0)
                t_in_wall = (-b_in + np.sqrt(disc_in)) / a_in
                p_in_wall = ro_k + t_in_wall[:, None] * rd_k
                n_in_wall = normalize(
                    np.stack([-(p_in_wall[:, 0] - 0.24), np.zeros_like(t_in_wall), -(p_in_wall[:, 2] - 1.78)], axis=-1)
                )
                hits_in_wall = in_hollow & (~hits_meniscus) & (t_in_wall > t_cup)

                # Concave capillary meniscus normal near the inner ceramic wall
                men_lip = smoothstep(r_inner - 0.007, r_inner, men_r) * 0.28
                n_men = normalize(
                    np.stack(
                        [
                            -men_lip * men_rx / np.maximum(men_r, 1e-5),
                            np.ones_like(men_r),
                            -men_lip * men_rz / np.maximum(men_r, 1e-5),
                        ],
                        axis=-1,
                    )
                )

                t_cup_final = np.where(hits_meniscus, t_men, np.where(hits_in_wall, t_in_wall, t_cup))
                n_cup_final = np.where(
                    hits_meniscus[:, None],
                    n_men,
                    np.where(hits_in_wall[:, None], n_in_wall, n_cup),
                )
                m_cup_final = np.where(hits_meniscus, MAT_ESPRESSO_CREMA, MAT_ESPRESSO_CUP)

                tb_k = np.where(take_cup, t_cup_final, tb_k)
                nb_k = np.where(take_cup[:, None], n_cup_final, nb_k)
                mb_k = np.where(take_cup, m_cup_final, mb_k)

            # Prominent stoneware loop handle + brushed brass spoon resting on saucer rim
            hnd_c = esp_base + np.array([-0.058, 0.052, -0.010], dtype=np.float64)
            t_hnd, hit_hnd, n_hnd = intersect_ellipsoid(
                ro_k, rd_k, hnd_c, np.array([0.021, 0.027, 0.0085], dtype=np.float64), yaw_rad=0.18
            )
            take_hnd = hit_hnd & (t_hnd < tb_k)
            tb_k = np.where(take_hnd, t_hnd, tb_k)
            nb_k = np.where(take_hnd[:, None], n_hnd, nb_k)
            mb_k = np.where(take_hnd, MAT_ESPRESSO_CUP, mb_k)

            sp_a = esp_base + np.array([0.030, 0.018, -0.062], dtype=np.float64)
            sp_b = esp_base + np.array([0.086, 0.022, 0.044], dtype=np.float64)
            t_spn, hit_spn, n_spn = intersect_cylinder_segment(ro_k, rd_k, sp_a, sp_b, radius=0.0038)
            take_spn = hit_spn & (t_spn < tb_k)
            tb_k = np.where(take_spn, t_spn, tb_k)
            nb_k = np.where(take_spn[:, None], n_spn, nb_k)
            mb_k = np.where(take_spn, MAT_BRASS_SCONCE, mb_k)

            # Elliptical brass spoon bowl at sp_a
            t_spb, hit_spb, n_spb = intersect_ellipsoid(
                ro_k, rd_k, sp_a, np.array([0.013, 0.0040, 0.017], dtype=np.float64), yaw_rad=0.48
            )
            take_spb = hit_spb & (t_spb < tb_k)
            tb_k = np.where(take_spb, t_spb, tb_k)
            nb_k = np.where(take_spb[:, None], n_spb, nb_k)
            mb_k = np.where(take_spb, MAT_BRASS_SCONCE, mb_k)

        t_best[idx_bk2] = tb_k
        n_best[idx_bk2] = nb_k
        mat_best[idx_bk2] = mb_k
        alb_best[idx_bk2] = ab_k
        rough_best[idx_bk2] = rb_k
        uv_best[idx_bk2] = uv_k

    # ------------------------------------------------------------------
    # H & I. Separate Hand-Thrown Ribbed Stoneware Olive Vase with Connected
    #        Silver-Dollar Eucalyptus/Olive Branches + Separate Clear Crystalline
    #        Glass Water Carafe & Faceted Crystal Water Tumbler on Brass Coaster
    # ------------------------------------------------------------------
    mask_vas = rays_hitting_sphere(
        ro_flat, rd_flat, np.array([0.52, 0.66, 1.72], dtype=np.float64), 0.62, t_best
    )
    idx_vas = np.where(mask_vas)[0]
    if include_coffee_table and idx_vas.size > 0:
        ro_v = ro_flat[idx_vas]
        rd_v = rd_flat[idx_vas]
        tb_v = t_best[idx_vas]
        nb_v = n_best[idx_vas]
        mb_v = mat_best[idx_vas]
        ab_v = alb_best[idx_vas]
        rb_v = rough_best[idx_vas]
        uv_v = uv_best[idx_vas]

        # 1. Hand-Thrown Ribbed Stoneware Olive Vase at [0.44, 0.390, 1.98] (rear-center right)
        vase_base = np.array([0.44, 0.390, 1.98], dtype=np.float64)
        v_belly_c = vase_base + np.array([0.0, 0.098, 0.0], dtype=np.float64)
        v_belly_r = np.array([0.086, 0.098, 0.086], dtype=np.float64)
        t_vb, hit_vb, n_vb = intersect_ellipsoid(ro_v, rd_v, v_belly_c, v_belly_r)

        v_neck_a = vase_base + np.array([0.0, 0.165, 0.0], dtype=np.float64)
        v_neck_b = vase_base + np.array([0.0, 0.222, 0.0], dtype=np.float64)
        t_vn, hit_vn, n_vn = intersect_cylinder_segment(ro_v, rd_v, v_neck_a, v_neck_b, radius=0.040, capped=False)

        v_lip_c = vase_base + np.array([0.0, 0.222, 0.0], dtype=np.float64)
        v_lip_r = np.array([0.048, 0.010, 0.048], dtype=np.float64)
        t_vl, hit_vl, n_vl = intersect_ellipsoid(ro_v, rd_v, v_lip_c, v_lip_r)

        t_ovase = np.where(hit_vb, t_vb, 1e9)
        n_ovase = n_vb.copy()
        use_vn = hit_vn & (t_vn < t_ovase)
        t_ovase = np.where(use_vn, t_vn, t_ovase)
        n_ovase = np.where(use_vn[:, None], n_vn, n_ovase)
        use_vl = hit_vl & (t_vl < t_ovase)
        t_ovase = np.where(use_vl, t_vl, t_ovase)
        n_ovase = np.where(use_vl[:, None], n_vl, n_ovase)

        take_ovase = (t_ovase < tb_v) & (hit_vb | hit_vn | hit_vl)
        if np.any(take_ovase):
            p_ov = ro_v + t_ovase[:, None] * rd_v
            # Wheel-thrown horizontal lathe throwing rings & warm speckled stoneware glaze
            ring_ny = np.sin(p_ov[:, 1] * 260.0) * 0.12
            sp_ov = value_noise_2d(p_ov[:, 0] * 160.0, p_ov[:, 2] * 160.0 + p_ov[:, 1] * 160.0, seed=947)
            ov_col = np.array([0.74, 0.69, 0.59], dtype=np.float64) * (0.90 + 0.16 * sp_ov[:, None])
            n_ov_ring = normalize(n_ovase + np.stack([np.zeros_like(ring_ny), ring_ny, np.zeros_like(ring_ny)], axis=-1))
            tb_v = np.where(take_ovase, t_ovase, tb_v)
            nb_v = np.where(take_ovase[:, None], n_ov_ring, nb_v)
            mb_v = np.where(take_ovase, MAT_CERAMIC_VESSEL, mb_v)
            ab_v = np.where(take_ovase[:, None], ov_col, ab_v)
            rb_v = np.where(take_ovase, 0.28, rb_v)

        # 2. Botanical Olive / Silver-Dollar Eucalyptus Stems & 11 Leaves emerging from the Stoneware Vase
        stem_a = vase_base + np.array([0.0, 0.195, 0.0], dtype=np.float64)
        stem_b = vase_base + np.array([0.048, 0.565, -0.022], dtype=np.float64)
        twig1_a = vase_base + np.array([0.012, 0.265, -0.006], dtype=np.float64)
        twig1_b = vase_base + np.array([-0.088, 0.495, 0.032], dtype=np.float64)
        twig2_a = vase_base + np.array([0.024, 0.345, -0.011], dtype=np.float64)
        twig2_b = vase_base + np.array([0.088, 0.515, -0.026], dtype=np.float64)

        for st_p0, st_p1, st_r in (
            (stem_a, stem_b, 0.0046),
            (twig1_a, twig1_b, 0.0035),
            (twig2_a, twig2_b, 0.0033),
        ):
            t_s, hit_s, n_s = intersect_cylinder_segment(ro_v, rd_v, st_p0, st_p1, radius=st_r)
            take_s = hit_s & (t_s < tb_v)
            tb_v = np.where(take_s, t_s, tb_v)
            nb_v = np.where(take_s[:, None], n_s, nb_v)
            mb_v = np.where(take_s, MAT_EUCALYPTUS_STEM, mb_v)

        leaves_connected_spec = [
            # Lower pair along main stem & left branchlet
            (np.array([0.011, 0.285, -0.005]), np.array([-0.048, 0.302, -0.028]), np.array([0.040, 0.0065, 0.033]), 0.62),
            (np.array([0.019, 0.325, -0.008]), np.array([0.056, 0.338, -0.036]), np.array([0.042, 0.0065, 0.034]), -0.58),
            # Mid-tier leaves along left branchlet (twig1)
            (np.array([-0.018, 0.340, 0.007]), np.array([-0.064, 0.365, -0.018]), np.array([0.044, 0.0068, 0.036]), 0.54),
            (np.array([-0.046, 0.402, 0.017]), np.array([-0.095, 0.418, -0.008]), np.array([0.041, 0.0065, 0.034]), -0.48),
            (np.array([-0.072, 0.458, 0.026]), np.array([-0.115, 0.485, 0.006]), np.array([0.038, 0.0062, 0.031]), 0.65),
            # Mid-tier leaves along right branchlet (twig2)
            (np.array([0.052, 0.415, -0.017]), np.array([0.096, 0.422, -0.042]), np.array([0.043, 0.0068, 0.035]), -0.64),
            (np.array([0.072, 0.472, -0.022]), np.array([0.114, 0.498, -0.045]), np.array([0.039, 0.0065, 0.032]), 0.52),
            # Upper main stem leaves & apical crown pair
            (np.array([0.028, 0.405, -0.012]), np.array([-0.012, 0.428, -0.045]), np.array([0.040, 0.0065, 0.033]), 0.68),
            (np.array([0.036, 0.455, -0.015]), np.array([0.068, 0.478, -0.048]), np.array([0.044, 0.0068, 0.036]), -0.55),
            (np.array([0.045, 0.515, -0.019]), np.array([0.008, 0.542, -0.042]), np.array([0.037, 0.0062, 0.030]), 0.58),
            (np.array([0.051, 0.555, -0.021]), np.array([0.068, 0.586, -0.040]), np.array([0.035, 0.0060, 0.029]), -0.52),
        ]
        for node_off, lf_off, lf_rad, lf_roll in leaves_connected_spec:
            node_p = vase_base + node_off
            lf_c = vase_base + lf_off

            u_ax = normalize((lf_c - node_p)[None, :])[0]
            ref_up = np.array([0.0, 1.0, -0.45], dtype=np.float64)
            v_base = normalize(np.cross(u_ax, ref_up)[None, :])[0]
            n_base = np.cross(v_base, u_ax)
            cr, sr = math.cos(lf_roll), math.sin(lf_roll)
            v_ax = cr * v_base + sr * n_base
            n_ax = -sr * v_base + cr * n_base

            leaf_base_p = lf_c - u_ax * (lf_rad[0] * 0.72)
            t_pet, hit_pet, n_pet = intersect_cylinder_segment(ro_v, rd_v, node_p, leaf_base_p, radius=0.0024)
            take_pet = hit_pet & (t_pet < tb_v)
            tb_v = np.where(take_pet, t_pet, tb_v)
            nb_v = np.where(take_pet[:, None], n_pet, nb_v)
            mb_v = np.where(take_pet, MAT_EUCALYPTUS_STEM, mb_v)

            oc_lf = ro_v - lf_c[None, :]
            ro_loc = np.stack([oc_lf @ u_ax, oc_lf @ n_ax, oc_lf @ v_ax], axis=-1) / lf_rad[None, :]
            rd_loc = np.stack([rd_v @ u_ax, rd_v @ n_ax, rd_v @ v_ax], axis=-1) / lf_rad[None, :]

            a_q = np.sum(rd_loc * rd_loc, axis=-1)
            b_q = np.sum(ro_loc * rd_loc, axis=-1)
            c_q = np.sum(ro_loc * ro_loc, axis=-1) - 1.0
            disc_lf = b_q * b_q - a_q * c_q
            hit_lf = disc_lf > 0.0
            sqrt_d_lf = np.sqrt(np.maximum(disc_lf, 0.0))
            t0_lf = (-b_q - sqrt_d_lf) / np.maximum(a_q, 1e-12)
            t1_lf = (-b_q + sqrt_d_lf) / np.maximum(a_q, 1e-12)
            t_lf = np.where(t0_lf > 1e-4, t0_lf, t1_lf)
            hit_lf = hit_lf & (t_lf > 1e-4)

            take_lf = hit_lf & (t_lf < tb_v)
            if np.any(take_lf):
                p_loc_hit = ro_loc + t_lf[:, None] * rd_loc
                lu = p_loc_hit[:, 0]
                lv = p_loc_hit[:, 2]
                g_loc = p_loc_hit / lf_rad[None, :]
                n_lf = normalize(
                    g_loc[:, 0:1] * u_ax[None, :]
                    + g_loc[:, 1:2] * n_ax[None, :]
                    + g_loc[:, 2:3] * v_ax[None, :]
                )
                tb_v = np.where(take_lf, t_lf, tb_v)
                nb_v = np.where(take_lf[:, None], n_lf, nb_v)
                mb_v = np.where(take_lf, MAT_EUCALYPTUS_LEAF, mb_v)
                uv_v = np.where(take_lf[:, None], np.stack([lu, lv], axis=-1), uv_v)

        # 3. Turned Brushed-Brass Coaster under the Crystal Water Tumbler at [0.70, 0.390, 1.42]
        if not fast_secondary:
            coa_a = np.array([0.70, 0.390, 1.42], dtype=np.float64)
            coa_b = np.array([0.70, 0.3955, 1.42], dtype=np.float64)
            t_coa, hit_coa, n_coa = intersect_cylinder_segment(ro_v, rd_v, coa_a, coa_b, radius=0.050, capped=True)
            take_coa = hit_coa & (t_coa < tb_v)
            tb_v = np.where(take_coa, t_coa, tb_v)
            nb_v = np.where(take_coa[:, None], n_coa, nb_v)
            mb_v = np.where(take_coa, MAT_BRASS_SCONCE, mb_v)

        # 4. Separate Clear Crystalline Glass Water Carafe at [0.52, 0.390, 1.52] + Crystal Tumbler at [0.70, 0.3955, 1.42]
        if include_glass_vessels:
            carafe_base = np.array([0.52, 0.390, 1.52], dtype=np.float64)
            bulb_c = carafe_base + np.array([0.0, 0.095, 0.0], dtype=np.float64)
            bulb_r = np.array([0.088, 0.095, 0.088], dtype=np.float64)
            t_cb, hit_cb, n_cb = intersect_ellipsoid(ro_v, rd_v, bulb_c, bulb_r)

            neck_a = carafe_base + np.array([0.0, 0.155, 0.0], dtype=np.float64)
            neck_b = carafe_base + np.array([0.0, 0.265, 0.0], dtype=np.float64)
            t_cn, hit_cn, n_cn = intersect_cylinder_segment(ro_v, rd_v, neck_a, neck_b, radius=0.036, capped=False)

            flare_c = carafe_base + np.array([0.0, 0.278, 0.0], dtype=np.float64)
            flare_r = np.array([0.046, 0.022, 0.046], dtype=np.float64)
            t_cf, hit_cf, n_cf = intersect_ellipsoid(ro_v, rd_v, flare_c, flare_r)

            lip_c = carafe_base + np.array([0.0, 0.296, 0.0], dtype=np.float64)
            lip_r = np.array([0.050, 0.008, 0.050], dtype=np.float64)
            t_cl, hit_cl, n_cl = intersect_ellipsoid(ro_v, rd_v, lip_c, lip_r)

            t_carafe = np.where(hit_cb, t_cb, 1e9)
            n_carafe = n_cb.copy()
            use_cn = hit_cn & (t_cn < t_carafe)
            t_carafe = np.where(use_cn, t_cn, t_carafe)
            n_carafe = np.where(use_cn[:, None], n_cn, n_carafe)
            use_cf = hit_cf & (t_cf < t_carafe)
            t_carafe = np.where(use_cf, t_cf, t_carafe)
            n_carafe = np.where(use_cf[:, None], n_cf, n_carafe)
            use_cl = hit_cl & (t_cl < t_carafe)
            t_carafe = np.where(use_cl, t_cl, t_carafe)
            n_carafe = np.where(use_cl[:, None], n_cl, n_carafe)

            take_carafe = (t_carafe < tb_v) & (hit_cb | hit_cn | hit_cf | hit_cl)
            if np.any(take_carafe):
                # Optic-mold vertical fluting ribs on the hand-blown clear glass water carafe
                p_crf = ro_v + t_carafe[:, None] * rd_v
                theta_crf = np.arctan2(p_crf[:, 2] - 1.52, p_crf[:, 0] - 0.52)
                rib_w = smoothstep(0.41, 0.58, p_crf[:, 1]) * (1.0 - smoothstep(0.62, 0.68, p_crf[:, 1]))
                flute_nx = np.cos(theta_crf * 20.0) * 0.058 * rib_w
                flute_nz = -np.sin(theta_crf * 20.0) * 0.058 * rib_w
                n_carafe_rib = normalize(
                    n_carafe + np.stack([flute_nx, np.zeros_like(flute_nx), flute_nz], axis=-1)
                )
                tb_v = np.where(take_carafe, t_carafe, tb_v)
                nb_v = np.where(take_carafe[:, None], n_carafe_rib, nb_v)
                mb_v = np.where(take_carafe, MAT_GLASS_CARAFE, mb_v)

            tumb_a = np.array([0.70, 0.3955, 1.42], dtype=np.float64)
            tumb_b = np.array([0.70, 0.515, 1.42], dtype=np.float64)
            t_tb, hit_tb, n_tb = intersect_cylinder_segment(ro_v, rd_v, tumb_a, tumb_b, radius=0.043, capped=True)
            take_tb = hit_tb & (t_tb < tb_v)
            if np.any(take_tb):
                p_tmb = ro_v + t_tb[:, None] * rd_v
                r_tmb_top = np.hypot(p_tmb[:, 0] - 0.70, p_tmb[:, 2] - 1.42)
                r_tmb_in = 0.0385
                y_tmb_water = 0.468
                in_tmb_hollow = take_tb & (n_tb[:, 1] > 0.85) & (r_tmb_top < r_tmb_in)

                t_tw = (y_tmb_water - ro_v[:, 1]) / np.where(np.abs(rd_v[:, 1]) < 1e-8, -1e-8, rd_v[:, 1])
                p_tw = ro_v + t_tw[:, None] * rd_v
                tw_rx = p_tw[:, 0] - 0.70
                tw_rz = p_tw[:, 2] - 1.42
                tw_r = np.hypot(tw_rx, tw_rz)
                hits_tw_men = in_tmb_hollow & (t_tw > t_tb) & (tw_r <= r_tmb_in)

                oc_tx = ro_v[:, 0] - 0.70
                oc_tz = ro_v[:, 2] - 1.42
                a_tin = np.maximum(rd_v[:, 0] ** 2 + rd_v[:, 2] ** 2, 1e-12)
                b_tin = oc_tx * rd_v[:, 0] + oc_tz * rd_v[:, 2]
                c_tin = oc_tx * oc_tx + oc_tz * oc_tz - r_tmb_in * r_tmb_in
                disc_tin = np.maximum(b_tin * b_tin - a_tin * c_tin, 0.0)
                t_tin_wall = (-b_tin + np.sqrt(disc_tin)) / a_tin
                p_tin_wall = ro_v + t_tin_wall[:, None] * rd_v
                n_tin_wall = normalize(
                    np.stack([-(p_tin_wall[:, 0] - 0.70), np.zeros_like(t_tin_wall), -(p_tin_wall[:, 2] - 1.42)], axis=-1)
                )
                hits_tin_wall = in_tmb_hollow & (~hits_tw_men) & (t_tin_wall > t_tb)

                tw_lip = smoothstep(r_tmb_in - 0.006, r_tmb_in, tw_r) * 0.24
                n_tw_men = normalize(
                    np.stack(
                        [
                            -tw_lip * tw_rx / np.maximum(tw_r, 1e-5),
                            np.ones_like(tw_r),
                            -tw_lip * tw_rz / np.maximum(tw_r, 1e-5),
                        ],
                        axis=-1,
                    )
                )

                on_tmb_rim = take_tb & (n_tb[:, 1] > 0.85) & (r_tmb_top >= r_tmb_in)
                tmb_rim_u = np.clip((r_tmb_top - 0.5 * (r_tmb_in + 0.043)) / (0.5 * (0.043 - r_tmb_in)), -0.96, 0.96)
                tmb_rim_ny = np.sqrt(np.maximum(1.0 - tmb_rim_u * tmb_rim_u, 0.08))
                n_tmb_rim = normalize(
                    np.stack(
                        [
                            tmb_rim_u * (p_tmb[:, 0] - 0.70) / np.maximum(r_tmb_top, 1e-5),
                            tmb_rim_ny,
                            tmb_rim_u * (p_tmb[:, 2] - 1.42) / np.maximum(r_tmb_top, 1e-5),
                        ],
                        axis=-1,
                    )
                )

                t_tb_eff = np.where(hits_tw_men, t_tw, np.where(hits_tin_wall, t_tin_wall, t_tb))
                n_tb_eff = np.where(
                    hits_tw_men[:, None],
                    n_tw_men,
                    np.where(hits_tin_wall[:, None], n_tin_wall, np.where(on_tmb_rim[:, None], n_tmb_rim, n_tb)),
                )
                p_tmb_eff = ro_v + t_tb_eff[:, None] * rd_v
                theta_tmb = np.arctan2(p_tmb_eff[:, 2] - 1.42, p_tmb_eff[:, 0] - 0.70)
                cut_w = (1.0 - smoothstep(0.45, 0.48, p_tmb_eff[:, 1])) * (~hits_tw_men) * (~on_tmb_rim)
                tfl_nx = np.cos(theta_tmb * 16.0) * 0.065 * cut_w
                tfl_nz = -np.sin(theta_tmb * 16.0) * 0.065 * cut_w
                n_tb_cut = normalize(n_tb_eff + np.stack([tfl_nx, np.zeros_like(tfl_nx), tfl_nz], axis=-1))
                tb_v = np.where(take_tb, t_tb_eff, tb_v)
                nb_v = np.where(take_tb[:, None], n_tb_cut, nb_v)
                mb_v = np.where(take_tb, MAT_WATER_TUMBLER, mb_v)

        t_best[idx_vas] = tb_v
        n_best[idx_vas] = nb_v
        mat_best[idx_vas] = mb_v
        alb_best[idx_vas] = ab_v
        rough_best[idx_vas] = rb_v
        uv_best[idx_vas] = uv_v

    pos_best = ro_flat + t_best[:, None] * rd_flat

    # ------------------------------------------------------------------
    # J. Assign Detailed Material Albedo, Roughness, Normals & F0
    # ------------------------------------------------------------------
    mask_floor = mat_best == MAT_FLOOR_OAK
    if np.any(mask_floor):
        p_fl = pos_best[mask_floor]
        in_trav = (
            (p_fl[:, 0] >= -1.68)
            & (p_fl[:, 0] <= 1.56)
            & (p_fl[:, 2] >= 1.38)
            & (p_fl[:, 2] <= 4.38)
        )
        dist_border_x = np.minimum(np.abs(p_fl[:, 0] - (-1.68)), np.abs(p_fl[:, 0] - 1.56))
        dist_border_z = np.minimum(np.abs(p_fl[:, 2] - 1.38), np.abs(p_fl[:, 2] - 4.38))
        on_brass_inlay = (
            ((dist_border_x < 0.0085) & (p_fl[:, 2] >= 1.37) & (p_fl[:, 2] <= 4.39))
            | ((dist_border_z < 0.0085) & (p_fl[:, 0] >= -1.69) & (p_fl[:, 0] <= 1.57))
        )

        fl_alb, fl_rough, fl_aniso, fl_n = material_oiled_oak_and_travertine_floor(p_fl, in_trav)
        brass_col = np.array([0.82, 0.62, 0.28], dtype=np.float64)
        fl_alb = np.where(on_brass_inlay[:, None], brass_col[None, :], fl_alb)
        fl_rough = np.where(on_brass_inlay, 0.14, fl_rough)

        alb_best[mask_floor] = fl_alb
        rough_best[mask_floor] = fl_rough
        n_best[mask_floor] = fl_n
        uv_best[mask_floor] = fl_aniso
        mat_best[mask_floor] = np.where(
            on_brass_inlay,
            MAT_BRASS_INLAY,
            np.where(in_trav, MAT_FLOOR_TRAVERTINE, MAT_FLOOR_OAK),
        )
        f0_best[mask_floor] = np.where(
            on_brass_inlay[:, None],
            np.array([[0.85, 0.65, 0.30]]),
            np.where(in_trav[:, None], 0.048, 0.044),
        )
        metal_best[mask_floor] = np.where(on_brass_inlay, 0.90, 0.0)

    mask_rug = (mat_best == MAT_BOUCLE_RUG) | (mat_best == MAT_RUG_LINEN_BORDER)
    if np.any(mask_rug):
        p_rg = pos_best[mask_rug]
        n_rg = n_best[mask_rug]
        is_bdr = mat_best[mask_rug] == MAT_RUG_LINEN_BORDER
        rg_alb, rg_rough, rg_n = material_woven_boucle_rug(p_rg, n_rg, is_bdr, fast_mode=fast_secondary)
        alb_best[mask_rug] = rg_alb
        rough_best[mask_rug] = rg_rough
        n_best[mask_rug] = rg_n
        f0_best[mask_rug] = 0.032

    mask_conc = mat_best == MAT_BACK_CONCRETE
    if np.any(mask_conc):
        p_c = pos_best[mask_conc]
        n_c = n_best[mask_conc]
        c_alb, c_rough, c_n = material_board_formed_concrete(
            p_c, n_c, is_ceiling=False, fast_mode=fast_secondary
        )
        alb_best[mask_conc] = c_alb
        rough_best[mask_conc] = c_rough
        n_best[mask_conc] = c_n
        f0_best[mask_conc] = 0.042

    mask_ceil = mat_best == MAT_CEILING_CONCRETE
    if np.any(mask_ceil):
        p_ce = pos_best[mask_ceil]
        n_ce = n_best[mask_ceil]
        ce_alb, ce_rough, ce_n = material_board_formed_concrete(
            p_ce, n_ce, is_ceiling=True, fast_mode=fast_secondary
        )
        alb_best[mask_ceil] = ce_alb
        rough_best[mask_ceil] = ce_rough
        n_best[mask_ceil] = ce_n
        f0_best[mask_ceil] = 0.040

    mask_wal = (mat_best == MAT_WALNUT_CASEWORK) | (mat_best == MAT_COFFEE_TABLE_BASE)
    if np.any(mask_wal):
        p_w = pos_best[mask_wal]
        n_w = n_best[mask_wal]
        m_w = mat_best[mask_wal]
        w_alb, w_rough, w_n = material_american_walnut(p_w, n_w, fast_mode=fast_secondary)

        # Architectural Fluted Walnut Tambour Staves + Brushed Brass Plinth Toe-Kick on Foreground Table Base
        is_tbase = m_w == MAT_COFFEE_TABLE_BASE
        if np.any(is_tbase):
            on_toekick = is_tbase & (p_w[:, 1] < 0.032)
            on_tambour = is_tbase & (~on_toekick)
            flute_phase = p_w[:, 0] * 235.0
            stave_crown = 0.5 + 0.5 * np.cos(flute_phase)
            stave_nx = -np.sin(flute_phase) * 0.46 * on_tambour
            w_n = normalize(w_n + np.stack([stave_nx, np.zeros_like(stave_nx), np.zeros_like(stave_nx)], axis=-1))
            w_alb = np.where(
                on_tambour[:, None],
                w_alb * (0.76 + 0.42 * stave_crown[:, None]),
                w_alb,
            )
            w_rough = np.where(on_tambour, 0.22 + 0.08 * (1.0 - stave_crown), w_rough)

            brass_plinth_col = np.array([0.82, 0.62, 0.28], dtype=np.float64)
            w_alb = np.where(on_toekick[:, None], brass_plinth_col[None, :], w_alb)
            w_rough = np.where(on_toekick, 0.17, w_rough)

        alb_best[mask_wal] = w_alb
        rough_best[mask_wal] = w_rough
        n_best[mask_wal] = w_n
        f0_best[mask_wal] = np.where(
            (is_tbase & (p_w[:, 1] < 0.032))[:, None],
            np.array([[0.85, 0.65, 0.30]], dtype=np.float64),
            0.046,
        )
        metal_best[mask_wal] = np.where(is_tbase & (p_w[:, 1] < 0.032), 0.90, 0.0)

    mask_tt = mat_best == MAT_COFFEE_TABLE_TOP
    if np.any(mask_tt):
        p_t = pos_best[mask_tt]
        n_t = n_best[mask_tt]
        tx, ty, tz = p_t[:, 0], p_t[:, 1], p_t[:, 2]
        tb = fbm2d(tx * 4.2 + ty * 9.0, tz * 3.6, octaves=2 if fast_secondary else 3, seed=853)
        tv1 = 0.5 + 0.5 * np.sin(tx * 14.0 + tb * 6.8 + tz * 3.8 + ty * 22.0)
        tv2 = 0.5 + 0.5 * np.cos(tx * 32.0 - tb * 8.2 + tz * 5.2)
        tv = 0.65 * tv1 + 0.35 * tv2

        # Calcite veinlets + high-frequency sedimentary micro-vugs on the travertine tabletop
        calcite_t = np.exp(-((np.abs(np.sin(tx * 22.0 + tb * 7.5 + tz * 4.5)) / 0.05) ** 2))
        vug_dt, vug_nxt, vug_nzt = cellular_bugholes_2d(
            tx * 190.0 + ty * 80.0, tz * 62.0 + ty * 45.0, density_prob=0.38, seed=861
        )
        pits = vug_dt * smoothstep(0.25, 0.72, 1.0 - tv1)

        t_col1 = np.array([0.65, 0.58, 0.475], dtype=np.float64)
        t_col2 = np.array([0.50, 0.42, 0.33], dtype=np.float64)
        t_calc = np.array([0.75, 0.69, 0.59], dtype=np.float64)
        t_alb = (
            t_col1[None, :] * tv[:, None]
            + t_col2[None, :] * (1.0 - tv[:, None])
            + t_calc[None, :] * (0.14 * calcite_t[:, None])
        )
        t_alb *= (1.0 - 0.45 * pits)[:, None]

        alb_best[mask_tt] = t_alb
        rough_best[mask_tt] = np.clip(0.125 + 0.075 * (1.0 - tv) + 0.28 * pits, 0.10, 0.44)
        f0_best[mask_tt] = 0.048
        if not fast_secondary:
            n_best[mask_tt] = normalize(
                n_t + np.stack([vug_nxt * 0.16, np.zeros_like(vug_nxt), vug_nzt * 0.14], axis=-1)
            )

    mask_lth = mat_best == MAT_LEATHER_CHAIR
    if np.any(mask_lth):
        p_l = pos_best[mask_lth]
        n_l = n_best[mask_lth]
        uv_l = uv_best[mask_lth]
        l_alb, l_rough, l_n = material_cognac_saddle_leather(p_l, n_l, uv_l, fast_mode=fast_secondary)
        alb_best[mask_lth] = l_alb
        rough_best[mask_lth] = l_rough
        f0_best[mask_lth] = 0.048
        n_best[mask_lth] = l_n

    mask_strp = mat_best == MAT_LEATHER_STRAP
    if np.any(mask_strp):
        alb_best[mask_strp] = np.array([0.14, 0.055, 0.022], dtype=np.float64)
        rough_best[mask_strp] = 0.30
        f0_best[mask_strp] = 0.045

    mask_chr = mat_best == MAT_CHAIR_STEEL
    if np.any(mask_chr):
        alb_best[mask_chr] = np.array([0.75, 0.76, 0.78], dtype=np.float64)
        rough_best[mask_chr] = 0.17
        f0_best[mask_chr] = np.array([0.74, 0.75, 0.77], dtype=np.float64)
        metal_best[mask_chr] = 0.92

    mask_mul = mat_best == MAT_STEEL_MULLION
    if np.any(mask_mul):
        alb_best[mask_mul] = np.array([0.058, 0.062, 0.068], dtype=np.float64)
        rough_best[mask_mul] = 0.25
        f0_best[mask_mul] = np.array([0.16, 0.16, 0.17], dtype=np.float64)
        metal_best[mask_mul] = 0.68

    mask_gsk = mat_best == MAT_GLAZING_GASKET
    if np.any(mask_gsk):
        alb_best[mask_gsk] = np.array([0.025, 0.026, 0.028], dtype=np.float64)
        rough_best[mask_gsk] = 0.65
        f0_best[mask_gsk] = 0.032

    mask_brs = mat_best == MAT_BRASS_SCONCE
    if np.any(mask_brs):
        alb_best[mask_brs] = np.array([0.83, 0.63, 0.29], dtype=np.float64)
        rough_best[mask_brs] = 0.19
        f0_best[mask_brs] = np.array([0.86, 0.66, 0.31], dtype=np.float64)
        metal_best[mask_brs] = 0.94

    # Framed Architectural Bas-Relief / Charcoal Artwork on Back Credenza
    mask_afr = mat_best == MAT_ART_FRAME
    if np.any(mask_afr):
        alb_best[mask_afr] = np.array([0.07, 0.055, 0.045], dtype=np.float64)
        rough_best[mask_afr] = 0.28
        f0_best[mask_afr] = 0.042

    mask_acv = mat_best == MAT_ART_CANVAS
    if np.any(mask_acv):
        uv_a = uv_best[mask_acv]
        n_a = n_best[mask_acv]
        ua, va = uv_a[:, 0], uv_a[:, 1]
        in_inner_art = (np.abs(ua) < 0.72) & (np.abs(va) < 0.74)
        # Geometric architectural bas-relief & charcoal composition (Ando/Chillida inspired)
        arch_circle = np.hypot(ua + 0.12, va - 0.08) < 0.38
        charcoal_block = (ua > -0.05) & (ua < 0.58) & (va > -0.56) & (va < 0.32)
        plaster_rib = np.sin(ua * 45.0) * ((ua < 0.05) & (np.abs(va) < 0.60))

        mat_col = np.array([0.68, 0.64, 0.57], dtype=np.float64)
        plaster_col = np.array([0.60, 0.56, 0.49], dtype=np.float64)
        ochre_col = np.array([0.48, 0.35, 0.23], dtype=np.float64)
        charcoal_col = np.array([0.13, 0.12, 0.11], dtype=np.float64)

        art_alb = np.where(in_inner_art[:, None], plaster_col[None, :], mat_col[None, :])
        art_alb = np.where((in_inner_art & arch_circle)[:, None], ochre_col[None, :], art_alb)
        art_alb = np.where((in_inner_art & charcoal_block)[:, None], charcoal_col[None, :], art_alb)
        alb_best[mask_acv] = art_alb
        rough_best[mask_acv] = np.where(in_inner_art & charcoal_block, 0.72, 0.54)
        n_best[mask_acv] = normalize(
            n_a + np.stack([plaster_rib * 0.18, np.zeros_like(plaster_rib), np.zeros_like(plaster_rib)], axis=-1)
        )

    # Open Hardcover Art Monograph Pages & Stacked Page-Block Striations
    mask_opg = mat_best == MAT_OPEN_BOOK_PAGE
    if np.any(mask_opg):
        p_pg = pos_best[mask_opg]
        n_pg = n_best[mask_opg]
        uv_p = uv_best[mask_opg]
        u_p, v_p = uv_p[:, 0], uv_p[:, 1]
        paper_base = np.array([0.77, 0.73, 0.65], dtype=np.float64)
        gutter = 0.48 + 0.52 * smoothstep(0.0, 0.115, np.abs(u_p))

        # Stacked paper page-block striations on the exposed side & bottom fore-edges
        on_fore_edge = n_pg[:, 1] < 0.72
        page_striation = 0.84 + 0.16 * np.sin(p_pg[:, 1] * 1150.0)

        # Left page: Architectural section drawing (top) + duotone concrete pavilion plate (bottom)
        in_left_plate = (u_p > -0.84) & (u_p < -0.14) & (v_p > -0.10) & (v_p < 0.72)
        in_left_drawing = (u_p > -0.84) & (u_p < -0.14) & (v_p > -0.72) & (v_p < -0.16)
        colonnade_grid = (np.abs(np.sin(u_p * 38.0)) < 0.14) | (np.abs(np.sin(v_p * 28.0)) < 0.12)
        plate_tone = np.where(
            v_p < 0.32,
            0.20 + 0.14 * np.sin(u_p * 22.0) + 0.08 * np.cos(v_p * 18.0),
            0.52 - 0.18 * (v_p - 0.32),
        )
        plate_col = np.stack([plate_tone * 0.96, plate_tone * 0.89, plate_tone * 0.79], axis=-1)

        # Right page: Venetian red rubricated title header + two-column Swiss typography
        in_rubric = (u_p > 0.15) & (u_p < 0.58) & (v_p > -0.68) & (v_p < -0.58) & (np.sin(v_p * 120.0) > -0.2)
        in_col1 = (u_p > 0.15) & (u_p < 0.46) & (v_p > -0.50) & (v_p < 0.58)
        in_col2 = (u_p > 0.50) & (u_p < 0.81) & (v_p > -0.50) & (v_p < 0.42)
        text_line = (np.sin(v_p * 115.0) > 0.12) & (np.cos(u_p * 65.0 + v_p * 30.0) > -0.65)
        in_text = (in_col1 | in_col2) & text_line

        pg_alb = paper_base[None, :] * gutter[:, None]
        pg_alb = np.where(in_left_plate[:, None], plate_col * gutter[:, None], pg_alb)
        pg_alb = np.where((in_left_drawing & colonnade_grid)[:, None], pg_alb * 0.48, pg_alb)
        pg_alb = np.where(
            in_rubric[:, None], np.array([[0.58, 0.16, 0.12]], dtype=np.float64) * gutter[:, None], pg_alb
        )
        pg_alb = np.where(in_text[:, None], pg_alb * 0.55, pg_alb)
        pg_alb = np.where(on_fore_edge[:, None], paper_base[None, :] * (0.78 * page_striation[:, None]), pg_alb)

        alb_best[mask_opg] = pg_alb
        rough_best[mask_opg] = np.where(in_left_plate, 0.36, 0.58)
        f0_best[mask_opg] = 0.036

    mask_ocv = mat_best == MAT_OPEN_BOOK_COVER
    if np.any(mask_ocv):
        alb_best[mask_ocv] = np.array([0.14, 0.16, 0.19], dtype=np.float64)
        rough_best[mask_ocv] = 0.48

    mask_rib = mat_best == MAT_BOOKMARK_RIBBON
    if np.any(mask_rib):
        alb_best[mask_rib] = np.array([0.48, 0.065, 0.085], dtype=np.float64)
        rough_best[mask_rib] = 0.24
        f0_best[mask_rib] = 0.048

    mask_esp = mat_best == MAT_ESPRESSO_CUP
    if np.any(mask_esp):
        p_e = pos_best[mask_esp]
        sp_e = value_noise_2d(p_e[:, 0] * 180.0, p_e[:, 2] * 180.0 + p_e[:, 1] * 180.0, seed=929)
        alb_best[mask_esp] = np.array([0.62, 0.57, 0.49], dtype=np.float64) * (0.90 + 0.18 * sp_e[:, None])
        rough_best[mask_esp] = 0.22
        f0_best[mask_esp] = 0.046

    mask_crm = mat_best == MAT_ESPRESSO_CREMA
    if np.any(mask_crm):
        alb_best[mask_crm] = np.array([0.22, 0.11, 0.045], dtype=np.float64)
        rough_best[mask_crm] = 0.08
        f0_best[mask_crm] = 0.042

    # Silver-Dollar Eucalyptus Leaves with Embossed Midrib & Lateral Veins + Glaucous Waxy Bloom
    mask_lf = mat_best == MAT_EUCALYPTUS_LEAF
    if np.any(mask_lf):
        p_lf = pos_best[mask_lf]
        n_lf = n_best[mask_lf]
        uv_lf = uv_best[mask_lf]
        lu, lv = uv_lf[:, 0], uv_lf[:, 1]

        vn = value_noise_2d(p_lf[:, 0] * 95.0, p_lf[:, 2] * 95.0, seed=941)
        midrib = np.exp(-((lv / 0.085) ** 2)) * (1.0 - smoothstep(0.75, 1.0, np.abs(lu)))
        lat_vein_phase = np.abs(np.sin(lu * 9.5 - np.abs(lv) * 6.2))
        lat_vein = np.exp(-((lat_vein_phase / 0.14) ** 2)) * (1.0 - midrib) * smoothstep(0.05, 0.75, np.abs(lv))
        vein_mask = np.clip(midrib + 0.55 * lat_vein, 0.0, 1.0)

        leaf_glaucous = np.array([0.23, 0.35, 0.29], dtype=np.float64)
        vein_sage = np.array([0.36, 0.46, 0.34], dtype=np.float64)
        lf_alb = (
            leaf_glaucous[None, :] * (1.0 - 0.45 * vein_mask[:, None])
            + vein_sage[None, :] * (0.45 * vein_mask[:, None])
        ) * (0.88 + 0.22 * vn[:, None])

        vein_nx = np.cos(lu * 9.5 - np.abs(lv) * 6.2) * lat_vein * 0.16
        vein_nz = -np.sign(lv) * midrib * 0.22
        alb_best[mask_lf] = lf_alb
        rough_best[mask_lf] = np.clip(0.28 + 0.08 * (1.0 - vn) - 0.06 * midrib, 0.20, 0.38)
        f0_best[mask_lf] = 0.050
        n_best[mask_lf] = normalize(n_lf + np.stack([vein_nx, np.zeros_like(vein_nx), vein_nz], axis=-1))

    mask_st = mat_best == MAT_EUCALYPTUS_STEM
    if np.any(mask_st):
        alb_best[mask_st] = np.array([0.20, 0.145, 0.10], dtype=np.float64)
        rough_best[mask_st] = 0.48

    return {
        "t": t_best.reshape(orig_shape),
        "pos": pos_best.reshape((*orig_shape, 3)),
        "normal": n_best.reshape((*orig_shape, 3)),
        "mat_id": mat_best.reshape(orig_shape),
        "albedo": alb_best.reshape((*orig_shape, 3)),
        "roughness": rough_best.reshape(orig_shape),
        "f0": f0_best.reshape((*orig_shape, 3)),
        "metallic": metal_best.reshape(orig_shape),
        "uv": uv_best.reshape((*orig_shape, 2)),
    }



# ==============================================================================
# 11. DIRECT + INDIRECT GLOBAL ILLUMINATION, CAUSTICS & SHADING ENGINE
# ==============================================================================


def compute_analytic_ambient_occlusion(pos: np.ndarray, normal: np.ndarray, mat_id: np.ndarray) -> np.ndarray:
    """
    Compute 3D multi-scale architectural & furniture contact ambient occlusion:
      - Room macro corner/wall-floor/wall-ceiling crease darkening
      - Deep walnut bookshelf alcove & recessed shelf-header cavity occlusion
      - Floor contact shadows under the travertine coffee table, leather lounge
        chair, back credenza, and around the 3D woven bouclé rug bevel
      - Tabletop contact occlusion under the glass carafe, brass coaster & tumbler,
        open art monograph, stoneware espresso cup/saucer, and stacked books
      - Credenza contact occlusion under the leaning framed artwork, folio stack,
        and fluted travertine vessels
    """
    px, py, pz = pos[..., 0], pos[..., 1], pos[..., 2]

    d_floor = np.clip(py / 0.38, 0.0, 1.0)
    d_ceil = np.clip((ROOM_Y_CEIL - py) / 0.45, 0.0, 1.0)
    d_back = np.clip((ROOM_Z_BACK - pz) / 0.45, 0.0, 1.0)
    d_left = np.clip((px - (-2.72)) / 0.42, 0.0, 1.0)

    corner_ao = (
        (0.74 + 0.26 * np.sqrt(d_floor * d_back))
        * (0.76 + 0.24 * np.sqrt(d_ceil * d_back))
        * (0.72 + 0.28 * np.sqrt(np.maximum(0.0, d_left) * d_floor))
    )

    alcove_depth = np.clip((-2.745 - px) / 0.36, 0.0, 1.0)
    shelf_dist = np.min(np.abs(py[..., None] - (BOOKCASE_SHELF_Y[None, :] + 0.021)), axis=-1)
    stile_dist = np.min(np.abs(pz[..., None] - BOOKCASE_STILE_Z[None, :]), axis=-1)
    shelf_gap = np.maximum(0.0, shelf_dist - 0.021)
    stile_gap = np.maximum(0.0, stile_dist - 0.026)
    ao_case = (0.76 + 0.24 * smoothstep(0.0, 0.09, shelf_gap)) * (0.82 + 0.18 * smoothstep(0.0, 0.07, stile_gap))
    alcove_ao = np.where(px < -2.745, (1.0 - 0.42 * (alcove_depth ** 1.2)) * ao_case, 1.0)

    on_floor_or_rug = py < 0.016
    dx_tab = np.maximum(np.abs(px - 0.33) - 0.56, 0.0)
    dz_tab = np.maximum(np.abs(pz - 1.77) - 0.62, 0.0)
    r_tab = np.hypot(dx_tab, dz_tab)
    ao_under_table = np.where(on_floor_or_rug, 0.30 + 0.70 * smoothstep(0.0, 0.36, r_tab), 1.0)

    r_chair = np.hypot(px - (-1.18), pz - 2.28)
    cy_ch_ao, sy_ch_ao = math.cos(-0.56), math.sin(-0.56)
    lcx_ao = cy_ch_ao * (px - (-1.18)) - sy_ch_ao * (pz - 2.28)
    lcz_ao = sy_ch_ao * (px - (-1.18)) + cy_ch_ao * (pz - 2.28)
    d_runner_ao = np.hypot(np.abs(np.abs(lcx_ao) - 0.425), np.maximum(np.abs(lcz_ao - 0.005) - 0.382, 0.0))
    ao_chair_runners = (
        (0.07 + 0.93 * smoothstep(0.008, 0.030, d_runner_ao))
        * (0.64 + 0.36 * smoothstep(0.012, 0.075, d_runner_ao))
    )
    ao_under_chair = np.where(
        on_floor_or_rug,
        (0.32 + 0.68 * smoothstep(0.18, 0.64, r_chair)) * ao_chair_runners,
        1.0,
    )

    dx_bn = np.maximum(np.abs(px - (-0.85)) - 1.72, 0.0)
    dz_bn = np.maximum(np.abs(pz - 7.22) - 0.32, 0.0)
    ao_under_bench = np.where(on_floor_or_rug, 0.34 + 0.66 * smoothstep(0.0, 0.28, np.hypot(dx_bn, dz_bn)), 1.0)

    # Contact crease shadow on the oak/travertine floor right along the 13mm 3D rug edge
    on_bare_floor = py < 0.004
    dx_rug = np.maximum(np.maximum(-1.88 - px, px - 1.04), 0.0)
    dz_rug = np.maximum(np.maximum(0.94 - pz, pz - 3.10), 0.0)
    d_rug_edge = np.hypot(dx_rug, dz_rug)
    ao_rug_edge = np.where(
        on_bare_floor & (d_rug_edge > 0.0) & (d_rug_edge < 0.055),
        0.48 + 0.52 * smoothstep(0.0, 0.050, d_rug_edge),
        1.0,
    )

    on_tabletop = (mat_id == MAT_COFFEE_TABLE_TOP) & (normal[..., 1] > 0.8)
    r_carafe = np.hypot(px - 0.52, pz - 1.52)
    ao_carafe = np.where(on_tabletop, 0.18 + 0.82 * smoothstep(0.055, 0.135, r_carafe), 1.0)

    r_tumbler = np.hypot(px - 0.70, pz - 1.42)
    ao_tumbler = np.where(on_tabletop, 0.22 + 0.78 * smoothstep(0.045, 0.095, r_tumbler), 1.0)

    r_ovase = np.hypot(px - 0.44, pz - 1.98)
    ao_ovase = np.where(on_tabletop, 0.16 + 0.84 * smoothstep(0.068, 0.145, r_ovase), 1.0)

    r_espresso = np.hypot(px - 0.23, pz - 1.84)
    ao_espresso = np.where(on_tabletop, 0.35 + 0.65 * smoothstep(0.068, 0.084, r_espresso), 1.0)

    cy_b, sy_b = math.cos(-0.19), math.sin(-0.19)
    bx_l = cy_b * (px - 0.04) - sy_b * (pz - 1.54)
    bz_l = sy_b * (px - 0.04) + cy_b * (pz - 1.54)
    d_book = np.hypot(np.maximum(np.abs(bx_l) - 0.232, 0.0), np.maximum(np.abs(bz_l) - 0.165, 0.0))
    ao_book = np.where(on_tabletop, 0.35 + 0.65 * smoothstep(0.0, 0.024, d_book), 1.0)

    cy_s, sy_s = math.cos(0.22), math.sin(0.22)
    sx_l = cy_s * (px - (-0.11)) - sy_s * (pz - 2.04)
    sz_l = sy_s * (px - (-0.11)) + cy_s * (pz - 2.04)
    d_sbook = np.hypot(np.maximum(np.abs(sx_l) - 0.162, 0.0), np.maximum(np.abs(sz_l) - 0.210, 0.0))
    ao_sbook = np.where(on_tabletop, 0.35 + 0.65 * smoothstep(0.0, 0.022, d_sbook), 1.0)

    # Contact AO on the back credenza top under the leaning frame, folio stack & vessels
    on_credenza = (py > 0.515) & (py < 0.526) & (pz > 6.85) & (normal[..., 1] > 0.8)
    d_art_cred = np.hypot(np.maximum(np.abs(px - (-1.45)) - 0.44, 0.0), np.maximum(np.abs(pz - 7.18) - 0.06, 0.0))
    d_fol_cred = np.hypot(np.maximum(np.abs(px - (-0.52)) - 0.20, 0.0), np.maximum(np.abs(pz - 7.18) - 0.15, 0.0))
    r_v1_cred = np.hypot(px - 0.12, pz - 7.16)
    r_v2_cred = np.hypot(px - 0.34, pz - 7.24)
    ao_credenza = np.where(
        on_credenza,
        (0.30 + 0.70 * smoothstep(0.0, 0.055, d_art_cred))
        * (0.26 + 0.74 * smoothstep(0.0, 0.055, d_fol_cred))
        * (0.25 + 0.75 * smoothstep(0.065, 0.14, r_v1_cred))
        * (0.28 + 0.72 * smoothstep(0.050, 0.12, r_v2_cred)),
        1.0,
    )

    return (
        corner_ao
        * alcove_ao
        * ao_under_table
        * ao_under_chair
        * ao_under_bench
        * ao_rug_edge
        * ao_carafe
        * ao_tumbler
        * ao_ovase
        * ao_espresso
        * ao_book
        * ao_sbook
        * ao_credenza
    )


def evaluate_refractive_caustics(pos: np.ndarray, normal: np.ndarray, mat_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the focused refractive optical caustic irradiance (`n = 1.52` hand-blown
    18-ribbed clear glass carafe + `n = 1.333` water meniscus + 12-faceted crystal
    water tumbler) with Cauchy spectral dispersion fringes (`n_R = 1.512`, `n_G = 1.520`,
    `n_B = 1.531`) cast onto the sunlit travertine tabletop and open art monograph.
    """
    is_receiver = (
        (
            (mat_id == MAT_COFFEE_TABLE_TOP)
            | (mat_id == MAT_OPEN_BOOK_PAGE)
            | (mat_id == MAT_OPEN_BOOK_COVER)
            | (mat_id == MAT_BRASS_SCONCE)
        )
        & (normal[..., 1] > 0.5)
    )
    caustic_rad = np.zeros((*pos.shape[:-1], 3), dtype=np.float64)
    refr_shadow = np.ones(pos.shape[:-1], dtype=np.float64)
    if not np.any(is_receiver):
        return caustic_rad, refr_shadow

    sun_horiz = normalize(np.array([-SUN_DIR[0], -SUN_DIR[2]], dtype=np.float64))
    sun_perp = np.array([-sun_horiz[1], sun_horiz[0]], dtype=np.float64)

    p_rec = pos[is_receiver]
    xz = p_rec[:, [0, 2]]

    def vessel_caustic_pattern(
        vessel_xz: np.ndarray,
        radius: float,
        focal_dist: float,
        tint_core: np.ndarray,
        tint_body: np.ndarray,
        rib_freq: float,
        intensity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rel = xz - vessel_xz[None, :]
        u_along = rel @ sun_horiz
        v_perp = rel @ sun_perp

        u_n = u_along / focal_dist
        v_n = v_perp / radius

        ell_r = np.hypot((u_along - 0.85 * focal_dist) / (1.35 * focal_dist), v_perp / (1.05 * radius))
        umbra = 1.0 - 0.64 * (1.0 - smoothstep(0.45, 1.15, ell_r))

        in_downstream = smoothstep(0.06, 0.30, u_n) * (1.0 - smoothstep(1.68, 2.40, u_n))
        envelope_w = np.maximum(0.08, 0.78 * np.sqrt(np.clip(1.55 - u_n * 0.72, 0.0, 2.0)))
        edge_dist = np.abs(np.abs(v_n) - envelope_w * 0.68)

        # Primary cardioid caustic fold + secondary water-meniscus inner loop
        caustic_arc = np.exp(-((edge_dist / 0.125) ** 2)) * in_downstream
        inner_loop_dist = np.abs(np.abs(v_n) - envelope_w * 0.36)
        inner_arc = np.exp(-((inner_loop_dist / 0.11) ** 2)) * in_downstream * 0.68
        cusp_focus = np.exp(-(((u_n - 1.02) / 0.26) ** 2 + (v_n / 0.22) ** 2)) * 1.95

        # Optic-mold vertical rib / crystal facet caustic striations
        striation = (
            0.52 + 0.48 * np.cos(u_n * rib_freq - v_n * v_n * 7.2)
        ) * np.exp(-((v_n / (envelope_w + 0.15)) ** 2)) * in_downstream

        # Cauchy chromatic dispersion fringes (warm ruby-amber outer, emerald-cyan inner)
        disp_warm = np.exp(-(((np.abs(v_n) - envelope_w * 0.78) / 0.075) ** 2)) * in_downstream
        disp_cool = np.exp(-(((np.abs(v_n) - envelope_w * 0.57) / 0.070) ** 2)) * in_downstream

        c_rgb = (
            (caustic_arc + cusp_focus)[:, None] * tint_core[None, :]
            + (inner_arc + striation * 0.55)[:, None] * tint_body[None, :]
            + (disp_warm * 0.52)[:, None] * np.array([2.10, 0.95, 0.22], dtype=np.float64)[None, :]
            + (disp_cool * 0.44)[:, None] * np.array([0.22, 1.45, 1.65], dtype=np.float64)[None, :]
        ) * intensity
        return c_rgb, umbra

    c_carafe, u_carafe = vessel_caustic_pattern(
        vessel_xz=np.array([0.52, 1.52], dtype=np.float64),
        radius=0.092,
        focal_dist=0.23,
        tint_core=np.array([15.8, 13.8, 5.8], dtype=np.float64),
        tint_body=np.array([5.4, 10.2, 4.2], dtype=np.float64),
        rib_freq=22.0,
        intensity=1.24,
    )

    c_tumb, u_tumb = vessel_caustic_pattern(
        vessel_xz=np.array([0.70, 1.42], dtype=np.float64),
        radius=0.045,
        focal_dist=0.14,
        tint_core=np.array([16.2, 12.6, 6.8], dtype=np.float64),
        tint_body=np.array([9.2, 7.4, 3.8], dtype=np.float64),
        rib_freq=14.0,
        intensity=1.15,
    )

    caustic_rad[is_receiver] = c_carafe + c_tumb
    refr_shadow[is_receiver] = u_carafe * u_tumb
    return caustic_rad, refr_shadow


def shade_surfaces_radiosity(
    hit_buf: dict[str, np.ndarray],
    rd: np.ndarray,
    is_secondary: bool = False,
) -> np.ndarray:
    """
    Evaluate Kajiya's rendering equation in linear float64 HDR radiance:
      - Direct late-afternoon solar irradiance (`SUN_RADIANCE`) with 3D mullion
        & furniture soft penumbra shadows, Oren-Nayar rough micro-cavity diffuse,
        and GGX microfacet specular
      - Focused refractive glass & water meniscus caustics with Cauchy dispersion
      - Anisotropic/Velvet grazing fiber backscatter on the woven wool bouclé rug
        and cloth book bindings
      - Dual-lobe wax/oil top-coat specular + subsurface scatter on cognac leather
      - Area-window cool cerulean sky dome irradiance from right & back glazing
      - Multi-patch warm bounce radiosity from sunlit oak/travertine floor & walnut
      - Warm 2700K recessed linear shelf-header LED grazing wash inside the 4x5
        walnut library bays + articulated brass reading sconce illumination
    """
    pos = hit_buf["pos"]
    normal = hit_buf["normal"]
    mat_id = hit_buf["mat_id"]
    albedo = hit_buf["albedo"]
    roughness = hit_buf["roughness"]
    f0 = hit_buf["f0"]
    metallic = hit_buf["metallic"]
    view_dir = -rd

    rad = np.zeros_like(pos, dtype=np.float64)

    is_win = mat_id == MAT_WINDOW_GLASS
    if np.any(is_win):
        p_w = pos[is_win]
        d_w = rd[is_win]
        n_w = normal[is_win]
        ext_rad = evaluate_exterior_courtyard(p_w, d_w)
        cos_v = np.clip(dot(n_w, view_dir[is_win]), 0.0, 1.0)
        f_glass = dielectric_fresnel_exact(cos_v, 1.0, 1.52)

        # Double-Pane Low-E Interior Ghost Reflection (warm sunlit floor, walnut shelves & double-pane sconce ghosts)
        r_int = reflect(d_w, n_w)
        shelf_band1 = 0.5 + 0.5 * np.sin((p_w[:, 1] + r_int[:, 1] * 2.8) * 9.5)
        shelf_band2 = 0.5 + 0.5 * np.sin((p_w[:, 1] + r_int[:, 1] * 2.8 + 0.045) * 9.5)
        floor_ghost = smoothstep(0.0, 0.35, -r_int[:, 1]) * (0.55 + 0.45 * np.sin(p_w[:, 2] * 3.2))

        int_refl_rgb = (
            np.array([0.34, 0.23, 0.13], dtype=np.float64)[None, :] * (0.55 + 0.45 * (0.6 * shelf_band1 + 0.4 * shelf_band2))[:, None]
            + np.array([0.62, 0.44, 0.24], dtype=np.float64)[None, :] * floor_ghost[:, None]
        )
        rad[is_win] = ext_rad * (1.0 - 0.86 * f_glass[:, None]) + f_glass[:, None] * int_refl_rgb

    is_bulb = mat_id == MAT_SCONCE_BULB
    if np.any(is_bulb):
        rad[is_bulb] = np.array([5.20, 2.25, 0.72], dtype=np.float64)

    opaque = (~is_win) & (~is_bulb)
    if not np.any(opaque):
        return rad

    p_op = pos[opaque]
    n_op = normal[opaque]
    v_op = view_dir[opaque]
    alb_op = albedo[opaque]
    rgh_op = roughness[opaque]
    f0_op = f0[opaque]
    met_op = metallic[opaque]
    mid_op = mat_id[opaque]

    ao = compute_analytic_ambient_occlusion(p_op, n_op, mid_op)

    sun_vis = evaluate_window_sun_transmission(p_op)
    if is_secondary:
        caustic_contrib = 0.0
        sun_vis_eff = sun_vis
    else:
        caustic_irrad, refr_umbra = evaluate_refractive_caustics(p_op, n_op, mid_op)
        sun_vis_eff = sun_vis * refr_umbra
        caustic_contrib = (alb_op / PI) * caustic_irrad * sun_vis[:, None]

    diff_sun, spec_sun = cook_torrance_brdf(n_op, v_op, SUN_DIR, alb_op, rgh_op, f0_op, met_op)

    # Enhance micro-cavity retro-reflection on rough porous mineral & woven textile surfaces
    is_oren = (
        (mid_op == MAT_BACK_CONCRETE)
        | (mid_op == MAT_CEILING_CONCRETE)
        | (mid_op == MAT_BOUCLE_RUG)
        | (mid_op == MAT_RUG_LINEN_BORDER)
        | (mid_op == MAT_BOOK_SPINE)
        | (mid_op == MAT_OPEN_BOOK_PAGE)
        | (mid_op == MAT_ART_CANVAS)
    )
    if np.any(is_oren):
        retro = 1.0 + 0.12 * np.clip(dot(v_op[is_oren], SUN_DIR), 0.0, 1.0) * rgh_op[is_oren]
        diff_sun[is_oren] *= retro[:, None]

    direct_sun_rad = (diff_sun + spec_sun) * SUN_RADIANCE[None, :] * sun_vis_eff[:, None]

    # Botanical translucent subsurface scattering on backlit eucalyptus leaves
    is_leaf = mid_op == MAT_EUCALYPTUS_LEAF
    if np.any(is_leaf):
        backlit = np.clip(dot(-n_op[is_leaf], SUN_DIR), 0.0, 1.0)
        sss_leaf = (
            backlit[:, None]
            * sun_vis_eff[is_leaf, None]
            * np.array([1.65, 2.45, 0.95], dtype=np.float64)[None, :]
        )
        direct_sun_rad[is_leaf] += sss_leaf

    # Dual-lobe wax/oil top-coat specular + warm cognac subsurface wrap on lounge chair leather
    is_chair = mid_op == MAT_LEATHER_CHAIR
    if np.any(is_chair):
        n_ch = n_op[is_chair]
        v_ch = v_op[is_chair]
        n_dot_l_ch = np.clip(dot(n_ch, SUN_DIR), 0.0, 1.0)
        wrap_lth = np.clip((dot(n_ch, SUN_DIR) + 0.38) / 1.38, 0.0, 1.0)
        sss_lth = (
            (wrap_lth - n_dot_l_ch)[:, None]
            * sun_vis_eff[is_chair, None]
            * np.array([1.35, 0.42, 0.09], dtype=np.float64)[None, :]
        )
        _, wax_spec = cook_torrance_brdf(
            n_ch,
            v_ch,
            SUN_DIR,
            np.zeros_like(n_ch),
            np.full(n_ch.shape[0], 0.16, dtype=np.float64),
            np.full((n_ch.shape[0], 3), 0.052, dtype=np.float64),
            0.0,
        )
        direct_sun_rad[is_chair] += sss_lth + 0.35 * wax_spec * SUN_RADIANCE[None, :] * sun_vis_eff[is_chair, None]

    # Velvet / grazing wool-fiber sheen on the hand-loomed bouclé rug & cloth book bindings
    is_textile = (mid_op == MAT_BOUCLE_RUG) | (mid_op == MAT_RUG_LINEN_BORDER) | (mid_op == MAT_BOOK_SPINE)
    if np.any(is_textile):
        n_tx = n_op[is_textile]
        v_tx = v_op[is_textile]
        cos_nv_tx = np.clip(dot(n_tx, v_tx), 0.0, 1.0)
        cos_nl_tx = np.clip(dot(n_tx, SUN_DIR), 0.0, 1.0)
        grazing_fiber = ((1.0 - cos_nv_tx) ** 3.2) * (0.25 + 0.75 * cos_nl_tx)
        sheen_tint = np.where(
            (mid_op[is_textile] == MAT_BOOK_SPINE)[:, None],
            alb_op[is_textile] * 0.55 + 0.12,
            np.array([[0.86, 0.80, 0.70]], dtype=np.float64),
        )
        direct_sun_rad[is_textile] += (
            grazing_fiber[:, None]
            * sheen_tint
            * (0.28 + 0.72 * sun_vis_eff[is_textile, None])
        )

    # Area-Window Cool Cerulean Sky Irradiance
    dist_right_win = np.maximum(ROOM_X_RIGHT - p_op[:, 0], 0.35)
    sky_dir_right = normalize(
        np.stack(
            [
                np.full_like(dist_right_win, 0.86),
                0.36 - 0.07 * p_op[:, 1],
                0.22 - 0.03 * p_op[:, 2],
            ],
            axis=-1,
        )
    )
    form_right = (1.0 / (1.0 + 0.11 * dist_right_win ** 2)) * (0.45 + 0.55 * np.clip(dot(n_op, sky_dir_right), 0.0, 1.0))

    dist_back_win = np.hypot(np.maximum(p_op[:, 0] - 2.15, 0.0), np.maximum(ROOM_Z_BACK - p_op[:, 2], 0.45))
    sky_dir_back = normalize(
        np.stack(
            [
                (2.15 - p_op[:, 0]) * 0.25,
                0.28 - 0.05 * p_op[:, 1],
                np.ones_like(dist_back_win),
            ],
            axis=-1,
        )
    )
    form_back = (0.65 / (1.0 + 0.10 * dist_back_win ** 2)) * (0.35 + 0.65 * np.clip(dot(n_op, sky_dir_back), 0.0, 1.0))

    sky_irrad = (
        form_right[:, None] * np.array([0.82, 1.02, 1.38], dtype=np.float64)[None, :]
        + form_back[:, None] * np.array([0.68, 0.84, 1.12], dtype=np.float64)[None, :]
    )
    diff_sky, spec_sky = cook_torrance_brdf(n_op, v_op, sky_dir_right, alb_op, np.maximum(rgh_op, 0.28), f0_op, met_op)
    sky_contrib = (diff_sky * PI + 0.42 * spec_sky) * sky_irrad * ao[:, None]

    # Multi-Patch Warm Bounce Radiosity
    bounce_floor_pos = np.array([0.15, 0.0, 2.85], dtype=np.float64)
    to_floor_patch = bounce_floor_pos[None, :] - p_op
    dist_fl_patch = np.sqrt(np.maximum(np.sum(to_floor_patch * to_floor_patch, axis=-1), 0.12))
    dir_fl_patch = to_floor_patch / dist_fl_patch[:, None]
    cos_fl_rec = np.clip(dot(n_op, dir_fl_patch), 0.0, 1.0)
    cos_fl_emit = np.clip(-dir_fl_patch[:, 1], 0.0, 1.0)
    form_fl_bounce = (cos_fl_rec * cos_fl_emit) / (1.4 + 0.16 * dist_fl_patch ** 2)
    hemi_bounce = 0.35 + 0.65 * np.clip(0.5 - 0.5 * n_op[:, 1], 0.0, 1.0)

    warm_floor_radiosity = (
        (form_fl_bounce * 1.45 + hemi_bounce * 0.28)[:, None]
        * np.array([1.52, 0.96, 0.52], dtype=np.float64)[None, :]
    )

    dist_walnut = np.maximum(p_op[:, 0] - (-2.75), 0.35)
    cos_wal_rec = np.clip(-n_op[:, 0] * 0.75 + 0.25, 0.0, 1.0)
    walnut_radiosity = (
        (cos_wal_rec / (1.2 + 0.28 * dist_walnut ** 2))[:, None]
        * np.array([0.62, 0.32, 0.15], dtype=np.float64)[None, :]
    )

    ambient_gi = np.array([0.24, 0.22, 0.20], dtype=np.float64)[None, :]
    indirect_bounce = (alb_op / PI) * (warm_floor_radiosity + walnut_radiosity + ambient_gi) * ao[:, None]

    # Warm Architectural Brass Reading Sconce & Back-Wall Downlight Sconces (BOTH dir = [0, -1, 0])
    to_sconce = SCONCE_POS[None, :] - p_op
    dist_sc = np.sqrt(np.maximum(np.sum(to_sconce * to_sconce, axis=-1), 0.02))
    dir_sc = to_sconce / dist_sc[:, None]
    att_sc = 1.0 / (0.28 + 1.15 * dist_sc ** 2)
    cone_sc = smoothstep(-0.65, 0.35, -dir_sc[:, 1])
    diff_sc, spec_sc = cook_torrance_brdf(n_op, v_op, dir_sc, alb_op, rgh_op, f0_op, met_op)
    sconce_contrib = (diff_sc + spec_sc) * SCONCE_RADIANCE[None, :] * (att_sc * cone_sc)[:, None]

    # Left Horizontal Gallery Picture Light (linear diffuser X in [-1.79, -1.37] at Y=1.925)
    # angled toward the leaning framed bas-relief canvas (X=-1.58, Y=0.53..1.65, Z=7.18..7.30)
    pic_x_clamped = np.clip(p_op[:, 0], -1.79, -1.37)
    to_pic = np.stack(
        [
            pic_x_clamped - p_op[:, 0],
            1.918 - p_op[:, 1],
            7.00 - p_op[:, 2],
        ],
        axis=-1,
    )
    dist_pic = np.sqrt(np.maximum(np.sum(to_pic * to_pic, axis=-1), 0.025))
    dir_pic = to_pic / dist_pic[:, None]
    below_pic = 1.0 - smoothstep(1.895, 1.932, p_op[:, 1])
    lat_pic = np.abs(p_op[:, 0] - (-1.58))
    horiz_wash = 1.0 - smoothstep(0.34, 0.92, lat_pic)
    cone_pic = (
        smoothstep(0.06, 0.58, dir_pic[:, 1]) * below_pic * (0.28 + 0.72 * horiz_wash)
        + 0.03 * smoothstep(-0.40, 0.12, dir_pic[:, 1]) * (1.0 - below_pic) * horiz_wash
    )
    att_pic = 1.0 / (0.18 + 1.05 * dist_pic ** 2)
    diff_pic, spec_pic = cook_torrance_brdf(n_op, v_op, dir_pic, alb_op, rgh_op, f0_op, met_op)
    pic_rad = np.array([10.40, 6.85, 3.12], dtype=np.float64)
    sconce_contrib = sconce_contrib + (diff_pic + spec_pic) * pic_rad[None, :] * (att_pic * cone_pic)[:, None]

    # Right Brushed-Brass Cylindrical Downlight Sconce at [0.25, 1.843, 7.26] casting a warm
    # parabolic gallery spotlight cone onto the concrete wall and credenza vessels below
    to_dl = np.array([[0.25, 1.842, 7.06]], dtype=np.float64) - p_op
    dist_dl = np.sqrt(np.maximum(np.sum(to_dl * to_dl, axis=-1), 0.025))
    dir_dl = to_dl / dist_dl[:, None]
    below_dl = 1.0 - smoothstep(1.815, 1.852, p_op[:, 1])
    cone_dl = (
        smoothstep(0.28, 0.78, dir_dl[:, 1]) * below_dl
        + 0.03 * smoothstep(-0.40, 0.12, dir_dl[:, 1]) * (1.0 - below_dl)
    )
    att_dl = 1.0 / (0.20 + 1.25 * dist_dl ** 2)
    diff_dl, spec_dl = cook_torrance_brdf(n_op, v_op, dir_dl, alb_op, rgh_op, f0_op, met_op)
    dl_rad = np.array([8.40, 5.48, 2.48], dtype=np.float64)
    sconce_contrib = sconce_contrib + (diff_dl + spec_dl) * dl_rad[None, :] * (att_dl * cone_dl)[:, None]

    # Warm 2700K Recessed Linear Shelf-Header LED Grazing Light inside the Walnut Library Bays
    in_shelf_zone = (p_op[:, 0] < -2.745) & (p_op[:, 1] > 0.11) & (p_op[:, 1] < 4.02) & (p_op[:, 2] > -0.76) & (p_op[:, 2] < 7.54)
    shelf_led_contrib = np.zeros_like(direct_sun_rad)
    if np.any(in_shelf_zone):
        p_sh = p_op[in_shelf_zone]
        n_sh = n_op[in_shelf_zone]
        v_sh = v_op[in_shelf_zone]
        alb_sh = alb_op[in_shelf_zone]
        rgh_sh = rgh_op[in_shelf_zone]
        f0_sh = f0_op[in_shelf_zone]
        met_sh = met_op[in_shelf_zone]

        # Exact vertical distance below the nearest walnut shelf underside (BOOKCASE_SHELF_Y[1:] - 0.021)
        header_ys = BOOKCASE_SHELF_Y[1:] - 0.021
        dy_below = header_ys[None, :] - p_sh[:, 1:2]
        dy_below = np.where(dy_below >= -0.01, np.maximum(dy_below, 0.0), 10.0)
        dist_from_header = np.clip(np.min(dy_below, axis=1), 0.015, 0.56)
        led_x_dist = np.abs(p_sh[:, 0] - (-2.80))
        led_atten = (
            (1.0 / (0.14 + 3.6 * (dist_from_header ** 2) + 8.5 * (led_x_dist ** 2)))
            * smoothstep(0.0, 0.04, dist_from_header)
            * (1.0 - smoothstep(0.46, 0.55, dist_from_header))
        )

        # Grazing light vector pointing steeply downward and slightly inward from the front shelf header
        led_dir = normalize(
            np.stack(
                [
                    (-2.78 - p_sh[:, 0]) * 1.4,
                    np.full_like(dist_from_header, 0.92),
                    np.zeros_like(dist_from_header),
                ],
                axis=-1,
            )
        )
        diff_led, spec_led = cook_torrance_brdf(n_sh, v_sh, led_dir, alb_sh, rgh_sh, f0_sh, met_sh)
        led_radiance = np.array([1.28, 0.78, 0.34], dtype=np.float64)
        shelf_led_contrib[in_shelf_zone] = (diff_led + 1.35 * spec_led) * led_radiance[None, :] * led_atten[:, None]

    rad[opaque] = (
        direct_sun_rad
        + caustic_contrib
        + sky_contrib
        + indirect_bounce
        + sconce_contrib
        + shelf_led_contrib
    )
    return rad


# ==============================================================================
# 12. ANISOTROPIC GGX FLOOR REFLECTION & SNELL GLASS REFRACTION PASSES
# ==============================================================================


def compute_anisotropic_floor_reflections(
    base_rad: np.ndarray,
    hit_buf: dict[str, np.ndarray],
    rd: np.ndarray,
    catalog: WalnutLibraryCatalog,
) -> np.ndarray:
    """
    For all bare floor pixels (Wide-Plank Oiled European Oak, Honed Roman Travertine,
    and Brushed Brass Inlay), trace exact 3D mirror reflection rays into the
    upper room and convolve the reflected radiance buffer with an anisotropic
    vertically-stretched GGX microfacet roughness kernel modulated by local plank
    roughness and exact Schlick Fresnel reflectance.
    """
    mat_id = hit_buf["mat_id"]
    is_floor = (
        (mat_id == MAT_FLOOR_OAK)
        | (mat_id == MAT_FLOOR_TRAVERTINE)
        | (mat_id == MAT_BRASS_INLAY)
    )
    if not np.any(is_floor):
        return base_rad

    p_fl = hit_buf["pos"][is_floor]
    n_fl = hit_buf["normal"][is_floor]
    rgh_fl = hit_buf["roughness"][is_floor]
    f0_fl = hit_buf["f0"][is_floor]
    d_in = rd[is_floor]
    v_fl = -d_in

    r_mirror = reflect(d_in, n_fl)
    r_mirror[:, 1] = np.maximum(r_mirror[:, 1], 0.015)
    r_mirror = normalize(r_mirror)

    ro_offset = p_fl + np.array([0.0, 0.003, 0.0], dtype=np.float64)
    sec_hit = trace_room_and_furniture(
        ro_offset, r_mirror, catalog, include_glass_vessels=True, fast_secondary=True
    )
    spec_sharp = shade_surfaces_radiosity(sec_hit, r_mirror, is_secondary=True)

    # Embed reflected radiance into 2D screen buffer to apply vertically-stretched
    # anisotropic GGX microfacet roughness lobes (sigma_y = 3.2 * sigma_x)
    refl_grid = base_rad.copy()
    refl_grid[is_floor] = spec_sharp

    # Anisotropic GGX lobe 1 (medium roughness vertical stretch)
    p_v1 = np.pad(refl_grid, ((18, 18), (0, 0), (0, 0)), mode="edge")
    ks_v1 = np.exp(-0.5 * (np.arange(-18, 19, dtype=np.float64) / 7.5) ** 2)
    ks_v1 /= np.sum(ks_v1)
    lobe1 = np.zeros_like(refl_grid)
    for k, w in enumerate(ks_v1):
        lobe1 += w * p_v1[k : k + refl_grid.shape[0], :, :]
    lobe1 = gaussian_blur_rgb(lobe1, radius=2.2)

    # Anisotropic GGX lobe 2 (wide roughness vertical stretch for satin oak/travertine tail)
    sub_grid = refl_grid[::2, ::2, :]
    p_v2 = np.pad(sub_grid, ((22, 22), (0, 0), (0, 0)), mode="edge")
    ks_v2 = np.exp(-0.5 * (np.arange(-22, 23, dtype=np.float64) / 11.0) ** 2)
    ks_v2 /= np.sum(ks_v2)
    lobe2_sub = np.zeros_like(sub_grid)
    for k, w in enumerate(ks_v2):
        lobe2_sub += w * p_v2[k : k + sub_grid.shape[0], :, :]
    lobe2_sub = gaussian_blur_rgb(lobe2_sub, radius=2.4)
    lobe2 = np.repeat(np.repeat(lobe2_sub, 2, axis=0), 2, axis=1)[: refl_grid.shape[0], : refl_grid.shape[1], :]

    # Blend sharp core, medium anisotropic lobe, and wide anisotropic tail by local roughness
    w_sharp = np.clip(1.0 - (rgh_fl - 0.08) * 5.5, 0.12, 0.78)[:, None]
    w_wide = np.clip((rgh_fl - 0.09) * 4.2, 0.15, 0.65)[:, None]
    w_mid = np.maximum(0.0, 1.0 - w_sharp - w_wide)

    accum_refl = (
        w_sharp * spec_sharp
        + w_mid * lobe1[is_floor]
        + w_wide * lobe2[is_floor]
    )

    cos_theta = np.clip(dot(n_fl, v_fl), 0.0, 1.0)
    f_refl = schlick_fresnel(cos_theta, f0_fl)
    gloss_atten = np.clip(1.0 - 1.25 * (rgh_fl - 0.08), 0.36, 0.96)[:, None]
    f_eff = f_refl * gloss_atten

    out_rad = base_rad.copy()
    out_rad[is_floor] = base_rad[is_floor] * (1.0 - 0.55 * f_eff) + accum_refl * f_eff * 1.32
    return out_rad


def compute_glass_carafe_and_tumbler_optics(
    base_rad: np.ndarray,
    hit_buf: dict[str, np.ndarray],
    rd: np.ndarray,
    catalog: WalnutLibraryCatalog,
) -> np.ndarray:
    """
    Exact physically-based optical rendering for the hand-blown 18-ribbed clear
    glass carafe (`n = 1.52`, water meniscus at `Y = 0.508m`, `n = 1.333`, and
    clinging micro-air bubbles) and the 12-faceted crystal water tumbler
    (`n = 1.52`, water meniscus at `Y = 0.465m`).
    """
    mat_id = hit_buf["mat_id"]
    is_glass = (mat_id == MAT_GLASS_CARAFE) | (mat_id == MAT_WATER_TUMBLER)
    if not np.any(is_glass):
        return base_rad

    p_g = hit_buf["pos"][is_glass]
    n_g = hit_buf["normal"][is_glass]
    m_g = mat_id[is_glass]
    d_in = rd[is_glass]
    v_g = -d_in

    cos_i = np.clip(dot(n_g, v_g), 0.0, 1.0)
    f_dielectric = dielectric_fresnel_exact(cos_i, 1.0, 1.52)

    refl_dir = normalize(reflect(d_in, n_g))
    ro_refl = p_g + n_g * 0.004
    refl_hit = trace_room_and_furniture(
        ro_refl, refl_dir, catalog, include_glass_vessels=False, fast_secondary=True
    )
    refl_rad = shade_surfaces_radiosity(refl_hit, refl_dir, is_secondary=True)

    sun_vis_g = evaluate_window_sun_transmission(p_g)
    _, spec_sun_g = cook_torrance_brdf(
        n_g,
        v_g,
        SUN_DIR,
        np.zeros_like(p_g),
        np.full(p_g.shape[0], 0.028, dtype=np.float64),
        np.full((p_g.shape[0], 3), 0.045, dtype=np.float64),
        0.0,
    )
    sun_spec_rad = spec_sun_g * SUN_RADIANCE[None, :] * sun_vis_g[:, None] * 1.52

    is_carafe = m_g == MAT_GLASS_CARAFE
    water_level = np.where(is_carafe, 0.508, 0.468)
    below_water = p_g[:, 1] < water_level

    eta_in = np.where(below_water, 1.0 / 1.44, 1.0 / 1.24)
    refr_dir1, _ = refract(d_in, n_g, eta_in)

    ro_inside = p_g + refr_dir1 * 0.004
    chord_len = np.where(is_carafe, 0.13, 0.075) * (0.35 + 0.65 * cos_i)
    p_exit = p_g + refr_dir1 * chord_len[:, None]

    vessel_c_xz = np.where(is_carafe[:, None], np.array([[0.47, 1.58]]), np.array([[0.68, 1.44]]))
    radial_xz = p_exit[:, [0, 2]] - vessel_c_xz
    n_exit = normalize(np.stack([radial_xz[:, 0], np.zeros_like(cos_i), radial_xz[:, 1]], axis=-1))
    refr_dir_out, valid_out = refract(refr_dir1, -n_exit, 1.18)
    refr_dir_final = np.where(valid_out[:, None], refr_dir_out, refr_dir1)

    refr_hit = trace_room_and_furniture(
        ro_inside, refr_dir_final, catalog, include_glass_vessels=False, fast_secondary=True
    )
    refr_rad = shade_surfaces_radiosity(refr_hit, refr_dir_final, is_secondary=True)

    base_y = np.where(is_carafe, 0.39, 0.3955)
    base_thick = np.exp(-((p_g[:, 1] - base_y) / 0.022) ** 2) * 0.18
    wall_path = 0.022 / np.maximum(cos_i, 0.14) + base_thick

    sigma_emerald = np.array([6.6, 2.0, 7.4], dtype=np.float64)
    sigma_crystal = np.array([0.28, 0.18, 0.22], dtype=np.float64)
    sigma_a = np.where(is_carafe[:, None], sigma_emerald[None, :], sigma_crystal[None, :])
    transmittance = np.exp(-sigma_a * wall_path[:, None])

    # Double-ring concave capillary water meniscus glow
    meniscus_dist = p_g[:, 1] - water_level
    men_w = np.where(is_carafe, 0.0028, 0.0014)
    meniscus_outer = np.exp(-(((meniscus_dist - 0.0016) / men_w) ** 2))
    meniscus_inner = np.exp(-(((meniscus_dist + 0.0013) / (men_w * 0.85)) ** 2)) * 0.68
    meniscus_tint = np.where(
        is_carafe[:, None],
        np.array([3.1, 2.6, 1.75], dtype=np.float64)[None, :],
        np.array([1.05, 1.12, 1.15], dtype=np.float64)[None, :],
    )
    meniscus_glow = (
        (meniscus_outer + meniscus_inner)[:, None]
        * (0.42 + 0.58 * sun_vis_g[:, None])
        * meniscus_tint
    )

    # Heavy 18mm architectural lead-crystal sham base internal TIR rings & 1.4mm fire-polished top lip at Y=0.5142
    sham_top = (~is_carafe) * np.exp(-(((p_g[:, 1] - 0.4135) / 0.0016) ** 2)) * 0.45
    sham_bot = (~is_carafe) * np.exp(-(((p_g[:, 1] - 0.3975) / 0.0015) ** 2)) * 0.38
    lip_ring = (~is_carafe) * np.exp(-(((p_g[:, 1] - 0.5142) / 0.0013) ** 2)) * 0.55
    crystal_rings = (
        (sham_top + sham_bot + lip_ring)[:, None]
        * (0.35 + 0.65 * sun_vis_g[:, None])
        * np.array([1.25, 1.32, 1.40], dtype=np.float64)[None, :]
    )

    # Tiny clinging air-bubble specular pearls below the water meniscus
    bubble_noise1 = value_noise_2d(p_g[:, 0] * 420.0 + p_g[:, 2] * 310.0, p_g[:, 1] * 480.0, seed=971)
    bubble_noise2 = value_noise_2d(p_g[:, 2] * 390.0 - p_g[:, 0] * 280.0, p_g[:, 1] * 510.0, seed=977)
    in_water_column = below_water & (p_g[:, 1] > base_y + 0.018)
    bubble_spark = smoothstep(0.955, 0.992, bubble_noise1 * (0.55 + 0.45 * bubble_noise2)) * in_water_column
    bubble_glint = (
        bubble_spark[:, None]
        * (0.40 + 0.60 * sun_vis_g[:, None])
        * np.array([3.4, 3.2, 2.6], dtype=np.float64)[None, :]
    )

    rim_caustic = (
        ((1.0 - cos_i) ** 3.2)[:, None]
        * sun_vis_g[:, None]
        * np.where(
            is_carafe[:, None],
            np.array([1.9, 3.1, 1.2], dtype=np.float64)[None, :],
            np.array([2.25, 2.38, 2.55], dtype=np.float64)[None, :],
        )
    )

    glass_total = (
        (1.0 - f_dielectric[:, None]) * refr_rad * transmittance
        + f_dielectric[:, None] * refl_rad * 1.22
        + sun_spec_rad
        + meniscus_glow
        + crystal_rings
        + bubble_glint
        + rim_caustic
    )

    out_rad = base_rad.copy()
    out_rad[is_glass] = glass_total
    return out_rad


# ==============================================================================
# 13. VOLUMETRIC TYNDALL SUNBEAMS & MIE FORWARD-SCATTERING DUST SHAFTS
# ==============================================================================


def integrate_volumetric_tyndall_sunbeams(
    ro: np.ndarray,
    rd: np.ndarray,
    t_hit: np.ndarray,
    seed: int = 20301,
    hdr_bg: np.ndarray | None = None,
    mat_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Ray-march volumetric Mie forward scattering along camera rays `ro + t * rd`
    evaluating the exact 3D window-mullion sunlight shadow frustums so that
    diagonal golden Tyndall light shafts and ultra-fine sub-pixel sunlit dust
    pinpoints slice across the shadowed room interior.
    """
    height, width, _ = rd.shape
    step_y, step_x = 2, 2
    rd_sub = rd[::step_y, ::step_x, :]
    t_max_sub = np.minimum(t_hit[::step_y, ::step_x], 8.4)

    cos_scatter = dot(rd_sub, -SUN_DIR)
    g = 0.68
    hg_phase = (1.0 - g * g) / (4.0 * PI * np.maximum(1.0 + g * g - 2.0 * g * cos_scatter, 0.08) ** 1.5)
    phase_weight = 0.45 + 0.95 * hg_phase

    n_steps = 28
    rng = np.random.default_rng(seed + 999)
    jitter = rng.uniform(0.0, 1.0, size=t_max_sub.shape)

    vol_accum = np.zeros(rd_sub.shape[:-1], dtype=np.float64)
    t_start = 0.42
    dt = np.maximum((t_max_sub - t_start) / float(n_steps), 0.0)

    for s in range(n_steps):
        frac = (s + jitter) / float(n_steps)
        t_cur = t_start + frac * (t_max_sub - t_start)
        p_sample = ro[None, None, :] + t_cur[..., None] * rd_sub

        px = p_sample[..., 0]
        py = p_sample[..., 1]
        pz = p_sample[..., 2]

        in_room = (
            (px > ROOM_X_LEFT)
            & (px < ROOM_X_RIGHT)
            & (py > 0.02)
            & (py < ROOM_Y_CEIL)
            & (pz < ROOM_Z_BACK)
        )

        # Trace 3D solar shadow ray from air sample to right & back window planes
        tx = (ROOM_X_RIGHT - px) / SUN_DIR[0]
        tz = (ROOM_Z_BACK - pz) / SUN_DIR[2]
        t_exit = np.minimum(tx, tz)
        on_right = tx <= tz

        wx = px + t_exit * SUN_DIR[0]
        wy = py + t_exit * SUN_DIR[1]
        wz = pz + t_exit * SUN_DIR[2]

        # Window aperture bounds + roof overhang shading upper clerestory into deep interior
        overhang_cut = 1.0 - smoothstep(2.92, 3.35, wy) * smoothstep(1.35, 2.65, t_exit)
        in_y = smoothstep(0.09, 0.14, wy) * (1.0 - smoothstep(3.75, 3.86, wy)) * overhang_cut
        in_z = smoothstep(-0.40, -0.05, wz) * (1.0 - smoothstep(7.52, 7.60, wz))
        in_x = smoothstep(1.14, 1.22, wx) * (1.0 - smoothstep(3.14, 3.20, wx))
        win_open = in_y * np.where(on_right, in_z, in_x)

        # Crisp 3D mullion & transom shadow blades carving distinct diagonal Tyndall shafts
        d_mull_z = np.minimum(
            np.minimum(np.abs(wz - 1.65), np.abs(wz - 3.55)),
            np.minimum(np.abs(wz - 5.50), np.abs(wz - 7.52)),
        )
        d_mull_x = np.minimum(np.abs(wx - 1.14), np.minimum(np.abs(wx - 2.18), np.abs(wx - 3.14)))
        d_mull = np.where(on_right, d_mull_z, d_mull_x)
        vis_mull = smoothstep(0.038, 0.145, d_mull)

        d_trans = np.minimum(np.abs(wy - 1.08), np.abs(wy - 2.98))
        vis_trans = smoothstep(0.032, 0.125, d_trans)

        d_corner = np.hypot(wx - ROOM_X_RIGHT, wz - ROOM_Z_BACK)
        vis_corner = smoothstep(0.065, 0.185, d_corner)

        # Striated diagonal shaft modulation along solar rays so individual sunbeams stand out clearly
        u_beam = px * SUN_DIR[1] - py * SUN_DIR[0]
        v_beam = pz * SUN_DIR[1] - py * SUN_DIR[2]
        shaft_ribbons = (
            0.55
            + 0.28 * np.sin(u_beam * 5.8 + v_beam * 3.4)
            + 0.22 * np.cos(u_beam * 11.2 - v_beam * 4.8)
        )

        u_air = px * 0.70 + py * 0.50
        v_air = pz * 0.60 - py * 0.40
        air_density = (0.72 + 0.48 * value_noise_2d(u_air, v_air, seed=1103 + s * 7)) * np.clip(shaft_ribbons, 0.35, 1.35)
        h_weight = np.exp(-0.26 * np.maximum(py - 0.38, 0.0)) * (1.0 - 0.65 * smoothstep(1.55, 2.25, py) * smoothstep(5.2, 7.2, pz))

        sun_shaft = win_open * vis_mull * vis_trans * vis_corner * (t_exit > 0.0) * in_room
        scatter_step = sun_shaft * air_density * h_weight * dt * 0.0245
        vol_accum += scatter_step * phase_weight

    vol_u8 = np.clip(vol_accum * 255.0 * 3.0, 0.0, 255.0).astype(np.uint8)
    vol_up = np.asarray(
        Image.fromarray(vol_u8, mode="L").resize((width, height), resample=Image.Resampling.BICUBIC),
        dtype=np.float64,
    ) / (255.0 * 3.0)
    vol_scalar = gaussian_blur_rgb(vol_up, radius=2.4)

    # Compute background luminance and interior mask so sky/water stay crisp & unwashed
    if hdr_bg is not None:
        rad_bg = np.maximum(hdr_bg, 0.0) * 0.88
        mapped_bg = np.clip(
            (rad_bg * (2.51 * rad_bg + 0.03)) / (rad_bg * (2.43 * rad_bg + 0.59) + 0.14),
            0.0,
            1.0,
        )
        srgb_bg = np.where(mapped_bg <= 0.0031308, 12.92 * mapped_bg, 1.055 * (mapped_bg ** (1.0 / 2.24)) - 0.055)
        bg_luminance = (
            0.2126 * srgb_bg[..., 0] + 0.7152 * srgb_bg[..., 1] + 0.0722 * srgb_bg[..., 2]
        )
    else:
        bg_luminance = np.full((height, width), 0.20, dtype=np.float64)

    if mat_id is not None:
        is_window_or_frame = (
            (mat_id == MAT_WINDOW_GLASS)
            | (mat_id == MAT_STEEL_MULLION)
            | (mat_id == MAT_GLAZING_GASKET)
        )
        is_upper_back_wall = (mat_id == MAT_BACK_CONCRETE) & (
            (ro[1] + t_hit * rd[..., 1]) > 1.48
        )
    else:
        is_window_or_frame = np.zeros((height, width), dtype=bool)
        is_upper_back_wall = np.zeros((height, width), dtype=bool)

    # Attenuate volumetric haze over bright exterior window sky/water and upper sconce wall
    shaft_interior_weight = np.where(is_window_or_frame, 0.18, 1.0) * np.where(is_upper_back_wall, 0.15, 1.0)
    shaft_contrast_boost = 0.65 + 0.75 * np.clip(1.0 - bg_luminance, 0.0, 1.0)
    vol_full = (
        (vol_scalar * shaft_interior_weight * shaft_contrast_boost)[..., None]
        * np.array([1.62, 1.14, 0.60], dtype=np.float64)[None, None, :]
    )

    # Normalized sunbeam intensity in [0, 1] strictly inside the diagonal shafts
    p98 = max(float(np.percentile(vol_scalar, 98.5)), 0.045)
    sunbeam_intensity = smoothstep(0.24 * p98, 0.92 * p98, vol_scalar)
    sunbeam_intensity = np.where(is_window_or_frame | is_upper_back_wall | (t_hit < 1.35), 0.0, sunbeam_intensity)

    # Physical dust visibility mask: inside diagonal sunbeams against shaded interior backgrounds
    dark_bg_gate = np.clip((0.36 - bg_luminance) / 0.22, 0.0, 1.0)
    dust_visibility_mask = (sunbeam_intensity ** 1.6) * dark_bg_gate

    # Suspended air dust specks: sub-pixel / 0.65px-1.05px 2D Gaussian warm golden pinpoints
    mote_layer = np.zeros_like(vol_full)
    rng_motes = np.random.default_rng(seed + 707)
    n_specks = 260
    sp_u = rng_motes.uniform(0.24, 0.94, size=n_specks)
    sp_v = rng_motes.uniform(0.34, 0.92, size=n_specks)
    sp_sigma = rng_motes.uniform(0.65, 1.05, size=n_specks) * (width / 1600.0)
    sp_peak_hdr = rng_motes.uniform(0.05, 0.15, size=n_specks)

    warm_pinpoint_tint = np.array([1.0, 0.85, 0.58], dtype=np.float64)
    for k in range(n_specks):
        cx = sp_u[k] * (width - 1)
        cy = sp_v[k] * (height - 1)
        px_i = round(cx)
        py_i = round(cy)
        if float(dust_visibility_mask[py_i, px_i]) < 1e-3:
            continue
        sigma = float(sp_sigma[k])
        pad = math.ceil(sigma * 3.0) + 1
        x0, x1 = max(0, px_i - pad), min(width, px_i + pad + 1)
        y0, y1 = max(0, py_i - pad), min(height, py_i + pad + 1)
        dx = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
        dy = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
        d2 = dx * dx + dy * dy
        # 2D Gaussian radial falloff
        kern = np.exp(-d2 / (2.0 * sigma * sigma))
        local_gate = dust_visibility_mask[y0:y1, x0:x1]
        pinpoint = kern * float(sp_peak_hdr[k]) * local_gate
        mote_layer[y0:y1, x0:x1, :] += pinpoint[..., None] * warm_pinpoint_tint[None, None, :]

    return vol_full + mote_layer


# ==============================================================================
# 14. 35MM TILT-SHIFT OPTICAL LENS, ACES FILMIC TONE-MAPPING & FILM EMULSION
# ==============================================================================


def apply_35mm_camera_optics_and_aces(
    hdr_radiance: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    seed: int = 20301,
) -> np.ndarray:
    """
    Transform linear 64-bit HDR scene radiance through a realistic 35mm architectural
    tilt-shift camera pipeline.
    """
    rad = np.maximum(hdr_radiance, 0.0) * 0.88

    lum = 0.2126 * rad[..., 0] + 0.7152 * rad[..., 1] + 0.0722 * rad[..., 2]
    bright_mask = smoothstep(1.35, 4.20, lum)
    highlights = rad * bright_mask[..., None]

    h_sub2 = highlights[::2, ::2, :]
    b_tight_sub = gaussian_blur_rgb(h_sub2, radius=1.8)
    bloom_tight = np.repeat(np.repeat(b_tight_sub, 2, axis=0), 2, axis=1)[: rad.shape[0], : rad.shape[1], :]

    h_sub4 = highlights[::4, ::4, :]
    b_wide_sub = gaussian_blur_rgb(h_sub4, radius=3.5)
    b_wide = np.repeat(np.repeat(b_wide_sub, 4, axis=0), 4, axis=1)[: rad.shape[0], : rad.shape[1], :]

    rad = (
        rad
        + 0.065 * bloom_tight
        + 0.045 * b_wide * np.array([1.12, 0.96, 0.82], dtype=np.float64)[None, None, :]
    )

    u_max_ref = max(float(np.max(np.abs(uu))), 1.0)
    u_opt = uu / u_max_ref
    r2_lens = (u_opt * 0.62) ** 2 + ((vv - CAM_V_SHIFT) * 0.62) ** 2
    cos_lens = 1.0 / np.sqrt(1.0 + r2_lens)
    vignette = 0.42 + 0.58 * (cos_lens ** 3.4)
    rad *= vignette[..., None]

    # ACES Filmic Tone-Mapping
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    mapped = np.clip((rad * (a * rad + b)) / (rad * (c * rad + d) + e), 0.0, 1.0)

    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * (mapped ** (1.0 / 2.24)) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)

    # Subtle Lateral Chromatic Aberration near extreme wide-angle corners
    height, width, _ = srgb.shape
    r_corner = np.hypot(u_opt, vv)
    ca_shift = np.clip((r_corner - 0.45) * 1.35, 0.0, 1.4)
    _iy_grid, ix_grid = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    dx_ca = np.clip(ix_grid + u_opt * ca_shift, 0.0, width - 1.001)
    ix0 = np.floor(dx_ca).astype(np.int64)
    fx_ca = dx_ca - ix0
    iy_int = np.arange(height, dtype=np.int64)[:, None]
    r_chan = (1.0 - fx_ca) * srgb[iy_int, ix0, 0] + fx_ca * srgb[iy_int, np.minimum(ix0 + 1, width - 1), 0]

    dx_ca_b = np.clip(ix_grid - u_opt * ca_shift * 0.85, 0.0, width - 1.001)
    ix0_b = np.floor(dx_ca_b).astype(np.int64)
    fx_b = dx_ca_b - ix0_b
    b_chan = (1.0 - fx_b) * srgb[iy_int, ix0_b, 2] + fx_b * srgb[iy_int, np.minimum(ix0_b + 1, width - 1), 2]

    srgb = np.stack([r_chan, srgb[..., 1], b_chan], axis=-1)

    # Subtle 35mm Kodak Portra 160 Architectural Film Silver-Halide Micro-Grain
    rng = np.random.default_rng(seed + 4040)
    grain_raw = rng.standard_normal(size=(height, width)).astype(np.float64)
    midtone_weight = 4.0 * srgb * (1.0 - srgb)
    return np.clip(srgb + grain_raw[..., None] * midtone_weight * 0.0085, 0.0, 1.0)


# ==============================================================================
# 15. MAIN RENDERING ENTRY POINT (3-TAP ROTATED-GRID SUPERSAMPLING)
# ==============================================================================


def render_interior_photograph(width: int = 1600, height: int = 1200, seed: int = 20301) -> Image.Image:
    """
    Render the complete architectural interior photograph (1600x1200 4:3 or 1920x1080 16:9).
    """
    catalog = WalnutLibraryCatalog(seed=seed)

    rd, uu, vv = generate_camera_rays(width, height)
    hit_buf = trace_room_and_furniture(CAM_POS, rd, catalog, include_glass_vessels=True, fast_secondary=False)
    hdr_rad = shade_surfaces_radiosity(hit_buf, rd, is_secondary=False)
    hdr_rad = compute_anisotropic_floor_reflections(hdr_rad, hit_buf, rd, catalog)
    hdr_rad = compute_glass_carafe_and_tumbler_optics(hdr_rad, hit_buf, rd, catalog)

    # Selective 3-tap Rotated-Grid Supersampling on High-Contrast Geometric Silhouettes
    mat_id = hit_buf["mat_id"]
    t_depth = hit_buf["t"]
    edge_mat = (
        (mat_id != np.roll(mat_id, 1, axis=0))
        | (mat_id != np.roll(mat_id, 1, axis=1))
    )
    edge_depth = (
        (np.abs(t_depth - np.roll(t_depth, 1, axis=0)) > 0.09)
        | (np.abs(t_depth - np.roll(t_depth, 1, axis=1)) > 0.09)
    )
    edge_mask = edge_mat | edge_depth
    edge_mask[0, :] = False
    edge_mask[-1, :] = False
    edge_mask[:, 0] = False
    edge_mask[:, -1] = False

    if np.any(edge_mask):
        forward, right, up = build_camera_basis()
        du = float(uu[0, -1] - uu[0, 0]) / float(max(width - 1, 1))
        dv = float(vv[-1, 0] - vv[0, 0]) / float(max(height - 1, 1))
        u_edge = uu[edge_mask]
        v_edge = vv[edge_mask]

        sub_offsets = [(-0.35, -0.15), (+0.35, +0.15)]
        accum_edge = hdr_rad[edge_mask] * 0.50

        for ox, oy in sub_offsets:
            u_s = u_edge + ox * du
            v_s = v_edge + oy * dv
            d_s = normalize(
                forward[None, :]
                + (u_s * CAM_FOV_SCALE)[:, None] * right[None, :]
                + (v_s * CAM_FOV_SCALE)[:, None] * up[None, :]
            )
            h_s = trace_room_and_furniture(
                CAM_POS, d_s, catalog, include_glass_vessels=True, fast_secondary=True
            )
            r_s = shade_surfaces_radiosity(h_s, d_s, is_secondary=False)
            r_s = compute_glass_carafe_and_tumbler_optics(r_s, h_s, d_s, catalog)
            accum_edge += 0.25 * r_s

        hdr_rad[edge_mask] = accum_edge

    vol_rad = integrate_volumetric_tyndall_sunbeams(
        CAM_POS, rd, hit_buf["t"], seed=seed, hdr_bg=hdr_rad, mat_id=hit_buf["mat_id"]
    )
    hdr_rad = hdr_rad + vol_rad

    srgb_float = apply_35mm_camera_optics_and_aces(hdr_rad, uu, vv, seed=seed)
    rgb_u8 = np.clip(np.round(srgb_float * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def main() -> None:
    is_16x9 = ("--16x9" in sys.argv) or (os.environ.get("ASPECT_16X9", "0") == "1")
    pos_args = [a for a in sys.argv[1:] if a != "--16x9"]

    default_out = (
        "renders/sunlit_library_1920x1080.png"
        if is_16x9
        else "renders/sunlit_library_1600x1200.png"
    )
    out_arg = pos_args[0] if pos_args else os.environ.get("PAINT_OUTPUT", default_out)
    width = int(os.environ.get("PAINT_WIDTH", "1920" if is_16x9 else "1600"))
    height = int(os.environ.get("PAINT_HEIGHT", "1080" if is_16x9 else "1200"))
    seed = int(os.environ.get("PAINT_SEED", "20301"))

    out_path = Path(out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = render_interior_photograph(width=width, height=height, seed=seed)
    img.save(out_path, format="PNG", optimize=False)
    print(f"Saved {out_path} ({width}x{height}, seed={seed})")


if __name__ == "__main__":
    main()
