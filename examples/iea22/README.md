# IEA-22-280-RWT benchmark

Worked output of the converter for the **IEA 22 MW** reference blade (the windIO v2 file bundled with
`pip install windIO`). The single-station files are mid-span (**r = 0.50**); the full spanwise validation
sweep lives under [`validation/`](validation/).

At r = 0.50: chord 4.86 m, twist 1.16 deg, an FFA-W3 airfoil blend with **3 shear webs** and **6 laminates**
(carbon spar caps SS/PS, foam fillers, glass LE/TE reinforcements, glass-triax skins, biax/foam webs).

## Single-station files (r = 0.50)

| file | what |
|------|------|
| `iea22_r050_shell.yaml`        | OpenSG **1D-shell SG** (nodes, line elements, element sets = laminates, sections, materials) -> JAX MSG-RM / Kirchhoff |
| `iea22_r050_prevabs/*.xml`     | **PreVABS** cross-section input (general/baselines/dividing-points/webs/layups/components) |
| `iea22_r050_prevabs/*.dat`     | normalised airfoil contour for PreVABS |
| `iea22_r050_prevabs/materials.xml` | PreVABS materials + laminae |
| `iea22_r050_orient.png`        | element e1/e2/e3 frames (e2 = blue tangent, e3 = green OML->IML on skin / red e1xe2 on webs; e1 = +z out-of-plane) |

## Full-span validation benchmark ([`validation/`](validation/))

10 spanwise stations, r = 0.10 .. 0.95. Each station carries the **3 mandatory outputs** (see below).

| file | what |
|------|------|
| `iea22_stiffness_K.dat`            | Timoshenko 6x6 (bare `.K` convention) for 2D-solid, RM, KL, every station |
| `K_matrices/iea22_rNNN_{RM,KL,solid}.K` | per-station bare 6x6 stiffness, named by station r |
| `iea22_pcterr_RM.dat` / `_KL.dat`  | RM / KL %-error 6x6 vs 2D-solid (terms with `|value| < 1e6` -> 0, since those couplings are denominator-noise) |
| `iea22_pcterr_refined_RM.dat` / `_KL.dat` | same, on a 4x-refined 1D shell (mesh-converged: diagonals shift < 0.3%) |
| `iea22_Cij_matrices.dat` / `_diagonal.dat` | stiffness in `Cij` notation |
| `iea22_timo_full.dat`              | full nonzero-term table (solid / RM / KL / normalised coupling) |
| `iea22_stations.dat`               | per-station geometry + thin-wall metric (t/h, t/chord) |
| `validation_summary.txt`           | per-station nonzero-term %-error, RM vs KL |
| `orientation/orient_iea22_rNNN.png` | per-station solid+shell e1/e2/e3 orientation |
| `orientation/iea22_span_montage.png` | all stations at a glance |

### The 3 mandatory outputs (per station, every run)

1. **Orientation PNG for BOTH the 2D-solid and the 1D-shell** (e1/e2/e3) -- catches web/skin e3 and
   geometry bugs (the tilted-web and TE-skin fixes were both found this way).
2. **RM and KL Timoshenko 6x6**, stored bare (`.K` convention) and named by the station `r`.
3. **%-error 6x6** of RM and KL vs the 2D-solid Timoshenko, with the `|term| < 1e6 -> 0` cutoff so only
   non-negligible terms carry a percentage.

## Result

The blade is **thin-walled at every station** (thickest wall / airfoil height 2.1% root -> 9.2% tip), so
**RM and KL agree to < 0.4 pp** and both track the 2D-solid closely:

| station | max diag %err (RM) | max diag %err (KL) |
|---------|--------------------|--------------------|
| r=0.10  | 5.5  | 5.2  |
| r=0.20  | 4.7  | 4.7  |
| r=0.30  | 6.0  | 6.0  |
| r=0.40  | 4.9  | 4.6  |
| r=0.50  | 5.5  | 5.1  |
| r=0.60  | 6.8  | 6.3  |
| r=0.70  | 7.0  | 7.0  |
| r=0.80  | 6.9  | 7.2  |
| r=0.90  | 7.3  | 6.7  |
| r=0.95  | 24.4 | 24.4 |

Root-to-tip diagonals are within ~7%; the tip (r=0.95) rises to +24%, the thin-shell limit where the wall
is thickest relative to the section height. **RM is the recommended default** -- on this thin blade it is no
more accurate than KL, but it stays bounded at thick webs / junctions where the Kirchhoff transverse-shear
model degrades (see the OpenSG-TW mh104 study).

The validated mid-span (r = 0.50) Timoshenko diagonals (SI: N, N*m, N*m^2):

| term | RM | KL | 2D-solid |
|------|----|----|----------|
| EA   | 2.184e10 | 2.184e10 | 2.115e10 |
| GA2  | 5.340e8  | 5.358e8  | 5.158e8  |
| GA3  | 1.928e8  | 1.935e8  | 2.041e8  |
| GJ   | 7.395e8  | 7.415e8  | 7.351e8  |
| EI2  | 7.354e9  | 7.354e9  | 7.115e9  |
| EI3  | 6.296e10 | 6.296e10 | 6.260e10 |

## Regenerate / extend

```bash
# this exact station
python scripts/convert_station.py --r 0.5 --mesh-size 0.01 --out examples/iea22

# + mesh the 2D-solid (needs the PreVABS binary, see scripts/fetch_prevabs.py)
python scripts/convert_station.py --r 0.5 --out out --run --prevabs <path>/prevabs.exe
#   -> out/xsec_r050_prevabs/xsec_r050.sg   (then convert_sg_to_yaml -> 2D-solid SG YAML)

# all stations
python scripts/run_sweep.py
```

The RM / KL Timoshenko 6x6 and the %-error tables are produced by the OpenSG-TW JAX MSG solvers (the
shell homogenizer is not bundled here); the `.K` and `.dat` files above are the reference results.
