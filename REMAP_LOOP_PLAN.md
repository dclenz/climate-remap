# MFA Remap Round-Trip Plan And Work Record

## Purpose

This document is the plan and implementation record for pointwise round-trip error analysis of the MFA climate remapper. It begins as the full implementation reference. As work proceeds, it must be updated with completed work, verification results, remaining work, discoveries, and justified changes to the plan.

The target workflow is:

1. Select MPAS cells that loosely intersect the ROMS region and include the existing one-ring MPAS halo.
2. Fit an MFA to MPAS temperature and salinity.
3. Decode the forward MFA at all wet ROMS three-dimensional sample locations.
4. Fit a second, higher-resolution MFA to those decoded ROMS values.
5. Decode the reverse MFA at the exact active MPAS cell centroids used by the forward fit.
6. Compare the round-trip values pointwise with the original MPAS values, subject to strict horizontal and vertical support criteria.

## Agreed Scope

- The round-trip workflow is implemented by a new executable, `remap_loop`, whose entry point is `src/remap_loop.cpp`.
- The existing `mfa_remap` executable and its behavior remain unchanged.
- Reusable mesh loading, point construction, fitting, profile packing, coverage classification, reverse decoding, and error logic belongs in `src/cblock2.hpp` where practical.
- `src/remap_loop.cpp` should contain orchestration, command-line parsing, timing, and top-level output calls rather than duplicate numerical logic.
- The first supported fields are:
  - `timeDaily_avg_activeTracers_temperature_3d`
  - `timeDaily_avg_activeTracers_salinity_3d`
- The default forward science-variable MFA resolution is `11 x 11 x 11` control points.
- The reverse MFA has independent controls and defaults to `17 x 17 x 17` control points.
- Both legs run in one invocation so exact entity/level indexing and in-memory decoded data can be preserved.
- The primary quantitative outputs are pointwise signed error, pointwise absolute error, bias, RMSE, and maximum absolute error.
- Headline errors include only MPAS cells fully covered by the union of wet ROMS cells (`ROMS_Mask == 1`). Bounding-box containment is not sufficient.
- Halo cells, partially covered cells, cells crossing masked gaps, and uncovered cells are excluded from headline metrics.
- An active MPAS level is eligible only when its cell centroid depth has local vertical support from the overlapping wet ROMS profiles.
- Excluded cells and levels remain classified in outputs for diagnosis.
- ParaView verification uses `/Users/dlenz/workspace/moab_plugin/moab_h5m_reader.py`.

## Output Contract

The intended stable outputs from `remap_loop` are:

- `forward_remap.h5m`: ROMS topology with complete array-valued forward temperature and salinity profiles.
- `roundtrip_mpas.h5m`: selected MPAS topology with original, round-trip, signed-error, absolute-error, coverage, and eligibility tags.
- `roundtrip_metrics.csv`: overall and per-level error metrics.
- `forward_remap.mfa` and `reverse_remap.mfa`, or equivalently named serialized MFA artifacts if serialization remains useful.
- A preserved run log containing settings, entity counts, eligibility counts, timings, model sizes, and metric summaries.
- Optional VTK diagnostics may remain, but H5M is the authoritative profile output because fixed-width MOAB tags preserve all vertical levels.

## Status

| Phase | State | Summary |
| --- | --- | --- |
| 0. Plan and architecture | Complete | Added and installed `remap_loop`; existing `mfa_remap` entry point and help remain unchanged. |
| 1. Forward artifact | Complete | Canonical run wrote finite, complete 100-level temperature and salinity tags to `forward_remap.h5m`. |
| 2. Coverage and support | Complete | Exact projected polygon-union coverage and conservative local vertical support are written to MPAS tags. |
| 3. Reverse MFA | Complete | `17 x 17 x 17` reverse fit decodes at exact original MPAS parameters and writes `reverse_remap.mfa`. |
| 4. Error products | Complete | MPAS round-trip/error tags and overall/per-level CSV metrics are generated. |
| 5. Canonical validation | Complete | Two authoritative runs produced identical counts and metrics; scripted ParaView checks pass. |

## Phase 0: Plan And Architecture

### Tasks

- Add this living plan at the repository root.
- Add and install a `remap_loop` executable without modifying the existing `mfa_remap` entry point.
- Keep existing forward behavior behind the existing executable while adding loop-specific capabilities through explicit `CBlock2` methods or mode flags.
- Add early loop-mode validation for dimensions, variables, files, tag lengths, and MFA controls.

### Verification

- `mfa_remap --help` and its existing output behavior remain unchanged.
- `remap_loop --help` documents forward and reverse settings.
- Both executables build and install under `$MFA_REMAP_PATH/bin`.

## Phase 1: Valid Full-Profile Forward Artifact

### Implementation

- Preserve the existing MPAS fitting selection: loose ROMS-region intersection plus the one-ring MPAS halo.
- Separate MPAS geometry construction from field loading where required for reverse reuse.
- Correct MPAS vertical centroids using the dataset semantics: level zero spans from `SSH` to `-layerThickness[0]`; each lower level starts at the preceding bottom and extends downward by its own layer thickness.
- Pack every forward-decoded ROMS level into one fixed-width MOAB tag per field. The tag width must equal `ROMS3DLocations` length rather than the current hard-coded 15-level diagnostic limit.
- Keep decoded forward arrays in memory for the reverse fit.
- Write `forward_remap.h5m` from loop mode.
- Do not rely on the current scalar `<field>_remap` tag, whose 3D point count does not match the 2D target entity count.

### Verification

- Rebuild with `source load-env.sh && spack install`.
- Run the installed executable from a dedicated scratch directory.
- Assert profile width equals `ROMS3DLocations` width, currently 100 for the canonical data.
- Assert the packed values exactly match the in-memory decoded point ordering.
- Assert all wet-cell decoded values are finite.
- Open `forward_remap.h5m` with the MOAB ParaView plugin and inspect representative surface, middle, and deep levels with `ROMS_Mask == 1`.
- Check for profile-index striping, NaNs, extreme overshoot, and masked-cell contamination.

## Phase 2: Exact Horizontal Coverage And Vertical Support

### Horizontal classification

For every selected MPAS polygon:

- Project the MPAS polygon and candidate wet ROMS quads into the existing common local plane.
- Use spatial bounds only to obtain candidate ROMS cells.
- Compute actual polygon intersections with wet ROMS cells.
- Determine coverage by the union of wet ROMS cells, avoiding double-counting shared or overlapping regions.
- Compare covered area with MPAS polygon area using a documented numerical tolerance.
- Classify the MPAS cell as halo, uncovered, partially covered, or fully covered.
- Store coverage fraction and classification as MOAB tags.
- Exclude halo, partial, and uncovered cells from headline metrics.

A cell lying inside the ROMS bounding box but crossing a masked or geometric gap must not be classified as fully covered.

### Vertical classification

- Retain the wet ROMS cells overlapping each MPAS polygon.
- Determine the local sampled vertical support represented by those ROMS profiles.
- Mark an MPAS active-level centroid eligible only when its depth lies within that local support.
- Store per-level eligibility so exclusions can be inspected.
- Use conservative support rules at cells touched by multiple ROMS profiles and document the exact rule after implementation testing.

### Verification

- Add small deterministic geometry cases for full coverage, partial coverage, masked gaps, and boundary touching.
- Verify cells inside only the rectangular bounds are not falsely accepted.
- Verify no halo cell becomes error-eligible.
- Inspect coverage fraction and class tags in ParaView.
- Confirm accepted cells follow wet ROMS coastlines and holes.

## Phase 3: Higher-Resolution Reverse MFA

### Implementation

- Use complete forward-decoded ROMS profiles as reverse source samples without re-reading lossy scalar VTK output.
- Build reverse source geometry from wet ROMS cell centroids and `ROMS3DLocations`.
- Decode at the exact MPAS active cell centroids and preserve the original `(MPAS entity, active level)` mapping.
- Add independent reverse MFA CLI settings.
- Default reverse science-variable controls to `17 x 17 x 17`.
- Generate reverse vertical knot placement from ROMS source locations rather than reusing the MPAS-source-only knot policy.
- Decode all selected MPAS cells for visualization, while applying coverage and support masks only during metric accumulation.
- Write complete fixed-width round-trip profile tags on the selected MPAS topology.

### Verification

- Reverse source count equals wet ROMS cells multiplied by ROMS profile length.
- Reverse target count equals the sum of active levels across selected MPAS cells.
- Every reverse point maps to the original MPAS entity and active level expected by the forward source indexing.
- Compare a diagnostic reverse `11 x 11 x 11` run with the default `17 x 17 x 17` run. Higher resolution is expected to exploit the finer ROMS sampling, but acceptance does not assume every metric must monotonically improve.
- Inspect original and round-trip fields at representative MPAS levels.

## Phase 4: Error Tags And Metrics

### Pointwise output

For each supported field, write:

- round-trip value;
- signed error, defined as `roundtrip - original`;
- absolute error;
- per-level eligibility.

Use `NaN` for inactive or ineligible error entries unless MOAB/H5M constraints require a documented alternative fill value.

### Metrics

For temperature and salinity, report overall and per-level:

- eligible point count;
- bias;
- RMSE;
- maximum absolute error;
- MPAS entity and level at maximum absolute error;
- original and round-trip value ranges.

Write metrics to stderr/run log and `roundtrip_metrics.csv`.

### Verification

- Unit-test formulas with known values and exclusions.
- Confirm overall eligible count equals the sum of per-level counts.
- Recompute selected errors from output tags and compare with reported metrics.
- Confirm no excluded cell or level enters accumulation.
- Inspect signed and absolute error fields in ParaView, including maxima, coastlines, bathymetric transitions, and the accepted-domain edge.

## Phase 5: Canonical End-To-End Validation

Use:

- `data/mpas_grid_tstep0.h5m`
- `data/roms_3d_projected_BoB.h5m`
- forward controls `11 x 11 x 11`;
- reverse controls `17 x 17 x 17`;
- temperature and salinity;
- the installed `$MFA_REMAP_PATH/bin/remap_loop`;
- a dedicated scratch directory and preserved log.

### Acceptance Criteria

- Both MFAs complete without NaNs at supported points.
- Full ROMS profiles are preserved in H5M.
- Reverse values map to the exact original MPAS entity/level indices.
- Headline metrics contain only non-halo MPAS points with complete wet-ROMS horizontal coverage and local vertical support.
- Metrics can be reproduced from H5M output.
- ParaView inspection shows coherent fields and errors without indexing artifacts or masked-region contamination.
- Existing `mfa_remap` behavior remains unchanged.

## Known Initial Risks

- MPAS SSH/layer-thickness semantics were initially unclear; the shared extrusion is now corrected and documented.
- The current unsuffixed remap tag has a profile-to-entity size mismatch.
- Existing usable VTK output exposes only 15 of 100 ROMS levels.
- Source-subset VTK output hard-codes temperature and salinity and 15 levels.
- Current selection uses projected bounding-box overlap and an opposite-side-of-globe heuristic; this is insufficient for the error eligibility domain.
- Existing vertical knots are explicitly MPAS-source-driven and must not be reused blindly for the reverse fit.
- There is no project-level automated test suite yet.
- The local tangent-plane approximation must be checked for the canonical regional domain.
- Velocity basis transformation is intentionally outside this first temperature/salinity milestone.

## Plan Adjustments

### 2026-08-13: Correct MPAS height semantics in both executables

The initial plan incorrectly treated SSH as an offset applied to every layer. The clarified dataset semantics are that the top cell spans from positive `SSH` to `-layerThickness[0]`, while every lower cell starts at the bottom of the cell above it and has exactly its configured layer thickness. The shared `CBlock2` geometry construction was corrected accordingly, intentionally changing both `mfa_remap` and `remap_loop`. Canonical runs made with the earlier interpretation are invalidated and must be repeated.

### 2026-08-13: Preserve full target topology in forward H5M

`forward_remap.h5m` contains all ROMS cells so existing topology, masks, and metadata remain intact. Complete decoded profile tags are populated on wet cells; masked cells retain the dense-tag default value. Quantitative and visual checks must apply `ROMS_Mask == 1`.

### 2026-08-13: Resolve local vertical-support rule

For a horizontally eligible MPAS polygon, a level is vertically eligible only when its centroid height lies in the intersection of sampled height ranges of all wet ROMS cells with positive-area overlap. This conservative rule ensures the entire horizontal MPAS cell has sampled vertical support at the reported level.

### 2026-08-13: Use exact polygon-union area

The first coverage implementation summed MPAS/ROMS intersection areas under the assumption that projected ROMS cells do not overlap. Review showed that projection-induced or malformed overlaps could conceal an uncovered gap. Coverage now computes the actual union of clipped wet-ROMS regions with Boost.Geometry. This changed two canonical cells from full to partial and reduced the eligible set from 210,781 to 210,667 points. Boost is now an explicit header-only build dependency.

### 2026-08-13: Deterministic plane is loop-only

MFA's sampled normal estimator produced slightly different source selections between processes. `remap_loop` now uses the normalized mean radial ROMS centroid as a deterministic regional plane normal. The legacy `mfa_remap` retains its previous sampled-normal behavior; only the explicitly requested shared cell-height correction changes that executable.

## Work Log

### 2026-08-13

- Agreed on a dedicated `remap_loop` executable and reusable implementation in `src/cblock2.hpp`.
- Agreed to leave `mfa_remap` behavior unchanged.
- Agreed on temperature and salinity, a single-process round-trip invocation, explicit reverse controls, and `17 x 17 x 17` reverse defaults.
- Defined headline horizontal eligibility as full coverage by wet ROMS cells, not ROMS bounding-box containment.
- Defined vertical eligibility as local ROMS sampled-depth support.
- Created this plan and work-record document.
- Added `src/remap_loop.cpp`, registered and installed `remap_loop`, and verified both executable help surfaces after a managed Spack build.
- Added loop-only complete profile packing and H5M output in `src/cblock2.hpp`; the existing legacy tag path remains selected by `mfa_remap`.
- Initially changed MPAS layer geometry under an incorrect SSH interpretation; that result was superseded after clarification.
- Corrected shared MPAS geometry for both executables: the top cell spans from `SSH` to `-layerThickness[0]`, and lower cells are contiguous with their configured thicknesses.
- Ran the canonical forward stage: 6,538 selected MPAS cells (6,396 core and 142 halo), 335,463 active MPAS points, 68,921 wet ROMS cells, and 6,892,100 ROMS decode points.
- Verified `forward_remap.h5m` contains two fixed-width 100-level profile tags and opens through the ParaView MOAB plugin. Scripted plugin checks loaded levels 0, 50, and 99.
- Observed substantial unregularized forward overshoot. With the final corrected MPAS heights, decoded temperature reaches about `54.66` despite a documented active source maximum near `31.71`. This is a scientific fitting result to investigate, not a profile-packing failure.
- Added an independent reverse MFA with default `17 x 17 x 17` controls, independent degree/regularization options, ROMS-derived vertical knots, and `reverse_remap.mfa` serialization.
- Stabilized loop-mode selection with a deterministic radial plane normal. The canonical selection is 6,538 MPAS cells: 6,396 core and 142 halo, with 335,463 active levels.
- Implemented actual wet-ROMS polygon-union coverage using a spatial bin index and Boost.Geometry. The canonical classification is 4,221 full, 347 partial, 1,828 uncovered, and 142 halo cells.
- Implemented conservative local vertical support, producing 210,667 eligible active levels in the canonical run.
- Added `roundtrip_mpas.h5m` with coverage fraction/class, per-level eligibility, complete round-trip profiles, signed errors, and absolute errors.
- Added `roundtrip_metrics.csv` with overall and per-level counts, bias, RMSE, maximum absolute error, persistent MPAS `GLOBAL_ID` and level at the maximum, and original/round-trip ranges.
- The corrected authoritative canonical metrics are temperature RMSE `0.1965704585`, bias `-0.0008287612`, max absolute error `1.41108667`; salinity RMSE `0.0880278759`, bias `0.0000991428`, max absolute error `0.685817743`.
- Scripted ParaView plugin checks loaded `roundtrip_mpas.h5m` coverage, eligibility, and representative error components. Coverage ranges from 0 to 1, classes from 0 to 3, and error arrays expose the reported maxima.
- A code review identified and prompted fixes for exact union coverage, loop-only deterministic plane behavior, stable IDs, finite checks, reverse settings, and reverse serialization.
- Repeated the authoritative exact-union run after all fixes. Selection counts, coverage classes, eligible-point count, extrema locations, and every reported aggregate metric matched exactly.
