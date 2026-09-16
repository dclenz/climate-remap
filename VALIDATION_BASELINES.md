# Validation Baselines

This ledger distinguishes numerical configurations that were previously described
collectively as the canonical result. Both use the Bay of Bengal inputs, degree-2
science models, `11 x 11 x 11` forward controls, `17 x 17 x 17` reverse controls,
temperature and salinity, and the same coverage/eligibility rules.

## Known Results

| Baseline | Forward / reverse regularization | Eligible points | Temperature bias | Temperature RMSE | Temperature max abs. | Salinity bias | Salinity RMSE | Salinity max abs. | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Unregularized exact-union | `0 / 0` | 210,667 | -0.0008287612 | 0.1965704585 | 1.41108667 | 0.0000991428 | 0.0880278759 | 0.685817743 | Repeated canonical runs recorded in `REMAP_LOOP_PLAN.md`; the retained `runs/run.log` is from this configuration. |
| Regularized defaults | `1 / 1` | 210,667 | -0.0000915246 | 0.0156696743 | 0.344090204 | 0.0000139794 | 0.0149424929 | 0.238294726 | Retained `runs/roundtrip_metrics.csv`; produced after regularization became the default. |

The regularized result is the current default baseline. The unregularized result is
retained as a diagnostic comparison because it exposes the forward temperature
overshoot discussed in the work record. Results from the two configurations must not
be compared as if they came from identical settings.

The historical files under `runs/` predate complete machine-readable provenance:
the log and CSV represent different configurations, and hashes of their original
inputs and executable were not captured. They should therefore be treated as
supporting evidence, not as fully reproducible run bundles.

## Reproducing A Baseline

Activate the managed environment, rebuild the installed develop package after source
changes, and run one of:

```bash
source load-env.sh
spack install
scripts/run-canonical-validation.sh regularized
scripts/run-canonical-validation.sh unregularized
```

The runner refuses to overwrite a non-empty output directory. By default it writes
to `runs/bob-regularized` or `runs/bob-unregularized`. An explicit empty output
directory may be supplied as the second argument.

Each run writes `run-manifest.txt` beside its outputs. The manifest records:

- the selected baseline and complete numerical settings;
- start/end timestamps and process status;
- Git branch, revision, and dirty state;
- Spack environment and installed executable;
- SHA-256 hashes of the executable and both input datasets;
- `MFA_NUM_THREADS`;
- the exact invocation; and
- sizes and SHA-256 hashes of all generated artifacts and the run log.

For an authoritative baseline, rebuild from a clean checkout and require
`git_dirty=false` in the manifest. Preserve the entire output directory rather than
copying the metrics CSV alone.
