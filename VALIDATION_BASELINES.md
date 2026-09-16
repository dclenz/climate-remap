# Validation Baselines

The supported canonical workflow always uses positive regularization. Its strength
may vary between experiments and is part of each run's recorded provenance. The
current reference result uses the Bay of Bengal inputs, degree-2 science models,
`11 x 11 x 11` forward controls, `17 x 17 x 17` reverse controls, temperature and
salinity, and regularization strength `1` on both legs.

## Known Results

| Baseline | Forward / reverse regularization | Eligible points | Temperature bias | Temperature RMSE | Temperature max abs. | Salinity bias | Salinity RMSE | Salinity max abs. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | `1 / 1` | 210,667 | -0.0000915246 | 0.0156696743 | 0.344090204 | 0.0000139794 | 0.0149424929 | 0.238294726 |

The old flat `runs/` artifacts predated machine-readable provenance and were removed.
The table preserves their regularized summary as a historical reference, not a
fully reproducible run bundle. Future accepted results should be backed by a complete
manifest-bearing run directory.

## Reproducing A Baseline

Activate the managed environment, rebuild the installed develop package after source
changes, and run with a positive regularization strength:

```bash
source load-env.sh
spack install
scripts/run-canonical-validation.sh
scripts/run-canonical-validation.sh 0.5
```

The runner refuses to overwrite a non-empty output directory. By default it writes
to a timestamped run directory such as `runs/bob-20260916T183000Z-reg-1`.
An explicit empty output directory may be supplied as the second argument. A zero or
negative strength is rejected.

Each run writes `run-manifest.txt` beside its outputs. The manifest records:

- the complete numerical settings, including regularization strength;
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
