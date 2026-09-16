#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'Usage: %s [regularized|unregularized] [output-directory]\n' "$0" >&2
}

profile="${1:-regularized}"
case "${profile}" in
    regularized)
        forward_regularization=1
        reverse_regularization=1
        ;;
    unregularized)
        forward_regularization=0
        reverse_regularization=0
        ;;
    *)
        usage
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
output_dir="${2:-${project_root}/runs/bob-${profile}}"
mpas_file="${project_root}/data/mpas_grid_tstep0.h5m"
roms_file="${project_root}/data/roms_3d_projected_BoB.h5m"

if [[ -z "${MFA_REMAP_PATH:-}" || ! -x "${MFA_REMAP_PATH}/bin/remap_loop" ]]; then
    printf 'Activate the managed environment first: source %s/load-env.sh\n' "${project_root}" >&2
    exit 1
fi

for required_file in "${mpas_file}" "${roms_file}"; do
    if [[ ! -f "${required_file}" ]]; then
        printf 'Required canonical input is missing: %s\n' "${required_file}" >&2
        exit 1
    fi
done

if [[ -e "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Output directory must be absent or empty: %s\n' "${output_dir}" >&2
    exit 1
fi
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

manifest="${output_dir}/run-manifest.txt"
log="${output_dir}/run.log"
executable="${MFA_REMAP_PATH}/bin/remap_loop"
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_branch="$(git -C "${project_root}" branch --show-current)"
if [[ -n "$(git -C "${project_root}" status --porcelain --untracked-files=normal)" ]]; then
    git_dirty=true
else
    git_dirty=false
fi
start_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

{
    printf 'manifest_version=1\n'
    printf 'initial_status=running\n'
    printf 'profile=%s\n' "${profile}"
    printf 'started_utc=%s\n' "${start_utc}"
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_branch=%s\n' "${git_branch}"
    printf 'git_dirty=%s\n' "${git_dirty}"
    printf 'spack_environment=%s\n' "${SPACKENV:-unknown}"
    printf 'mfa_remap_prefix=%s\n' "${MFA_REMAP_PATH}"
    printf 'executable=%s\n' "${executable}"
    printf 'executable_sha256=%s\n' "$(sha256_file "${executable}")"
    printf 'mpas_file=%s\n' "${mpas_file}"
    printf 'mpas_sha256=%s\n' "$(sha256_file "${mpas_file}")"
    printf 'roms_file=%s\n' "${roms_file}"
    printf 'roms_sha256=%s\n' "$(sha256_file "${roms_file}")"
    printf 'forward_degree=2\n'
    printf 'forward_controls=11,11,11\n'
    printf 'forward_regularization=%s\n' "${forward_regularization}"
    printf 'forward_regularize_first_and_second=0\n'
    printf 'reverse_degree=2\n'
    printf 'reverse_controls=17,17,17\n'
    printf 'reverse_regularization=%s\n' "${reverse_regularization}"
    printf 'reverse_regularize_first_and_second=0\n'
    printf 'variables=timeDaily_avg_activeTracers_temperature_3d,timeDaily_avg_activeTracers_salinity_3d\n'
    printf 'mfa_num_threads=%s\n' "${MFA_NUM_THREADS:-unlimited}"
    printf 'command=%q --dom_dim 3 --vars_degree 2 --vars_nctrl 11 --reverse_vars_degree 2 --reverse_vars_nctrl 17 --regularization %q --reverse_regularization %q --reg1and2 0 --reverse_reg1and2 0 --varNames timeDaily_avg_activeTracers_temperature_3d --varNames timeDaily_avg_activeTracers_salinity_3d --mpasfile %q --romsfile %q\n' \
        "${executable}" "${forward_regularization}" "${reverse_regularization}" \
        "${mpas_file}" "${roms_file}"
} > "${manifest}"

set +e
(
    cd -- "${output_dir}"
    "${executable}" \
        --dom_dim 3 \
        --vars_degree 2 \
        --vars_nctrl 11 \
        --reverse_vars_degree 2 \
        --reverse_vars_nctrl 17 \
        --reg1and2 0 \
        --reverse_reg1and2 0 \
        --varNames timeDaily_avg_activeTracers_temperature_3d \
        --varNames timeDaily_avg_activeTracers_salinity_3d \
        --mpasfile "${mpas_file}" \
        --romsfile "${roms_file}" \
        --regularization "${forward_regularization}" \
        --reverse_regularization "${reverse_regularization}"
) 2>&1 | tee "${log}"
run_status=${PIPESTATUS[0]}
set -e

{
    printf 'finished_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'exit_status=%s\n' "${run_status}"
    if [[ "${run_status}" -eq 0 ]]; then
        printf 'status=complete\n'
    else
        printf 'status=failed\n'
    fi
    for output_name in forward_remap.h5m roundtrip_mpas.h5m roundtrip_metrics.csv \
                       forward_remap.mfa reverse_remap.mfa run.log; do
        output_path="${output_dir}/${output_name}"
        if [[ -f "${output_path}" ]]; then
            key="${output_name//[^[:alnum:]]/_}"
            printf '%s_bytes=%s\n' "${key}" "$(wc -c < "${output_path}" | tr -d ' ')"
            printf '%s_sha256=%s\n' "${key}" "$(sha256_file "${output_path}")"
        fi
    done
} >> "${manifest}"

printf 'Run manifest: %s\n' "${manifest}"
exit "${run_status}"
