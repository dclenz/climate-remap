#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'This installer must be executed: %s\n' "${BASH_SOURCE[0]}" >&2
    return 1
fi

_mfa_remap_setup_spack() {
    if [[ -n "${SPACK_ROOT:-}" && -f "${SPACK_ROOT}/share/spack/setup-env.sh" ]]; then
        source "${SPACK_ROOT}/share/spack/setup-env.sh"
        return
    fi

    local spack_command
    spack_command="$(command -v spack 2>/dev/null || true)"
    if [[ -n "${spack_command}" ]]; then
        local spack_root
        spack_root="$(cd -- "$(dirname -- "${spack_command}")/.." && pwd -P)"
        if [[ -f "${spack_root}/share/spack/setup-env.sh" ]]; then
            source "${spack_root}/share/spack/setup-env.sh"
            return
        fi
    fi

    if [[ -f "${HOME}/spack/share/spack/setup-env.sh" ]]; then
        source "${HOME}/spack/share/spack/setup-env.sh"
        return
    fi

    printf 'Unable to find Spack. Set SPACK_ROOT or put spack on PATH.\n' >&2
    return 1
}

_mfa_remap_create_env() {
    _mfa_remap_setup_spack || return 1

    local project_root
    project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    local environment_name="${SPACK_ENV_NAME:-mfa-remap-env}"

    if spack env activate "${environment_name}" >/dev/null 2>&1; then
        printf 'Spack environment %s already exists; updating it.\n' "${environment_name}"
    else
        printf 'Creating Spack environment %s.\n' "${environment_name}"
        spack env create "${environment_name}" "${project_root}/env.yaml" || return 1
        spack env activate "${environment_name}" || return 1
    fi

    # Register this checkout only in this environment, then use it as the
    # mfa-remap develop source. MFA remains in develop mode under the environment.
    spack repo remove --scope "env:${environment_name}" mfa-remap >/dev/null 2>&1 || true
    spack repo add --scope "env:${environment_name}" "${project_root}" || return 1
    spack develop --path "${project_root}" --no-clone \
        'mfa-remap@master build_type=RelWithDebInfo' || return 1
    spack develop 'mfa@master~examples~tests thread=serial build_type=RelWithDebInfo' || return 1

    # This project owns the environment's root specs. Reset roots left by older
    # versions of this installer without uninstalling packages from Spack.
    spack remove --all || return 1
    spack add 'mfa-remap@master build_type=RelWithDebInfo' || return 1
    spack add 'mpich@4' || return 1

    printf 'Concretizing and installing %s.\n' "${environment_name}"
    spack concretize --force || return 1
    spack install || return 1

    local install_prefix
    install_prefix="$(spack location -i mfa-remap)" || return 1
    printf 'Installed mfa-remap in %s\n' "${install_prefix}"
    printf 'Activate it with: source %s/load-env.sh\n' "${project_root}"
}

_mfa_remap_create_env
status=$?
unset -f _mfa_remap_create_env _mfa_remap_setup_spack
exit "${status}"
