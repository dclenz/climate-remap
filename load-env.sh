#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'This script must be sourced: source %s\n' "$0" >&2
    exit 1
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

_mfa_remap_load_env() {
    _mfa_remap_setup_spack || return 1

    export MFA_REMAP_ROOT
    MFA_REMAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    export SPACKENV="${SPACK_ENV_NAME:-mfa-remap-env}"

    if ! spack env activate "${SPACKENV}"; then
        printf 'Unable to activate %s. Run ./create-env.sh first.\n' "${SPACKENV}" >&2
        return 1
    fi

    export MFA_REMAP_PATH
    MFA_REMAP_PATH="$(spack location -i mfa-remap)" || return 1
    printf 'Activated %s (mfa-remap: %s)\n' "${SPACKENV}" "${MFA_REMAP_PATH}"
}

_mfa_remap_load_env
_mfa_remap_status=$?
unset -f _mfa_remap_load_env _mfa_remap_setup_spack
return "${_mfa_remap_status}"
