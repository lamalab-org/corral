

set -ex



test -f "${PREFIX}/bin/micromamba"
test ! -f "${PREFIX}/etc/profile.d/mamba.sh"
micromamba --help
export MAMBA_ROOT_PREFIX="$(mktemp -d)"
micromamba create -n test --override-channels -c conda-forge --yes python=3.9
"${MAMBA_ROOT_PREFIX}/envs/test/bin/python" --version
"${MAMBA_ROOT_PREFIX}/envs/test/bin/python" -c "import os"
exit 0
