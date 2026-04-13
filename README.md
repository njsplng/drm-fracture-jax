# drm-fracture-jax

Companion repository for the paper "[Deep learning-based phase-field modelling of brittle fracture in anisotropic media](https://doi.org/10.48550/arXiv.2603.20120)".

> **Please read this file in its entirety before attempting to run the code.**

---

## Setup

### Environment

Run the following from the repository root to create the virtual environment and install all dependencies:

```bash
bash bash/setup_venv.sh [--no-petsc]
```

The optional `--no-petsc` flag skips the build of `petsc` and `petsc4py`, which are only required by the FEM solver. If you are only interested in the neural network component, this flag is recommended.

> **Note:** Building `petsc4py` typically takes 5–10 minutes. Minimal terminal output during this step is expected and does not indicate an error.

Once setup is complete, initialise the virtual environment and repository aliases by running:

```bash
source ./initialise.sh
```

Optionally, ensure that the tests included with the package pass by running `pytest` for the full testing suite (unit tests + FEM integration tests, will take about 5 minutes) or `pytest tests` (from repository's root, should take <1 minute) to run just the unit tests.

### HPC Systems

If running on an HPC cluster, ensure that `bash/module_loads.sh` is updated to reflect the modules available on your system. When building PETSc on HPC systems, you may need to adjust the module imports to avoid conflicts with system-provided MPI and BLAS installations. Furthermore, update the SLURM submission headers under `.hpc_submit` according to your system submission script requirements.

### GPU Support

GPU-compatible JAX is installed automatically **on Linux only**. If you are using a different platform or a custom CUDA setup, adjust `requirements.txt` accordingly before running the setup script.

### ParaView Rendering

To enable automatic GIF rendering of simulation outputs on HPC systems, update `bash/module_loads_pvrender.sh` with the relevant modules available on your system. On local machines, set the `PARAVIEW_PYTHON` environment variable (see [Variables](#environment-variables)).

### Input File Reference

Input file structure and schema documentation is available at [Input Schemas](reference/schemas/SUMMARY.md).

---

## Aliases

After sourcing `initialise.sh`, the following aliases become available. Tab-completion is enabled for `run`, `submit` and `submit_array` commands. Respective input files `{input-file-name}` are available based on the current directory. SLURM header files `{header}` are located under `.hpc_submit`.

| Alias | Description |
|---|---|
| `run {input-file-name}` | Runs a simulation interactively. Must be executed from within the `nn` or `fem` directory. |
| `submit [--dry-run] {header} {input-file-name}` | Submits a simulation to a SLURM scheduler. Use `--dry-run` to preview the generated submission script. |
| `submit_array [--dry-run] {header} {range} {input-file-name}` | Submits an array of simulations to SLURM. Seeds are assigned from the SLURM array task IDs, making this ideal for repeated runs with varying random seeds. Use `--dry-run` to preview. |
| `format` | Runs the `pre-commit` formatting routine over code and input files. |
| `diagnostics` | Launches an MLflow server for inspecting neural network training diagnostics. |
| `documentation` | Starts a live MkDocs session for browsing the repository documentation. |
| `lint` | Runs a linting session using `skylos`. |

---

## Environment Variables

### General

- **`SCAN_ALL_FILES`** — Set to `true` to run the input file validator against all input files in the repository, rather than only files changed since the last Git commit.

### ParaView-Specific

- **`PARAVIEW_PYTHON`** — Path to the `pvpython` executable of your local ParaView installation. Required for local ParaView image rendering.
- **`RENDER_OUTPUT`** — Set to `true` to automatically render ParaView output as a GIF after each simulation. This requires either `PARAVIEW_PYTHON` (local) or the ParaView modules in `bash/module_loads_pvrender.sh` (HPC) to be configured correctly. Rendering will **not** occur unless this variable is explicitly set.

---

## Running Simulations

To run either an FEM or NN simulation, navigate to the relevant subdirectory (`fem` or `nn`) and use the `run` or `submit` alias with the appropriate input file name.

Input files are organised into `parent` files, which define shared material parameters, loading conditions, and output settings common to both solvers, and solver-specific `nn` or `fem` files, which configure the respective solution method. Full documentation for all input file parameters is available under `input/schemas`.

Files are automatically renamed based on their parameter values when `format` is run.

---

## Paper Examples

The input files corresponding to the numerical examples presented in the paper are listed below.

| Example | Input File |
|---|---|
| Isotropic 4th order | `square_201_IGA2x2_9IP_PE_NO_PF4_ND_RN4_24x300_RFF192x0_1_tanh2-0TG_TLO_rprop_DF200E6_0e-3` |
| Cubic anisotropy, −30° | `cubic_an_square_201_IGA2x2_9IP_PE_CA_AN-30_ND_RN4_24x300_RFF192x0_1_tanh2-0TG_TLO_rprop_DF200E6_0e-3` |
| Cubic anisotropy, −50° | `cubic_an_square_201_IGA2x2_9IP_PE_CA_AN-50_ND_RN4_24x300_RFF192x0_1_tanh2-0TG_TLO_rprop_DF200E6_0e-3` |
| Orthotropic anisotropy, −30° | `ortho_an_square_201_IGA2x2_9IP_PE_OA_AN-30_ND_RN4_24x300_RFF192x0_1_tanh2-0TG_TLO_rprop_DF200E0_015` |
| Orthotropic anisotropy, −50° | `ortho_an_square_201_IGA2x2_9IP_PE_OA_AN-50_ND_RN4_24x300_RFF192x0_1_tanh2-0TG_TLO_rprop_DF200E0_015` |
| Crack kinking | `square_201_IGA2x2_9IP_PE_OA_AN+20_3X_ND_RN4_24x300_RFF192x0_1_tanh2-0TG_TLO_rprop_DF200E0_035` |

---

## API Reference

- [Shared Functions](reference/Shared/SUMMARY.md)
- [FEM-only](reference/FEM-only/SUMMARY.md)
- [NN-only](reference/NN-only/SUMMARY.md)

---

## Note

- Docstrings throughout the codebase were generated with the assistance of an AI agent to improve repository accessibility. If you encounter any inaccuracies or issues with the code, please open a GitHub issue or contact [nojus.plunge@warwick.ac.uk](mailto:nojus.plunge@warwick.ac.uk).
