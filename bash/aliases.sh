#!/usr/bin/env bash
script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
project_root="$(cd "$script_dir/.." >/dev/null 2>&1 && pwd -P)"

DISABLE_MKDOCS_2_WARNING=true

alias format="pre-commit run --all-files"
alias lint="skylos ${project_root} -c 80 --exclude-folder .local --exclude-folder tests"
alias documentation="mkdocs serve"
diagnostics() {
  MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true \
  mlflow ui \
    --host 0.0.0.0 \
    --port 5001 \
    --backend-store-uri "sqlite:////${project_root}/output/mlflow/mlflow.db" \
    --default-artifact-root "file://${project_root}/output/mlflow/artifacts"
}
