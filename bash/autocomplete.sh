#!/usr/bin/env bash

# ─────────────────────────────────────────────────────────────────────────────
# initialise.sh — project shell helpers for running and submitting jobs
#
# Source this file to get: run, submit, submit_array
# Tab-completion is registered automatically for bash and zsh.
# ─────────────────────────────────────────────────────────────────────────────

# Zsh compatibility: load bash-style completion
if [[ -n "$ZSH_VERSION" ]]; then
    autoload -U +X bashcompinit && bashcompinit
fi

# Resolve project root
_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -z "$ZSH_VERSION" ]] && _PROJECT_ROOT="$(dirname "$_PROJECT_ROOT")"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# _require_dir — assert cwd is one of the allowed directories.
# Sets `dir` and `input_dir` in the caller's scope.
# Usage: _require_dir "command_name" fem nn
_require_dir() {
    local cmd="$1"; shift
    dir=$(basename "$PWD")
    local allowed=("$@")

    for a in "${allowed[@]}"; do
        [[ "$dir" == "$a" ]] && { input_dir="$_PROJECT_ROOT/input/$dir"; return 0; }
    done

    echo "Error: '$cmd' must be called inside one of: ${allowed[*]}" >&2
    return 1
}

# _expand_names — expand a list of name/glob arguments into job names.
# Populates the `expanded` array in the caller's scope.
# Usage: _expand_names "command_name" "$input_dir" "$@"
_expand_names() {
    local cmd="$1" search_dir="$2"; shift 2
    expanded=()
    local used_wildcard=0

    for name in "$@"; do
        if [[ "$name" == *[\*\?\[]* ]]; then
            used_wildcard=1
            local pat matches=()
            [[ "$name" == *.json ]] && pat="$name" || pat="${name}.json"

            local f
            while IFS= read -r f; do
                f=${f##*/}; f=${f%.json}
                matches+=("$f")
            done < <(find "$search_dir" -maxdepth 1 -type f -name "$pat" -print 2>/dev/null | LC_ALL=C sort)

            if [[ ${#matches[@]} -eq 0 ]]; then
                echo "[$cmd] pattern '$name' matched 0 files in $search_dir" >&2
                return 3
            fi
            expanded+=("${matches[@]}")
        else
            expanded+=("${name%.json}")
        fi
    done

    if [[ $used_wildcard -eq 1 ]]; then
        echo "[$cmd] ${#expanded[@]} item(s) matched:"
        local i=0
        for name in "${expanded[@]}"; do
            i=$((i + 1))
            echo "  $i) $name"
        done
    fi
}

# _build_jobfile — write a SLURM job script from a header + payload commands.
# Usage: _build_jobfile "$header" "$jobfile" "payload line 1" "payload line 2" ...
_build_jobfile() {
    local header="$1" jobfile="$2"; shift 2

    {
        if head -n1 "$header" | grep -q '^#!'; then
            cat "$header"
        else
            echo '#!/bin/bash -l'
            cat "$header"
        fi
        echo
        for line in "$@"; do echo "$line"; done
    } > "$jobfile"
    chmod 750 "$jobfile"
}

# _sbatch_submit — submit a job file (or print it in dry-run mode).
# Usage: _sbatch_submit "$cmd" "$i" "$total" "$jobname" "$logdir" "$jobfile" \
#            ["$out_pat" "$err_pat"] ["$dry_run"]
# dry_run is the 9th arg; pass "1" to skip sbatch and print the script instead.
_sbatch_submit() {
    local cmd="$1" i="$2" total="$3" jobname="$4" logdir="$5" jobfile="$6"
    local out_pat="${7:-%x_%j.out}" err_pat="${8:-%x_%j.err}"
    local dry_run="${9:-0}"

    if [[ $dry_run -eq 1 ]]; then
        echo "[$cmd] ($i/$total) [dry-run] would sbatch -> $jobname"
        echo "--- $jobfile ---"
        cat "$jobfile"
        echo "--- end ---"
        echo
        return 0
    fi

    echo "[$cmd] ($i/$total) sbatch -> $jobname"
    local sbout jid
    sbout=$(sbatch --job-name="$jobname" \
                   --output="$logdir/$out_pat" \
                   --error="$logdir/$err_pat" \
                   "$jobfile")
    jid=${sbout##* }
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "[$cmd] queued: $jobname (job $jid)"
    else
        echo "[$cmd] submitted: $sbout"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 1) run — execute jobs locally via main.py
#    Must be called from fem/ or nn/
# ─────────────────────────────────────────────────────────────────────────────
run() {
    local dir input_dir
    _require_dir "run" fem nn || return $?

    # Parse options
    local nobar=0 OPTIND=1
    while getopts ":q-:" opt; do
        case "$opt" in
            q) nobar=1 ;;
            -)  case "$OPTARG" in
                    nobar) nobar=1 ;;
                    *)     echo "Unknown option: --$OPTARG" >&2; return 2 ;;
                esac ;;
            \?) echo "Unknown option: -$OPTARG" >&2; return 2 ;;
        esac
    done
    shift $((OPTIND - 1))
    [[ "${1:-}" == "--" ]] && shift

    if [[ $# -eq 0 ]]; then
        echo "Usage: run [-q|--nobar] <name|pattern> [more...]" >&2
        return 2
    fi

    # Expand names/globs
    local expanded
    _expand_names "run" "$input_dir" "$@" || return $?

    # Build extra args for python
    local extra_args=()
    [[ $nobar -eq 1 ]] && extra_args+=(nobar)

    # Execute each job
    local rc=0 i=0
    for name in "${expanded[@]}"; do
        i=$((i + 1))
        echo "[run] ($i/${#expanded[@]}) starting: $name"
        python3 -m main "$name" "${extra_args[@]}"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "[run] ($i/${#expanded[@]}) failed: $name (exit $rc)" >&2
            return $rc
        fi
        echo "[run] ($i/${#expanded[@]}) finished: $name"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# 2) submit — submit individual SLURM jobs
#    Must be called from fem/ or nn/
# ─────────────────────────────────────────────────────────────────────────────
submit() {
    local dir input_dir
    _require_dir "submit" fem nn || return $?

    # Parse leading --dry-run flag
    local dry_run=0
    if [[ "${1:-}" == "--dry-run" ]]; then dry_run=1; shift; fi

    if [[ $# -lt 2 ]]; then
        echo "Usage: submit [--dry-run] <header_file> <name|pattern> [more...]" >&2
        return 2
    fi

    local header_file="$1"; shift
    local header="$_PROJECT_ROOT/.hpc_submit/$header_file"
    [[ -f "$header" ]] || { echo "[submit] Header not found: $header" >&2; return 1; }

    local jobdir="$_PROJECT_ROOT/input/slurm"
    local logdir="$_PROJECT_ROOT/output/slurm_log"
    mkdir -p "$jobdir" "$logdir"

    local expanded
    _expand_names "submit" "$input_dir" "$@" || return $?

    local i=0 stamp jobfile jobname header_name
    header_name="${header_file%.sh}"

    for name in "${expanded[@]}"; do
        i=$((i + 1))
        stamp=$(date +%Y%m%d_%H%M)
        jobname="${header_name}_${dir}_${name}"
        jobfile="${jobdir}/${jobname}_${stamp}"

        _build_jobfile "$header" "$jobfile" \
            "cd $_PROJECT_ROOT" \
            "source ./initialise.sh" \
            "cd $dir" \
            "$(printf 'run -q %q' "$name")"

        _sbatch_submit "submit" "$i" "${#expanded[@]}" "$jobname" "$logdir" "$jobfile" \
            "%x_%j.out" "%x_%j.err" "$dry_run"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# 3) submit_array — submit SLURM array jobs (one array per run name)
#    Must be called from fem/ or nn/
# ─────────────────────────────────────────────────────────────────────────────
submit_array() {
    local dir input_dir
    _require_dir "submit_array" fem nn || return $?

    # Parse leading --dry-run flag
    local dry_run=0
    if [[ "${1:-}" == "--dry-run" ]]; then dry_run=1; shift; fi

    if [[ $# -lt 3 ]]; then
        echo "Usage: submit_array [--dry-run] <header_file> <array_spec> <name|pattern> [more...]" >&2
        echo "  array_spec examples: 1-10 | 1-10%2 | 1-10:2 | 1,3,5,7" >&2
        return 2
    fi

    local header_file="$1"; shift
    local array_spec="$1"; shift

    local header="$_PROJECT_ROOT/.hpc_submit/$header_file"
    [[ -f "$header" ]] || { echo "[submit_array] Header not found: $header" >&2; return 1; }

    local jobdir="$_PROJECT_ROOT/input/slurm"
    local logdir="$_PROJECT_ROOT/output/slurm_log"
    mkdir -p "$jobdir" "$logdir"

    local expanded
    _expand_names "submit_array" "$input_dir" "$@" || return $?

    local i=0 stamp jobfile jobname header_name
    header_name="${header_file%.sh}"

    for name in "${expanded[@]}"; do
        i=$((i + 1))
        stamp=$(date +%Y%m%d_%H%M%S)
        jobname="${header_name}_${dir}_${name}_A${array_spec}"
        jobfile="${jobdir}/${jobname}_${stamp}.sh"

        _build_jobfile "$header" "$jobfile" \
            "#SBATCH --array=${array_spec}" \
            "" \
            "cd $_PROJECT_ROOT" \
            "source ./initialise.sh" \
            "cd $dir" \
            'export SEED="$SLURM_ARRAY_TASK_ID"' \
            "$(printf 'run -q %q' "$name")"

        _sbatch_submit "submit_array" "$i" "${#expanded[@]}" \
            "$jobname" "$logdir" "$jobfile" "%x_%A_%a.out" "%x_%A_%a.err" "$dry_run"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Zsh: disable globbing for commands that accept wildcards
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "$ZSH_VERSION" ]]; then
    alias run='noglob run'
    alias submit='noglob submit'
    alias submit_array='noglob submit_array'
fi

# ─────────────────────────────────────────────────────────────────────────────
# Tab completion
# ─────────────────────────────────────────────────────────────────────────────

# Complete run names from JSON files in the appropriate input/ directory
_run_complete() {
    local cur dir input_dir opts
    cur=${COMP_WORDS[COMP_CWORD]}

    [[ "$cur" == -* ]] && return 0

    dir=$(basename "$PWD")
    case "$dir" in
        fem) input_dir="$_PROJECT_ROOT/input/fem" ;;
        nn)  input_dir="$_PROJECT_ROOT/input/nn"  ;;
        *)   return 0 ;;
    esac

    opts=$(ls "$input_dir"/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//')
    COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
}

# Complete header files (1st real arg) then run names (subsequent args).
# Accounts for an optional leading --dry-run flag.
_submit_complete() {
    local cur cword
    cur=${COMP_WORDS[COMP_CWORD]}
    cword=${COMP_CWORD}

    # Offer --dry-run if typing a flag
    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "--dry-run" -- "${cur}"))
        return 0
    fi

    # Skip --dry-run when calculating effective position
    local offset=1
    if [[ "${COMP_WORDS[1]:-}" == "--dry-run" ]]; then offset=2; fi

    # First real arg -> header files
    if [[ $cword -eq $offset ]]; then
        local headers_dir="${SLURM_HEADERS_DIR:-$_PROJECT_ROOT/.hpc_submit}"
        local opts
        opts=$(ls -1 "$headers_dir" 2>/dev/null | grep -v '/$')
        COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
        return 0
    fi

    # Subsequent args -> run names
    _run_complete
}

complete -o default -F _run_complete run
complete -o default -F _submit_complete submit
complete -o default -F _submit_complete submit_array
