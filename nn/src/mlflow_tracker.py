"""MLflow experiment tracker for DRM simulations."""

import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import equinox as eqx
import mlflow
import pandas as pd

# -----------------------------------------------------------
# Byte-level serialisation
# -----------------------------------------------------------


def _serialise_to_bytes(pytree: object) -> bytes:
    """Serialise an Equinox-compatible PyTree to bytes."""
    # Allocate workspace
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, pytree)
    return buf.getvalue()


def _deserialise_from_bytes(data: bytes, skeleton: object) -> object:
    """Deserialise bytes into a PyTree, overwriting leaves in the skeleton."""
    # Deserialize from bytes
    buf = io.BytesIO(data)
    return eqx.tree_deserialise_leaves(buf, skeleton)


# -----------------------------------------------------------
# Public API — log to MLflow
# -----------------------------------------------------------


def log_network_to_mlflow(
    network: eqx.Module,
    step: int,
    optimiser_states: Optional[List[eqx.Module]] = None,
    artifact_subdir: str = "model_checkpoints",
) -> None:
    """Log network weights and optionally optimiser states to MLflow.

    Saves checkpoint artifacts with filenames based on the step number.

    Parameters
    ----------
    network : eqx.Module
        The neural network to serialize and log.
    step : int
        The training step number, used in the artifact filename.
    optimiser_states : List[eqx.Module], optional
        Optimiser state objects to log alongside the network.
    artifact_subdir : str, optional
        Subdirectory for artifacts. Default is "model_checkpoints".
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Network ---
        # Network checkpoint
        net_path = os.path.join(tmpdir, f"network_step_{step}.eqx")
        with open(net_path, "wb") as f:
            f.write(_serialise_to_bytes(network))
        mlflow.log_artifact(net_path, artifact_path=artifact_subdir)
        logging.info(f"Logged network checkpoint for step {step}.")

        # --- Optimiser states ---
        # Optimiser checkpoint
        if optimiser_states is not None:
            optim_path = os.path.join(tmpdir, f"optim_step_{step}.eqx")
            container = {"states": optimiser_states}
            with open(optim_path, "wb") as f:
                f.write(_serialise_to_bytes(container))
            mlflow.log_artifact(optim_path, artifact_path=artifact_subdir)
            logging.info(f"Logged optimiser states for step {step}.")


# -----------------------------------------------------------
# Public API — load from MLflow
# -----------------------------------------------------------


def load_network_from_mlflow(
    run_id: str,
    step: int,
    network_skeleton: eqx.Module,
    artifact_subdir: str = "model_checkpoints",
) -> eqx.Module:
    """Load a network from a previous MLflow run.

    The skeleton must have an identical PyTree structure to the saved
    network. Build it with construct_network(config.network, seed=config.seed).

    Parameters
    ----------
    run_id : str
        The MLflow run ID containing the checkpoint.
    step : int
        The training step number of the checkpoint to load.
    network_skeleton : eqx.Module
        An empty network with matching PyTree structure.
    artifact_subdir : str, optional
        Subdirectory where artifacts were saved. Default is
        "model_checkpoints".

    Returns
    -------
    eqx.Module
        The restored network with loaded weights.
    """
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(
        run_id, f"{artifact_subdir}/network_step_{step}.eqx"
    )
    with open(local_path, "rb") as f:
        data = f.read()

    restored = _deserialise_from_bytes(data, network_skeleton)

    # Debug: confirm trainable activation coefficients were restored
    if hasattr(restored, "activation_list"):
        for i, act in enumerate(restored.activation_list):
            if hasattr(act, "coeff"):
                logging.debug(
                    f"Restored activation_list[{i}].coeff = {float(act.coeff):.6f}"
                )

    return restored


def load_optimiser_states_from_mlflow(
    run_id: str,
    step: int,
    skeleton_states: List[eqx.Module],
    artifact_subdir: str = "model_checkpoints",
) -> List[eqx.Module]:
    """Load optimiser states from a previous MLflow run.

    The skeleton_states must have matching PyTree structure. Create via
    [opt.init(eqx.filter(network, eqx.is_array)) for opt in optimiser_list].

    Parameters
    ----------
    run_id : str
        The MLflow run ID containing the checkpoint.
    step : int
        The training step number of the checkpoint to load.
    skeleton_states : List[eqx.Module]
        Empty optimiser state objects with matching PyTree structure.
    artifact_subdir : str, optional
        Subdirectory where artifacts were saved. Default is
        "model_checkpoints".

    Returns
    -------
    List[eqx.Module]
        The restored optimiser states with loaded data.
    """
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(
        run_id, f"{artifact_subdir}/optim_step_{step}.eqx"
    )
    with open(local_path, "rb") as f:
        data = f.read()

    container_skeleton = {"states": skeleton_states}
    restored = _deserialise_from_bytes(data, container_skeleton)
    return restored["states"]


# -----------------------------------------------------------
# Data structures returned to the simulation
# -----------------------------------------------------------


@dataclass
class ResumeInfo:
    """Data structure for resuming an interrupted simulation run.

    Attributes
    ----------
    run_id : str
        The MLflow run ID to resume.
    latest_step : int
        The last completed training step.
    epoch_counter : int
        The epoch counter value to restore.
    title : str
        The reconstructed run title.
    """

    run_id: str
    latest_step: int
    epoch_counter: int
    title: str  # reconstructed from the old run name


# -----------------------------------------------------------
# Diagnostics payload — plain data, no model references
# -----------------------------------------------------------


@dataclass
class DiagnosticsPayload:
    """Pre-collected diagnostics data for logging.

    Built by the simulation so the tracker never touches the model.

    Attributes
    ----------
    sparsity_rows : List[Dict[str, object]], optional
        Rows of sparsity data to log as a table.
    activation_coefficients : Dict[int, float], optional
        Activation coefficient values keyed by index.
    extra_attribute_rows : List[Dict[str, object]], optional
        Additional attribute rows to log as a table.
    """

    sparsity_rows: Optional[List[Dict[str, object]]] = None
    activation_coefficients: Optional[Dict[int, float]] = None
    extra_attribute_rows: Optional[List[Dict[str, object]]] = None


# -----------------------------------------------------------
# Tracker
# -----------------------------------------------------------


class MLflowTracker:
    """Manages the MLflow lifecycle for a single simulation run.

    Parameters
    ----------
    config : object
        Configuration object containing simulation parameters.
    title : str
        The title of the simulation run.
    n_increments : int
        The number of training increments.
    """

    def __init__(self, config: object, title: str, n_increments: int) -> None:
        """Initialise the tracker with configuration and run metadata."""
        self._config = config
        self._title = title
        self._n_increments = n_increments
        self._run: Optional[mlflow.ActiveRun] = None
        self._epoch_counter: int = 0

    # -----------------------------------------------------------
    # Properties exposed to the simulation
    # -----------------------------------------------------------

    @property
    def epoch_counter(self) -> int:
        """Return the current epoch counter value."""
        return self._epoch_counter

    @property
    def run_id(self) -> Optional[str]:
        """Return the active MLflow run ID, or None if no run is active."""
        if self._run is not None:
            return self._run.info.run_id
        return None

    # -----------------------------------------------------------
    # Setup / teardown
    # -----------------------------------------------------------

    def setup(self) -> Optional[ResumeInfo]:
        """Initialise MLflow tracking for the simulation.

        Checks for incomplete runs and returns resume information if
        one is found. Otherwise starts a fresh run.

        Returns
        -------
        ResumeInfo or None
            ResumeInfo if an incomplete run was found and should be
            resumed. None if a fresh run was started.
        """
        tracking_uri, artifact_root = self._resolve_paths()
        mlflow.set_tracking_uri(tracking_uri)
        os.makedirs(artifact_root, exist_ok=True)

        experiment_name, run_name = self._derive_names()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=f"file://{artifact_root}",
            )
        mlflow.set_experiment(experiment_name)

        # --- guard against duplicate finished runs -----------------------
        if experiment is not None:
            self._guard_duplicate_finished(experiment)

        # --- resume check ------------------------------------------------
        resume = self._try_resume(experiment_name)
        if resume is not None:
            return resume

        # --- fresh run ---------------------------------------------------
        self._run = mlflow.start_run(run_name=run_name, log_system_metrics=True)
        self._epoch_counter = 0
        self._log_initial_params()
        logging.info("MLflow tracking initialised.")
        return None

    def teardown(self) -> None:
        """End the active MLflow run. Safe to call multiple times."""
        if self._run is not None:
            mlflow.end_run()
            self._run = None
            logging.info("MLflow run ended.")

    # -----------------------------------------------------------
    # Per-step logging
    # -----------------------------------------------------------

    def log_step(
        self,
        step: int,
        loss_history: List[object],
        aux_history: List[object],
        network: eqx.Module,
        optimiser_states: Optional[List[eqx.Module]] = None,
        diagnostics: Optional[DiagnosticsPayload] = None,
    ) -> None:
        """Log all data for one training increment.

        Parameters
        ----------
        step : int
            The current training step number.
        loss_history : List[object]
            History of loss values for this increment.
        aux_history : List[object]
            History of auxiliary metric values.
        network : eqx.Module
            The neural network to checkpoint.
        optimiser_states : List[eqx.Module], optional
            Optimiser states to checkpoint.
        diagnostics : DiagnosticsPayload, optional
            Additional diagnostic data to log.
        """
        self._log_metrics(loss_history, aux_history, step)
        self._log_diagnostics(diagnostics, step)
        log_network_to_mlflow(
            network=network,
            step=step,
            optimiser_states=optimiser_states,
        )

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active MLflow run.

        Parameters
        ----------
        key : str
            The tag key.
        value : str
            The tag value.
        """
        mlflow.set_tag(key, value)

    # -----------------------------------------------------------
    # Resume helpers (return plain data, never mutate sim)
    # -----------------------------------------------------------

    def get_network_loader(
        self, run_id: str, step: int
    ) -> Callable[[eqx.Module], eqx.Module]:
        """Return a callable that loads a network from a checkpoint.

        Parameters
        ----------
        run_id : str
            The MLflow run ID containing the checkpoint.
        step : int
            The training step of the checkpoint.

        Returns
        -------
        Callable[[eqx.Module], eqx.Module]
            A function that takes a network skeleton and returns
            the restored network.
        """

        def _load(skeleton: eqx.Module) -> eqx.Module:
            return load_network_from_mlflow(run_id, step, skeleton)

        return _load

    def get_optimiser_loader(
        self, run_id: str, step: int
    ) -> Callable[[List[eqx.Module]], List[eqx.Module]]:
        """Return a callable that loads optimiser states from a checkpoint.

        Parameters
        ----------
        run_id : str
            The MLflow run ID containing the checkpoint.
        step : int
            The training step of the checkpoint.

        Returns
        -------
        Callable[[List[eqx.Module]], List[eqx.Module]]
            A function that takes skeleton states and returns
            the restored optimiser states.
        """

        def _load(skeleton_states: List[eqx.Module]) -> List[eqx.Module]:
            return load_optimiser_states_from_mlflow(run_id, step, skeleton_states)

        return _load

    # -----------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------

    def _resolve_paths(self) -> Tuple[str, str]:
        """Resolve MLflow tracking URI and artifact root path.

        Returns
        -------
        Tuple[str, str]
            A tuple of (tracking_uri, artifact_root).
        """
        # Get repo root
        repo_root = os.path.dirname(os.path.abspath(os.getcwd()))
        # Mlflow directory
        mlflow_dir = os.path.join(repo_root, "output", "mlflow")
        os.makedirs(mlflow_dir, exist_ok=True)
        # Database path
        db_path = os.path.join(mlflow_dir, "mlflow.db")
        # Artifacts directory
        artifact_root = os.path.join(mlflow_dir, "artifacts")
        return f"sqlite:///{db_path}", artifact_root

    def _derive_names(self) -> Tuple[str, str]:
        """Derive experiment and run names from the title.

        Returns
        -------
        Tuple[str, str]
            A tuple of (experiment_name, run_name).
        """
        # Extract suffix
        match = re.search(r"(_S\d+_.+)$", self._title)
        if match:
            # Strip leading underscore
            suffix = match.group(1).lstrip("_")
            # Extract prefix
            prefix = self._title[3 : match.start()]
            return prefix, suffix
        return self._title, self._title

    # -----------------------------------------------------------
    # Duplicate / resume guards
    # -----------------------------------------------------------

    def _guard_duplicate_finished(self, experiment: object) -> None:
        """Check for existing finished runs with the same seed.

        Exits if a finished run is found and timestamp is disabled.

        Parameters
        ----------
        experiment : object
            The MLflow experiment object.
        """
        finished = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                # Seed filter
                f"params.seed = '{self._config.seed}' "
                # Status filter
                f"AND attributes.status = 'FINISHED'"
            ),
        )
        if len(finished) > 0 and not self._config.output_parameters.timestamp:
            logging.info(
                f"Found existing finished run with seed {self._config.seed} "
                "and no timestamp — exiting to avoid overwriting."
            )
            exit()

    def _try_resume(self, experiment_name: str) -> Optional[ResumeInfo]:
        """Attempt to resume an incomplete MLflow run.

        Searches for running runs with matching seed and valid checkpoints.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment to search.

        Returns
        -------
        ResumeInfo or None
            ResumeInfo if a resumable run is found. None otherwise.
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        run_id = self._find_incomplete_run(experiment)
        if run_id is None:
            return None

        latest_step = self._latest_checkpoint_step(run_id)
        # Subtract pretraining
        n_total = self._n_increments - (
            1 if self._config.pretraining_parameters.enabled else 0
        )
        if latest_step is None or latest_step >= n_total:
            logging.info(
                "[Resume] Incomplete run found but no usable checkpoints — "
                "starting fresh."
            )
            return None

        # Re-open the interrupted run
        self._run = mlflow.start_run(run_id=run_id, log_system_metrics=True)
        # Mark as resumed
        mlflow.set_tag("resumed", "true")
        # Track resume point
        mlflow.set_tag("resumed_from_step", str(latest_step))

        # Restore epoch counter from metric history
        # Restore counter
        epoch_counter = self._restore_epoch_counter(run_id)
        self._epoch_counter = epoch_counter

        # Reconstruct original title
        client = mlflow.tracking.MlflowClient()
        old_run_name = client.get_run(run_id).info.run_name
        # Derive name components
        stem, _ = self._derive_names()
        title = "nn_" + stem + "_" + old_run_name

        logging.info(
            f"[Resume] Attaching to run {run_id} (last checkpoint: step {latest_step})."
        )

        return ResumeInfo(
            run_id=run_id,
            latest_step=latest_step,
            epoch_counter=epoch_counter,
            title=title,
        )

    def _find_incomplete_run(self, experiment: object) -> Optional[str]:
        """Find an incomplete (running) run with the matching seed.

        Parameters
        ----------
        experiment : object
            The MLflow experiment object.

        Returns
        -------
        str or None
            The run ID of the most recent incomplete run, or None.
        """
        # Convert seed to string
        seed_str = str(self._config.seed)
        try:
            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    # Seed filter
                    f"params.seed = '{seed_str}' "
                    # Status filter
                    f"AND attributes.status = 'RUNNING'"
                ),
            )
        except Exception as e:
            # Handle error
            logging.warning(f"[Resume] MLflow run search failed: {e}")
            return None

        if runs_df.empty:
            return None

        runs_df = runs_df.sort_values("start_time", ascending=False)
        # Get first run
        run_id = str(runs_df.iloc[0]["run_id"])
        logging.info(f"[Resume] Found incomplete run: {run_id}")
        return run_id

    def _latest_checkpoint_step(self, run_id: str) -> Optional[int]:
        """Find the latest checkpoint step for a given run.

        Parameters
        ----------
        run_id : str
            The MLflow run ID.

        Returns
        -------
        int or None
            The step number of the latest checkpoint, or None.
        """
        client = mlflow.tracking.MlflowClient()
        try:
            # List checkpoints
            artifacts = client.list_artifacts(run_id, path="model_checkpoints")
        except Exception as e:
            logging.warning(f"[Resume] Could not list artifacts for run {run_id}: {e}")
            return None

        steps = []
        for art in artifacts:
            # Extract step number
            m = re.match(r".*/network_step_(-?\d+)\.eqx$", art.path)
            if m:
                steps.append(int(m.group(1)))

        if not steps:
            return None
        return max(steps)

    def _restore_epoch_counter(self, run_id: str) -> int:
        """Restore the epoch counter from a run's metric history.

        Parameters
        ----------
        run_id : str
            The MLflow run ID.

        Returns
        -------
        int
            The restored epoch counter value, or 0 if not found.
        """
        client = mlflow.tracking.MlflowClient()
        try:
            # Get loss history
            history = client.get_metric_history(run_id, "loss/total")
            if history:
                return max(m.step for m in history) + 1
        except Exception as e:
            logging.warning(f"[Resume] Could not restore epoch counter: {e}")
        return 0

    # -----------------------------------------------------------
    # Initial params
    # -----------------------------------------------------------

    @staticmethod
    def _flatten_config(d: Dict[str, object], prefix: str = "") -> Dict[str, str]:
        """Recursively flatten a nested dict into dot-separated keys.

        Lists and tuples are stored as their repr. Values are truncated
        to 500 characters to comply with MLflow limits.

        Parameters
        ----------
        d : Dict[str, object]
            The nested dictionary to flatten.
        prefix : str, optional
            The prefix for keys. Default is empty string.

        Returns
        -------
        Dict[str, str]
            The flattened dictionary with string keys and values.
        """
        flat = {}
        for key, val in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                flat.update(MLflowTracker._flatten_config(val, full_key))
            else:
                s = str(val)
                if len(s) > 500:
                    s = s[:497] + "..."
                flat[full_key] = s
        return flat

    def _log_initial_params(self) -> None:
        """Log all configuration parameters at the start of the run."""
        flat = self._flatten_config(dict(self._config))
        # MLflow log_params has a 100-param-per-call limit
        items = list(flat.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i : i + 100]))

    # -----------------------------------------------------------
    # Logging
    # -----------------------------------------------------------

    def _log_metrics(
        self, loss_history: List[object], aux_history: List[object], step: int
    ) -> None:
        """Log loss and auxiliary metrics for one training increment.

        Parameters
        ----------
        loss_history : List[object]
            History of loss values per epoch.
        aux_history : List[object]
            History of auxiliary metric values.
        step : int
            The current training step number.
        """
        loss_term_keys = [
            k for k, v in self._config.training_parameters.loss_terms.items() if v
        ]
        step_label = f"increment_{step}"

        # --- Per-epoch loss: global + per-increment curves ---
        for epoch_idx, loss_val in enumerate(loss_history):
            global_metrics = {}
            local_metrics = {}

            if isinstance(loss_val, dict):
                for key, val in loss_val.items():
                    fval = float(val)
                    global_metrics[f"loss/{key}"] = fval
                    local_metrics[f"loss/{step_label}/{key}"] = fval
                total = float(sum(loss_val.values()))
                global_metrics["loss/total"] = total
                local_metrics[f"loss/{step_label}/total"] = total
            else:
                fval = float(loss_val)
                global_metrics["loss/total"] = fval
                local_metrics[f"loss/{step_label}/total"] = fval

            mlflow.log_metrics(global_metrics, step=self._epoch_counter)
            mlflow.log_metrics(local_metrics, step=epoch_idx)
            self._epoch_counter += 1

        # --- Per-epoch aux: global + per-increment curves ---
        for epoch_idx, aux_val in enumerate(aux_history):
            global_step = self._epoch_counter - len(loss_history) + epoch_idx
            global_metrics = {}
            local_metrics = {}

            if isinstance(aux_val, (list, tuple)):
                for i, val in enumerate(aux_val):
                    key = loss_term_keys[i] if i < len(loss_term_keys) else f"term_{i}"
                    fval = float(val)
                    global_metrics[f"aux/{key}"] = fval
                    local_metrics[f"aux/{step_label}/{key}"] = fval
            elif isinstance(aux_val, dict):
                for key, val in aux_val.items():
                    fval = float(val)
                    global_metrics[f"aux/{key}"] = fval
                    local_metrics[f"aux/{step_label}/{key}"] = fval
            else:
                fval = float(aux_val)
                global_metrics["aux/value"] = fval
                local_metrics[f"aux/{step_label}/value"] = fval

            mlflow.log_metrics(global_metrics, step=global_step)
            mlflow.log_metrics(local_metrics, step=epoch_idx)

        # --- Per-increment final aux (energy evolution) ---
        if aux_history:
            final_aux = aux_history[-1]
            summary = {}
            if isinstance(final_aux, (list, tuple)):
                for i, val in enumerate(final_aux):
                    key = loss_term_keys[i] if i < len(loss_term_keys) else f"term_{i}"
                    summary[f"final_energy/{key}"] = float(val)
            elif isinstance(final_aux, dict):
                for key, val in final_aux.items():
                    summary[f"final_energy/{key}"] = float(val)
            mlflow.log_metrics(summary, step=max(step, 0))

    def _log_diagnostics(
        self, payload: Optional[DiagnosticsPayload], step: int
    ) -> None:
        """Log diagnostic data for one training increment.

        Parameters
        ----------
        payload : DiagnosticsPayload, optional
            The diagnostic data to log.
        step : int
            The current training step number.
        """
        if payload is None:
            return

        if payload.sparsity_rows:
            mlflow.log_table(
                pd.DataFrame(payload.sparsity_rows),
                artifact_file="diagnostics/sparsity.json",
            )

        if payload.activation_coefficients:
            for i, val in payload.activation_coefficients.items():
                mlflow.log_metric(f"activation_coeff/{i}", val, step=step)

        if payload.extra_attribute_rows:
            mlflow.log_table(
                pd.DataFrame(payload.extra_attribute_rows),
                artifact_file="diagnostics/extra_attributes.json",
            )
