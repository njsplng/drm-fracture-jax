"""Network training utilities for JAX-based neural networks."""

import logging
from typing import Callable, Dict, List, Literal, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from jaxtyping import Array
from tqdm import tqdm
from utils_nn import (
    output_training_snapshot,
    predict_model_output,
    update_displacement_increment,
    update_previous_phasefield,
)

from data_handling import MeshData
from log import clear_prefix, set_prefix
from utils import timed_filter_jit


class EarlyStopping:
    """Early stopping callback for neural network training.

    Monitors training loss and stops training when improvement
    falls below a threshold for a specified number of epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping.
    relative_threshold : float, optional
        Relative improvement threshold. Default is 1e-4.
    absolute_threshold : float, optional
        Absolute improvement threshold. Default is 1e-6.
    mode : {"relative", "absolute"}, optional
        Comparison mode. Default is "relative".
    """

    def __init__(
        self,
        patience: int,
        relative_threshold: float = 1e-4,
        absolute_threshold: float = 1e-6,
        mode: Literal["relative", "absolute"] = "relative",
    ) -> None:
        """Initialize early stopping callback."""
        self.patience: int = patience
        self.relative_threshold: float = relative_threshold
        self.absolute_threshold: float = absolute_threshold
        self.best_loss: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.eps: float = 1e-12
        self.to_burn: int = 1
        self.mode: Literal["relative", "absolute"] = mode

        if self.mode == "relative":
            self.update = self._relative_update
        elif self.mode == "absolute":
            self.update = self._absolute_update

    def _absolute_update(self, current_loss: float) -> bool:
        """Update early stopping state using absolute improvement threshold.

        Parameters
        ----------
        current_loss : float
            Current epoch loss value.

        Returns
        -------
        bool
            True if training should stop early, False otherwise.
        """
        if self.best_loss is None:
            if self.to_burn > 0:
                self.to_burn -= 1
                return False
            self.best_loss = current_loss
            self.counter = 0
            return False

        abs_impr = self.best_loss - current_loss
        if abs_impr > self.absolute_threshold:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _relative_improvement(self, best: float, current: float) -> float:
        """Calculate relative improvement between best and current loss.

        Parameters
        ----------
        best : float
            Best loss value seen so far.
        current : float
            Current loss value.

        Returns
        -------
        float
            Relative improvement (positive if current is better).
        """
        improvement = best - current
        denom = (abs(best) + abs(current)) * 0.5
        denom = denom if denom > self.eps else self.eps
        return improvement / denom

    def _relative_update(self, current_loss: float) -> bool:
        """Update early stopping state using relative improvement threshold.

        Parameters
        ----------
        current_loss : float
            Current epoch loss value.

        Returns
        -------
        bool
            True if training should stop early, False otherwise.
        """
        if self.best_loss is None:
            if self.to_burn > 0:
                self.to_burn -= 1
                return False
            self.best_loss = current_loss
            self.counter = 0
            return False

        rel_impr = self._relative_improvement(self.best_loss, current_loss)

        if rel_impr > self.relative_threshold:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def reset(self) -> None:
        """Reset early stopping state to initial values."""
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.to_burn = 1


class TrainState(eqx.Module):
    """Container for model parameters and optimizer state.

    Parameters
    ----------
    params : eqx.Module
        Trainable network parameters.
    static : eqx.Module
        Non-trainable static components.
    opt_state : optax.OptState
        Optimizer state from Optax.
    """

    params: eqx.Module
    static: eqx.Module
    opt_state: optax.OptState


@timed_filter_jit
def compute_loss(
    params: eqx.Module,
    static: eqx.Module,
    data: Array,
    weight_decay: float = 0,
    energy_scale: str = "linear",
    loss_terms: Optional[Dict] = None,
    model_kwargs: Optional[Dict] = None,
) -> Tuple[float, Tuple[Array, ...]]:
    """Compute the total loss and auxiliary outputs for a forward pass.

    Evaluate elastic energy, damage energy, irreversibility energy,
    and weight decay terms according to the specified loss terms.

    Parameters
    ----------
    params : eqx.Module
        Trainable network parameters.
    static : eqx.Module
        Non-trainable static components.
    data : Array
        Input data (nodal coordinates).
    weight_decay : float, optional
        Weight decay coefficient. Default is 0.
    energy_scale : {"linear", "log"}, optional
        Loss scaling mode. Default is "linear".
    loss_terms : dict, optional
        Dictionary specifying which loss terms to include.
    model_kwargs : dict, optional
        Additional keyword arguments for the model.

    Returns
    -------
    Tuple[float, Tuple[Array, ...]]
        Total loss value and tuple of auxiliary outputs.
    """
    model = eqx.tree_at(
        lambda m: m.network, static, params, is_leaf=lambda x: x is None
    )

    u, v, c = model(data, **(model_kwargs or {}))

    E_elastic, E_damage, E_irreversibility = model.loss_energy((u, v, c))

    loss = 0.0
    return_list = []

    if loss_terms["energy_elastic"]:
        loss += E_elastic
        return_list.append(E_elastic)

    if loss_terms["energy_phasefield"]:
        loss += E_damage
        return_list.append(E_damage)

    if loss_terms["energy_irreversibility"]:
        loss += E_irreversibility
        return_list.append(E_irreversibility)

    if energy_scale == "log":
        loss = jnp.log10(loss)

    if loss_terms["weight_decay"]:
        loss_weight_decay = model.loss_weight_decay(weight_decay)
        loss += loss_weight_decay
        return_list.append(loss_weight_decay)

    return loss, tuple(return_list)


def make_update_step(
    optimiser: optax.GradientTransformation,
    optimiser_name: str,
) -> Callable:
    """Create an update step function for the training process.

    Returns a JIT-compiled function that computes gradients and
    updates model parameters using the specified optimizer.

    Parameters
    ----------
    optimiser : optax.GradientTransformation
        Optax optimizer instance.
    optimiser_name : str
        Name of the optimizer (used for special handling).

    Returns
    -------
    Callable
        Update step function accepting train_state and data.
    """

    @timed_filter_jit
    def update_step(
        train_state: TrainState,
        data: Array,
        *,
        weight_decay: float = 0.0,
        energy_scale: str = "linear",
        loss_terms: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[TrainState, float, Tuple[Array, ...]]:
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            train_state.params,
            train_state.static,
            data,
            weight_decay=weight_decay,
            energy_scale=energy_scale,
            loss_terms=loss_terms,
            model_kwargs=model_kwargs,
        )

        prefix = optimiser_name.split("_")[0]
        if prefix in ["lbfgs", "soap"]:

            def f(p: eqx.Module) -> float:
                return compute_loss(
                    params=p,
                    static=train_state.static,
                    data=data,
                    weight_decay=weight_decay,
                    energy_scale=energy_scale,
                    loss_terms=loss_terms,
                    model_kwargs=model_kwargs,
                )[0]

            updates, new_opt_state = optimiser.update(
                grads,
                train_state.opt_state,
                train_state.params,
                grad=grads,
                value=loss,
                value_fn=f,
            )

        elif prefix.startswith("ss"):

            def fn_for_optimistix(p: Callable, args_: Dict) -> float:
                return compute_loss(
                    params=p,
                    static=args_["static"],
                    data=args_["data"],
                    weight_decay=args_["weight_decay"],
                    energy_scale=args_["energy_scale"],
                    loss_terms=args_["loss_terms"],
                    model_kwargs=args_["model_kwargs"],
                )[0]

            args = dict(
                __fn__=fn_for_optimistix,
                static=train_state.static,
                data=data,
                weight_decay=weight_decay,
                energy_scale=energy_scale,
                loss_terms=loss_terms,
                model_kwargs=model_kwargs,
            )

            updates, new_opt_state = optimiser.update(
                grads,
                train_state.opt_state,
                train_state.params,
                args=args,
            )

        else:
            updates, new_opt_state = optimiser.update(
                grads, train_state.opt_state, train_state.params
            )

        new_params = eqx.apply_updates(train_state.params, updates)
        return (
            TrainState(
                params=new_params, static=train_state.static, opt_state=new_opt_state
            ),
            loss,
            aux,
        )

    return update_step


def train_model(
    iterator: range,
    update_step: Callable,
    train_state: TrainState,
    nodal_coordinates: Array,
    input_dict: Dict,
    loss_terms: Dict,
    early_stop: Optional[EarlyStopping] = None,
    silent: bool = True,
    loss_history: Optional[List[float]] = None,
    aux_history: Optional[List[Tuple[Array, ...]]] = None,
    best_loss: Optional[float] = None,
    best_model: Optional[TrainState] = None,
    track_training_output: bool = False,
    training_output_frequency: int = 50,
    model_kwargs: Optional[Dict] = None,
) -> Tuple[
    List[float],
    List[Tuple[Array, ...]],
    eqx.Module,
    EarlyStopping,
    bool,
    float,
    Dict,
    optax.OptState,
]:
    """Train the model for the specified number of epochs.

    Performs gradient descent updates, tracks loss history, monitors
    for divergence, and optionally applies early stopping.

    Parameters
    ----------
    iterator : range
        Epoch iterator.
    update_step : Callable
        Function performing one training step.
    train_state : TrainState
        Current model and optimizer state.
    nodal_coordinates : Array
        Input nodal coordinates.
    input_dict : dict
        Dictionary of input parameters.
    loss_terms : dict
        Dictionary specifying loss terms.
    early_stop : EarlyStopping, optional
        Early stopping callback.
    silent : bool, optional
        Suppress progress bar if True. Default is True.
    loss_history : list, optional
        Existing loss history to append to.
    aux_history : list, optional
        Existing auxiliary outputs history.
    best_loss : float, optional
        Best loss seen so far.
    best_model : TrainState, optional
        Best model state seen so far.
    track_training_output : bool, optional
        Track model outputs during training. Default is False.
    training_output_frequency : int, optional
        Frequency for saving outputs. Default is 50.
    model_kwargs : dict, optional
        Additional keyword arguments for the model.

    Returns
    -------
    Tuple[List[float], List[Tuple], Module, EarlyStopping, bool, float, Dict]
        Loss history, aux history, best model, early stop state,
        divergence flag, best loss, training output, optimizer state.
    """
    if loss_history is None:
        loss_history = []
    if aux_history is None:
        aux_history = []
    if best_loss is None:
        best_loss = float("inf")
    if best_model is None:
        best_model = None
    training_output = {"displacement": [], "phasefield": []}
    diverged = False
    best_optimiser_state = None

    for epoch in iterator:
        train_state, loss, aux = update_step(
            train_state=train_state,
            data=nodal_coordinates,
            weight_decay=input_dict["training_parameters"]["weight_decay"],
            energy_scale=input_dict["model_energy_scale"],
            loss_terms=FrozenDict(loss_terms),
            model_kwargs=model_kwargs,
        )

        if jnp.isnan(loss):
            logging.error("Loss is nan, stopping execution.")
            diverged = True
            print("Loss is nan, terminating execution.")
            break

        loss_history.append(loss)
        aux_history.append(aux)

        if best_optimiser_state is None:
            best_optimiser_state = train_state.opt_state

        if loss < best_loss and epoch > 0:
            best_loss = loss
            full_model = eqx.tree_at(
                lambda m: m.network,
                train_state.static,
                train_state.params,
                is_leaf=lambda x: x is None,
            )
            best_model = full_model
            best_optimiser_state = train_state.opt_state

        if (epoch + 1) % input_dict["training_parameters"][
            "loss_reporting_frequency"
        ] == 0:
            if not silent:
                iterator.set_postfix(
                    {"loss": f"{loss:.4e}, best_loss: {best_loss:.4e}"}
                )
            logging.info(
                f"Epoch: {epoch + 1}, Loss: {loss:.4e}, Best loss: {best_loss:.4e}"
            )

        if track_training_output and (epoch + 1) % training_output_frequency == 0:
            full_model = eqx.tree_at(
                lambda m: m.network,
                train_state.static,
                train_state.params,
                is_leaf=lambda x: x is None,
            )
            dfield, pfield = predict_model_output(
                model=full_model, inp=nodal_coordinates
            )
            training_output["displacement"].append(dfield)
            training_output["phasefield"].append(pfield)
            logging.info(f"Training output appended at epoch {epoch + 1}")

        if (
            early_stop is not None
            and early_stop.update(float(loss))
            and input_dict["training_parameters"]["early_stopping"]["enabled"]
        ):
            logging.info(
                f"Epoch: {epoch + 1}, Loss: {loss:.4e}, Best loss: {best_loss:.4e}"
            )
            logging.info(f"Early stopping triggered at epoch {epoch}")
            early_stop.reset()
            full_model = eqx.tree_at(
                lambda m: m.network,
                train_state.static,
                train_state.params,
                is_leaf=lambda x: x is None,
            )
            best_model = full_model
            break

    return (
        loss_history,
        aux_history,
        best_model,
        early_stop,
        diverged,
        best_loss,
        training_output,
        best_optimiser_state,
    )


def training_step(
    model: eqx.Module,
    optimiser_list: List[optax.GradientTransformation],
    optimiser_name_list: List[str],
    input_dict: Dict,
    title: str,
    silent: bool,
    displacement_increment: float,
    increment_index: int,
    mesh_data: MeshData,
    number_of_epochs_list: List[int],
    loss_terms: Dict,
    early_stop: Optional[EarlyStopping],
    previous_phasefield: Array,
    tqdm_training_string: str = "Training",
    track_training_output: bool = False,
    training_output_frequency: int = 50,
    transferred_optimiser_list: Optional[List[optax.GradientTransformation]] = None,
    transfer_optimiser_states: bool = False,
    model_kwargs: Optional[Dict] = None,
) -> Tuple[
    Optional[eqx.Module],
    Optional[List[optax.OptState]],
    Tuple[Optional[List[float]], Optional[List[Tuple[Array, ...]]]],
]:
    """Execute a training step with optional multi-phase optimization.

    Updates displacement and phasefield state, then trains the model
    using one or more optimizers in sequence (warmup pattern).

    Parameters
    ----------
    model : eqx.Module
        Neural network model.
    optimiser_list : list
        List of Optax optimizers.
    optimiser_name_list : list
        List of optimizer names.
    input_dict : dict
        Dictionary of input parameters.
    title : str
        Title for output files.
    silent : bool
        Suppress progress bars if True.
    displacement_increment : float
        Displacement increment for this step.
    increment_index : int
        Current increment index.
    mesh_data : MeshData
        Mesh and boundary condition data.
    number_of_epochs_list : list
        Epochs for each optimizer phase.
    loss_terms : dict
        Loss term configuration.
    early_stop : EarlyStopping, optional
        Early stopping callback.
    previous_phasefield : Array
        Phasefield from previous increment.
    tqdm_training_string : str, optional
        Progress bar description. Default is "Training".
    track_training_output : bool, optional
        Track outputs during training. Default is False.
    training_output_frequency : int, optional
        Output tracking frequency. Default is 50.
    transferred_optimiser_list : list, optional
        Optimizer states to transfer.
    transfer_optimiser_states : bool, optional
        Transfer optimizer states between phases.
    model_kwargs : dict, optional
        Additional model keyword arguments.

    Returns
    -------
    Tuple[Module, List[OptState], Tuple[List, List]]
        Trained model, optimizer states, and (loss_history, aux_history).
    """
    assert len(optimiser_list) == len(
        optimiser_name_list
    ), "Optimiser and optimiser_name lists must have the same length."

    model = update_displacement_increment(
        model=model, displacement_increment=displacement_increment
    )
    model = update_previous_phasefield(
        model=model, previous_phasefield=previous_phasefield
    )

    loss_history, aux_history, best_loss = None, None, None
    training_output = {"displacement": [], "phasefield": []}
    optimiser_states = []

    for i, (optimiser, optimiser_name, number_of_epochs) in enumerate(
        zip(optimiser_list, optimiser_name_list, number_of_epochs_list)
    ):
        set_prefix(optimiser_name.split("_")[0])
        logging.info("Starting training step")

        update_step = make_update_step(
            optimiser=optimiser, optimiser_name=optimiser_name
        )

        params = model.network
        static = eqx.tree_at(
            lambda m: m.network, model, None, is_leaf=lambda x: x is None
        )

        opt_state = optimiser.init(params)
        if (
            transferred_optimiser_list is not None
            and len(transferred_optimiser_list) >= i + 1
        ):
            opt_state = transferred_optimiser_list[i]

        train_state = TrainState(params=params, static=static, opt_state=opt_state)

        iterator = range(number_of_epochs)
        if not silent:
            iterator = tqdm(iterator)
            iterator.set_description(
                f"({optimiser_name.split('_')[0]}) {tqdm_training_string}"
            )

        if early_stop:
            early_stop.reset()

        (
            loss_history,
            aux_history,
            model,
            early_stop,
            diverged,
            best_loss,
            training_step_output,
            optimiser_state,
        ) = train_model(
            iterator=iterator,
            update_step=update_step,
            train_state=train_state,
            nodal_coordinates=mesh_data.nodal_coordinates,
            input_dict=input_dict,
            early_stop=early_stop,
            silent=silent,
            loss_history=loss_history,
            aux_history=aux_history,
            best_loss=best_loss,
            best_model=model,
            track_training_output=track_training_output,
            training_output_frequency=training_output_frequency,
            loss_terms=loss_terms,
            model_kwargs=model_kwargs,
        )

        if track_training_output:
            training_output["displacement"].extend(training_step_output["displacement"])
            training_output["phasefield"].extend(training_step_output["phasefield"])

        if diverged:
            logging.error("Training diverged, stopping execution.")
            return (None, None, (None, None))

        clear_prefix()

        if transfer_optimiser_states:
            optimiser_states.append(optimiser_state)

    if track_training_output:
        output_training_snapshot(
            output_dict=training_output,
            filename=title,
            increment_index=increment_index,
        )

    return model, optimiser_states, (loss_history, aux_history)
