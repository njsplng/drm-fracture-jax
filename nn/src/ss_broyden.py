"""Self-scaled Broyden-family optimization solver implementation."""

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array


# -----------------------------------------------------------
# Implement new optimisers
# -----------------------------------------------------------
@dataclass
class _State(eqx.Module):
    """State for the SSBroyden family of optimisers.

    Parameters
    ----------
    y_flat : Array
        Flattened parameter vector.
    f : Array
        Current function value.
    g_flat : Array
        Flattened gradient vector.
    backend_state : object
        Internal state for the inverse Hessian approximation.
    iter : Array
        Current iteration count.
    result : optx.RESULTS
        Termination status.
    unravel : Callable[[Array], object]
        Function to unflatten parameters back to pytree.
    g0_norm : Array
        Initial gradient norm for convergence testing.
    """

    y_flat: Array
    f: Array
    g_flat: Array
    backend_state: object
    iter: Array
    result: optx.RESULTS
    unravel: Callable[[Array], object]
    g0_norm: Array


@dataclass
class _LMState(eqx.Module):
    """State for the limited memory Broyden updates.

    Parameters
    ----------
    S : Array
        Ring buffer storing s_i vectors of shape [m, n].
    Y : Array
        Ring buffer storing y_i vectors (gradient differences) of shape [m, n].
    phi : Array
        Ring buffer storing phi scalars of shape [m].
    tau : Array
        Ring buffer storing tau scalars of shape [m].
    head : Array
        Next write position in ring buffer (jnp.int32 scalar).
    count : Array
        Number of valid entries in ring buffer (jnp.int32 scalar).
    n : int
        Dimension of parameter space (static).
    m : int
        Memory size / ring buffer capacity (static).
    gamma : Array
        Initial scaling factor for inverse Hessian approximation.
    """

    # ring buffers [m, n]
    S: Array
    Y: Array  # y_i (== g_{k+1} - g_k)
    # scalars [m]
    phi: Array
    tau: Array
    # ring meta (JAX scalars, not Python ints)
    head: Array  # jnp.int32 scalar (next write position)
    count: Array  # jnp.int32 scalar (#valid entries)
    # shape/meta
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    gamma: Array


# Consumed by the SSBroyden class so need to tell the linter to ignore these explicitly
# skylos: ignore-start
class _LMFuncs:
    @staticmethod
    def init_state(n: int, m: int, *, gamma: float = 1.0) -> _LMState:
        """Initialize the state for the limited memory Broyden updates.

        Parameters
        ----------
        n : int
            Dimension of parameter space.
        m : int
            Memory size / ring buffer capacity.
        gamma : float, optional
            Initial scaling factor. Default is 1.0.

        Returns
        -------
        _LMState
            Initialized limited memory state.
        """
        dtype = jnp.float64
        zeros_mn = jnp.zeros((m, n), dtype=dtype)
        zeros_m = jnp.zeros((m,), dtype=dtype)
        return _LMState(
            S=zeros_mn,
            Y=zeros_mn,
            phi=zeros_m,
            tau=zeros_m,
            head=jnp.array(0, dtype=jnp.int32),
            count=jnp.array(0, dtype=jnp.int32),
            n=int(n),
            m=int(m),
            gamma=jnp.asarray(gamma, dtype=dtype),
        )

    @staticmethod
    def apply_H(st: _LMState, x: Array) -> Array:
        """Compute r = H_k x using limited memory inverse Hessian.

        The computation involves:
            - base metric: gamma * I
            - rank-two self-scaled updates stored as (S_i, Y_i, phi_i, tau_i)
        Replay order is oldest -> newest.

        Parameters
        ----------
        st : _LMState
            Limited memory state containing update history.
        x : Array
            Vector to apply inverse Hessian to.

        Returns
        -------
        Array
            Result of applying inverse Hessian approximation to x.
        """
        # Extract sizes from state
        m, n = st.m, st.n

        def phys(i: Array) -> Array:
            """Map logical index to physical index in ring buffer."""
            return (st.head - st.count + i) % m

        # Phase A: build Hy[i], v[i], ys[i], yHy[i] with prefix replay
        # Temporary caches
        HYc = jnp.zeros((m, n), dtype=x.dtype)
        Vc = jnp.zeros((m, n), dtype=x.dtype)
        ysc = jnp.zeros((m,), dtype=x.dtype)
        yHyc = jnp.zeros((m,), dtype=x.dtype)

        def outer_cond(carry: tuple[Array, ...]) -> bool:
            """Outer loop condition: i < count."""
            i, *_ = carry
            return i < st.count

        def outer_body(
            carry: tuple[Array, Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array, Array]:
            """Outer loop body: compute Hy[i], v[i], ys[i], yHy[i] for each i."""
            # Extract
            i, HYc, Vc, ysc, yHyc = carry
            idx_i = phys(i)

            # Store current S_i, Y_i
            S_i = jax.lax.dynamic_index_in_dim(st.S, idx_i, axis=0, keepdims=False)
            Y_i = jax.lax.dynamic_index_in_dim(st.Y, idx_i, axis=0, keepdims=False)

            # Compute Hy_i via inner loop over j < i
            q0 = st.gamma * Y_i

            def inner_cond(inner_carry: tuple[Array, ...]) -> bool:
                """Inner loop condition: j < i."""
                j, *_ = inner_carry
                return j < i

            def inner_body(inner_carry: tuple[Array, Array]) -> tuple[Array, Array]:
                """Inner body: apply update j to vector q."""
                # Extract
                j, q = inner_carry
                idx_j = phys(j)

                # Extract cached values
                S_j = jax.lax.dynamic_index_in_dim(st.S, idx_j, axis=0, keepdims=False)
                Hy_j = jax.lax.dynamic_index_in_dim(HYc, idx_j, axis=0, keepdims=False)
                V_j = jax.lax.dynamic_index_in_dim(Vc, idx_j, axis=0, keepdims=False)
                ys_j = jax.lax.dynamic_index_in_dim(ysc, idx_j, axis=0, keepdims=False)
                yHy_j = jax.lax.dynamic_index_in_dim(
                    yHyc, idx_j, axis=0, keepdims=False
                )
                phi_j = jax.lax.dynamic_index_in_dim(
                    st.phi, idx_j, axis=0, keepdims=False
                )
                tau_j = jax.lax.dynamic_index_in_dim(
                    st.tau, idx_j, axis=0, keepdims=False
                )

                # Stabilize denominators
                ys_j = jnp.maximum(ys_j, 1e-32)
                yHy_j = jnp.maximum(yHy_j, 1e-32)
                tau_j = jnp.maximum(tau_j, 1e-12)

                # Apply update j to vector q
                t1 = jnp.dot(Hy_j, Y_i)
                t2 = jnp.dot(V_j, Y_i)
                t3 = jnp.dot(S_j, Y_i)
                q = q - (t1 / yHy_j) * Hy_j + (phi_j * yHy_j * t2) * V_j
                q = q / tau_j + (t3 / ys_j) * S_j

                return (j + jnp.array(1, jnp.int32), q)

            # Run inner loop to compute Hy_i
            _, Hy_i = jax.lax.while_loop(
                inner_cond, inner_body, (jnp.array(0, jnp.int32), q0)
            )

            # Compute v_i, ys_i, yHy_i
            ys_i = jnp.dot(Y_i, S_i)
            yHy_i = jnp.dot(Y_i, Hy_i)
            ys_i = jnp.maximum(ys_i, 1e-32)
            yHy_i = jnp.maximum(yHy_i, 1e-32)
            v_i = S_i / ys_i - Hy_i / yHy_i

            # Cache results
            HYc = HYc.at[idx_i].set(Hy_i)
            Vc = Vc.at[idx_i].set(v_i)
            ysc = ysc.at[idx_i].set(ys_i)
            yHyc = yHyc.at[idx_i].set(yHy_i)

            return (i + jnp.array(1, jnp.int32), HYc, Vc, ysc, yHyc)

        # Run outer loop to fill caches
        _, HYc, Vc, ysc, yHyc = jax.lax.while_loop(
            outer_cond,
            outer_body,
            (jnp.array(0, jnp.int32), HYc, Vc, ysc, yHyc),
        )

        # Phase B: apply all updates to x (oldest -> newest)
        r = st.gamma * x

        def apply_cond(carry: tuple[Array, Array]) -> bool:
            """Apply loop condition: i < count."""
            r, i = carry
            return i < st.count

        def apply_body(carry: tuple[Array, Array]) -> tuple[Array, Array]:
            """Apply update i to r."""
            r, i = carry
            idx_i = phys(i)

            # Extract cached values
            S_i = jax.lax.dynamic_index_in_dim(st.S, idx_i, axis=0, keepdims=False)
            Hy_i = jax.lax.dynamic_index_in_dim(HYc, idx_i, axis=0, keepdims=False)
            V_i = jax.lax.dynamic_index_in_dim(Vc, idx_i, axis=0, keepdims=False)
            ys_i = jax.lax.dynamic_index_in_dim(ysc, idx_i, axis=0, keepdims=False)
            yHy_i = jax.lax.dynamic_index_in_dim(yHyc, idx_i, axis=0, keepdims=False)
            phi_i = jax.lax.dynamic_index_in_dim(st.phi, idx_i, axis=0, keepdims=False)
            tau_i = jax.lax.dynamic_index_in_dim(st.tau, idx_i, axis=0, keepdims=False)

            # Stabilize denominators
            ys_i = jnp.maximum(ys_i, 1e-32)
            yHy_i = jnp.maximum(yHy_i, 1e-32)
            tau_i = jnp.maximum(tau_i, 1e-12)

            # Inner products with original input x
            t1 = jnp.dot(Hy_i, x)
            t2 = jnp.dot(V_i, x)
            t3 = jnp.dot(S_i, x)

            # Apply update i to r
            r = r - (t1 / yHy_i) * Hy_i + (phi_i * yHy_i * t2) * V_i
            r = r / tau_i + (t3 / ys_i) * S_i

            return (r, i + jnp.array(1, jnp.int32))

        r, _ = jax.lax.while_loop(apply_cond, apply_body, (r, jnp.array(0, jnp.int32)))
        return r

    @staticmethod
    def update(
        st: _LMState,
        *,
        s: Array,
        y: Array,
        phi: Array,
        tau: Array,
        **_: dict[object, object],
    ) -> _LMState:
        """Update the limited memory state with new iteration information.

        Parameters
        ----------
        st : _LMState
            Current limited memory state.
        s : Array
            Step vector (change in parameters).
        y : Array
            Gradient difference vector.
        phi : Array
            Scaling parameter phi.
        tau : Array
            Scaling parameter tau.

        Returns
        -------
        _LMState
            Updated limited memory state.
        """
        i = st.head
        S = st.S.at[i].set(s)
        Y = st.Y.at[i].set(y)
        phiA = st.phi.at[i].set(phi)
        tauA = st.tau.at[i].set(tau)

        one = jnp.array(1, dtype=jnp.int32)
        mjax = jnp.array(st.m, dtype=jnp.int32)
        head = (st.head + one) % mjax
        count = jnp.minimum(st.count + one, mjax)

        return _LMState(
            S=S,
            Y=Y,
            phi=phiA,
            tau=tauA,
            head=head,
            count=count,
            n=st.n,
            m=st.m,
            gamma=st.gamma,
        )

    @staticmethod
    def maybe_rescale_H0(st: _LMState, gamma: Array) -> _LMState:
        """Rescale the gamma parameter in the limited memory state.

        Parameters
        ----------
        st : _LMState
            Current limited memory state.
        gamma : Array
            New gamma value for scaling.

        Returns
        -------
        _LMState
            State with updated gamma value.
        """
        gamma = jnp.asarray(gamma, dtype=st.gamma.dtype)
        return eqx.tree_at(lambda ss: ss.gamma, st, gamma)

    @staticmethod
    def nudge_towards_I(st: _LMState, eta: float) -> _LMState:
        """Nudge the gamma parameter towards identity for numerical stability.

        Parameters
        ----------
        st : _LMState
            Current limited memory state.
        eta : float
            Nudge factor between 0 and 1.

        Returns
        -------
        _LMState
            State with gamma nudged towards 1.0.
        """
        gamma_new = (1.0 - eta) * st.gamma + eta * 1.0
        return eqx.tree_at(lambda ss: ss.gamma, st, gamma_new)


class _DenseFuncs:
    @staticmethod
    def apply_H(H: Array, v: Array) -> Array:
        """Apply dense inverse Hessian approximation to vector v.

        Parameters
        ----------
        H : Array
            Dense inverse Hessian approximation matrix.
        v : Array
            Vector to apply H to.

        Returns
        -------
        Array
            Result of matrix-vector multiplication H @ v.
        """
        return H @ v

    @staticmethod
    def update(
        H: Array,
        *,
        s: Array,
        y: Array,
        phi: Array,
        tau: Array,
        yHy: Array,
        ys: Array,
        Hy: Array,
    ) -> Array:
        """Update dense inverse Hessian using self-scaled Broyden formula.

        Parameters
        ----------
        H : Array
            Current inverse Hessian approximation.
        s : Array
            Step vector (change in parameters).
        y : Array
            Gradient difference vector.
        phi : Array
            Scaling parameter phi.
        tau : Array
            Scaling parameter tau.
        yHy : Array
            Scalar y^T H y.
        ys : Array
            Scalar y^T s.
        Hy : Array
            Vector H y.

        Returns
        -------
        Array
            Updated inverse Hessian approximation.
        """
        ys = jnp.maximum(ys, 1e-32)
        yHy = jnp.maximum(yHy, 1e-32)
        tau = jnp.maximum(tau, 1e-12)
        v = s / ys - Hy / yHy
        term1 = H - jnp.outer(Hy, Hy) / yHy
        term2 = phi * yHy * jnp.outer(v, v)
        term3 = jnp.outer(s, s) / ys
        Hn = (term1 + term2) / tau + term3
        return 0.5 * (Hn + Hn.T)

    @staticmethod
    def maybe_rescale_H0(H: Array, gamma: float) -> Array:
        """Rescale the initial inverse Hessian approximation by gamma.

        Parameters
        ----------
        H : Array
            Current inverse Hessian (used to determine dimension).
        gamma : float
            Scaling factor.

        Returns
        -------
        Array
            Scaled identity matrix of appropriate dimension.
        """
        n = H.shape[0]
        return gamma * jnp.eye(n, dtype=H.dtype)

    @staticmethod
    def nudge_towards_I(H: Array, eta: float) -> Array:
        """Nudge the inverse Hessian towards identity for numerical stability.

        Parameters
        ----------
        H : Array
            Current inverse Hessian approximation.
        eta : float
            Nudge factor between 0 and 1.

        Returns
        -------
        Array
            Hessian nudged towards identity matrix.
        """
        n = H.shape[0]
        return (1.0 - eta) * H + eta * jnp.eye(n, dtype=H.dtype)


# skylos: ignore-end


class _BaseSSBroyden(optx.AbstractMinimiser, eqx.Module):
    """Base class for Self-Scaled Broyden (SSBroyden) family of optimisers.

    Parameters
    ----------
    rtol : float
        Relative tolerance for convergence.
    atol : float
            Absolute tolerance for convergence.
    norm : Callable
        Norm function for convergence testing.
    theta_mode : Literal["auto", "fixed"]
        Mode for theta parameter computation.
    theta_fixed : float
            Fixed theta value when theta_mode is "fixed".
    eps : float
        Small value for numerical stability.
    H_mixing_ratio : float
        Ratio for nudging H towards identity.
    bt_alpha_init : float
        Initial step size for Armijo line search.
    bt_c1 : float
            Armijo sufficient decrease parameter.
    bt_beta : float
            Step size reduction factor.
    bt_max_steps : int
        Maximum backtracking iterations.
    memory : Optional[int]
        Memory size for limited memory variant (None for dense).
    """

    # Base options
    # skylos: ignore-start
    rtol = 1e-6
    atol = 1e-8
    # skylos: ignore-end
    norm: Callable = optx.max_norm
    theta_mode: Literal["auto", "fixed"] = "auto"
    theta_fixed: float = 0.0
    eps: float = 1e-12
    H_mixing_ratio: float = 0.1

    # Armijo params
    bt_alpha_init: float = 1.0
    bt_c1: float = 1e-4
    bt_beta: float = 0.5
    bt_max_steps: int = 20

    _apply_H: Callable = eqx.field(static=True, init=False, default=None)
    _update_H: Callable = eqx.field(static=True, init=False, default=None)
    _maybe_rescale: Callable = eqx.field(static=True, init=False, default=None)
    _nudge: Callable = eqx.field(static=True, init=False, default=None)
    memory: Optional[int] = eqx.field(default=None, static=True)
    _init_state: Callable = eqx.field(static=True, init=False, default=None)

    def __check_init__(self) -> None:
        """Setup internal functions based on memory configuration."""
        if self.memory and self.memory > 0:
            m = int(self.memory)
            object.__setattr__(self, "_apply_H", _LMFuncs.apply_H)
            object.__setattr__(self, "_update_H", _LMFuncs.update)
            object.__setattr__(self, "_maybe_rescale", _LMFuncs.maybe_rescale_H0)
            object.__setattr__(self, "_nudge", _LMFuncs.nudge_towards_I)
            object.__setattr__(
                self, "_init_state", lambda n, g: _LMFuncs.init_state(n, m, gamma=g)
            )
        else:
            object.__setattr__(self, "_apply_H", _DenseFuncs.apply_H)
            object.__setattr__(self, "_update_H", _DenseFuncs.update)
            object.__setattr__(self, "_maybe_rescale", _DenseFuncs.maybe_rescale_H0)
            object.__setattr__(self, "_nudge", _DenseFuncs.nudge_towards_I)
            object.__setattr__(
                self, "_init_state", lambda n, g: g * jnp.eye(n, dtype=jnp.float64)
            )

    @staticmethod
    def _n_from_state(st_or_H: object) -> int:
        """Extract the number of variables from the state or dense H.

        Parameters
        ----------
        st_or_H : object
            Either a JAX Array (dense H) or _LMState object.

        Returns
        -------
        int
            Number of variables / dimension.

        Raises
        ------
        ValueError
            If unable to extract dimension from input.
        """
        if isinstance(st_or_H, jax.Array):
            return int(st_or_H.shape[0])
        elif hasattr(st_or_H, "n"):
            return int(st_or_H.n)
        else:
            raise ValueError("Cannot extract number of variables from state or H.")

    def _flatten(self, y: Array) -> tuple[Array, Callable[[Array], object]]:
        """Flatten a pytree and return the unravel function.

        Parameters
        ----------
        y : Array
            Pytree to flatten.

        Returns
        -------
        tuple[Array, Callable[[Array], object]]
            Flattened array and unravel function.
        """
        y_flat, unravel = ravel_pytree(y)
        return y_flat, unravel

    def _value_grad(
        self, fn: Callable, y: Array, args: dict[str, object]
    ) -> tuple[Array, Array]:
        """Evaluate function value and gradient at point y.

        Parameters
        ----------
        fn : Callable
            Function to evaluate.
        y : Array
            Point at which to evaluate.
        args : dict[str, object]
            Additional arguments for the function.

        Returns
        -------
        tuple[Array, Array]
            Function value and gradient.
        """
        f, g = jax.value_and_grad(lambda _y: fn(_y, args))(y)
        return f, g

    def _value_only(self, fn: Callable, y: Array, args: dict[str, object]) -> Array:
        """Evaluate function value only (no gradient).

        Parameters
        ----------
        fn : Callable
            Function to evaluate.
        y : Array
            Point at which to evaluate.
        args : dict[str, object]
            Additional arguments for the function.

        Returns
        -------
        Array
            Function value.
        """
        return fn(y, args)

    # -----------------------------------------------------------
    # Armijo backtracking line search
    # -----------------------------------------------------------
    def _armijo_backtracking(
        self,
        fn: Callable,
        yk_flat: Array,
        unravel: Callable[[Array], object],
        args: dict[str, object],
        f_k: Callable[[], Array],
        gk: Array,
        p: Array,
    ) -> float:
        """Armijo backtracking line search.

        Parameters
        ----------
        fn : Callable
            Objective function.
        yk_flat : Array
            Current flattened parameters.
        unravel : Callable[[Array], object]
            Function to unflatten parameters.
        args : dict[str, object]
            Additional function arguments.
        f_k : Callable[[], Array]
            Current function value accessor.
        gk : Array
            Current gradient.
        p : Array
            Search direction.

        Returns
        -------
        float
            Step size alpha, or 0.0 if search failed.
        """
        # Get the directional derivative at the start
        dg0 = jnp.dot(gk, p)

        def reject_alpha() -> Array:
            """Reject alpha when directional derivative is non-negative."""
            return jnp.array(0.0, dtype=jnp.float64)

        def search_alpha() -> float:
            """Search for a suitable alpha using backtracking."""
            # Start off with the initial prediction
            alpha0 = jnp.asarray(self.bt_alpha_init, dtype=jnp.float64)

            def cond_fun(val: tuple[Array, Array, int]) -> bool:
                """Return True if we need to shrink alpha further."""
                # Unpack
                alpha, f_trial, steps = val

                # Check if we need to shrink
                need_shrink = (f_trial > f_k + self.bt_c1 * alpha * dg0) | (
                    ~jnp.isfinite(f_trial)
                )

                # Return True if we need to shrink and haven't exceeded max steps
                return (need_shrink) & (steps < self.bt_max_steps)

            def body_fun(val: tuple[Array, Array, int]) -> tuple[Array, Array, int]:
                """Body function for the while loop."""
                # Unpack
                alpha, _f_trial, steps = val

                # Compute new alpha and evaluate new trial point
                alpha = alpha * self.bt_beta
                y_trial = unravel(yk_flat + alpha * p)
                f_trial = self._value_only(fn, y_trial, args)
                return (alpha, f_trial, steps + 1)

            # Execute first trial
            y_trial0 = unravel(yk_flat + alpha0 * p)
            f_trial0 = self._value_only(fn, y_trial0, args)

            # Iterate until we find a suitable alpha or exhaust max steps
            alpha, f_trial, _ = jax.lax.while_loop(
                cond_fun,
                body_fun,
                (alpha0, f_trial0, jnp.array(0, dtype=jnp.int32)),
            )

            # Check if we succeeded
            fail = (f_trial > f_k + self.bt_c1 * alpha * dg0) | (~jnp.isfinite(f_trial))

            # Set alpha to zero if failed to find a suitable step
            alpha = jnp.where(fail, 0.0, alpha)
            return alpha

        # Return alpha, either zero (reject) or found (search)
        return jax.lax.cond(dg0 >= 0.0, reject_alpha, search_alpha)

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def init(
        self, fn: Callable[..., Array], y: object, args: dict[str, object]
    ) -> _State:
        """Initialize the SSBroyden optimiser state.

        Parameters
        ----------
        fn : Callable[..., Array]
            Objective function.
        y : object
            Initial parameters (pytree).
        args : dict[str, object]
            Additional function arguments.

        Returns
        -------
        _State
            Initialized optimiser state.
        """
        # Cast non-f64 into f64
        y = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), y)

        # Get the initial function value and gradient, flatten the gradient
        f, g = self._value_grad(fn, y, args)
        y_flat, unravel = self._flatten(y)
        g_flat, _ = self._flatten(g)

        # Get the initial inverse Hessian approximation (identity)
        n = y_flat.shape[0]
        H0 = self._init_state(n, 1.0)

        # Compute the initial gradient norm for convergence tests
        g0_norm = jnp.linalg.norm(g_flat) / jnp.sqrt(g_flat.size)

        # Build the state
        state = _State(
            y_flat=y_flat,
            f=jnp.asarray(f, dtype=jnp.float64),
            g_flat=jnp.asarray(g_flat, dtype=jnp.float64),
            backend_state=H0,
            iter=jnp.array(0, dtype=jnp.int32),
            result=optx.RESULTS.successful,
            unravel=unravel,
            g0_norm=jnp.asarray(g0_norm, dtype=jnp.float64),
        )
        return state

    def step(
        self, fn: Callable[..., Array], args: dict[str, object], state: _State
    ) -> tuple[Array, _State]:
        """Perform one iteration of the SSBroyden optimiser.

        Parameters
        ----------
        fn : Callable[..., Array]
            Objective function.
        args : dict[str, object]
            Additional function arguments.
        state : _State
            Current optimiser state.

        Returns
        -------
        tuple[Array, _State]
            Next parameters and updated state.
        """
        # Extract the variables from the state
        yk = state.y_flat
        gk = state.g_flat
        Hk = state.backend_state

        # Get the direction
        pk = -self._apply_H(Hk, gk)

        # Line search to determine step length alpha
        alpha = self._armijo_backtracking(fn, yk, state.unravel, args, state.f, gk, pk)

        # Produce the next state
        no_move = alpha <= 0.0
        s = alpha * pk
        y_next_flat = jnp.where(no_move, yk, yk + s)
        y_next = state.unravel(y_next_flat)

        def _eval_new(_: None) -> tuple[Array, Array]:
            """Produce new function value and gradient at y_next."""
            f_new, g_new = self._value_grad(fn, y_next, args)
            g_new_flat, _ = self._flatten(g_new)
            return (f_new, g_new_flat)

        # If no_move: reuse (state.f, gk). Else: evaluate at y_next and flatten gradient.
        f_next, g_next_flat = jax.lax.cond(
            no_move,
            lambda _: (state.f, gk),
            _eval_new,
            operand=None,
        )

        # Change in gradient
        y_vec = g_next_flat - gk

        Hy_unscaled = self._apply_H(Hk, y_vec)
        yHy_unscaled = jnp.dot(y_vec, Hy_unscaled)

        ys = jnp.dot(y_vec, s)
        ys_safe = jnp.maximum(ys, 1e-32)
        sBg = -alpha * jnp.dot(s, gk)
        bk = sBg / ys_safe

        # Determine conditions for update
        good_curv = ys > self.eps * jnp.linalg.norm(s) * jnp.linalg.norm(y_vec)
        positive_denoms = (ys > self.eps) & (yHy_unscaled > self.eps) & (bk > self.eps)
        do_update = (~no_move) & good_curv & positive_denoms

        # Compute scaling for first iteration
        scale = jnp.maximum(ys, 1e-32) / jnp.maximum(jnp.dot(y_vec, y_vec), 1e-32)

        def first_iter_branch(H_in: object) -> tuple[Array, Array, Array]:
            """Rescale H and apply once."""
            H_scaled = self._maybe_rescale(H_in, scale)
            Hy = self._apply_H(H_scaled, y_vec)
            yHy = jnp.dot(y_vec, Hy)
            return H_scaled, Hy, yHy

        def later_iter_branch(H_in: object) -> tuple[Array, Array, Array]:
            """Leave H as-is; apply once."""
            Hy = self._apply_H(H_in, y_vec)
            yHy = jnp.dot(y_vec, Hy)
            return H_in, Hy, yHy

        # Get previous H, Hy, yHy (with possible rescaling on first iter)
        H_prev, Hy_used, yHy_used = jax.lax.cond(
            (state.iter == 0) & do_update, first_iter_branch, later_iter_branch, Hk
        )
        yHy_safe = jnp.maximum(yHy_used, 1e-32)

        # Compute scalars
        hk = yHy_safe / ys_safe
        ak = bk * hk - 1.0

        def _compute_scalars(
            _: object = None,
        ) -> tuple[Array, Array]:
            """Compute scaling scalars for the Broyden update."""
            # Bounded computation of ck
            ck = jnp.sqrt(jnp.maximum(0.0, ak / (1.0 + ak)))

            # Negative/positive rho, theta
            rho_minus = jnp.minimum(1.0, hk * (1.0 - ck))
            theta_minus = (rho_minus - 1.0) / jnp.maximum(ak, 1e-32)
            rho_plus = jnp.maximum(
                0.0, jnp.minimum(1.0, 1.0 / jnp.maximum(bk, 1e-32))
            )  # bind to [0,1]
            theta_plus = 1.0 / jnp.maximum(rho_minus, 1e-32)

            # Update theta if allowed
            if self.theta_mode == "fixed":
                theta_k = jnp.clip(self.theta_fixed, 0.0, 1.0)
            else:
                theta_extra = (1.0 - bk) / jnp.maximum(bk, 1e-32)
                theta_k = jnp.maximum(theta_minus, jnp.minimum(theta_plus, theta_extra))

            # Compute scaling phi_k and tau_k
            sigma_k = 1.0 + theta_k * ak
            phi_k = (1.0 - theta_k) / (1.0 + ak * theta_k)
            n = _BaseSSBroyden._n_from_state(Hk)
            inv_1_minus_n = 1.0 / (1.0 - float(n))
            sigma_pow = jnp.abs(sigma_k) ** inv_1_minus_n
            tau_case1 = jnp.minimum(rho_plus * sigma_pow, jnp.maximum(sigma_k, 0.0))
            tau_case2 = rho_plus * jnp.minimum(
                sigma_pow, 1.0 / jnp.maximum(theta_k, 1e-32)
            )
            tau_k = jnp.where(theta_k <= 0.0, tau_case1, tau_case2)
            tau_k = jnp.maximum(tau_k, 1e-12)

            return (phi_k, tau_k)

        phi_k, tau_k = jax.lax.cond(
            do_update,
            _compute_scalars,
            lambda _: (jnp.array(0.0, jnp.float64), jnp.array(1.0, jnp.float64)),
            operand=None,
        )

        # Apply update if conditions met; otherwise passthrough
        def _do_update(H: object) -> Array:
            return self._update_H(
                H,
                s=s,
                y=y_vec,
                phi=phi_k,
                tau=tau_k,
                yHy=yHy_safe,
                ys=ys_safe,
                Hy=Hy_used,
            )

        # Update H
        H_next = jax.lax.cond(do_update, _do_update, lambda H: H, H_prev)

        # Nudge H slightly towards identity to avoid pathological cases
        H_next = jax.lax.cond(
            no_move, lambda H: self._nudge(H, self.H_mixing_ratio), lambda H: H, H_next
        )

        # Produce the next state
        next_state = _State(
            y_flat=y_next_flat,
            f=jnp.asarray(
                jax.lax.cond(no_move, lambda _: state.f, lambda _: f_next, None),
                dtype=jnp.float64,
            ),
            g_flat=jnp.asarray(g_next_flat, dtype=jnp.float64),
            backend_state=H_next,
            iter=state.iter + jnp.array(1, dtype=jnp.int32),
            result=state.result,
            unravel=state.unravel,
            g0_norm=state.g0_norm,
        )
        return y_next, next_state

    # Fulfil optimistix AbstractMinimiser requirements. Optax is not using these hooks but we need them to be present.
    # skylos: ignore-start
    terminate = lambda: None
    postprocess = lambda: None
    # skylos: ignore-end


class SSBroydenArmijo(_BaseSSBroyden):
    """Self-Scaled Broyden with Armijo backtracking line search.

    Parameters
    ----------
    norm : Callable
        Norm function for convergence testing.
    eps : float
        Small value for numerical stability.
    bt_alpha_init : float
        Initial step size for Armijo line search.
    bt_c1 : float
        Armijo sufficient decrease parameter.
    bt_beta : float
        Step size reduction factor.
    bt_max_steps : int
        Maximum backtracking iterations.
    memory : Optional[int]
        Memory size for limited memory variant.
    """

    def __init__(
        self,
        norm: Callable = optx.max_norm,
        eps: float = 1e-12,
        bt_alpha_init: float = 1.0,
        bt_c1: float = 1e-4,
        bt_beta: float = 0.5,
        bt_max_steps: int = 20,
        memory: Optional[int] = None,
    ) -> None:
        """Initialize the SSBroydenArmijo optimiser.

        Parameters
        ----------
        norm : Callable, optional
            Norm function. Default is optx.max_norm.
        eps : float, optional
            Small value for numerical stability. Default is 1e-12.
        bt_alpha_init : float, optional
            Initial step size. Default is 1.0.
        bt_c1 : float, optional
            Armijo parameter. Default is 1e-4.
        bt_beta : float, optional
            Step reduction factor. Default is 0.5.
        bt_max_steps : int, optional
            Max backtracking steps. Default is 20.
        memory : Optional[int], optional
            Memory size for L-M variant. Default is None (dense).
        """
        super().__init__(
            norm=norm,
            theta_mode="auto",
            theta_fixed=0.0,
            eps=eps,
            bt_alpha_init=bt_alpha_init,
            bt_c1=bt_c1,
            bt_beta=bt_beta,
            bt_max_steps=bt_max_steps,
            memory=memory,
        )


# skylos: ignore-start
class OptimistixAsOptax:
    """Adapter making an Optimistix minimiser look like an Optax optimiser.

    Parameters
    ----------
    solver : object
        Optimistix solver instance (e.g., SSBroyden or SSBFGS).
    """

    def __init__(self, solver: object) -> None:
        """Wrap an Optimistix solver to look like an Optax optimiser.

        Parameters
        ----------
        solver : object
            Optimistix solver instance (e.g., SSBroyden or SSBFGS).
        """
        self.solver = solver

    def init(self, *_: Tuple[()], **__: Dict[str, object]) -> Dict[str, object]:
        """Lazy initialization of the solver state.

        Returns
        -------
        Dict[str, object]
            Empty state dictionary with solver_state set to None.
        """
        return {"solver_state": None}

    def update(
        self,
        grads: Array,
        opt_state: Dict[str, object],
        params: Dict[str, object],
        *,
        args: Optional[Dict[str, object]] = None,
        **kw,
    ) -> Tuple[Array, Dict[str, object]]:
        """Perform one optimisation step using the Optimistix solver.

        Parameters
        ----------
        grads : Array
            Gradients (not used directly; solver computes them).
        opt_state : Dict[str, object]
            Current optimiser state.
        params : Dict[str, object]
            Current parameters.
        args : Optional[Dict[str, object]]
            Must contain '__fn__' callable. Required.

        Returns
        -------
        Tuple[Array, Dict[str, object]]
            Updates to apply and updated optimiser state.

        Raises
        ------
        AssertionError
            If args is None or '__fn__' is not callable.
        """
        assert args is not None, "args must be provided and include '__fn__'."
        fn = args.get("__fn__")
        assert callable(
            fn
        ), "args['__fn__'] must be a callable: fn(params, args)->scalar."

        st = opt_state.get("solver_state", None)
        if st is None:
            # First call: initialize the solver state now that we have fn/args.
            st = self.solver.init(fn, params, args)

        # One Optimistix step
        y_next, new_state = self.solver.step(fn, args, st)

        # Convert to Optax-style "updates"
        updates = jax.tree_util.tree_map(lambda a, b: a - b, y_next, params)

        return updates, {"solver_state": new_state}


# skylos: ignore-end
