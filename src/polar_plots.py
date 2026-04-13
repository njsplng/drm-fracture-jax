"""Polar plot functionality for the anisotropic model."""

import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.integrate import solve_bvp

from plots_functions import linestyle_list

plt.style.use("seaborn-v0_8-bright")


# Only used manually, tell linter to ignore
# skylos: ignore-start
def sweep_and_plot_polar_energy_surfaces(
    params_list: list[dict],
    l_0: float = 1.0,
    figsize: tuple[int, int] = (12, 6),
    rmax: float = 1.5,
    radial_ticks: tuple[float, float] = (0.5, 1.0),
    differentiate_linestyles: bool = False,
    title_fontsize: int = 14,
    label_offset: float = 0.25,
    label_fontsize: int = 10,
    threshold_radius: Optional[float] = None,
    filename: Optional[str] = None,
    separate_plots: bool = False,
) -> None:
    """Generate and plot polar energy surfaces for multiple parameter sets.

    Sweep the angular direction and compute the surface energy using
    a BVP solver for the 1D interfacial profile, then display the
    results in polar coordinates.

    Parameters
    ----------
    params_list : list[dict]
        List of parameter dictionaries for each material configuration.
    l_0 : float, optional
        Characteristic length scale. Default is 1.0.
    figsize : tuple[int, int], optional
        Figure size for the plot. Default is (12, 6).
    rmax : float, optional
        Maximum radial extent for the polar plots. Default is 1.5.
    radial_ticks : tuple[float, float], optional
        Radial tick positions. Default is (0.5, 1.0).
    differentiate_linestyles : bool, optional
        Whether to use different linestyles for each curve.
        Default is False.
    title_fontsize : int, optional
        Font size for the plot title. Default is 14.
    label_offset : float, optional
        Offset for the legend position. Default is 0.25.
    label_fontsize : int, optional
        Font size for legend labels. Default is 10.
    threshold_radius : Optional[float], optional
        Threshold value for threshold-based coloring.
    filename : Optional[str], optional
        Output filename for saving the figure.
    separate_plots : bool, optional
        Whether to save plots individually. Default is False.

    Raises
    ------
    ValueError
        If threshold_radius is set with multiple parameter entries.
    """
    if threshold_radius is not None and len(params_list) > 1:
        raise ValueError(
            "When threshold_radius is set, only one parameter list entry can be provided."
        )

    def voigt_K_2d(phi: float) -> np.ndarray:
        """Compute the 2D Voigt rotation matrix for a given angle.

        Parameters
        ----------
        phi : float
            Rotation angle in radians.

        Returns
        -------
        np.ndarray
            3x2 transformation matrix for Voigt notation.
        """
        c, s = np.cos(phi), np.sin(phi)
        return np.array(
            [
                [c * c, s * s, 2 * c * s],
                [s * s, c * c, -2 * c * s],
                [-c * s, c * s, c * c - s * s],
            ]
        )

    def gamma_voigt_rotated(
        gamma11: float,
        gamma22: float,
        gamma12: float,
        gamma44: float,
        material_angle: float,
    ) -> np.ndarray:
        """Rotate the gamma tensor to a material coordinate system.

        Transform the orthorhombic gamma tensor from its principal axes
        to the material orientation using Voigt notation.

        Parameters
        ----------
        gamma11 : float
            Normal gamma component in direction 1.
        gamma22 : float
            Normal gamma component in direction 2.
        gamma12 : float
            Coupling gamma component between directions 1 and 2.
        gamma44 : float
            Shear gamma component.
        material_angle : float
            Rotation angle of the material frame in radians.

        Returns
        -------
        np.ndarray
            Rotated gamma tensor in Voigt notation (3x3 matrix).
        """
        # Principal-axes Voigt matrix for orthorhombic γ (2D): diag shear entry is 4*γ44
        # Principal-axes Voigt matrix for orthorhombic γ (2D): diag shear entry is 4*γ44
        Gamma_mat = np.array(
            [[gamma11, gamma12, 0.0], [gamma12, gamma22, 0.0], [0.0, 0.0, gamma44]]
        )
        K = voigt_K_2d(material_angle)
        return K @ Gamma_mat @ K.T

    def m_theta(theta: float, Gamma_voigt_rot: np.ndarray) -> float:
        """Compute the directional surface energy coefficient.

        Parameters
        ----------
        theta : float
            Angle of the interface normal.
        Gamma_voigt_rot : np.ndarray
            Rotated gamma tensor in Voigt notation.

        Returns
        -------
        float
            Directional surface energy coefficient.
        """
        theta = theta + 0.5 * np.pi
        nx, ny = np.cos(theta), np.sin(theta)
        v = np.array([nx * nx, ny * ny, 2 * nx * ny])
        return float(v @ Gamma_voigt_rot @ v)

    # 1D interfacial profile solver for a given theta
    def gc_for_theta(
        theta: float,
        Gamma_voigt_rot: np.ndarray,
        l0: float,
        xi_max: float = 50.0,
        npts: int = 600,
    ) -> float:
        """Compute the surface energy for a given theta direction.

        Solve the 4th-order Euler–Lagrange differential equation for
        the interfacial profile using a boundary value problem solver.

        Parameters
        ----------
        theta : float
            Direction angle of the interface.
        Gamma_voigt_rot : np.ndarray
            Rotated gamma tensor.
        l0 : float
            Characteristic length scale.
        xi_max : float, optional
            Maximum extent of the computational domain. Default is 50.0.
        npts : int, optional
            Number of discretization points. Default is 600.

        Returns
        -------
        float
            Surface energy for the given direction.

        Raises
        ------
        RuntimeError
            If the BVP solver fails to converge.
        """
        m = m_theta(theta, Gamma_voigt_rot)

        # 4th-order Euler–Lagrange ODE in state-space form
        # y = [c, c', c'', c'''], with independent var ξ
        # From δ/δc of the functional:
        #   l0^3*m * c'''' - l0 * c'' + (c - 1)/(4*l0) = 0
        # -> c'''' = [ l0*c'' - (c - 1)/(4*l0) ] / (l0^3*m)
        def ode(x: float, y: np.ndarray) -> np.ndarray:
            c, c1, c2, c3 = y
            rhs4 = (l0 * c2 - (c - 1.0) / (4.0 * l0)) / (l0**3 * m)
            return np.vstack((c1, c2, c3, rhs4))

        # Boundary conditions (planar interface):
        # c(-∞)=0, c(+∞)=1, c'(-∞)=0, c'(+∞)=0
        # On finite domain [-xi_max, +xi_max]:
        def bc(yL: np.ndarray, yR: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    # c(-∞)=0
                    yL[0] - 0.0,
                    # c'(-∞)=0
                    yL[1] - 0.0,
                    # c(+∞)=1
                    yR[0] - 1.0,
                    # c'(+∞)=0
                    yR[1] - 0.0,
                ]
            )

        x = np.linspace(-xi_max, xi_max, npts)
        # Smooth monotone initial guess
        c_guess = 0.5 * (1.0 + np.tanh(x / (np.sqrt(2) * l0)))
        y_init = np.vstack(
            (c_guess, np.gradient(c_guess, x), np.zeros_like(x), np.zeros_like(x))
        )
        sol = solve_bvp(ode, bc, x, y_init, max_nodes=20000)
        if not sol.success:
            raise RuntimeError("BVP solver did not converge for theta=%.4f rad" % theta)

        # Compute surface energy (per unit length) at optimal profile
        c = sol.sol(x)[0]
        c1 = sol.sol(x)[1]
        c2 = sol.sol(x)[2]

        integrand = (
            ((c - 1.0) ** 2 / (4.0 * l0)) + (l0 * c1**2) + ((l0**3) * m * (c2**2))
        )
        Gc = np.trapezoid(integrand, x)
        return Gc

    # Driver: sweep theta, make polar
    def sweep_polar(
        ell0: float,
        gamma11: float,
        gamma22: float,
        gamma12: float,
        gamma44: float,
        material_angle_gamma: float,
        n_angles: int = 361,
        xi_max: float = 50.0,
        npts: int = 600,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sweep angular direction and generate polar representation.

        Compute the surface energy at multiple angles and return
        the angle-energy pairs for visualization.

        Parameters
        ----------
        ell0 : float
            Characteristic length scale.
        gamma11 : float
            Normal gamma component in direction 1.
        gamma22 : float
            Normal gamma component in direction 2.
        gamma12 : float
            Coupling gamma component.
        gamma44 : float
            Shear gamma component.
        material_angle_gamma : float
            Material orientation angle.
        n_angles : int, optional
            Number of angles to evaluate. Default is 361.
        xi_max : float, optional
            Domain extent for BVP solver. Default is 50.0.
        npts : int, optional
            Number of discretization points. Default is 600.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Pair of (angles, surface_energies) arrays.
        """
        Gam_rot = gamma_voigt_rotated(
            gamma11, gamma22, gamma12, gamma44, material_angle_gamma
        )
        thetas = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=True)
        Gc_vals = np.array(
            [gc_for_theta(th, Gam_rot, ell0, xi_max, npts) for th in thetas]
        )
        return thetas, Gc_vals

    def gamma_label(params: dict) -> str:
        """Generate a LaTeX-formatted label from gamma parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing gamma11, gamma22, gamma12, gamma44,
            and optionally theta.

        Returns
        -------
        str
            LaTeX-formatted string with gamma parameter values.
        """
        base_str = (
            rf'$\gamma_{{11}}$={params["gamma11"]:.1f}, '
            rf'$\gamma_{{22}}$={params["gamma22"]:.1f}, '
            rf'$\gamma_{{12}}$={params["gamma12"]:.1f}, '
            rf'$\gamma_{{44}}$={params["gamma44"]:.1f}'
        )
        additional_str = (
            rf', $\theta$={np.rad2deg(params.get("theta", 0.0)):.1f}$^\circ$'
            if params.get("theta")
            else ""
        )
        return base_str + additional_str

    def set_radial_ticks(
        ax: "axes",
        ticks: tuple[float, float] = (0.5, 1.0),
        rmax: Optional[float] = None,
        labels: Optional[list[str]] = None,
        fontsize: int = 10,
    ) -> None:
        """Format radial ticks on a polar axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The polar axes to modify.
        ticks : tuple[float, float], optional
            Radial tick positions. Default is (0.5, 1.0).
        rmax : Optional[float], optional
            Maximum radial value to display.
        labels : Optional[list[str]], optional
            Custom labels for tick marks.
        fontsize : int, optional
            Font size for tick labels. Default is 10.
        """
        if rmax is not None:
            # Ensure the outer limit matches what you want
            ax.set_rlim(0, rmax)
        # Place gridlines at these radii
        ax.set_rticks(ticks)
        if labels is not None:
            # Custom text (otherwise matplotlib auto-formats)
            ax.set_yticklabels(labels, fontsize=fontsize)
        # ax.set_rlabel_position(90)  # optional: move radial labels to a nicer angle
        # Grid is usually on by default for polar
        ax.grid(True)
        # Positions in radians
        theta_locs = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ax.set_xticks(theta_locs)
        # Your custom labels
        ax.set_xticklabels(
            [f"{np.rad2deg(t):.0f}" for t in theta_locs], fontsize=fontsize
        )

    def add_colored_polar_line(
        ax: plt.Axes,
        theta: float,
        r: float,
        color_values: list[float],
        threshold: float,
        cmap: str,
        norm: BoundaryNorm,
    ) -> None:
        """Add a multicolored line collection to a polar plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The polar axes to add the line to.
        theta : float
            Angular coordinate of the line endpoint.
        r : float
            Radial coordinate of the line endpoint.
        color_values : list[float]
            Values to map to colors along the line.
        threshold : float
            Threshold value for color mapping.
        cmap : str
            Colormap name.
        norm : BoundaryNorm
            Normalization object for the colormap.
        """
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Use average value of segment to determine color
        mid_values = (color_values[:-1] + color_values[1:]) / 2

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(mid_values)
        lc.set_linewidth(2)
        ax.add_collection(lc)

    def plot_polar_with_bottom_legend(
        entries: list[Tuple[np.ndarray, np.ndarray, dict]],
        title_left: str = "$\\mathcal{G}_c$($\\theta$)",
        title_right: str = "1 / $\\mathcal{G}_c$($\\theta$)",
    ) -> None:
        """Plot polar coordinates with a legend centered below.

        Create two polar subplots showing the surface energy and
        its inverse, with a shared legend positioned at the bottom.

        Parameters
        ----------
        entries : list[Tuple[np.ndarray, np.ndarray, dict]]
            List of (thetas, Gc_vals, params_dict) tuples.
        title_left : str, optional
            Title for the left subplot. Default is the Gc expression.
        title_right : str, optional
            Title for the right subplot. Default is the inverse Gc
            expression.
        """
        fig, ax = plt.subplots(
            1,
            2,
            subplot_kw={"projection": "polar"},
            figsize=figsize,
            constrained_layout=False,
        )

        handles = []
        labels = []

        for i, (thetas, Gc_vals, p) in enumerate(entries):
            if threshold_radius is not None:
                # if below the threshold, blue, otherwise, red
                norm = BoundaryNorm([0, threshold_radius, float("inf")], ncolors=2)
                cmap = ListedColormap(["blue", "red"])

                # plot Gc vs theta
                add_colored_polar_line(
                    ax[0], thetas, Gc_vals, Gc_vals, threshold_radius, cmap, norm
                )

                # plot 1/Gc vs theta
                add_colored_polar_line(
                    ax[1], thetas, 1.0 / Gc_vals, Gc_vals, threshold_radius, cmap, norm
                )

                # Dummy handle for legend
                # (h,) = ax[0].plot([],[],color="blue",lw=2)
                # handles.append(h)
                # labels.append(gamma_label(p))

            else:
                ls = (
                    linestyle_list[i % len(linestyle_list)]
                    if differentiate_linestyles
                    else "-"
                )
                # Left plot
                (h,) = ax[0].plot(thetas, Gc_vals, lw=2, linestyle=ls)
                # Right plot: reuse same color
                ax[1].plot(
                    thetas, 1.0 / Gc_vals, lw=2, color=h.get_color(), linestyle=ls
                )

                handles.append(h)
                labels.append(gamma_label(p))

        ax[0].set_title(title_left, fontsize=title_fontsize)
        ax[1].set_title(title_right, fontsize=title_fontsize)

        set_radial_ticks(
            ax[0],
            ticks=radial_ticks,
            rmax=rmax,
            labels=[str(t) for t in radial_ticks],
            fontsize=label_fontsize,
        )
        set_radial_ticks(
            ax[1],
            ticks=radial_ticks,
            rmax=rmax,
            labels=[str(t) for t in radial_ticks],
            fontsize=label_fontsize,
        )

        # Build a single legend centered under both plots
        # Tweak ncol depending on how many curves you have
        ncols = 1
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=ncols,
            frameon=False,
            fontsize=label_fontsize,
        )

        if threshold_radius is None:
            # Leave room at the bottom for the legend
            fig.subplots_adjust(bottom=label_offset, wspace=0.15)

        # If filename is provided, save the fig
        if filename is not None:
            # Get the path for saving
            current_path = pathlib.Path(__file__).parent.resolve()
            project_root = current_path.parent
            output_path = project_root / "output" / "plots"
            output_path.mkdir(parents=True, exist_ok=True)

            # Base path for the combined file
            file_path = output_path / filename

            if separate_plots:
                # We need to construct new filenames: filename_a.png, filename_b.png
                stem = file_path.stem
                # e.g., .png
                suffix = file_path.suffix

                # Define the extent for Left Plot (ax[0]) and Right Plot (ax[1])
                # This is a robust way to save subplots individually
                for idx, suffix_char in enumerate(["a", "b"]):
                    extent = (
                        ax[idx]
                        .get_tightbbox(fig.canvas.get_renderer())
                        .transformed(fig.dpi_scale_trans.inverted())
                    )
                    # Add a little padding if needed, or use bbox_inches='tight' on the extent
                    part_filename = output_path / f"{stem}_{suffix_char}{suffix}"

                    # Save just that axes' bbox
                    fig.savefig(
                        part_filename,
                        bbox_inches=extent.expanded(1, 1),
                        format=str(suffix)[1:],
                        dpi=300,
                    )
                    print(f"Saved split figure to {part_filename}")
            else:
                # Standard single file save
                fig.savefig(file_path, format="png", dpi=300)
                print(f"Saved figure to {file_path}")

        plt.show()
        plt.tight_layout()

    plot_thetas = []
    plot_Gc = []
    for entry in params_list:
        thetas, Gc = sweep_polar(
            l_0,
            entry["gamma11"],
            entry["gamma22"],
            entry["gamma12"],
            entry["gamma44"],
            entry.get("theta", 0.0),
        )
        plot_thetas.append(thetas)
        plot_Gc.append(Gc)

    plot_list = []
    for i, entry in enumerate(params_list):
        plot_list.append((plot_thetas[i], plot_Gc[i], entry))

    plot_polar_with_bottom_legend(plot_list)


# skylos: ignore-end
