"""Tests for material models.

Tests strain energy models from src/strain_energy_models.py and
phase field models from src/phase_field_models.py.
"""

import jax.numpy as jnp
import pytest

from phase_field_models import (
    AT1PhasefieldModel,
    AT2PhasefieldModel,
    get_phasefield_model,
)
from strain_energy_models import (
    SpectralSplitEnergyModel,
    VolumetricDeviatoricSplitEnergyModel,
    get_energy_model,
)


class TestSpectralSplitEnergy:
    """Tests for spectral split energy model."""

    def test_spectral_split_energy(self):
        """Test 7.1: Create spectral energy model, compute parts for known strain, verify both ≥ 0."""
        # Create spectral energy model
        model = get_energy_model("spectral", E=210.0, nu=0.3, plane_mode="strain")

        # Create a known strain tensor (small strain)
        strain = jnp.array(
            [
                [0.01, 0.002, 0.0],
                [0.002, -0.005, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )

        # Compute energy parts
        e_pos, e_neg = model.energy_parts(strain[None, None, :, :])

        # Both parts should be non-negative
        assert e_pos >= 0, f"Positive energy part should be ≥ 0, got {e_pos}"
        assert e_neg >= 0, f"Negative energy part should be ≥ 0, got {e_neg}"

        # Both should be finite
        assert jnp.isfinite(e_pos), "Positive energy part should be finite"
        assert jnp.isfinite(e_neg), "Negative energy part should be finite"

    def test_spectral_split_energy_plane_stress(self):
        """Test 7.1 variant: Plane stress assumption."""
        model = get_energy_model("spectral", E=210.0, nu=0.3, plane_mode="stress")

        strain = jnp.array(
            [
                [0.01, 0.002, 0.0],
                [0.002, -0.005, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )

        e_pos, e_neg = model.energy_parts(strain[None, None, :, :])

        assert e_pos >= 0, "Positive energy part should be ≥ 0"
        assert e_neg >= 0, "Negative energy part should be ≥ 0"
        assert jnp.isfinite(e_pos), "Positive energy part should be finite"
        assert jnp.isfinite(e_neg), "Negative energy part should be finite"

    def test_volumetric_deviatoric_split_energy(self):
        """Test 7.1 variant: Volumetric-deviatoric split."""
        model = get_energy_model("volumetric", E=210.0, nu=0.3, plane_mode="strain")

        strain = jnp.array(
            [
                [0.01, 0.002, 0.0],
                [0.002, -0.005, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )

        e_pos, e_neg = model.energy_parts(strain[None, None, :, :])

        assert e_pos >= 0, "Positive energy part should be ≥ 0"
        assert e_neg >= 0, "Negative energy part should be ≥ 0"


class TestConstitutiveMatrix:
    """Tests for constitutive matrix properties."""

    def test_constitutive_matrix_symmetry(self):
        """Test 7.2: Verify D = D.T and all eigenvalues positive for isotropic case."""
        # Create energy model using factory function
        model = get_energy_model("spectral", E=210.0, nu=0.3, plane_mode="strain")

        # Get constitutive matrix
        point_volumes = jnp.array([1.0])
        C = model.initialise_constitutive_matrices(point_volumes)

        # Squeeze to get the actual matrix
        C_matrix = C.squeeze()

        # Check symmetry: C = C.T
        assert jnp.allclose(
            C_matrix, C_matrix.T, atol=1e-10
        ), "Constitutive matrix should be symmetric"

        # Check positive definiteness: all eigenvalues > 0
        eigenvalues = jnp.linalg.eigvalsh(C_matrix)
        assert jnp.all(
            eigenvalues > 0
        ), f"All eigenvalues should be positive, got {eigenvalues}"

        # Check eigenvalues are finite
        assert jnp.all(jnp.isfinite(eigenvalues)), "Eigenvalues should be finite"

    def test_constitutive_matrix_plane_stress(self):
        """Test 7.2 variant: Plane stress constitutive matrix."""
        model = get_energy_model("spectral", E=210.0, nu=0.3, plane_mode="stress")

        point_volumes = jnp.array([1.0])
        C = model.initialise_constitutive_matrices(point_volumes)
        C_matrix = C.squeeze()

        # Check symmetry
        assert jnp.allclose(
            C_matrix, C_matrix.T, atol=1e-10
        ), "Constitutive matrix should be symmetric"

        # Check positive definiteness
        eigenvalues = jnp.linalg.eigvalsh(C_matrix)
        assert jnp.all(
            eigenvalues > 0
        ), f"All eigenvalues should be positive, got {eigenvalues}"

    def test_constitutive_matrix_shape(self):
        """Test 7.2 variant: Constitutive matrix shape."""
        model = get_energy_model("spectral", E=210.0, nu=0.3, plane_mode="strain")

        point_volumes = jnp.array([1.0, 2.0, 3.0])
        C = model.initialise_constitutive_matrices(point_volumes)

        # For plane strain/stress, should be (3, 3) after reduction
        assert C.shape[-2:] == (
            3,
            3,
        ), f"Expected last two dims to be (3, 3), got {C.shape[-2:]}"


class TestPhasefieldModelDegradation:
    """Tests for phase field model degradation."""

    def test_phasefield_model_degradation_at2(self):
        """Test 7.3: For AT2: verify degradation function properties.

        The degradation function is g(c) = c^2, so:
        - g(0) = 0 (fully damaged state has zero stiffness contribution)
        - g(1) = 1 (undamaged state has full stiffness)
        - Monotonically increasing with c
        """
        # Create AT2 phase field model
        model = get_phasefield_model("at2", G_c=1e-4, l_0=0.5)

        # Test degradation at c = 0 (fully damaged)
        c_zero = jnp.array([0.0])
        g_zero = model._degradation(c_zero)
        assert jnp.allclose(g_zero, 0.0, atol=1e-10), f"g(0) should be 0, got {g_zero}"

        # Test degradation at c = 1 (undamaged)
        c_one = jnp.array([1.0])
        g_one = model._degradation(c_one)
        assert jnp.allclose(g_one, 1.0, atol=1e-10), f"g(1) should be 1, got {g_one}"

        # Test monotonicity: degradation should increase as c increases
        c_values = jnp.linspace(0.0, 1.0, 11)
        g_values = model._degradation(c_values)

        # Check monotonically increasing
        for i in range(len(g_values) - 1):
            assert (
                g_values[i] <= g_values[i + 1]
            ), f"Degradation should be monotonically increasing: g({c_values[i]})={g_values[i]} > g({c_values[i+1]})={g_values[i+1]}"

        # All values should be in [0, 1]
        assert jnp.all(g_values >= 0), "Degradation should be ≥ 0"
        assert jnp.all(g_values <= 1), "Degradation should be ≤ 1"

    def test_phasefield_model_degradation_at1(self):
        """Test 7.3 variant: AT1 model degradation."""
        model = get_phasefield_model("at1", G_c=1e-4, l_0=0.5)

        # Test degradation at c = 0
        g_zero = model._degradation(jnp.array([0.0]))
        assert jnp.allclose(g_zero, 0.0, atol=1e-10), "g(0) should be 0"

        # Test degradation at c = 1
        g_one = model._degradation(jnp.array([1.0]))
        assert jnp.allclose(g_one, 1.0, atol=1e-10), "g(1) should be 1"

        # Monotonicity
        c_values = jnp.linspace(0.0, 1.0, 11)
        g_values = model._degradation(c_values)
        for i in range(len(g_values) - 1):
            assert (
                g_values[i] <= g_values[i + 1]
            ), "Degradation should be monotonically increasing"

    def test_phasefield_model_degradation_intermediate(self):
        """Test 7.3 variant: Intermediate degradation values."""
        model = get_phasefield_model("at2", G_c=1e-4, l_0=0.5)

        # At c = 0.5, g = 0.5^2 = 0.25
        c_half = jnp.array([0.5])
        g_half = model._degradation(c_half)
        assert jnp.allclose(
            g_half, 0.25, atol=1e-10
        ), f"g(0.5) should be 0.25, got {g_half}"

        # At c = 0.25, g = 0.25^2 = 0.0625
        c_quarter = jnp.array([0.25])
        g_quarter = model._degradation(c_quarter)
        assert jnp.allclose(
            g_quarter, 0.0625, atol=1e-10
        ), f"g(0.25) should be 0.0625, got {g_quarter}"


class TestEnergyModelFactory:
    """Tests for energy model factory function."""

    def test_get_energy_model_spectral(self):
        """Test factory returns correct type for spectral."""
        model = get_energy_model("spectral", E=210.0, nu=0.3, plane_mode="strain")
        assert isinstance(
            model, SpectralSplitEnergyModel
        ), "Should return SpectralSplitEnergyModel"

    def test_get_energy_model_volumetric(self):
        """Test factory returns correct type for volumetric."""
        model = get_energy_model("volumetric", E=210.0, nu=0.3, plane_mode="strain")
        assert isinstance(
            model, VolumetricDeviatoricSplitEnergyModel
        ), "Should return VolumetricDeviatoricSplitEnergyModel"

    def test_get_energy_model_unknown(self):
        """Test factory raises for unknown model."""
        with pytest.raises(ValueError, match="Unknown energy model"):
            get_energy_model("unknown_model", E=210.0, nu=0.3, plane_mode="strain")


class TestPhasefieldModelFactory:
    """Tests for phase field model factory function."""

    def test_get_phasefield_model_at1(self):
        """Test factory returns correct type for AT1."""
        model = get_phasefield_model("at1", G_c=1e-4, l_0=0.5)
        assert isinstance(model, AT1PhasefieldModel), "Should return AT1PhasefieldModel"

    def test_get_phasefield_model_at2(self):
        """Test factory returns correct type for AT2."""
        model = get_phasefield_model("at2", G_c=1e-4, l_0=0.5)
        assert isinstance(model, AT2PhasefieldModel), "Should return AT2PhasefieldModel"

    def test_get_phasefield_model_unknown(self):
        """Test factory raises for unknown model."""
        with pytest.raises(ValueError, match="Unknown phasefield model"):
            get_phasefield_model("unknown_model", G_c=1e-4, l_0=0.5)


class TestMaterialModelProperties:
    """Tests for material model properties."""

    def test_lame_parameters_computation(self):
        """Test Lamé parameter computation."""
        E = 210.0
        nu = 0.3

        model = get_energy_model("spectral", E=E, nu=nu, plane_mode="strain")

        # Expected values
        mu_expected = E / (2.0 * (1.0 + nu))
        lambda_expected = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        assert jnp.allclose(
            model.lame_mu, mu_expected, rtol=1e-10
        ), "Shear modulus incorrect"
        assert jnp.allclose(
            model.lame_lambda, lambda_expected, rtol=1e-10
        ), "Lamé lambda incorrect"

    def test_energy_model_with_batched_strains(self):
        """Test energy computation with batched strain inputs."""
        model = get_energy_model("spectral", E=210.0, nu=0.3, plane_mode="strain")

        # Batch of strain tensors (E=2, G=3)
        strains = jnp.zeros((2, 3, 3, 3), dtype=jnp.float64)
        strains = strains.at[0, 0, 0, 0].set(0.01)
        strains = strains.at[1, 1, 1, 1].set(-0.005)

        e_pos, e_neg = model.energy_parts(strains)

        # Output should have shape (E, G)
        assert e_pos.shape == (2, 3), f"Expected shape (2, 3), got {e_pos.shape}"
        assert e_neg.shape == (2, 3), f"Expected shape (2, 3), got {e_neg.shape}"

        # All values should be non-negative and finite
        assert jnp.all(e_pos >= 0), "Positive energy should be ≥ 0"
        assert jnp.all(e_neg >= 0), "Negative energy should be ≥ 0"
        assert jnp.all(jnp.isfinite(e_pos)), "Positive energy should be finite"
        assert jnp.all(jnp.isfinite(e_neg)), "Negative energy should be finite"
