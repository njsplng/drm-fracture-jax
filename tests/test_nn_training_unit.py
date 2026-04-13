"""Unit tests for training utilities.

Tests for EarlyStopping class.
"""

from train_networks import EarlyStopping


class TestEarlyStopping:
    """Tests for the EarlyStopping class."""

    def test_early_stopping_relative(self) -> None:
        """Test 6.18: Feed improving -> stagnant losses, verify trigger. Reset, verify clean state."""
        patience = 3
        relative_threshold = 1e-4

        early_stop = EarlyStopping(
            patience=patience, relative_threshold=relative_threshold, mode="relative"
        )

        # Feed improving losses
        improving_losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in improving_losses:
            should_stop = early_stop.update(loss)
            assert not should_stop, "Should not stop on improving losses"

        # Verify best_loss was updated
        assert early_stop.best_loss == 0.6, "Best loss should be 0.6"
        assert early_stop.counter == 0, "Counter should be 0 after improvement"

        # Feed stagnant losses (improvement < threshold)
        stagnant_losses = [
            0.59999,
            0.59998,
            0.59997,
            0.59996,
        ]  # Very small improvements
        for loss in stagnant_losses:
            should_stop = early_stop.update(loss)
            if early_stop.counter >= patience:
                assert should_stop, "Should stop after patience stagnant updates"
                break
        else:
            # If we didn't break, check that early_stop flag is set
            assert early_stop.early_stop, "Early stop should be triggered"

        # Reset and verify clean state
        early_stop.reset()
        assert early_stop.best_loss is None, "Best loss should be None after reset"
        assert early_stop.counter == 0, "Counter should be 0 after reset"
        assert not early_stop.early_stop, "Early stop should be False after reset"

    def test_early_stopping_absolute(self) -> None:
        """Test 6.19: Feed losses improving by less than threshold, verify trigger after patience."""
        patience = 3
        absolute_threshold = 1e-6

        early_stop = EarlyStopping(
            patience=patience, absolute_threshold=absolute_threshold, mode="absolute"
        )

        # Initial loss - first update is "burned" (to_burn=1 by default)
        early_stop.update(1.0)
        assert early_stop.best_loss is None, "Best loss should be None after burn epoch"

        # Second update sets the best_loss
        early_stop.update(1.0)
        assert early_stop.best_loss == 1.0, "Best loss should be 1.0"

        # Feed losses improving by less than threshold (each improvement < 1e-6)
        # Improvement from 1.0 to 0.9999999 = 1e-7 < 1e-6
        small_improvements = [0.9999999, 0.9999998, 0.9999997, 0.9999996]
        for loss in small_improvements:
            should_stop = early_stop.update(loss)
            # Each improvement is 1e-7 < threshold of 1e-6, so counter increments
            if early_stop.counter >= patience:
                assert should_stop, "Should stop after patience small improvements"
                break
        else:
            assert early_stop.early_stop, "Early stop should be triggered"

    def test_early_stopping_no_stop_on_large_improvement(self) -> None:
        """Verify early stopping doesn't trigger on large improvements."""
        patience = 2
        absolute_threshold = 1e-6

        early_stop = EarlyStopping(
            patience=patience, absolute_threshold=absolute_threshold, mode="absolute"
        )

        # Initial loss
        early_stop.update(1.0)

        # Large improvement (much greater than threshold)
        early_stop.update(0.5)
        assert early_stop.counter == 0, "Counter should reset on large improvement"
        assert early_stop.best_loss == 0.5, "Best loss should update"

        # Another large improvement
        early_stop.update(0.1)
        assert early_stop.counter == 0, "Counter should stay at 0"
        assert early_stop.best_loss == 0.1, "Best loss should update again"
