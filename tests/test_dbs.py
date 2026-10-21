import math
import unittest

import torch

from benchmark import DBSMLP


class LayerBalancedDBSTests(unittest.TestCase):
    def test_zero_biases_and_reproducible_quantized_weights(self):
        a = DBSMLP(16, torch.device("cpu"), seed=7)
        b = DBSMLP(16, torch.device("cpu"), seed=7)
        self.assertTrue(torch.equal(a.q, b.q))
        self.assertEqual(a.q.dtype, torch.uint8)
        self.assertEqual(a.evidence.dtype, torch.int16)
        weights = a.values()
        self.assertTrue(torch.count_nonzero(weights[0]) > 0)
        self.assertTrue(torch.count_nonzero(weights[2]) > 0)
        self.assertTrue(torch.count_nonzero(weights[1]) == 0)
        self.assertTrue(torch.count_nonzero(weights[3]) == 0)

    def test_block_visits_every_layer_without_duplicate_indices(self):
        model = DBSMLP(16, torch.device("cpu"), block=256)
        seen = torch.zeros_like(model.q, dtype=torch.bool)
        for _ in range(150):
            idx = model.block_indices(torch.device("cpu"))
            self.assertEqual(idx.numel(), 256)
            self.assertEqual(torch.unique(idx).numel(), idx.numel())
            self.assertTrue((idx >= 0).all() and (idx < model.q.numel()).all())
            seen[idx] = True
        self.assertTrue(seen.all())

    def test_probe_step_keeps_integer_state_and_finite_outputs(self):
        model = DBSMLP(8, torch.device("cpu"), block=64, threshold=1, seed=3)
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        loss = model.train_step(x, y, probes=2)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(torch.isfinite(model.predict(x)).all())
        self.assertEqual(model.q.dtype, torch.uint8)
        self.assertGreaterEqual(int(model.q.min()), 0)
        self.assertLessEqual(int(model.q.max()), 255)

    def test_legacy_keeps_original_zero_mapping(self):
        model = DBSMLP(8, torch.device("cpu"), variant="legacy")
        q = torch.full_like(model.q, 127)
        self.assertAlmostEqual(float(model.values(q)[0].flatten()[0]),
                               127 * 2 / 255 - 1, places=6)

    def test_ablation_variants_change_only_the_selected_component(self):
        device = torch.device("cpu")
        legacy = DBSMLP(16, device, block=256, variant="legacy", seed=5)
        scaled = DBSMLP(16, device, block=256, variant="layer_scaled", seed=5)
        scheduled = DBSMLP(16, device, block=256, variant="layer_scheduled", seed=5)
        balanced = DBSMLP(16, device, block=256, variant="layer_balanced", seed=5)
        self.assertTrue(torch.equal(legacy.q, scheduled.q))
        self.assertTrue(torch.equal(scaled.q, balanced.q))
        self.assertTrue(torch.equal(legacy.block_indices(device), scaled.block_indices(device)))
        self.assertTrue(torch.equal(scheduled.block_indices(device), balanced.block_indices(device)))

    def test_residual_votes_preserve_overflow_and_reset_votes_do_not(self):
        idx = torch.tensor([0])
        for rule, expected in (("reset", 0), ("residual", 2)):
            model = DBSMLP(8, torch.device("cpu"), threshold=3, evidence_rule=rule)
            model.q[0] = 128
            model.apply_votes(idx, torch.tensor([5], dtype=torch.int16))
            self.assertEqual(int(model.q[0]), 129)
            self.assertEqual(int(model.evidence[0]), expected)
            self.assertEqual(model.moves, 1)

    def test_residual_votes_reset_at_clipped_boundary(self):
        model = DBSMLP(8, torch.device("cpu"), threshold=3, evidence_rule="residual")
        model.q[0] = 255
        model.apply_votes(torch.tensor([0]), torch.tensor([5], dtype=torch.int16))
        self.assertEqual(int(model.q[0]), 255)
        self.assertEqual(int(model.evidence[0]), 0)

    def test_residual_votes_work_in_negative_direction(self):
        model = DBSMLP(8, torch.device("cpu"), threshold=3, evidence_rule="residual")
        model.q[0] = 128
        model.apply_votes(torch.tensor([0]), torch.tensor([-5], dtype=torch.int16))
        self.assertEqual(int(model.q[0]), 127)
        self.assertEqual(int(model.evidence[0]), -2)


if __name__ == "__main__":
    unittest.main()
