import unittest

import numpy as np
import pandas as pd
import torch

from src.data.lag_injection import inject_lag_into_dataframe
from src.models.delay_alignment import DelayAlignment
from src.models.dimf import DIMF
from src.models.stda_lag_identifier import STDALagIdentifier, lag_identifier_loss
from src.postprocess.viterbi_lag_decoder import viterbi_decode_lag


class LagGroundedSmokeTest(unittest.TestCase):
    def _frame(self, n=160):
        rng = np.random.default_rng(7)
        t = pd.date_range("2024-01-01", periods=n, freq="15min")
        data = {"TimeStamp": t, "yield_flow": rng.normal(size=n)}
        for prefix in ["feed_", "stage1_", "stage2_", "stage3_"]:
            for idx in range(3):
                data[f"{prefix}{idx}"] = rng.normal(size=n).cumsum()
        return pd.DataFrame(data)

    def test_lag_injection_summary_and_soft_labels(self):
        result = inject_lag_into_dataframe(
            self._frame(),
            time_col="TimeStamp",
            target_cols=["yield_flow"],
            lag_injection_cfg={
                "enabled": True,
                "split_ratio": 0.8,
                "injection_ratio_train": 0.6,
                "injection_ratio_test": 0.6,
                "injection_granularity": "block",
                "max_lag": 6,
                "min_lag": 1,
                "random_seed": 11,
                "shapes": {"fixed": 1.0},
                "fixed_lags": [4],
                "inject_strength_range": [0.5, 0.5],
            },
        )
        meta = result.metadata
        valid_train = meta[(meta["split"] == "train") & (meta["valid_for_injection"] == 1)]
        valid_test = meta[(meta["split"] == "test") & (meta["valid_for_injection"] == 1)]
        self.assertAlmostEqual(valid_train["lag_flag"].mean(), 0.6, delta=0.08)
        self.assertAlmostEqual(valid_test["lag_flag"].mean(), 0.6, delta=0.12)
        soft_cols = [c for c in meta.columns if c.startswith("lag_soft_")]
        np.testing.assert_allclose(meta[soft_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-6)

    def test_lag_identifier_shapes_and_loss(self):
        model = STDALagIdentifier(d_source=4, d_target=3, max_lag=5, hidden_dim=8, num_layers=1)
        source = torch.randn(6, 12, 4)
        target = torch.randn(6, 12, 3)
        out = model(source, target_seq=target)
        self.assertEqual(tuple(out["pi_lag"].shape), (6, 4, 6))
        self.assertEqual(tuple(out["pi_edge"].shape), (6, 6))
        gt = torch.zeros(6, 6)
        gt[:, 2] = 1.0
        losses = lag_identifier_loss(out, gt, torch.ones(6))
        self.assertTrue(torch.isfinite(losses["loss"]))

        window_model = STDALagIdentifier(
            d_source=4,
            d_target=3,
            max_lag=5,
            hidden_dim=8,
            num_layers=1,
            use_candidate_window_encoder=True,
            lag_window_radius=2,
        )
        window_out = window_model(source, target_seq=target)
        self.assertEqual(tuple(window_out["pi_edge"].shape), (6, 6))
        self.assertEqual(tuple(window_out["expected_edge"].shape), (6,))
        self.assertTrue(bool(window_out["lag_candidate_valid"][5].item()))
        window_losses = lag_identifier_loss(
            window_out,
            gt,
            torch.ones(6),
            segment_id=torch.tensor([0, 0, 0, 1, 1, 1]),
            lag_gt=torch.full((6,), 2),
        )
        self.assertTrue(torch.isfinite(window_losses["loss"]))

    def test_delay_alignment_accepts_soft_prior(self):
        align = DelayAlignment(dim=8, attn_dim=8, L_max=5)
        down = torch.randn(4, 10, 8)
        up = torch.randn(4, 10, 8)
        prior = torch.zeros(4, 6)
        prior[:, 3] = 1.0
        msg, pi, raw = align.forward_seq(
            down,
            up,
            pi_prior=prior,
            lambda_prior=1.0,
            prior_mode="soft_distribution",
        )
        self.assertEqual(tuple(msg.shape), (4, 10, 8))
        self.assertEqual(tuple(pi.shape), (4, 10, 6))
        self.assertEqual(tuple(raw.shape), (4, 10, 8))

    def test_dimf_accepts_feature_level_prior(self):
        model = DIMF(
            group_dims={"feed": 2, "stage1": 3, "stage2": 4, "stage3": 2},
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            attn_dim=8,
            L_max=5,
            lead_steps=1,
            encoder_type="gru",
        )
        x = {
            "feed": torch.randn(3, 12, 2),
            "stage1": torch.randn(3, 12, 3),
            "stage2": torch.randn(3, 12, 4),
            "stage3": torch.randn(3, 12, 2),
        }
        prior = torch.zeros(3, 3, 6)
        prior[:, :, 2] = 1.0
        y_hat, pi = model(
            x,
            delay_priors={
                "stage1_to_stage2": {
                    "pi_prior": prior,
                    "prior_mode": "soft_distribution",
                    "lambda_prior": 0.5,
                }
            },
        )
        self.assertEqual(tuple(y_hat.shape), (3,))
        self.assertEqual(tuple(pi["stage1_to_stage2"].shape), (3, 12, 6))

    def test_dimf_can_own_lag_guided_prior_generator(self):
        model = DIMF(
            group_dims={"feed": 2, "stage1": 3, "stage2": 4, "stage3": 2},
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            attn_dim=8,
            L_max=5,
            lead_steps=1,
            encoder_type="gru",
        )
        identifier = STDALagIdentifier(d_source=3, d_target=4, max_lag=5, hidden_dim=8, num_layers=1)
        model.attach_lag_guided_prior_generator(
            identifier,
            edge_name="stage1_to_stage2",
            source_stage="stage1",
            target_stage="stage2",
            feature_mask=torch.tensor([True, False, True]),
            lambda_prior=0.5,
            prior_mode="soft_distribution",
        )
        x = {
            "feed": torch.randn(3, 12, 2),
            "stage1": torch.randn(3, 12, 3),
            "stage2": torch.randn(3, 12, 4),
            "stage3": torch.randn(3, 12, 2),
        }
        y_hat, pi, lag_out = model.forward_with_lag_guided_alignment(x)
        self.assertEqual(tuple(y_hat.shape), (3,))
        self.assertEqual(tuple(pi["stage1_to_stage2"].shape), (3, 12, 6))
        self.assertEqual(tuple(lag_out["pi_edge"].shape), (3, 6))
        self.assertIn("lag_identifier.lag_bias", model.state_dict())
        prior = model.build_lag_guided_delay_priors(lag_out)["stage1_to_stage2"]["pi_prior"]
        torch.testing.assert_close(prior[:, 1, :], torch.full_like(prior[:, 1, :], 1.0 / 6.0))

    def test_viterbi_respects_segment_boundaries(self):
        pred = np.full((3, 6), 1e-4, dtype=np.float64)
        pred[0, 5] = 0.9995
        pred[1, 5] = 0.9995
        pred[2, 0] = 0.9995
        no_segment = viterbi_decode_lag(
            pred,
            smooth_lambda=0.0,
            switch_penalty=0.0,
            pos_to_zero_penalty=20.0,
        )
        segmented = viterbi_decode_lag(
            pred,
            segment_id=np.asarray([0, 0, 1]),
            smooth_lambda=0.0,
            switch_penalty=0.0,
            pos_to_zero_penalty=20.0,
        )
        self.assertGreater(int(no_segment[-1]), 0)
        self.assertEqual(int(segmented[-1]), 0)


if __name__ == "__main__":
    unittest.main()
