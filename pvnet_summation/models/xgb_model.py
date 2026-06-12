"""XGBoost-based summation model"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from safetensors.torch import save_file

from pvnet_summation.data.datamodule import SumTensorBatch
from pvnet_summation.models.base_model import BaseModel
from pvnet_summation.utils import PYTORCH_WEIGHTS_NAME


class XGBModel(BaseModel):
    """XGBoost-based summation model.

    Flattens PVNet outputs to [batch*horizon, locs*(quantiles)] and trains one
    XGBoost model per output quantile (or a single regressor if no quantiles).
    Fits once on epoch 0 via the Lightning module's accumulation loop.
    """

    def __init__(
        self,
        output_quantiles: list[float] | None,
        num_input_locations: int,
        input_quantiles: list[float] | None,
        history_minutes: int,
        forecast_minutes: int,
        interval_minutes: int,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        predict_difference_from_sum: bool = False,
    ):
        """XGBoost-based summation model.

        Args:
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            num_input_locations: The number of input locations (e.g. number of GSPs)
            input_quantiles: A list of float (0.0, 1.0) quantiles which PVNet predicts for. If set
                to None we assume PVNet predicts a single value
            history_minutes: Length of the GSP history period in minutes
            forecast_minutes: Length of the GSP forecast period in minutes
            interval_minutes: The interval in minutes between each timestep in the data
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training data per tree
            colsample_bytree: Subsample ratio of features per tree
            predict_difference_from_sum: Whether to predict the difference from the sum of
                locations, else the total is predicted directly
        """
        super().__init__(
            output_quantiles,
            num_input_locations,
            input_quantiles,
            history_minutes,
            forecast_minutes,
            interval_minutes,
        )

        self.predict_difference_from_sum = predict_difference_from_sum
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        # One booster per output quantile (or one for mean regression)
        self.boosters: list[xgb.Booster] = []
        self._is_fitted = False

        # Dummy parameter so Lightning doesn't complain about no parameters
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def _make_booster_params(self, alpha: float | None = None) -> dict:
        """Return XGBoost params dict."""
        params = {
            "tree_method": "hist",
            "device": "cpu",
            "nthread": 1,
            "verbosity": 0,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
        if alpha is not None:
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = alpha
        else:
            params["objective"] = "reg:absoluteerror"
        return params

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all boosters on accumulated data."""
        self.boosters = []
        dtrain = xgb.DMatrix(X, label=y)
        if self.use_quantile_regression:
            for q in self.output_quantiles:
                booster = xgb.train(
                    self._make_booster_params(alpha=q),
                    dtrain,
                    num_boost_round=self.n_estimators,
                )
                self.boosters.append(booster)
        else:
            booster = xgb.train(
                self._make_booster_params(),
                dtrain,
                num_boost_round=self.n_estimators,
            )
            self.boosters.append(booster)
        self._is_fitted = True

    def _batch_to_numpy(self, x: SumTensorBatch) -> tuple[np.ndarray, np.ndarray]:
        """Convert a batch to (X, y) numpy arrays shaped [batch*horizon, features]."""
        # pvnet_outputs: [batch, locs, horizon, (quantiles)]
        pv = x["pvnet_outputs"]
        batch_size = pv.shape[0]
        pv = torch.swapaxes(pv, 1, 2)
        pv = torch.flatten(pv, start_dim=2)

        # Add horizon encoding
        horizon_enc = torch.linspace(0, 1, self.forecast_len, device=pv.device, dtype=pv.dtype)
        horizon_enc = horizon_enc.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        pv = torch.cat([pv, horizon_enc], dim=2)

        X = pv.flatten(0, 1).detach().cpu().numpy()
        y = x["target"].detach().cpu().numpy()

        if self.predict_difference_from_sum:
            loc_sum = self.sum_of_locations(x).detach().cpu().numpy()
            y = y - loc_sum

        y = y.reshape(-1)
        return X, y


    def forward(self, x: SumTensorBatch) -> torch.Tensor:
        """Run model forward"""
        X, _ = self._batch_to_numpy(x)
        batch_size = x["pvnet_outputs"].shape[0]

        if not self._is_fitted:
            if self.use_quantile_regression:
                return torch.zeros(batch_size, self.forecast_len, len(self.output_quantiles))
            return torch.zeros(batch_size, self.forecast_len)

        dtest = xgb.DMatrix(X)
        preds = np.stack([b.predict(dtest) for b in self.boosters], axis=-1)

        if self.use_quantile_regression:
            out = preds.reshape(batch_size, self.forecast_len, len(self.output_quantiles))
        else:
            out = preds.reshape(batch_size, self.forecast_len, 1).squeeze(-1)

        out = torch.tensor(out, dtype=torch.float32, device=self._dummy.device)

        if self.predict_difference_from_sum:
            loc_sum = self.sum_of_locations(x)
            if self.use_quantile_regression:
                loc_sum = loc_sum.unsqueeze(-1)
            out = loc_sum + out

        return F.leaky_relu(out, negative_slope=0.01)


    def _save_model_weights(self, save_directory: str) -> None:
        """Save boosters to pickle alongside the standard weights file."""
        save_file({"_dummy": self._dummy}, f"{save_directory}/{PYTORCH_WEIGHTS_NAME}")
        with open(f"{save_directory}/boosters.pkl", "wb") as f:
            pickle.dump(self.boosters, f)

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str, **kwargs) -> "XGBModel":
        """Load pretrained model weights and boosters from a local directory or HuggingFace."""
        model = super().from_pretrained(model_id, revision, **kwargs)
        booster_path = (
            f"{model_id}/boosters.pkl"
            if Path(model_id).is_dir()
            else None
        )
        if booster_path and Path(booster_path).exists():
            with open(booster_path, "rb") as f:
                model.boosters = pickle.load(f)
            model._is_fitted = True
        return model
