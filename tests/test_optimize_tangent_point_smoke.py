import numpy as np

from src.curvepack.tangent_point.optimize_tangent_point import (
    optimize_curves_tangent_point,
)


def test_optimize_tangent_point_smoke_no_nans() -> None:
    m = 8
    X0 = np.zeros((1, m, 2), dtype=np.float32)
    X0[0, :, 0] = np.linspace(0.0, 1.0, m, dtype=np.float32)

    X1, last_loss = optimize_curves_tangent_point(
        X0,
        steps=1,
        lr=1e-2,
        weight_decay=0.0,
        closed=False,
        ignore_k=2,
        len_final_mul=1.2,
        len_rho=1.0,
        seed=0,
        log_every=1000,
        checkpoint_every=None,
        checkpoint_svgs=False,
        checkpoint_csv=False,
    )

    assert X1.shape == X0.shape
    assert np.isfinite(X1).all()
    assert np.isfinite(float(last_loss))
