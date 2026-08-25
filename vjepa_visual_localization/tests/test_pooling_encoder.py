from __future__ import annotations

import numpy as np

from src.models.pooling import l2_normalize, mean_pool_tokens
from tests.helpers import MeanColorEncoder


def test_pooling_and_normalization_shapes() -> None:
    tokens = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    pooled = l2_normalize(mean_pool_tokens(tokens))
    assert pooled.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(pooled, axis=-1), 1.0, atol=1e-6)
    assert np.isfinite(pooled).all()


def test_identical_clips_have_identical_embeddings() -> None:
    encoder = MeanColorEncoder()
    clip = np.full((4, 16, 16, 3), [20, 80, 150], dtype=np.uint8)
    output = encoder.encode_video(np.stack([clip, clip]))
    assert output.global_embedding.shape == (2, 3)
    assert output.local_tokens is not None and output.local_tokens.shape == (2, 4, 3)
    np.testing.assert_allclose(output.global_embedding[0], output.global_embedding[1])
    np.testing.assert_allclose(np.linalg.norm(output.global_embedding, axis=-1), 1.0)
