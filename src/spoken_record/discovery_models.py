"""Settings for passage discovery. Similarity scores don't say that something is true."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DiscoveryConfig(BaseModel):
    """Pins the encoder. Kept separate from the entity-linking settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    repo_id: Literal["sentence-transformers/all-mpnet-base-v2"] = "sentence-transformers/all-mpnet-base-v2"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    cache_dir: str | None = None
    offline_only: Literal[True] = True
    device: Literal["cpu"] = "cpu"
    max_tokens: int = Field(default=384, ge=8, le=384)
    batch_size: int = Field(default=8, ge=1, le=64)
    dimension: Literal[768] = 768
    pooling: Literal["token-weighted-mean-of-normalized-chunks"] = "token-weighted-mean-of-normalized-chunks"
    top_k: int = Field(default=3, ge=1, le=20)
    minimum_cosine: float = Field(default=.60, ge=-1, le=1)
    minimum_jaccard: float = Field(default=.05, gt=0, le=1)
    cross_episode_only: Literal[True] = True

