"""
MIT License

Copyright (c) 2021 Valentino Maiorca, Luca Moschella

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code is adopted from Moschella et al.'s work: 
https://github.com/lucmos/relreps?tab=readme-ov-file, which is under MIT License.
"""

import torch
import random
from typing import *
from dataUtils import data
import torch.nn.functional as TF
from pytorch_lightning import seed_everything

def relative_projection(x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Compute the relative representation of x with the cosine similarity

    Args:
        x: the samples absolute latents [batch, hidden_dim]
        anchors: the anchors absolute latents [anchors, hidden_dim]

    Returns:
        the relative representation of x. The relative representation is *not* normalized,
        when training on relative representation it is useful to normalize it
    """
    x = TF.normalize(x, p=2, dim=-1)
    anchors = TF.normalize(anchors, p=2, dim=-1)
    return torch.einsum("bm, am -> ba", x, anchors)

class LatentSpace:
    def __init__(
        self,
        encoding_type: str,
        encoder_name: str,
        vectors: torch.Tensor,
        ids: Sequence[int],
    ):
        """Utility class to represent a generic latent space

        Args:
            encoding_type: the type of latent space, i.e. "absolute" or "relative" usually
            encoder_name: the name of the encoder used to obtain the vectors
            vectors: the latents that compose the latent space
            ids: the ids associated with the vectors
        """
        assert vectors.shape[0] == len(ids)

        self.encoding_type: str = encoding_type
        self.vectors: torch.Tensor = vectors
        self.ids: Sequence[int] = ids
        self.encoder_name: str = encoder_name

    def get_anchors(
        self, 
        anchor_choice: str, 
        num_anchors: int, 
        seed: int
    ) -> Sequence[int]:
        """Adopt some strategy to select the anchors.

        Args:
            anchor_choice: the selection strategy for the anchors
            seed: the random seed to use

        Returns:
            the ids of the chosen anchors
        """
        # Select anchors
        seed_everything(seed)
        anchor_set = []
        if anchor_choice == "uniform":
            while (len(anchor_set) != num_anchors):
              anchor_id = random.randint(0, len(self.ids))

              if anchor_id in anchor_set:
                # if the selected anchor is already in the set, pass
                continue
              elif anchor_id in en_index and anchor_id in zh_index:
                # if the selected anchor is in training set include it
                anchor_set.append(anchor_id)
              else:
                continue
        else:
            assert NotImplementedError

        result = sorted(anchor_set)
        return result

    def to_relative(
        self, 
        anchor_choice: str = None, 
        seed: int = None, 
        anchors: Optional[Sequence[int]] = None
    ) -> "RelativeSpace":
        """Compute the relative transformation on the current space returning a new one.

        Args:
            anchor_choice: the anchors selection strategy to use, if no anchors are provided
            seed: the random seed to use
            anchors: the ids of the anchors to use

        Returns:
            the RelativeSpace associated to the current LatentSpace
        """
        assert self.encoding_type != "relative"  # TODO: for now
        anchors = self.get_anchors(anchor_choice=anchor_choice, seed=seed) if anchors is None else anchors

        anchor_latents: torch.Tensor = self.vectors[anchors]

        relative_vectors = relative_projection(x=self.vectors, anchors=anchor_latents.cpu())

        return RelativeSpace(vectors=relative_vectors, encoder_name=self.encoder_name, anchors=anchors, ids=self.ids)

class RelativeSpace(LatentSpace):
    def __init__(
        self,
        vectors: torch.Tensor,
        ids: Sequence[int],
        anchors: Sequence[int],
        encoder_name: str = None,
    ):
        """Utility class to represent a relative latent space

        Args:
            vectors: the latents that compose the latent space
            ids: the ids associated ot the vectors
            encoder_name: the name of the encoder_name used to obtain the vectors
            anchors: the ids associated to the anchors to use
        """
        super().__init__(encoding_type="relative", vectors=vectors, encoder_name=encoder_name, ids=ids)
        self.anchors: Sequence[int] = anchors


def create_latent_space():
    """Create latent space from absolute embeddings"""

    NUM_SAMPLES_EN = data.vocab_size_en
    en_anchors_ids = data.anchors["en"]
    en_abs_embedding = torch.FloatTensor(data.en_embedding)

    en_abs_latent_space = LatentSpace(
        encoding_type="absolute",
        encoder_name="english abs embedding",
        vectors=en_abs_embedding,
        ids=list(range(NUM_SAMPLES_EN)),
    )

    en_rel_latent_space = en_abs_latent_space.to_relative(anchors=en_anchors_ids)

    NUM_SAMPLES_ZH = data.vocab_size_zh
    zh_anchors_ids = data.anchors["zh"]
    zh_abs_embedding = torch.FloatTensor(data.zh_embedding)

    zh_abs_latent_space = LatentSpace(
        encoding_type="absolute",
        encoder_name="chinese abs embedding",
        vectors=zh_abs_embedding,
        ids=list(range(NUM_SAMPLES_ZH)),
    )

    zh_rel_latent_space = zh_abs_latent_space.to_relative(anchors=zh_anchors_ids)
    return en_rel_latent_space, zh_rel_latent_space
