import pdb
from typing import List
import torch
from torch import Tensor

from layers.wrappers import nonzero_tuple


class Matcher:
    """
    This class assigns to each prediction "element" (e.g., a box) a
    ground-truth element. Each predicted element will have exactly zero
    or one matches. Each ground-truth element may be matched to zero or
    more predicted elements.

    The match is determined by the MxN `match_quality_matrix`, that
    characterizes how well each (ground-truth, prediction) pair match.

    i.e. In the case of boxes we can use the IOU between pairs.

    The matcher returns:
        1. A vector of length N containing the index of the ground-truth
           element m in [0, M) that matches to prediction n in [0, N).

        2. A vector of length N containing the labels for each prediction.
    """

    def __init__(
        self,
        thresholds: List[float],
        labels: List[int],
        allow_low_quality_matches: bool = False,
    ):
        """
        Args:
            thresholds: a list of thresholds used to stratify predictions
                into levels.
            labels: a list of values to lable predictions belonging at each level.
                A label can be one of {-1, 0, 1} signifying {ignore, negative class, positive class},
                respectively.
            allow_low_quality_matches: if True, produce additional matches for predictions
                with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.
        """
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))

        # Currently torchscript does not support all pos generator.
        assert all(
            [low <= high] for (low, high) in zip(thresholds[:-1], thresholds[1:])
        )
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor containing the pairwise quality between M
                ground-truth elements and N predicted elements. All elements must be >= 0
                (due to the use of `torch.nonzero` in `set_low_quality_matches_` methods.)
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M).
            match_labels (Tensor[int8]): a vector of length N where match_labels[i] indicates
                whether a prediction is a true or false positive or ignored.
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            pdb.set_trace(header="matcher.py -> __call__: match qual is numel!")
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )

            # No gt boxes exits. So set labels to `self.labels[0]` which is usally background.
            # To ignore instead make labels=[-1, 0, -1, 1].
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )

            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(
            self.labels, self.thresholds[:-1], self.thresholds[1:]
        ):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l
            ## pdb.set_trace()

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low_quality matches.

        Specifically, for each ground-truth label element find the set of predictions that have
        maximum overlap with it and set them to match ground-truth if unmatched previously.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )

        match_labels[pred_inds_with_highest_quality] = 1
