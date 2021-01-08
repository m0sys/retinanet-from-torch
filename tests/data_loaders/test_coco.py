import unittest
import pytest
import torch

from data_loader.data_loaders import load_sample_coco_dls


class TestCocoDataset(unittest.TestCase):
    @pytest.mark.skip("Run if you have time to kill.")
    def test_load_sample_coco_dls_is_not_nan(self):
        dls = load_sample_coco_dls()

        for batch in dls[0]:
            assert not torch.isnan(batch[0]).all()
            assert not torch.isnan(batch[1]).all()
