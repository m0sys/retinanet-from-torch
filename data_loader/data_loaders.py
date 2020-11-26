from torchvision import datasets, transforms
from fastai.data.external import URLs
from fastai.data.transforms import Categorize, parent_label
from fastai.vision.all import (
    imagenet_stats,
    PILImage,
    URLs,
    untar_data,
    get_image_files,
    GrandparentSplitter,
    Datasets,
    ToTensor,
    RandomResizedCrop,
    IntToFloatTensor,
    Normalize,
)

from base import BaseDataLoader

from utils.imagenette_utils import lbl_dict


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class ImagenetteDataLoaders:
    """FastAI dataloaders for Imagenette dataset."""

    def __init__(self, batch_size, num_workers=8):
        source = untar_data(URLs.IMAGENETTE)
        fnames = get_image_files(source)
        splits = GrandparentSplitter(valid_name="val")(fnames)
        dsets = Datasets(
            fnames,
            [[PILImage.create], [parent_label, lbl_dict.__getitem__, Categorize]],
            splits=splits,
        )

        item_tfms = [ToTensor, RandomResizedCrop(224, min_scale=0.35)]
        batch_tfms = [IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dls = dsets.dataloaders(
            after_item=item_tfms,
            after_batch=batch_tfms,
            bs=batch_size,
            num_workers=num_workers,
        )

    def __len__(self):
        return len(self.dls.loaders[0])

    def __iter__(self):
        return iter(self.dls.loaders[0])

    def split_validation(self):
        return self.dls.valid
