from torchvision import datasets, transforms
from fastai.torch_basics import set_seed
from fastai.data.block import DataBlock
from fastai.vision.core import get_annotations
from fastai.vision.data import BBoxBlock, BBoxLblBlock, ImageBlock
from fastai.data.external import URLs
from fastai.data.transforms import Categorize, RandomSplitter, parent_label
from fastai.vision.all import (
    imagenet_stats,
    PILImage,
    URLs,
    untar_data,
    get_image_files,
    get_annotations,
    GrandparentSplitter,
    Pipeline,
    Datasets,
    ToTensor,
    RandomResizedCrop,
    IntToFloatTensor,
    Normalize,
    Resize,
    Flip,
    DataBlock,
    ImageBlock,
    BBoxLblBlock,
    BBoxBlock,
    RandomSplitter,
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


def load_imagenette160_dls(image_size=224, bs=64):
    source = untar_data(URLs.IMAGENETTE_160)
    fnames = get_image_files(source)
    tfm = Pipeline(
        [parent_label, lbl_dict.__getitem__, Categorize(vocab=lbl_dict.values())]
    )
    splits = GrandparentSplitter(valid_name="val")(fnames)
    dsets = Datasets(fnames, [[PILImage.create], tfm], splits=splits)
    item_tfms = [ToTensor, RandomResizedCrop(image_size, min_scale=0.35)]
    batch_tfms = [IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]
    return dsets.dataloaders(
        after_item=item_tfms, after_batch=batch_tfms, bs=bs, num_workers=1
    )


def load_sample_coco_dls(img_size=512, bs=32, seed=None):
    # For reproducibility purposes.
    if seed is not None:
        set_seed(seed, True)

    path = untar_data(URLs.COCO_SAMPLE)
    imgs, lbl_bbox = get_annotations(path / "annotations/train_sample.json")
    img2bbox = dict(zip(imgs, lbl_bbox))
    getters = [
        lambda o: path / "train_sample" / o,
        lambda o: img2bbox[o][0],
        lambda o: img2bbox[o][1],
    ]
    item_tfms = [
        Resize(img_size, method="pad"),
    ]
    batch_tfms = [Flip(), Normalize.from_stats(*imagenet_stats)]
    sample_coco = DataBlock(
        blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
        splitter=RandomSplitter(seed=seed),
        get_items=lambda noop: imgs,
        getters=getters,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        n_inp=1,
    )
    return sample_coco.dataloaders(path / "train_sample", bs=bs)
