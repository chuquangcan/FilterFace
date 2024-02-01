from typing import Any, Dict, Optional

import pyrootutils as pyrootutils
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

import os
import numpy as np
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw


class DlibDatasetBox(Dataset):
    def __init__(self, data_dir, xml_file):
        self.data_dif = data_dir
        self.samples = self._load_data(data_dir, xml_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['filename']
        box_top: int = sample['box_top']
        box_left: int = sample['box_left']
        box_width: int = sample['box_width']
        box_height: int = sample['box_height']
        box = np.array(
            [[box_left, box_top], [box_left + box_width, box_top], [box_left + box_width, box_top + box_height],
             [box_left, box_top + box_height]])
        original_image: Image = Image.open(os.path.join(self.data_dif, filename)).convert('RGB')

        return original_image, box  # chua chuan hoa

    def _load_data(self, data_dif: str, xml_file: str):
        # lay du lieu tu file xml (anh va keypoint)
        xml_path = os.path.join(data_dif, xml_file)
        root = ET.parse(xml_path).getroot()
        samples = root.find('images')
        samples = [self._get_labeled_sample(sample) for sample in samples]
        return samples

    def _get_labeled_sample(self, sample: ET.Element) -> Dict:
        filename = sample.attrib['file']
        width = int(sample.attrib['width'])
        height = int(sample.attrib['height'])

        box = sample.find('box')
        box_top = int(box.attrib['top'])
        box_left = int(box.attrib['left'])
        box_width = int(box.attrib['width'])
        box_height = int(box.attrib['height'])

        return dict(
            filename=filename, width=width, height=height, box_top=box_top, box_left=box_left, box_width=box_width,
            box_height=box_height
        )

    # ve len anh (key point)
    @staticmethod
    def annotate_image(image: Image, box: np.ndarray) -> Image:
        draw = ImageDraw.Draw(image)
        draw.rectangle([box[0][0], box[0][1], box[2][0], box[2][1]], outline="red", width=2)
        return image


class TransformDataset(Dataset):
    def __init__(self, dataset: DlibDatasetBox, transfrom: Optional[Compose] = None):
        self.dataset = dataset
        if transfrom is not None:
            self.transform = transfrom
        else:
            self.transform = Compose([
                A.resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, box = self.dataset[idx]
        image = np.array(image)
        transformed = self.transform(image=image, keypoints=box)
        image, box = transformed['image'], transformed['keypoints']
        _, height, width = image.shape
        box = box / np.array([width, height]) - 0.5
        return image, box.astype(np.float32)

    @staticmethod
    def annotate_tensor(image: torch.Tensor, box: np.ndarray) -> Image:
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        # dao nguoc chuan hoa
        def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
            # 3, H, W, B
            ten = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

        images = denormalize(image)
        images_to_save = []
        for bx, img in zip(box, images):
            img = img.permute(1, 2, 0).numpy() * 255
            h, w, _ = img.shape
            bx = (bx + 0.5) * np.array([w, h])
            img = DlibDatasetBox.annotate_image(Image.fromarray(img.astype(np.uint8)), bx)
            images_to_save.append(torchvision.transforms.ToTensor()(img))
        return torch.stack(images_to_save)


class DlibDataModuleBox(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            data_train: DlibDatasetBox,
            data_test: DlibDatasetBox,
            data_dir: str = "C:\\Users\\ADMIN\\Downloads\\300W-20240111T021256Z-001\\300W",
            train_val_test_split=[5_600, 1_000],
            transform_train_box: Optional[Compose] = None,
            transform_val_box: Optional[Compose] = None,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # khai bao tham so va auto checkpoint
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None and self.data_val is None and self.data_test is None:
            data_train = self.hparams.data_train(
                data_dir=self.hparams.data_dir)
            data_test = self.hparams.data_test(
                data_dir=self.hparams.data_dir)
            data_train, data_val = random_split(
                dataset=data_train, lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = TransformDataset(data_train, self.hparams.transform_train_box)
            self.data_val = TransformDataset(data_val, self.hparams.transform_val_box)
            self.data_test = TransformDataset(data_test, self.hparams.transform_val_box)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: str) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    path = pyrootutils.find_root(search_from=__file__, indicator='.project-root')
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)


    def test_dataset(cfg: DictConfig):
        print("testing...")
        dataset: DlibDatasetBox = hydra.utils.instantiate(cfg.data_train)
        dataset = dataset(data_dir=cfg.data_dir)
        print("dataset", len(dataset))
        image, box = dataset[0]
        print("image", image.size, "box", box.shape)
        annotated_image = DlibDatasetBox.annotate_image(image, box)
        annotated_image.save(output_path / "test_dataset_result.png")


    def test_datamodule(cfg: DictConfig):
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        bx, by = next(iter(loader))
        print("n_batch", len(loader), bx.shape, by.shape, type(by))
        annotated_batch = TransformDataset.annotate_tensor(bx, by)
        print("annotated_batch", annotated_batch.shape)
        torchvision.utils.save_image(annotated_batch, output_path / "Test_datamodule_result.png")

        for bx, by in tqdm(datamodule.train_dataloader()):
            pass
        print("Train data pass")

        for bx, by in tqdm(datamodule.val_dataloader()):
            pass
        print("val data pass")

        for bx, by in tqdm(datamodule.test_dataloader()):
            pass
        print("test data pass")


    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib_databox.yaml")
    def main(cfg: DictConfig):
        # print(cfg)
        test_dataset(cfg)
        test_datamodule(cfg)


    main()
