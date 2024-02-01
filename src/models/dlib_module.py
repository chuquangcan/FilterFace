import os
from typing import Any, List

import hydra.utils
import numpy as np
import torch
import torchvision
from PIL import Image
from PIL import ImageDraw
from pytorch_lightning import LightningModule
import wandb
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mae import MeanAbsoluteError

from src.models.components.res_net_18 import ResNet as ResNet18

output_path = "C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\outputs"

def draw_batch(images, targets, preds) -> torch.Tensor:
    # helper function
    def denormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
        """Reverse COLOR transform"""
        # clone: make a copy
        # permute: [batch, 3, H, W] -> [3, H, W, batch]
        tmp = images.clone().permute(1, 2, 3, 0)

        # denormalize
        for t, m, s in zip(tmp, mean, std):
            t.mul_(s).add_(m)

        # clamp: limit value to [0, 1]
        # permute: [3, H, W, batch] -> [batch, 3, H, W]
        return torch.clamp(tmp, 0, 1).permute(3, 0, 1, 2)

    def annotate_image(image, targets, preds):
        """Draw target & pred landmarks on image"""
        # create an ImageDraw object
        draw = ImageDraw.Draw(image)
        if (targets.shape[0] == 68):
            # draw target_landmarks on image (green)
            for x, y in targets:
                draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(0, 255, 0))

            # draw pred_landmarks on image (red)
            for x, y in preds:
                draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))
        else:
            if preds[2][0] >= preds[0][0] and preds[0][1] <= preds[2][1]:
                draw.rectangle([preds[0][0], preds[0][1], preds[2][0], preds[2][1]], outline="red", width=2)
            else:
                for x, y in preds:
                    draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))
            draw.rectangle([targets[0][0], targets[0][1], targets[2][0], targets[2][1]], outline="green", width=2)

        return image

    # denormalize
    images = denormalize(images)

    # set an empty list
    images_to_save = []

    # loop through each sample in batch
    for i, t, p in zip(images, targets, preds):
        # get size of x
        i = i.cpu()
        t = t.cpu()
        p = p.cpu()
        img = i.permute(1, 2, 0).numpy() * 255
        height, width, color_channels = img.shape

        # denormalize landmarks -> pixel coordinates
        t = (t + 0.5) * np.array([width, height])
        p = (p + 0.5) * np.array([width, height])

        # draw landmarks on cropped image
        annotated_image = annotate_image(Image.fromarray(img.astype(np.uint8)), t, p)

        # save drawed cropped image
        images_to_save.append(torchvision.transforms.ToTensor()(annotated_image))

    return torch.stack(images_to_save)

class DlibLitModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
            - Computations (init)
            - Train loop (training_step)
            - Validation loop (validation_step)
            - Test loop (test_step)
            - Prediction Loop (predict_step)
            - Optimizers and LR Schedulers (configure_optimizers)

        Docs:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
        """

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler):
        super().__init__()

        # truy cap vao tham so tru module cua ham va tao checkpoint
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.validation_step_outputs=[]

        self.net = net

        # ham loss
        self.criterion = torch.nn.MSELoss()

        # tinh toan accuracy
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # tinh toan loss
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # luu vet accuracy cua valid
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.log("train/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append({"image": inputs, "targets": targets, "preds": preds})
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_mae.compute()
        self.val_mae_best(acc)
        self.log("val_mae_best", self.val_mae_best.compute(), prog_bar=True, sync_dist=True)
        first_test_val = self.validation_step_outputs[0]
        inputs = first_test_val["image"]
        targets = first_test_val["targets"]
        preds = first_test_val["preds"]

        annotated_batch = draw_batch(inputs, targets, preds)

        output_paths = os.path.join(output_path, "test_on_validation_epoch_end.png")

        torchvision.utils.save_image(annotated_batch, output_paths)

        self.validation_step_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.model_step(batch)

        self.test_loss(loss)
        self.test_mae(preds, targets)
        self.log("Test/acc", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        _, preds, _ = self.model_step(batch)
        return preds

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mae",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra

    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    output_path = path / "outputs"
    print(config_path)


    def test_net(cfg):
        net = hydra.utils.instantiate(cfg.net)
        print("*" * 20 + " net " + "*" * 20, "\n", net)
        output = net(torch.randn(16, 3, 224, 224))
        print(output.shape)


    def test_module(cfg):
        module = hydra.utils.instantiate(cfg)
        output = module(torch.randn(16, 3, 224, 224))
        print("output", output.shape)


    @hydra.main(version_base="1.3", config_path=config_path, config_name="resnet18.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_net(cfg)
        test_module(cfg)

    main()
