"""
M4TR1X Training Script - Fine-tuning EfficientNet-B0 per AI video detection.

Struttura dataset:
    data/train/real/  data/train/ai_generated/
        data/val/real/    data/val/ai_generated/

        Uso: python train_detector.py --epochs 15 --batch-size 16
        """

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from ai_detector import M4tr1xDetector, DEVICE, MODEL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m4tr1x.train")


def get_transforms():
      train_t = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ])
      val_t = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ])
      return train_t, val_t


def train_epoch(model, loader, criterion, optimizer, epoch):
      model.train()
      loss_sum, correct, total = 0.0, 0, 0
      for i, (x, y) in enumerate(loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
                if (i + 1) % 10 == 0:
                              logger.info(f"E{epoch} [{i+1}/{len(loader)}] Loss:{loss_sum/(i+1):.4f} Acc:{100.*correct/total:.1f}%")
                      return loss_sum / len(loader), 100.0 * correct / total


def validate(model, loader, criterion):
      model.eval()
      loss_sum, correct, total = 0.0, 0, 0
      with torch.no_grad():
                for x, y in loader:
                              x, y = x.to(DEVICE), y.to(DEVICE)
                              out = model(x)
                              loss_sum += criterion(out, y).item()
                              _, pred = out.max(1)
                              total += y.size(0)
                              correct += pred.eq(y).sum().item()
                      return loss_sum / len(loader), 100.0 * correct / total


def main():
      p = argparse.ArgumentParser()
      p.add_argument("--data-dir", default="data")
      p.add_argument("--epochs", type=int, default=15)
      p.add_argument("--batch-size", type=int, default=16)
      p.add_argument("--lr", type=float, default=0.0001)
      p.add_argument("--output", default=str(MODEL_DIR / "m4tr1x_detector.pt"))
      args = p.parse_args()

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    if not train_dir.exists():
              logger.error(f"Crea: {train_dir}/real/ e {train_dir}/ai_generated/")
              return

    train_t, val_t = get_transforms()
    train_ds = datasets.ImageFolder(str(train_dir), train_t)
    val_ds = datasets.ImageFolder(str(val_dir), val_t)
    logger.info(f"Dataset: {len(train_ds)} train, {len(val_ds)} val | Classi: {train_ds.classes}")

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=4, pin_memory=True)

    model = M4tr1xDetector(len(train_ds.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    best = 0.0
    for ep in range(1, args.epochs + 1):
              tl, ta = train_epoch(model, train_dl, criterion, optimizer, ep)
              vl, va = validate(model, val_dl, criterion)
              scheduler.step()
              logger.info(f"E{ep}/{args.epochs} Train:{ta:.1f}% Val:{va:.1f}%")
              if va > best:
                            best = va
                            torch.save(model.state_dict(), args.output)
                            logger.info(f"Salvato {args.output} (best: {va:.1f}%)")

          logger.info(f"Fine. Best val accuracy: {best:.1f}%")


if __name__ == "__main__":
      main()
