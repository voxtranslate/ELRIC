import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os

class ImageDataset(Dataset):
    """Training dataset. Strict enforcement of 0-1 range and NO synthetic data generation."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root_dir, cfg: Dict[str, Any], mode="train"):
        super().__init__()
        self.patch_size = cfg["patch_size"]
        self.mode       = mode
        self.paths      = self._collect(root_dir)
        
        if not self.paths:
            raise RuntimeError(f"No valid images found in {root_dir}. Synthetic data generation is strictly forbidden.")
        
        self.length = len(self.paths)
        if mode == "train":
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ColorJitter(brightness=cfg["color_jitter_b"], contrast=cfg["color_jitter_c"]),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(self.patch_size), transforms.ToTensor()])

    def _collect(self, root):
        root = Path(root); paths = []
        if root.exists():
            for ext in self.EXTENSIONS:
                paths.extend(str(p) for p in root.rglob(f"*{ext}"))
        return sorted(paths)

    def __len__(self): return self.length

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if w < self.patch_size or h < self.patch_size:
                resample_method = getattr(Image, 'Resampling', Image).BICUBIC
                img = img.resize((max(w, self.patch_size), max(h, self.patch_size)), resample_method)
            img_t = self.transform(img).clamp(0.0, 1.0) # Guaranteed 0-1
            name = Path(path).stem
            return {"image": img_t, "path": path, "name": name}
        except Exception as e:
            # Strictly NO synthetic random tensors! Recursively find the next valid image.
            return self.__getitem__((idx + 1) % len(self.paths))


class KodakDataset(Dataset):
    """Standard 24-image Kodak benchmark dataset at full resolution."""
    KODAK_BASE_URL = "http://r0k.us/graphics/kodak/kodak/"
    EXTENSIONS     = {".png", ".jpg", ".jpeg"}

    def __init__(self, root_dir: str, download: bool = True):
        super().__init__()
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        if download:
            self._download_if_needed()
            
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTENSIONS)
        if len(self.paths) < 1:
            raise RuntimeError("Kodak dataset is empty. Synthetic data fallback is strictly forbidden.")
            
        print(f"[KodakDataset] {len(self.paths)} images found in {self.root}")
        self.transform = transforms.ToTensor()

    def _download_if_needed(self):
        existing = list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        if len(existing) >= 24:
            return
        print("[KodakDataset] Downloading Kodak images …")
        for i in range(1, 25):
            fname = f"kodim{i:02d}.png"; fpath = self.root / fname
            if fpath.exists(): continue
            try:
                urllib.request.urlretrieve(self.KODAK_BASE_URL + fname, fpath)
            except Exception as e:
                print(f"  ✗ {fname}: {e}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        img_t = self.transform(img).clamp(0.0, 1.0) # Guaranteed 0-1
        return {"image": img_t, "path": str(path), "name": path.stem}


def build_dataloaders(cfg):
    full_ds = ImageDataset(cfg["data_dir"], cfg)
    n = len(full_ds)
    n_train = int(n * cfg["train_ratio"]); n_val = int(n * cfg["val_ratio"])
    n_test  = n - n_train - n_val
    g = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)
    
    for ds, mode in [(val_ds, "val"), (test_ds, "test")]:
        ds.dataset = copy.deepcopy(full_ds); ds.dataset.mode = mode
        ds.dataset.transform = transforms.Compose([
            transforms.CenterCrop(cfg["patch_size"]), transforms.ToTensor()])
            
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], 
              pin_memory=True, worker_init_fn=seed_worker)
              
    tl = DataLoader(train_ds, shuffle=True,  **kw)
    vl = DataLoader(val_ds,   shuffle=False, **kw)
    sl = DataLoader(test_ds,  shuffle=False, **kw)
    print(f"[Dataset] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    return tl, vl, sl
