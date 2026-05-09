"""
MobileNetV2 + TSM (Temporal Shift Module)
Bukva датасет + Ө, Ү үсгийг нэгтгэж сургах
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================== ТОХИРГОО =====================
BUKVA_DIR    = "bukva_data/trimmed"
ANNOTATIONS  = "bukva_data/annotations.tsv"
CUSTOM_DIR   = "bukva_extra"
OUTPUT_MODEL = "tsm_35classes.pth"
OUTPUT_ONNX  = "tsm_35classes.onnx"
NUM_FRAMES   = 8
IMG_SIZE     = 224
BATCH_SIZE   = 4
EPOCHS       = 30
LR           = 1e-4
NUM_WORKERS  = 0
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ====================================================

BUKVA_CLASSES = [
    "no_event","Ё","А","Б","В","Г","Д","Е","Ж","З","И",
    "Й","К","Л","М","Н","О","П","Р","С","Т","У","Ф","Х",
    "Ц","Ч","Ш","Щ","Ъ","Ы","Ь","Э","Ю","Я"
]
CUSTOM_CLASSES = ["Ө", "Ү"]
ALL_CLASSES    = BUKVA_CLASSES + CUSTOM_CLASSES
CLASS_TO_IDX   = {c: i for i, c in enumerate(ALL_CLASSES)}
NUM_CLASSES    = len(ALL_CLASSES)

print(f"Нийт класс: {NUM_CLASSES}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, fold_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = fold_div

    def forward(self, x):
        bt, c, h, w = x.shape
        t = self.n_segment
        b = bt // t
        x = x.view(b, t, c, h, w)
        fold = c // self.fold_div
        out  = torch.zeros_like(x)
        out[:, 1:,  :fold]       = x[:, :-1, :fold]
        out[:, :-1, fold:fold*2] = x[:, 1:,  fold:fold*2]
        out[:, :,   fold*2:]     = x[:, :,   fold*2:]
        return out.view(bt, c, h, w)


class TSMBlock(nn.Module):
    def __init__(self, block, n_segment=8):
        super().__init__()
        self.block = block
        self.tsm   = TemporalShift(n_segment)

    def forward(self, x):
        x = self.tsm(x)
        return self.block(x)


class MobileNetV2TSM(nn.Module):
    def __init__(self, num_classes, n_segment=8, pretrained=True):
        super().__init__()
        self.n_segment = n_segment
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        features = list(base.features.children())
        for i in range(len(features) - 6, len(features)):
            if hasattr(features[i], 'conv'):
                features[i] = TSMBlock(features[i], n_segment)
        self.features   = nn.Sequential(*features)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.features(x)
        x = self.pool(x).view(b, t, -1)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


def load_video_frames(video_path, num_frames=NUM_FRAMES):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        return None
    return np.stack(frames[:num_frames])


def build_dataset():
    samples = []
    print("Bukva датасет уншиж байна...")
    df = pd.read_csv(ANNOTATIONS, sep='\t')
    found = miss = 0
    for _, row in df.iterrows():
        label = str(row['text'])
        if label not in CLASS_TO_IDX:
            continue
        path = os.path.join(BUKVA_DIR, f"{row['attachment_id']}.mp4")
        if os.path.exists(path):
            samples.append((path, CLASS_TO_IDX[label]))
            found += 1
        else:
            miss += 1
    print(f"  Bukva: {found} олдлоо, {miss} олдсонгүй")

    print("Өөрийн датасет уншиж байна...")
    for letter in CUSTOM_CLASSES:
        d = os.path.join(CUSTOM_DIR, letter)
        if not os.path.exists(d):
            print(f"  ⚠️ {d} олдсонгүй")
            continue
        vids = [f for f in os.listdir(d) if f.endswith('.mp4')]
        for v in vids:
            samples.append((os.path.join(d, v), CLASS_TO_IDX[letter]))
        print(f"  {letter}: {len(vids)} видео")

    print(f"\nНийт: {len(samples)} видео")
    return samples


class SignDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples   = samples
        self.augment   = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def aug_frame(self, f):
        if np.random.rand() > 0.5:
            f = cv2.flip(f, 1)
        if np.random.rand() > 0.5:
            f = np.clip(f.astype(np.float32) * np.random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        if np.random.rand() > 0.5:
            h, w = f.shape[:2]
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            f = cv2.warpAffine(f, M, (w, h), borderValue=(114, 114, 114))
        return f

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = load_video_frames(path)
        if frames is None:
            frames = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if self.augment:
            frames = np.array([self.aug_frame(f) for f in frames])
        tensors = torch.stack([self.transform(f) for f in frames])
        return tensors, label


def export_onnx(model):
    model.eval()
    dummy = torch.zeros(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    torch.onnx.export(
        model, dummy, OUTPUT_ONNX,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )
    print(f"✅ ONNX хадгалагдлаа: {OUTPUT_ONNX}")


def train():
    print("\n" + "="*55)
    print("  MobileNetV2+TSM сургалт эхэлж байна")
    print("="*55)

    samples = build_dataset()
    if not samples:
        print("❌ Датасет хоосон!")
        return

    train_s, val_s = train_test_split(
        samples, test_size=0.15, random_state=42,
        stratify=[s[1] for s in samples]
    )
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    train_dl = DataLoader(SignDataset(train_s, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = DataLoader(SignDataset(val_s,   augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = MobileNetV2TSM(NUM_CLASSES, n_segment=NUM_FRAMES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = train_correct = train_total = 0

        for frames, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            frames = frames.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)
            optimizer.zero_grad()
            out  = model(frames)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for frames, labels in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                frames = frames.to(DEVICE)
                labels = torch.tensor(labels).to(DEVICE)
                out    = model(frames)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        print(f"\n✅ Epoch {epoch}: Train={train_acc:.1f}%  Val={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'classes':     ALL_CLASSES,
                'val_acc':     val_acc,
            }, OUTPUT_MODEL)
            print(f"  💾 Хамгийн сайн загвар хадгалагдлаа: {val_acc:.1f}%")

    print(f"\n🎉 Дууслаа! Хамгийн сайн: {best_acc:.1f}%")

    # ONNX руу хөрвүүлэх
    print("\nONNX руу хөрвүүлж байна...")
    ckpt = torch.load(OUTPUT_MODEL)
    model.load_state_dict(ckpt['model_state'])
    export_onnx(model)


if __name__ == "__main__":
    print("Checkpoint ачаалж байна...")

    model = MobileNetV2TSM(NUM_CLASSES, n_segment=NUM_FRAMES).to(DEVICE)

    ckpt = torch.load(OUTPUT_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    print("ONNX руу хөрвүүлж байна...")
    export_onnx(model)