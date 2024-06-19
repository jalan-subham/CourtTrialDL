import torch
from rich import print 
import random
# from utils import load_video
import glob 
import numpy as np
import gc

gc.collect()

torch.cuda.empty_cache()

files = glob.glob("CourtTrial/Clips/Identities/*/*/*/*/*aligned/video.npy")
random.shuffle(files)
device = torch.device("cuda:1")
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files 
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        # print("[red]Loading video", idx, "...")
        # load the video as an np array
        video_path = self.files[idx]
        video = np.load(video_path)
        video = video / 255.0
        # pad the video to the max number of frames
        video = torch.tensor(video, dtype=torch.float32)
        # change ordering
        video = video.permute(3, 0, 1, 2)

        y = 0 if "truth" in video_path else 1
        return video, y
    def truth_lies(self):
        lies = 0
        truths = 0
        for file in self.files:
            if "truth" in file:
                truths += 1
            elif "lie" in file:
                lies += 1
        return truths, lies

class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv3d(3, 32, kernel_size=(2,2,2)),
            torch.nn.ReLU(),

            torch.nn.Conv3d(32, 64, kernel_size=(2,2,2)),
            torch.nn.MaxPool3d((2,2,2)),
            torch.nn.ReLU(),

            torch.nn.Conv3d(64, 64, kernel_size=(2,2,2)),
            torch.nn.MaxPool3d((2,2,2)),
            torch.nn.ReLU(),

            torch.nn.Conv3d(64, 32, kernel_size=(2,2,2)),
            torch.nn.MaxPool3d((2,2,2)),
            torch.nn.ReLU(),

            torch.nn.Linear(11520, 5000),
            torch.nn.ReLU(),

            torch.nn.Linear(5000, 500),
            torch.nn.ReLU(),

            torch.nn.Linear(500, 2),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.cnn(x)
        return x
    
def collater(batch):
    X = []
    y = []
    for x, _y in batch:
        X.append(torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 2443 - x.shape[1])))
        y.append(_y)
    X = torch.stack(X).to(device)
    y = torch.tensor(y).to(device)
    return X, y
print("[red]Loading dataset...")
dataset = VideoDataset(files)
print("[green]Number of videos:", len(dataset))

# shape of first 5 videos 
for i in range(5):
    print("[blue]Shape of video", i, ":", dataset[i][0].shape)



print("[red]Making model...")
model = CNNModel()
model = torch.nn.DataParallel(model)
model.to(device)
print("[green]Model:", model)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset = VideoDataset(files[:train_size])
test_dataset = VideoDataset(files[train_size:])


print("[red]Training dataset size:", len(train_dataset))
print("[green]Test dataset size:", len(test_dataset))

print("[red]Number of truths and lies in training dataset:", train_dataset.truth_lies())
print("[green]Number of truths and lies in test dataset:", test_dataset.truth_lies())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collater)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collater)

print("[red]Training model...")

for epoch in range(10):
    model.train()
    for i, (X, y) in enumerate(train_loader):
        print("[red]Epoch", epoch, "Iteration", i, "...")
        # print("[green]X:", X.shape)
        # print("Size taken by X", X.element_size() * X.nelement() / 1e6, "MB")
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss {loss.item()}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (X, y) in enumerate(test_loader):
            y_pred = model(X)
            print(y_pred)
            correct += (y_pred.argmax(1) == y).sum().item()
            total += 1
        print(f"Epoch {epoch}, Test Accuracy: {correct / total}")
