import os

import torch
from torch import nn
from torch.utils.data import DataLoader

import cv2
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "D:/data/huggan-anime-faces"


def download_data(split):
    # 下载数据集
    if os.path.isdir(DATA_PATH):
        print(f"Data folder already exist at {DATA_PATH}. Skip downloading...")
        return

    os.mkdir(DATA_PATH)
    # 学习datasets的参数
    dataset = load_dataset("huggan/anime-faces", split=split, verification_mode="no_checks", download_mode="force_redownload")
    # 装载dataset
    print(f"Downloading data to {DATA_PATH}...")
    for i, d in tqdm(enumerate(dataset)):
        d["image"].save(os.path.join(DATA_PATH, f"{i}.jpg"), format="jpeg")
        # 下载到本地
    print(f"Data downloaded.")
    return


class AnimeFaceDataset(torch.utils.data.Dataset):
    # 因为下载到本地，所以需要重新构建数据集（tensor形式）
    def __init__(self, split="train"):
        self.dataset = load_dataset("D:/data/huggan-anime-faces", split=split, verification_mode="no_checks", download_mode="force_redownload")
    # 数据集
    def __len__(self):
        return len(self.dataset)
    # 数据集长度
    def __getitem__(self, idx):
        transform = transforms.ToTensor()
        return transform(self.dataset[idx]["image"])
    # 将image变为可读的tensor

# def collate_fn(batch):
#     processed_batch = [transform(e["image"]).unsqueeze(0) for e in batch]
#     processed_batch = torch.concat(processed_batch, dim=0)
#     return processed_batch


def corrupt(x, amount):
    # 制造噪点
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    # view方法？
    return x * (1 - amount) + noise * amount


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x


def train(batch_size=128, n_epochs=3, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Dataloader (you can mess with batch size)
    dataset = AnimeFaceDataset(split="train")
    train_dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)
    # 128x3x64x64
    # torch.utils.data学习
    n_batches = len(train_dataloader)

    # from IPython import embed
    # embed()
    # assert False

    # Create the network加载神经网络
    net = BasicUNet()
    net.to(device)

    # Our loss function计算损失函数
    loss_fn = nn.MSELoss()

    # The optimizer优化器
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # The training loop
    for epoch in range(n_epochs):
        losses = []
        for i, x in enumerate(train_dataloader):
            # Get some data and prepare the corrupted version
            x = x.to(device)  # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts 1x3x64x64
            noisy_x = corrupt(x, noise_amount)  # Create our noisy x

            # Get the model prediction将原数据过模型得到预测后结果
            pred = net(noisy_x)

            # Calculate the loss
            loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?

            # Backprop and update the params:
            opt.zero_grad()  # 梯度函数重置为0
            loss.backward()  # 前向传播
            opt.step()  # theta = theta - gradient * learning_rate

            # Store the loss for later
            losses.append(loss.item())
            print(f"Epoch {epoch}/{n_epochs}, step {i}/{n_batches}, loss: {loss.item()}")

        # Print our the average of the loss values for this epoch:
        avg_loss = sum(losses) / len(train_dataloader)
        print(f"Finished epoch {epoch}/{n_epochs}. Average loss for this epoch: {avg_loss:05f}")

    print("Saving model...")
    torch.save(net, "mnist.pth")
    return net


def infer(net, n_steps=5):
    # @markdown Sampling strategy: Break the process into 5 steps and move 1/5'th of the way there each time:
    x = torch.rand(1, 3, 64, 64).to(device)  # Start from random

    for i in tqdm(range(n_steps)):
        with torch.no_grad():  # No need to track gradients during inference
            pred = net(x)  # Predict the denoised x0

        mix_factor = 1 / (n_steps - i)  # How much we move towards the prediction
        x = x * (1 - mix_factor) + pred * mix_factor  # Move part of the way there
        img = x[0].permute(1,2,0).cpu().clip(0, 1).numpy()
        img = (img * 256).astype(np.uint8)
        img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("", img)
        cv2.waitKey(300)


def main():
    download_data(split="train")
    if not os.path.isfile("mnist.pth"):
        net = train(n_epochs=50)
    else:
        net = torch.load("mnist.pth", map_location=device)
    infer(net, n_steps=50)


if __name__ == "__main__":
    main()
