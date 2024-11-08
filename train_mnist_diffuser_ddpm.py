import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

import cv2
import numpy as np
from tqdm import tqdm

from diffusers import DDPMScheduler, UNet2DModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(train=True, batch_size=8):
    dataset = torchvision.datasets.MNIST(
        root="mnist/", train=train, download=True, transform=torchvision.transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


def train(batch_size=128, n_epochs=3, lr=1e-3):
    # Dataloader (you can mess with batch size)
    train_dataloader = build_dataloader(train=True, batch_size=batch_size)

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Create the network
    net = UNet2DModel(
        sample_size=28,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    net.to(device)

    # Our loss function
    loss_fn = nn.MSELoss()

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            # Get some data and prepare the corrupted version
            x = x.to(device)  # Data on the GPU
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Get the model prediction
            pred = net(noisy_x, timesteps).sample

            # Calculate the loss
            loss = loss_fn(pred, noise)  # How close is the output to the true 'clean' x?

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        # Print our the average of the loss values for this epoch:
        avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        print(f"Finished epoch {epoch}/{n_epochs}. Average loss for this epoch: {avg_loss:05f}")

    print("Saving model...")
    torch.save(net, "mnist_ddpm.pth")
    return net


def infer(net):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # @markdown Sampling strategy: Break the process into 5 steps and move 1/5'th of the way there each time:
    x = torch.rand(1, 1, 28, 28).to(device)  # Start from random

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler.set_timesteps(num_inference_steps=40)

    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():  # No need to track gradients during inference
            residual = net(x, t).sample  # Predict the denoised x0

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

        img = x[0, 0].cpu().clip(0, 1).numpy()
        img = (img * 256).astype(np.uint8)
        img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("", img)
        cv2.waitKey()


def main():
    if not os.path.isfile("mnist_ddpm.pth"):
        net = train(n_epochs=10)
    else:
        net = torch.load("mnist_ddpm.pth", map_location=device)
    infer(net)


if __name__ == "__main__":
    main()
