import os

import numpy as np
import cv2

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def build_dataloader(train=True, batch_size=8):
    dataset = torchvision.datasets.MNIST(
        root="mnist/", train=train, download=True, transform=torchvision.transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=28,  # the target image resolution
            in_channels=1 + class_emb_size,  # Additional input channels for class cond.
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
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

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels)  # Map to embedding dimension
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  # (bs, 1, 28, 28)


def train(n_epochs=10):
    train_dataloader = build_dataloader(batch_size=128)

    # noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # Our network
    net = ClassConditionedUnet().to(device)

    # Our loss function
    loss_fn = nn.MSELoss()

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            # Get some data and prepare the corrupted version
            x = x.to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
            y = y.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Get the model prediction
            pred = net(noisy_x, timesteps, y)  # Note that we pass in the labels y

            # Calculate the loss
            loss = loss_fn(pred, noise)  # How close is the output to the noise

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        # Print out the average of the last 100 loss values to get an idea of progress:
        avg_loss = sum(losses[-100:]) / 100
        print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")

    print("Saving model...")
    torch.save(net, "classed_mnist_ddpm.pth")
    return net


def infer(net):
    # Sampling some different digits:
    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(80, 1, 28, 28).to(device)
    y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

    # noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    noise_scheduler.set_timesteps(num_inference_steps=40)

    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residual = net(x, t, y)  # Again, note that we pass in our labels y

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    # Show the results
    x = x.reshape(10, 8, 28, 28).permute(0, 2, 1, 3).reshape(10*28, 8*28)
    img = x.cpu().clip(0, 1).numpy()
    img = (img * 256).astype(np.uint8)
    cv2.imshow("", img)
    cv2.waitKey()


def main():
    if not os.path.isfile("classed_mnist_ddpm.pth"):
        net = train(n_epochs=10)
    else:
        net = torch.load("classed_mnist_ddpm.pth", map_location=device)
    infer(net)


if __name__ == "__main__":
    main()
