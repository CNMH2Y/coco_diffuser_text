import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import cv2
import numpy as np
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, UNet2DModel

from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

text_encoder_path = ""
text_embedder_path = ""
json_path = "D:/MLproject/coco2017/annotations/captions_train2017.json"
img_path = "D:/MLproject/coco2017/train2017"

# pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
# pipe.unet()
# pipe = pipe.to("cuda")
'''
# coco_labels = dict(coco.anns.items())
# my_json = json.dumps('caption')
coco_labels = [y for x, y in coco.anns.items()]
labels_list = []
for i in range(len(coco_labels)):
    labels = coco_labels[i]
    labels_list.append(labels.get('caption'))
labels_array = np.array(labels_list)
labels_array = labels_array.reshape(-1, 1)
'''


def data_pre():
    coco = COCO(annotation_file=json_path)
    ids = list(sorted(coco.imgs.keys()))
    anns_list = []  # 加载标签
    img_list = {}
    img_tag_list = []  # 加载图片
    for img_ids in ids:
        anns_ids = coco.getAnnIds(img_ids)

        # 加载图片路径
        path = coco.loadImgs(img_ids)[0]['file_name']
        # cv2.imread(os.path.join(img_path, path))
        # print(img.shape)

        # 加载图片描述以及图片
        print(len(coco.loadAnns(anns_ids)))
        for e in range(len(coco.loadAnns(anns_ids))):  # len(coco.loadAnns(anns_ids))
            anns = coco.loadAnns(anns_ids)[e]
            anns = anns['caption']
            anns_list.append(anns)
            img_tag_list.append(path)
    anns_array = np.array(anns_list).reshape(-1, 1)
    #img_array = np.array(img_list).reshape(-1, 1)
    for i in range(len(img_tag_list)):  # len(img_tag_list)
        k = img_tag_list[i]
        # print(k)
        image = cv2.imread(os.path.join(img_path, k))
        # 480x480 cv2.resize 256x256
        image = torchvision.transforms.ToTensor()(image)

        # 裁剪
        # 640x 480 420 x640
        # = 0.18215 * pipe.vae.encode(image).latent_dist.mean
        # 1 4 64 64
        #img_list[i] = latent_x
    # res = list(img_list.values())
    # img_array = np.array(res)
        print(anns_array)
    # print(img_array.shape)
    # print(anns_array.shape)
    # print(img_array.shape)
    # return img_array, anns_array  # x(1000x3x640x480) , tag(1000,1)

# 何并 预处理
class CocoDataset(Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root # latent x
        self.label = data_label

    # 数据处理
    def __getitem__(self, index):
        # encoder
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


# 文字转化
def build_dataloader(batch_size=8):
    # data, label = data_pre()
    dataset = CocoDataset(data, label)
    print(dataset.data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, embedding_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        # self.class_emb = nn.Embedding(num_classes, class_emb_size)
        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information
        # (the class embedding)
        self.model = UNet2DModel(
            sample_size=64,  # the target image resolution
            in_channels=4 + embedding_size,  # Additional input channels for class cond.
            out_channels=4,  # the number of output channels
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
    def forward(self, x, t, prompt):
        # Shape of x:
        bs, ch, w, h = x.shape  # latent x

        # class conditioning in right shape to add as additional input channels

        '''Tokenizer - 将文本输入拆分为各个子单词，然后使用查找表将每个子单词转换为数字
           Token_To_Embedding Encoder - 将每个子单词的数字表示转换为包含该文本语义信息的特征表示'''

        tokenizer = CLIPTokenizer.from_pretrained(text_embedder_path,
                                                  local_files_only=True,
                                                  torch_dtype=torch.float16)
        token = tokenizer(prompt, padding="max_length",
                          max_length=tokenizer.model_max_length,  # 1 4 64 64
                          # tokenizer.model_max_length,
                          truncation=True,
                          return_tensors="pt")

        '''input_ids - 表示一个文本提示被转化为一个1X77的tensor,后面重复的多个49407为了padding至固定长度77
           attention_mask - 这里的1表示对应有效的embeded值，0表示对应的为padding'''

        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path,
                                                     local_files_only=True,
                                                     torch_dtype=torch.float16).to('cuda')

        emb = text_encoder(token.input_ids.to("cuda"))[0].half()
        # text 变tensor

        emb = emb.view(bs, emb.shape[1], 1, 1).expand(bs, emb.shape[1], w, h)
        # x is shape (bs, 4, 64, 64) and class_cond is now (bs, 4, 64, 64)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, emb), 1)  # (bs, x+text, 64, 64)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  # (bs, 4, 64, 64)


def train(batch_size=128, n_epochs=3, lr=1e-3):
    # Dataloader (you can mess with batch size)
    train_dataloader = build_dataloader(batch_size=batch_size)

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    net = pipe.unet()

    net.to(device)
    '''
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
    '''
    # Our loss function
    loss_fn = nn.MSELoss()

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            x = x.to(device)  # latent x



            # 处理text

            noise = torch.randn_like(x)
            timestamps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timestamps)

            # Get the model prediction
            pred = net(noisy_x, t, text_embedding).sample

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
    x = torch.rand(1, 4, 64, 64).to(device)  # Start from random

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler.set_timesteps(num_inference_steps=40)

    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():  # No need to track gradients during inference
            residual = net(x, t).sample  # Predict the denoised x0

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

        # x decode

        img = x[0, 0].cpu().clip(0, 1).numpy()
        img = (img * 256).astype(np.uint8)
        img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("", img)
        cv2.waitKey()


def main():
    data_pre()
    # dataloader = build_dataloader()
    # for x, y in tqdm(dataloader):
    # print(x)


'''
    if not os.path.isfile("mnist_ddpm.pth"):
        net = train(n_epochs=10)
    else:
        net = torch.load("mnist_ddpm.pth", map_location=device)
    infer(net)
'''


if __name__ == "__main__":
    main()
