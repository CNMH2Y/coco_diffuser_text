import os

import numpy as np
import cv2

import torch
import torchvision
from torch import nn
from torch.nn import GRU
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm
import pandas as pd

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class PulseDataset(Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


def build_dataloader(data, label, batch_size=8):
    dataset = PulseDataset(data, label)

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    return dataloader


# 构建GRU
class MyGru(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=4, num_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x):
        output, _ = self.gru(x, None)
        output = output[:, -1, :]
        output = self.linear(output)
        return output


def class_emb(y):
    emb_func = nn.Embedding(2, 4)
    result = emb_func(y)
    #print(result)
    return result


def train(dataloader, n_epochs=10):
    train_dataloader = dataloader

    # Our network
    net = MyGru().to(device)

    # Our loss function
    loss_fn = torch.nn.MSELoss().to(device)

    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=0.01)

    # Keeping a record of the losses for later viewing
    losses = []

    # The training loop
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            # Get some data and prepare the corrupted version
            x = x.unsqueeze(dim=2)
            # print(x)
            x = x.to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
            y = class_emb(y)
            # print(y)
            y = y.to(device)
            # timestamps = torch.randint(0, 999, (x.shape[0],)).long().to(device)

            # Get the model prediction
            pred = net(x)  # Note that we pass in the labels y
            # print(pred)
            # Calculate the loss
            loss = loss_fn(pred, y)  # How close is the output to the noise

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        # Print out the average of the last 100 loss values to get an idea of progress:
        avg_loss = sum(losses) / len(losses)
        #print(losses)
        print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")

    print("Saving model...")
    torch.save(net, "classed_mnist_ddpm.pth")
    return net


'''
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
'''


# 读取数据文件夹
def getdata(path):
    data_total = []
    label_total = []
    files = os.listdir(path.encode('utf-8').decode('utf-8'))
    b = 0
    for file in files:
        txt_path = path + '\\' + file
        name = file + '=' + str(b)
        print(name)
        f = open(txt_path, 'r', encoding='utf-8')
        line = f.readline()
        data_list = []
        while line:
            num = list(map(str, line.split()))
            data_list.append(num)
            line = f.readline()
        f.close()
        data_array = (np.array(data_list))
        # print(data_array)
        # print(data_array.shape)
        data_array_refined = []
        num = len(data_array)
        print(num)
        for i in range(num):
            for j in range(19):
                num1 = 0
                num2 = 0
                ten = data_array[i][j][0]
                if ten == '0':
                    num1 = 0
                elif ten == '1':
                    num1 = 16
                elif ten == '2':
                    num1 = 16 * 2
                elif ten == '3':
                    num1 = 16 * 3
                elif ten == '4':
                    num1 = 16 * 4
                elif ten == '5':
                    num1 = 16 * 5
                elif ten == '6':
                    num1 = 16 * 6
                elif ten == '7':
                    num1 = 16 * 7
                elif ten == '8':
                    num1 = 16 * 8
                elif ten == '9':
                    num1 = 16 * 9
                elif ten == 'A':
                    num1 = 16 * 10
                elif ten == 'B':
                    num1 = 16 * 11
                elif ten == 'C':
                    num1 = 16 * 12
                elif ten == 'D':
                    num1 = 16 * 13
                elif ten == 'E':
                    num1 = 16 * 14
                elif ten == 'F':
                    num1 = 16 * 15
                else:
                    print('error')
                single = data_array[i][j][1]
                if single == '0':
                    num2 = 0
                elif single == '1':
                    num2 = 1
                elif single == '2':
                    num2 = 2
                elif single == '3':
                    num2 = 3
                elif single == '4':
                    num2 = 4
                elif single == '5':
                    num2 = 5
                elif single == '6':
                    num2 = 6
                elif single == '7':
                    num2 = 7
                elif single == '8':
                    num2 = 8
                elif single == '9':
                    num2 = 9
                elif single == 'A':
                    num2 = 10
                elif single == 'B':
                    num2 = 11
                elif single == 'C':
                    num2 = 12
                elif single == 'D':
                    num2 = 13
                elif single == 'E':
                    num2 = 14
                elif single == 'F':
                    num2 = 15
                else:
                    print('error')
                total = num1 + num2
                data_array_refined.append(total)
        data_list_refined = np.array(data_array_refined)
        data_list_refined = data_list_refined.reshape(-1, 19)
        # print(data_list_refined)
        # print(data_list_refined.shape)

        # 将前四行没用的数据删掉
        data_del = np.delete(data_list_refined, [0, 1, 2, 3], axis=1)  # 10863x15
        # print(data_del)

        # 校验是否与校验和一致
        dataset_test = np.delete(data_del, 14, axis=1)
        dataset_sum = np.sum(dataset_test, axis=1)
        dataset_ver = []
        for i in range(len(dataset_sum)):
            s = int(dataset_sum[i]) % 256
            dataset_ver.append(s)
        dataset_final = []
        for i in range(len(data_del)):
            t = data_del[i][14]
            if t == dataset_ver[i]:
                dataset_final.append(data_del[i])
            else:
                print('error')

        # 最终数组
        data_list_final = np.array(dataset_final)
        data_list_final = np.delete(data_list_final, 14, axis=1)
        # print(data_list_final)
        data_total.extend(data_list_final)
        # data_list_final = data_list_final.reshape(-1,14,1)

        # 添加标签
        a = len(data_list_final)
        data_label = np.zeros((a, 1))
        data_label = data_label + b
        print('第' + str(b) + '次')
        label_total.extend(data_label)
        # print(label_total.shape)
        b = b + 1

    data_total_array = np.array(data_total)
    data_total_array = data_total_array.astype(np.float32)
    #tensor = torchvision.transforms.ToTensor()(data_total_array)
    #print(tensor)
    label_total_array = np.array(label_total)
    label_total_array = label_total_array.astype(np.int32)
    print(data_total_array)
    return data_total_array, label_total_array

    # txt_path = path + file
    # f = pd.read_csv(txt_path, header=None, encoding='utf-8')

    # dataList.append(f)
    # data = pd.concat(dataList)
    # return data


'''
# 单个文件
def get_one_data(file):
    f = pd.read_csv(file, header=None, encoding='utf-8')
    return f
'''


def main():
    relative_path = 'pulse_data'
    data_all, total_all = getdata(relative_path)
    print(total_all)
    # train_dataloader_test = build_dataloader(data_all, total_all, batch_size=8)
    # train(train_dataloader_test)


if __name__ == "__main__":
    main()
