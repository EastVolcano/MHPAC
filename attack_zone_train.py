from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import argparse
import os
import csv
import random
import time
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Model.missile_attack_zone_model import MissileHitProb
from Utils.optim_schedule import ScheduledOptim


def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # ��GPU�����
        return value

    with torch.no_grad():
        dist.all_reduce(value)  # �Բ�ͬ�豸֮���value���
        if average:  # �����Ҫ��ƽ������ö��GPU����loss�ľ�ֵ
            value /= world_size
    return value


def logistic(x, growth_rate, mid_point):
    if (-growth_rate * (x - mid_point)) > 700:
        S = 0
    else:
        S = 1 / (1 + np.exp(-growth_rate * (x - mid_point)))
    return S


class MissileDataset(Dataset):
    def __init__(self, data_path=".\\DataSet\\missile_data\\"):
        self.data_buff = []
        print("start loading missile attack data: ")
        name = os.listdir(data_path)
        file_bar = tqdm(name)
        for i, b in enumerate(file_bar):
            with open(data_path + name[i], "r", newline='') as csvfile:
                reader = csv.reader(csvfile)
                for j, row in enumerate(reader):
                    if j > 1:
                        self.data_buff.append(row)
            csvfile.close()
            file_bar.set_description('Processing ' + b)

    def __len__(self):
        return len(self.data_buff)

    def __getitem__(self, item):
        data_array = self.data_buff[item]
        # print(type(data_array))
        attack_zone_input = []
        attack_zone_label = []
        for i, data in enumerate(data_array):
            if i == 0:  # alt
                attack_zone_input.append(0.000105 * float(data) if float(data) < 2000
                                         else logistic(float(data), 0.00053, 4500))
            elif i == 1:  # fire_ma
                attack_zone_input.append(logistic(float(data), 7, 0.75))
            elif i == 2:  # fire_pitch
                attack_zone_input.append(float(data) / 90)
            elif i == 3:  # dist
                attack_zone_input.append(2 * logistic(float(data), 0.00076, 0) - 1)
            elif i == 4:  # body_q_t
                attack_zone_input.append(float(data) / 180)
            elif i == 5:  # body_q_d
                attack_zone_input.append(float(data) / 90)
            elif i == 6:  # target_ma
                attack_zone_input.append(logistic(float(data), 7, 0.75))
            elif i == 7:  # target_hori_AA
                attack_zone_input.append(float(data) / 180)
            elif i == 8:  # target_v_pitch
                attack_zone_input.append(float(data) / 90)
            elif i == 9:  # hit
                if data == 'False':
                    attack_zone_label.append([0, 1])
                elif data == 'True':
                    attack_zone_label.append([1, 0])

        output = {
            "input": torch.as_tensor(attack_zone_input, dtype=torch.float32).squeeze(),
            "label": torch.as_tensor(attack_zone_label, dtype=torch.float32).squeeze()
        }

        return output


class MissileTrainer:
    """
    MissileTrainer make the trained missile attack prob model using CrossEntropyLoss.
    """

    def __init__(self, rank, model: MissileHitProb, train_dataloader: DataLoader,
                 test_dataloader: DataLoader = None, lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 1e-8,
                 warmup_steps=10000, with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param model: attack zone model which you want to train
        :param train_dataloader:  dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.rank = rank

        # Initialize the Model
        self.model = model

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda:
            self.model.cuda(rank)
            self.model = DDP(self.model, device_ids=[rank])
        print(f"Using device {self.model.device} for missile attack prob training")

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, 32, n_warmup_steps=warmup_steps, torch_dist=True)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.CrossEntropyLoss()

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        loss, avc = self.iteration(epoch, self.train_data)
        return loss, avc

    def test(self, epoch):
        loss, avc = self.iteration(epoch, self.test_data, train=False)
        return loss, avc

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        if train:
            self.model.train()
        else:
            self.model.eval()

        # Setting the tqdm progress bar # ��ʾ�������� ����һ���������������ѭ����ʹ�á������ѭ���������Ķ�����ʾ����
        if self.rank == 0:
            data_iter = tqdm(enumerate(data_loader),
                             desc="EP_%s:%d" % (str_code, epoch),
                             total=len(data_loader),
                             bar_format="{l_bar}{r_bar}")
        else:
            data_iter = enumerate(data_loader)

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.cuda(self.rank) for key, value in
                    data.items()}  # ���ݼ����ֵ䣩��ÿһ��������һ����tensor�ļ�ֵ��

            if train:
                missile_hit_prob = self.model(data['input'])
                loss = self.criterion(missile_hit_prob, data['label'])
            else:
                with torch.no_grad():
                    missile_hit_prob = self.model(data['input'])
                    loss = self.criterion(missile_hit_prob, data['label'])

            loss_all = reduce_value(loss, average=True)

            # 3. backward and optimization only in train
            if train:
                self.optim.zero_grad()
                # self.optim_schedule.zero_grad()
                loss.backward()
                self.optim.step()
                # self.optim_schedule.step_and_update_lr()

            # print(self.optim.state_dict()['param_groups'][0]['lr'])

            # prediction accuracy
            with torch.no_grad():
                soft = nn.Softmax(-1)
                correct = soft(missile_hit_prob).argmax(dim=-1).eq(data['label'].argmax(dim=-1)).sum().item()
                # print(f"out size: {soft(missile_hit_prob).argmax(dim=-1).size()},  label size: {data['label'].size()}, nelement: {data['label'].size(0)}")

            correct_all = reduce_value(correct, average=True)
            avg_loss += loss_all.item()
            total_correct += correct_all
            total_element += data['label'].size(0)

            post_fix = {
                "rank": self.rank,
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if (i % self.log_freq == 0) and (self.rank == 0):
                data_iter.write(str(post_fix))

        if self.rank == 0:
            print("rank%d EP%d_%s, avg_loss=" % (self.rank, epoch, str_code), avg_loss / len(data_iter), "total_acc=",
                  total_correct * 100.0 / total_element)

        return avg_loss / len(data_iter), total_correct * 100.0 / total_element

    def save(self, epoch, file_path=".\\output\\MissileAttack\\missile_prob_model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """

        output_path_cpu = file_path + "_cpu_ep%d" % epoch + ".pt"
        torch.save(self.model.cpu().module.state_dict(), output_path_cpu)
        output_path_gpu = file_path + "_gpu_ep%d" % epoch + ".pt"
        torch.save(self.model.cuda(self.rank).module.state_dict(), output_path_gpu)
        print("EP:%d CPU Model Saved on:" % epoch, output_path_cpu)
        return output_path_cpu


def train(rank, world_size, pargs):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Set the device for this process
    torch.cuda.set_device(rank)

    test_dataset = MissileDataset(data_path=".\\DataSet\\missile_data_test\\")
    train_dataset = MissileDataset(data_path=".\\DataSet\\missile_data\\")

    print("Creating Dataloader")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    # dataloader: 将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练
    train_data_loader = DataLoader(train_dataset, batch_size=pargs.batch_size, num_workers=pargs.num_workers,
                                   sampler=train_sampler, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=5,
                                  sampler=test_sampler, pin_memory=True)

    print(f"Created Dataloader:  train: {len(train_data_loader)}, test: {len(test_data_loader)}")

    print("Building model")
    missile_prob = MissileHitProb(9, [512, 512, 512, 512, 512], activation=nn.LeakyReLU)

    print("Creating Trainer")
    trainer = MissileTrainer(rank, missile_prob, train_dataloader=train_data_loader,
                             test_dataloader=test_data_loader,
                             lr=pargs.lr, betas=(pargs.adam_beta1, pargs.adam_beta2),
                             weight_decay=pargs.adam_weight_decay,
                             with_cuda=pargs.with_cuda, cuda_devices=pargs.cuda_devices, log_freq=pargs.log_freq,
                             warmup_steps=pargs.warmup_steps)

    print("Training Start with testing")

    data_dir = '.\\output\\MissileAttack\\'
    os.makedirs(data_dir, exist_ok=True)
    time_str = time.strftime("%Y_%m_%d", time.localtime())
    with open(data_dir + 'missile_attack_train_test_' + time_str + '.csv', 'w') as csvfile:
        wirter = csv.writer(csvfile)
        fieldname = ['epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'learning rate']
        wirter.writerow(fieldname)

    for epoch in range(pargs.epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_avc = trainer.train(epoch)

        test_sampler.set_epoch(epoch)
        test_loss, test_avc = trainer.test(epoch)

        with open(data_dir + 'missile_attack_train_test_' + time_str + '.csv', 'a', newline='') as csvfile:
            writer_n = csv.writer(csvfile)
            writer_n.writerow([epoch, train_loss, test_loss, train_avc, test_avc,
                               trainer.optim.state_dict()['param_groups'][0]['lr']])

        if rank == 0:
            if epoch % 5 == 0:
                trainer.save(epoch, pargs.output_path)


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output_path", default=".\\output\\MissileAttack\\missile_attack",
                        type=str, help=".\\output\\MissileAttack\\missile_attack")

    parser.add_argument("-b", "--batch_size", type=int, default=512, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=20, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=50, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=8e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-8, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--warmup_steps", type=int, default=20000, help="adam lr warm step")

    pargs = parser.parse_args()

    world_size = 1
    mp.spawn(train,
             args=(world_size, pargs),
             nprocs=world_size,
             join=True)
