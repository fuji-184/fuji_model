# -*- coding: utf-8 -*-
"""Skripsi EfficientNet V2 S - VIT Encoder Fix.ipynb
"""

import torch
from torch import nn, optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from einops import rearrange
from einops.layers.torch import Rearrange
import time
import copy
from sklearn.metrics import classification_report, f1_score
import psutil
import os

def get_data_loaders(batch_size, train=False, valid=False, test=False, valid_size=0.5, path_train="", path_val="", path_test=""):
    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.2, brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_data = datasets.ImageFolder(path_train, transform=transform)
        train_data_len = len(train_data)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        return train_loader, train_data_len

    elif valid or test:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if valid:
            val_data = datasets.ImageFolder(path_val, transform=transform)
            val_data_len = len(val_data)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)
            return val_loader, val_data_len

        if test:
            test_data = datasets.ImageFolder(path_test, transform=transform)
            test_data_len = len(test_data)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
            return test_loader, test_data_len

    else:
        raise ValueError("Either train, valid, or test must be True.")

def get_classes(path_train):
    all_data = datasets.ImageFolder(path_train)
    return all_data.classes

def grafik_pelatihan(training_history, validation_history):
  import matplotlib.pyplot as plt

  train_accuracy = [acc.item() for acc in training_history['accuracy']]
  train_loss = training_history['loss']

  val_accuracy = [acc.item() for acc in validation_history['accuracy']]
  val_loss = validation_history['loss']

  epochs = range(1, len(train_accuracy) + 1)

  plt.figure(figsize=(14, 5))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
  plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
  plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.tight_layout()
  plt.show()

def multiclass_classification_report(model, dataloaders, classes):
    y_true = []
    y_pred = []
    class_names = classes
    model.eval()
    for data, target in dataloaders['test']:
        y_true += target.tolist()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output, 1)
            y_pred += pred.tolist()

    report = classification_report(y_true, y_pred, digits=4, target_names=class_names, output_dict=True)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, target_names=class_names))

    f1_scores = [report[key]['f1-score'] for key in class_names]
    y_pos = np.arange(len(class_names))

    plt.figure(figsize=(12, 6))
    plt.bar(y_pos, f1_scores, align='center', alpha=0.5)
    plt.xticks(y_pos, class_names, rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.xlabel('Class')
    plt.title('F1 Scores for Each Class')
    plt.tight_layout()
    plt.show()

    return classification_report(y_true, y_pred, digits=4, target_names=class_names)

def get_gpu_utilization():
    if torch.cuda.is_available():
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
            gpu_utilization = int(result.stdout.decode().strip())
            return gpu_utilization
        except Exception as e:
            print("Error getting GPU utilization:", e)
    return None

def model_report(model, dataloaders):
    y_true = []
    y_pred = []
    model.eval()
    for data, target in dataloaders['test']:
        y_true += target.tolist()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output, 1)
            y_pred += pred.tolist()

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = next(model.parameters()).device
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    input_memory = input_tensor.element_size() * input_tensor.nelement()
    memory_usage = input_memory + sum(p.element_size() * p.nelement() for p in model.parameters())

    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    end_time = time.time()
    inference_speed = (end_time - start_time) / 10

    cpu_percent = psutil.cpu_percent()

    gpu_utilization = get_gpu_utilization()

    print(f"Number of parameters: {num_params}")
    print(f"Memory usage: {memory_usage / 1024 / 1024:.2f} MB")
    print(f"Inference speed: {inference_speed:.5f} seconds per sample")
    print(f"CPU Utilization: {cpu_percent}%")
    if gpu_utilization is not None:
        print(f"GPU Utilization: {gpu_utilization}%")
    else:
        print("GPU Utilization: N/A")

    return report

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, dataloaders=None, dataset_sizes=None, classes=None):

    training_history = { 'accuracy':[],'loss':[]}
    validation_history = {'accuracy':[],'loss':[]}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                training_history['accuracy'].append(epoch_acc)
                training_history['loss'].append(epoch_loss)
            elif phase == 'val':
                validation_history['accuracy'].append(epoch_acc)
                validation_history['loss'].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    grafik_pelatihan(training_history, validation_history)
    multiclass_classification_report(model, dataloaders, classes)
    model_report(model, dataloaders)
    return model

class Conv2d(nn.Module):
  def __init__(self, in_chan, out_chan, kernel_size=3, padding=1, stride=1, groups=1, with_bn=True, with_act=True):
    super().__init__()
    self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
    self.with_bn = with_bn
    self.with_act = with_act

    if with_bn:
      self.bn = nn.BatchNorm2d(out_chan)

    if with_act:
      self.act = nn.SiLU()

  def forward(self, x):
    x = self.conv(x)

    if self.with_bn:
      x = self.bn(x)

    if self.with_act:
      x = self.act(x)

    return x

class StochasticDepth(nn.Module):
    def __init__(
        self,
        survival_prob = 0.8
    ):
        super(StochasticDepth, self).__init__()

        self.p =  survival_prob

    def forward(self, x):

        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p

        return torch.div(x, self.p) * binary_tensor

class SE(nn.Module):
  def __init__(self, in_chan, r=4):
    super().__init__()
    self.squeeze = nn.AdaptiveAvgPool2d(1)
    self.excitation = nn.Sequential(
        nn.Conv2d(in_chan, int(in_chan//r), kernel_size=1, padding=0),
        nn.SiLU(),
        nn.Conv2d(int(in_chan//r), in_chan, kernel_size=1, padding=0),
        nn.Sigmoid()
    )

  def forward(self, x):

    input = x.clone()

    x = self.squeeze(x)
    x = self.excitation(x)

    return input * x

class Fused_MBConv(nn.Module):
  def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, expansion=2, padding=1):
    super().__init__()
    self.skip_conn = (in_chan == out_chan) and (stride == 1)

    if expansion > 1:
      expansion = int(in_chan * expansion)
      self.conv = nn.Sequential(
          Conv2d(in_chan, expansion, kernel_size=kernel_size, stride=stride, padding=padding),
          SE(expansion),
          Conv2d(expansion, out_chan, kernel_size=1, padding=0, stride=1, with_act=False),
      )
    else:
      self.conv = nn.Sequential(
          Conv2d(in_chan, out_chan, kernel_size, stride=stride, padding=padding),
          SE(out_chan)
      )

    self.sd = StochasticDepth()

  def forward(self, x):

    residu = x.clone()

    x = self.conv(x)

    if self.skip_conn:
      x = self.sd(x)
      x = x + residu

    return x

class MBConv(nn.Module):
  def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, expansion=1, padding=1):
    super().__init__()
    self.skip_conn = (in_chan == out_chan) and (stride == 1)

    if expansion > 1:
      expansion = int(in_chan * expansion)
      self.conv = nn.Sequential(
          Conv2d(in_chan, expansion, kernel_size=1, padding=0, stride=1),
          Conv2d(expansion, expansion, kernel_size=kernel_size, stride=stride, padding=padding, groups=expansion),
          SE(expansion),
          Conv2d(expansion, out_chan, kernel_size=1, padding=0, stride=1, with_act=False)
      )
    else:
      self.conv = nn.Sequential(
          Conv2d(in_chan, in_chan, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_chan),
          SE(in_chan),
          Conv2d(in_chan, out_chan, kernel_size=1, padding=0, stride=1, with_act=False)
      )

    self.sd = StochasticDepth()

  def forward(self, x):

    residu = x.clone()

    x = self.conv(x)

    if self.skip_conn:
      x = self.sd(x)
      x = x + residu

    return x

class Embedding(nn.Module):
  def __init__(self, in_chan, embed_dim, kernel_size=8, stride=8, padding=0, expansion=1):
    super().__init__()
    self.conv = Fused_MBConv(in_chan, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

  def forward(self, x):
    b, c, h, w = x.shape
    x = self.conv(x)
    x = rearrange(x, 'b c h w -> b (h w) c')

    return x

class Repatch(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=2, padding=0):
        super().__init__()

        self.conv = MBConv(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
      _, n, _ = x.shape
      h = w = int(n**0.5)
      x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

      x = self.conv(x)

      x = rearrange(x, 'b c h w -> b (h w) c')

      return x

class Fused_MBConv_Layers(nn.Module):
    def __init__(self, in_chan, out_chan=None, kernel_size=3, stride=1, padding=1, expansion=1, jumlah=0, downsample=False):
        super().__init__()

        if jumlah > 0:
            fused_mbconv_layers = []
            for _ in range(jumlah):
                fused_mbconv_layers.append(Fused_MBConv(in_chan, in_chan, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion))
            self.fused_mbconv = nn.Sequential(*fused_mbconv_layers)
        else:
            self.fused_mbconv = nn.Identity()

        self.out_chan = out_chan
        if downsample:
          stride = 2
        if out_chan is not None:
            self.up_chan = Fused_MBConv(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion)

    def forward(self, x):

      x = self.fused_mbconv(x)

      if self.out_chan is not None:
        x = self.up_chan(x)

      return x

class MBConv_Layers(nn.Module):
    def __init__(self, in_chan, out_chan=None, kernel_size=3, stride=1, padding=1, expansion=1, jumlah=0):
        super().__init__()

        if jumlah > 0:
            mbconv_layers = []
            for _ in range(jumlah):
                mbconv_layers.append(MBConv(in_chan, in_chan, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion))
            self.mbconv = nn.Sequential(*mbconv_layers)
        else:
            self.mbconv = nn.Identity()

        self.out_chan = out_chan
        if out_chan is not None:
            self.up_chan = MBConv(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, expansion=expansion)

    def forward(self, x):

      _, n, _ = x.shape
      h = w = int(n**0.5)
      x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

      x = self.mbconv(x)

      if self.out_chan is not None:
        x = self.up_chan(x)

      x = rearrange(x, 'b c h w -> b (h w) c')

      return x

class MultiHeadAttention(nn.Module):
  def __init__(self, in_dim, num_heads=8, kernel_size=3, dropout=0.1):
    super().__init__()
    padding = (kernel_size - 1)//2
    self.forward_conv = self.forward_conv
    self.num_heads = num_heads
    self.head_dim = in_dim // num_heads
    self.conv = nn.Sequential(
        nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0),
        Rearrange('b c h w -> b (h w) c'),
    )
    self.att_drop = nn.Dropout(dropout)

  def forward_conv(self, x):
    B, hw, C = x.shape
    H = W = int(x.shape[1]**0.5)
    x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

    q = self.conv(x)
    k = self.conv(x)
    v = self.conv(x)

    return q, k, v

  def forward(self, x):

    q, k, v = self.forward_conv(x)

    q = rearrange(x, 'b t (d H) -> b H t d', H=self.num_heads)
    k = rearrange(x, 'b t (d H) -> b H t d', H=self.num_heads)
    v = rearrange(x, 'b t (d H) -> b H t d', H=self.num_heads)

    att_score = q@k.transpose(2, 3)/self.num_heads**0.5
    att_score = F.softmax(att_score, dim=-1)
    att_score = self.att_drop(att_score)

    x = att_score@v

    x = rearrange(x, 'b H t d -> b t (H d)')

    return x, att_score

class Encoder(nn.Module):
  def __init__(self, embed_dim, num_heads=8, dropout=0.1):
    super().__init__()
    self.norm1 = nn.LayerNorm(embed_dim)
    self.mhsa = MultiHeadAttention(embed_dim, dropout=dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(embed_dim)

    self.ffn = nn.Sequential(
        nn.Conv2d(embed_dim, int(embed_dim*2), kernel_size=1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(int(embed_dim*2), embed_dim, kernel_size=1),
        nn.Dropout(dropout),
    )

  def forward(self, x):

    residu = x.clone()

    x = self.norm1(x)

    x, attn_score = self.mhsa(x)

    x = residu + self.dropout(x)

    residu = x.clone()

    x = self.norm2(x)

    B, hw, C = x.shape

    H = W = int(x.shape[1]**0.5)
    x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

    x = self.ffn(x)

    x = rearrange(x, 'b c h w -> b (h w) c')

    x = residu + x

    return x, attn_score

class Encoder_Layers(nn.Module):
  def __init__(self, embed_dim, num_heads=8, dropout=0.1, jumlah=0):
    super().__init__()
    if jumlah > 0:
      self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(jumlah)
        ])
    else:
      self.encoder_layers = nn.Identity()

  def forward(self, x):

    attn_scores = []
    for encoder in self.encoder_layers:
      x, attn_score = encoder(x)
      attn_scores.append(attn_score)

    return x, attn_scores

class EfficientNetV2_VitEncoder(nn.Module):
  def __init__(self, num_classes, embed_dim=192, num_heads=8, patch_size=16, dropout=0.1):
    super().__init__()
    self.conv = nn.Sequential(
        Conv2d(3, 24, stride=2),
        Fused_MBConv(24, 32, stride=1, expansion=1),
    )
    self.embedding = Embedding(32, embed_dim, kernel_size=patch_size, stride=patch_size//2, padding=0)
    self.layers = nn.ModuleList([
        Encoder_Layers(embed_dim, num_heads=num_heads, dropout=dropout, jumlah=3),
        MBConv_Layers(embed_dim, kernel_size=3, padding=1, jumlah=6, expansion=2),

        Repatch(embed_dim, embed_dim*2, kernel_size=5, stride=2, padding=0),

        Encoder_Layers(embed_dim*2, num_heads=num_heads, dropout=dropout, jumlah=6),
        MBConv_Layers(embed_dim*2, kernel_size=3, padding=1, jumlah=6, expansion=2),

        Repatch(embed_dim*2, embed_dim*2*2, kernel_size=3, stride=1, padding=0),

        Encoder_Layers(embed_dim*2*2, num_heads=num_heads, dropout=dropout, jumlah=2),
    ])
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.head = nn.Linear(embed_dim*2*2, num_classes)

  def forward(self, x):
    x = self.conv(x)
    x = self.embedding(x)
    n = x.shape[0]

    attn_scores = []

    for layer in self.layers:
      if isinstance(layer, Encoder_Layers):
        x, attn_score = layer(x)
        attn_scores.extend(attn_score)
      else:
        x = layer(x)
    x = rearrange(x, "b s d -> b d s")

    x = self.pool(x)
    x = x.squeeze(-1)

    x = self.head(x)
    attn_w = None

    return x

def buat_model(jumlah_kelas):
    model = EfficientNetV2_VitEncoder(num_classes=jumlah_kelas)

    device = torch.device("cuda" if torch.cuda. is_available() else 'cpu')
    model = model.to(device)
    return model

def latih_model(model, epochs, path_train, path_val, path_test, buat_model=True, jumlah_kelas=1):
    (train_loader, train_data_len) = get_data_loaders(16, train=True, path_train=path_train, path_val=path_val, path_test=path_test)
    (val_loader, val_data_len) = get_data_loaders(batch_size=16, valid=True, path_train=path_train, path_val=path_val, path_test=path_test)
    (test_loader, test_data_len) = get_data_loaders(16, test=True, path_train=path_train, path_val=path_val, path_test=path_test)
    classes = get_classes(path_train=path_train)

    dataloaders = {
        "train":train_loader,
        "val": val_loader,
        "test": test_loader
    }
    dataset_sizes = {
        "train":train_data_len,
        "val": val_data_len,
        "test": test_data_len
    }
    
    device = torch.device("cuda" if torch.cuda. is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

    model_ft = train_model(model, num_epochs=epochs, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, dataloaders=dataloaders, dataset_sizes=dataset_sizes, classes=classes)
    return model_ft
