import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import torch.nn as nn
channel = 1


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.spatial_attention(out) * out

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def AResNet18(num_classes=8):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def train_model(model, train_loader, val_loader, criterion, optimizer, out_dir, num_epochs=10, patience=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = None
    best_acc = 0.0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    stime = time.time()
    end_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_preds.double() / len(train_loader.dataset)

        scheduler.step()

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Train_Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        correct_val_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val_preds += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val_preds.double() / len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        print(f'------------- Validation Loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, str(out_dir) + '/best_model.pth')  # Save the best model weights
            print('save best!')

        if epoch - best_epoch > patience:
            end_epoch = epoch
            print(f'early stopping at epoch {epoch}')
            break

    etime = time.time()
    alltime = etime - stime
    print('training time:', alltime)
    print(f'max_val_acc: ', max(history['val_acc']))

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model, history, alltime, end_epoch


def test_model(model, test_loader, out_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct_preds = 0
    all_preds = []
    all_labels = []
    stime = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    etime = time.time()
    alltime = etime - stime
    print('testing time:', alltime)
    test_acc = correct_preds.double() / len(test_loader.dataset)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Weighted F1 Score: {weighted_f1:.4f}')
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_loader.dataset.dataset.classes)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=90)
    plt.savefig(str(out_dir) + '/confusion_matrix.png', bbox_inches='tight', dpi=300)
    return alltime, report


def plot_training_history(history, out_dir):
    expected_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    for key in expected_keys:
        if key not in history:
            history[key] = [None] * len(history['train_loss'])

    df_history = pd.DataFrame(history)
    df_history['epoch'] = range(1, len(df_history) + 1)
    csv_file_path = str(out_dir) + '/training_history.csv'
    df_history.to_csv(csv_file_path, index=False)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(str(out_dir) + '/training_history.png', bbox_inches='tight', dpi=300)
    # plt.show()


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


def show_transformed_image(dataset, out_dir, index, mean, std):
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    img, label = dataset[index]
    img_denorm = denormalize(img.clone(), mean, std)
    img_np = img_denorm.numpy()
    img_np = np.clip(img_np, 0, 1)
    if img_np.shape[0] == 1:
        img_np = np.squeeze(img_np, axis=0)
        plt.imshow(img_np, cmap='gray')
    else:
        img_np = np.transpose(img_np, (1, 2, 0))
        plt.imshow(img_np)

    plt.axis('off')
    plt.savefig(str(out_dir) + '/noise.png', bbox_inches='tight', dpi=300)
    # plt.show()


result_dir = 'results/AResNet18_result/dataset-name/Tab2Img-method'
os.makedirs(result_dir, exist_ok=True)
data_dir = 'dataset/name'
batch_size = 128
seed = 36
epochs = 100
early_stop = 20

norm = [0.5] if channel == 1 else [0.5, 0.5, 0.5]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=channel), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # AddGaussianNoise(mean=0., std=0.15),  # Add Gaussian Noise
    transforms.Normalize(mean=norm, std=norm),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# show_transformed_image(dataset, result_dir, index=10, mean=norm, std=norm)

dataset_size = len(dataset)

train_size = int(0.6 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
all_labels = [idx_to_class[idx] for idx in dataset.targets]

train_idx, temp_idx = train_test_split(list(range(len(dataset))), test_size=0.4, stratify=all_labels, random_state=seed)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[all_labels[i] for i in temp_idx],
                                     random_state=seed)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

test_labels = [label for _, label in test_dataset]
label_counts = Counter(test_labels)
print("Test set class distribution:")
for label, count in label_counts.items():
    print(f"Class {label}: {count} samples")

# create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = AResNet18(num_classes=len(dataset.classes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, (channel, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# train
model, history, train_time, epoch_ = train_model(model, train_loader, val_loader, criterion, optimizer, result_dir,
                                                 num_epochs=epochs, patience=early_stop)

test_time, reports = test_model(model, test_loader, result_dir)

with open(str(result_dir) + "/report.txt", "w") as file:
    file.write(f"early stop epoch: {epoch_}\n")
    file.write(f"training time: {train_time}\n")
    file.write(f"testing time: {test_time}\n")
    file.write(reports)

plot_training_history(history, result_dir)



