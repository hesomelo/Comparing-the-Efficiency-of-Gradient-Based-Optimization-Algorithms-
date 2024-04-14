import torch as t
import torch.optim
import torch.nn as nn
from optimizers import Sgd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
import torch.nn.functional as F
from utils import get_mnist
from torch.utils.tensorboard import SummaryWriter



inp = 28**2
hid = 10
out = 10
device = t.device("cuda" if t.cuda.is_available() else "cpu")

MLP = nn.Sequential(
    nn.Flatten(),
    nn.Linear(inp,hid),
    nn.ReLU(),
    nn.Linear(hid,hid),
    nn.ReLU(),
    nn.Linear(hid,out),

)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # MNIST is grayscale, hence in_channels=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for the dense layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class trainargs():
    batch_size = 64
    learning_rate = 1e-3
    subset = 50
    epochs = 10

def train(args, model, optimizer_type="Adam", dataset="mnist"):
    model.to(device)

    if dataset == "mnist":
        train, test = get_mnist(args.subset)
        trainloader =DataLoader(train, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(test, batch_size= args.batch_size, shuffle=False)
    accuracy_list = []

    if optimizer_type == 'Adam':
        optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = t.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif optimizer_type == 'RMSProp':
        optimizer = t.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    # elif optimizer_type == 'Adagrad':
    #     optimizer = Sgd(model.parameters(), lr=args.learning_rate)
    # elif optimizer_type == 'Momentum':
    #     optimizer = Sgd(model.parameters(), lr=args.learning_rate)
   
    # optimizer = t.optim.Adam(model.parameters(), lr = args.learning_rate) # change this line

    loss = nn.CrossEntropyLoss()
    loss_list = []
    n = 0
    for epoch in tqdm(range(args.epochs)):
        for imgs, labels in trainloader:
            # print(imgs.shape)
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss_n = loss(logits,labels)
            loss_n.backward()
            optimizer.step()
            loss_list.append(loss_n.item())
           
        writer.add_scalar("Loss/train", loss_n.item(), epoch)
        
        
        num_correct_classifications = 0

        for imgs, labels in testloader:
            imgs = imgs.to(device)
            # print(imgs.shape)
            labels = labels.to(device)
            with t.inference_mode():
                logits = model(imgs)
            predictions = t.argmax(logits, dim=1)
            num_correct_classifications += (predictions == labels).sum().item()
  
        accuracy = num_correct_classifications / len(test)
        accuracy_list.append(accuracy)   
        writer.add_scalar("accuracy/valid", accuracy, epoch)

   
    print(accuracy_list)


writer = SummaryWriter("10 epoch test, 64 bs, rmsprop")
args = trainargs()
train(args, CNN(), optimizer_type="RMSProp")
writer.close()


