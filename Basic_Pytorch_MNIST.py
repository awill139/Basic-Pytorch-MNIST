import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 10
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = datasets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

class Model(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Model,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self,x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out 

model = Model(input_size, hidden_size, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train():
    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_gen):
            images = Variable(images.view(-1,28*28)).to(device)
            labels = Variable(labels).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                            %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.item()))


def evaluate():
    correct = 0
    total = 0
    for images,labels in test_gen:
    images = Variable(images.view(-1,28*28)).to(device)
    labels = labels.to(device)
    
    output = model(images)
    _, predicted = torch.max(output,1)
    correct += (predicted == labels).sum()
    total += labels.size(0)

    print('Accuracy of the model: %.3f %%' %((100*correct)/(total+1)))

if __name__ == '__main__':
    train()
    evaluate