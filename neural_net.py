import os
import pokebase as pb
import requests
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class PokemonNet(nn.Module):
    def __init__(self, num_classes):
        super(PokemonNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 24 * 24, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 24 * 24)  # Adjust this based on your image size
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class PokemonDataset(Dataset):
    def __init__(self, image_dir, types_file, classes_to_int, transform=None):
        self.image_dir = image_dir
        self.types_file = types_file
        self.transform = transform
        self.classes_to_int = classes_to_int
        self.images = self.load_images()
        self.labels = self.load_labels()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        label = self.labels[idx]
        raw_img = self.images[idx]
        to_tensor = transforms.ToTensor()
        image = to_tensor(raw_img)

        return image, label
    
    def load_images(self):
        images = []
        img_files = [file for file in os.listdir(self.image_dir) if file.lower().endswith('.png')]
        img_files = [os.path.join(self.image_dir, file) for file in img_files]

        for i in img_files:
            create_img = Image.open(i).convert('RGB')
            images.append(create_img)

        return images
    
    def load_labels(self):
        labels = []
        with open(self.types_file, 'r') as file:
            for line in file:
                labels.append(self.classes_to_int[line.strip()])
        return labels




#-------------------------------------------------- TRAINING -----------------------------------------------------------------------------------
def train():

    #------------------------ HYPERPARAMETERS ----------------------------------#

    batch_size = 32
    number_of_epochs = 14
    learning_rate = 0.001

    #------------------------ END HYPERPARAMETERS ------------------------------#

    types = ['fire', 'water', 'grass', 'bug', 'dark', 'fighting', 'psychic', 'rock', 'ground', 'steel', 'ice', 'dragon', 'flying', 'poison', 'electric', 'ghost', 'normal', 'fairy']
    class_to_int = {class_name: idx for idx, class_name in enumerate(types)}

    image_dir = os.path.abspath('pkmn_images')
    types_file = 'pkmn_types.txt'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    b_size = batch_size
    pkmn_dataset = PokemonDataset(image_dir, types_file, class_to_int, transform)
    pkmn_dataloader = DataLoader(pkmn_dataset, batch_size=b_size, shuffle=True)

    net = PokemonNet(len(types))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    losses = []

    num_epochs = number_of_epochs
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in pkmn_dataloader:
            
            optimizer.zero_grad()

            outputs = net(images)
            
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss per epoch
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(pkmn_dataloader)}")
        losses.append(running_loss / len(pkmn_dataloader))

    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

    torch.save(net.state_dict(), 'pokemon_net.pth')













