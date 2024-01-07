import pokebase as pb
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import neural_net
import matplotlib.pyplot as plt

class PokemonTypePredictor:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def train_neural_net(self):
        neural_net.train()

    def predict_pokemon_type(self, pkmn_name):
        pkmn_id = pb.APIResource('pokemon', pkmn_name).id
        sprite_url = pb.SpriteResource('pokemon', pkmn_id).url
        response = requests.get(sprite_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)
        with torch.no_grad():
            energies = self.net(image_tensor)
        pred = None
        for k,v in self.dataset.classes_to_int.items():
            if v == torch.argmax(energies):
                pred = k

        plt.imshow(image)
        plt.title(f'{pb.APIResource("pokemon", pkmn_id).name} is {pred} type.')
        plt.show()

    def predict_image_type(self, path):
        image = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0) 
        with torch.no_grad():
            energies = self.net(image_tensor)
        pred = None
        for k,v in self.dataset.classes_to_int.items():
            if v == torch.argmax(energies):
                pred = k

        plt.imshow(image)
        plt.title(f'This is a {pred} type.')
        plt.show()






        
