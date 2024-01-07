import torch
from torchvision import transforms
from neural_net import PokemonNet, PokemonDataset
from predictor import PokemonTypePredictor

#dictionary mapping types to labels
types = ['fire', 'water', 'grass', 'bug', 'dark', 'fighting', 'psychic', 'rock', 'ground', 'steel', 'ice', 'dragon', 'flying', 'poison', 'electric', 'ghost', 'normal', 'fairy']
class_to_int = {class_name: idx for idx, class_name in enumerate(types)}
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#initialize net
net = PokemonNet(len(types))
pokemon_dataset = PokemonDataset('pkmn_images', 'pkmn_types.txt', class_to_int, transform)
predictor = PokemonTypePredictor(pokemon_dataset, net)

# comment once training is finished to avoid re-training
predictor.train_neural_net() #saves model as 'pokemon_net.pth'
net.load_state_dict(torch.load('pokemon_net.pth'))

test_pkmn = ['centiskorch','avalugg','croagunk','flygon','butterfree','alakazam']
for p in test_pkmn:
    predictor.predict_pokemon_type(p)

