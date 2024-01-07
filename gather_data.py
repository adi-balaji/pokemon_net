import pokebase as pb
import requests
import os
from PIL import Image

def download_pkmn_img(url, path):
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)

def download_pkmn_img_flipped(path):

    #with augmetations
    
    raw_image = Image.open(path)
    root, ext = os.path.splitext(path)

    hflip = raw_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    hflip_path = f'{root}hf{ext}'
    hflip.save(hflip_path)
    
    vflip = raw_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    vflip_path = f'{root}vf{ext}'
    vflip.save(vflip_path)

    hvflip = raw_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    hvflip = hvflip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    hvflip_path = f'{root}hvf{ext}'
    hvflip.save(hvflip_path)

    rot = raw_image.transpose(Image.Transpose.ROTATE_270)
    rot_path = f'{root}r{ext}'
    rot.save(rot_path)

    tilt = raw_image.transpose(Image.Transpose.ROTATE_90)
    tilt_path = f'{root}t{ext}'
    tilt.save(tilt_path)



# Gather data in batches, may crash due to heap overflow. Start with 1-300, then 300-500, and so on..
bottom = 800
top = 1000
image_dir = 'pkmn_images'
os.makedirs(image_dir, exist_ok=True)
pokemon_ids = list(range(bottom,top))

with open('pkmn_types.txt', 'a') as types_file:

    for id in pokemon_ids:
        pkmn = pb.APIResource('pokemon', id)
        pkmn_sprite_url = pb.SpriteResource('pokemon', id).url
        pkmn_name = pkmn.name
        pkmn_types = [t.type.name for t in pkmn.types]

        image_path = os.path.join(image_dir, f'{id}.png')
        download_pkmn_img(pkmn_sprite_url, image_path)
        download_pkmn_img_flipped(image_path)

        #6 times for 1 original and 5 augmented
        types_file.write(f'{pkmn_types[0]}\n')
        types_file.write(f'{pkmn_types[0]}\n')
        types_file.write(f'{pkmn_types[0]}\n')
        types_file.write(f'{pkmn_types[0]}\n')
        types_file.write(f'{pkmn_types[0]}\n')
        types_file.write(f'{pkmn_types[0]}\n')
        

        if id % 5 == 0:
            print(f'Loaded {id} of {top} pokemon..')


