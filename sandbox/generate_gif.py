import os
from PIL import Image

from sandbox.constants import GIFS_DIR, SCREENSHOTS_DIR


def generate_gif(screenshots_folder_name, output_gif_name=None):
    image_folder = f'{SCREENSHOTS_DIR}/{screenshots_folder_name}'
    output_gif_name = output_gif_name if output_gif_name else screenshots_folder_name
    output_gif = f'{GIFS_DIR}/{output_gif_name}.gif'
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    frames = [Image.open(os.path.join(image_folder, image)) for image in images]
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], optimize=False, duration=150, loop=0)
    print(f"GIF created and saved as {output_gif}")

if __name__ == "__main__":
    screenshots_folder_name = '3_chefs_forced_coordination'
    generate_gif(screenshots_folder_name=screenshots_folder_name) # pass output_gif_name if you want to save the gif with a different name
