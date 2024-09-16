import os
from PIL import Image

# Path to the folder containing the images
name = '3_chefs_small_kitchen_all_SP'
image_folder = f'data/screenshots/{name}'
output_gif = f'data/gifs/{name}.gif'

# List all image files and sort them numerically (assuming file names like 1.png, 2.png, etc.)
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Sort numerically by the file name

# Load images into a list
frames = [Image.open(os.path.join(image_folder, image)) for image in images]

# Save as a GIF
frames[0].save(output_gif, save_all=True, append_images=frames[1:], optimize=False, duration=150, loop=0)

print(f"GIF created and saved as {output_gif}")
