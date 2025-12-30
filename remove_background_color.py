from PIL import Image
import numpy as np
import pathlib
from pathlib import Path
from dotenv import load_dotenv
import os
from rembg import remove
import onnxruntime

#this code take the choas data and tries to remove the colorful background to make it easier to sort
# sorted the images with the remove background into  a seperate file called "No_background"

load_dotenv()
# Grave path to choas_data from .env file
CHAOS_PATH = pathlib.Path(os.getenv("CHAOS_PATH"))

output_dir = Path("No_background")
extensions = {".png", ".jpg", ".jpeg"}
# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

#for is so that it goes through each image in the folder
for img_path in CHAOS_PATH.iterdir():
    if img_path.suffix.lower() in extensions:
        img = Image.open(img_path)
        #this remove function is what actually removes the background
        result = remove(img)
        #this .save is what stores the result into the No_background file
        result.save(output_dir / img_path.name)



