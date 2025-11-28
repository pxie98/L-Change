import torch
from PIL import Image

class dataGenerator:
    def __init__(self):
        self.func = 0

    def generate_data(self):
        image = Image.open('1.png').convert('L')
        pixel_data = list(image.getdata())
        width, height = image.size
        img_data = torch.zeros([width-9, height-9])
        for x in range(9, width):
            for y in range(9, height):
                pixel_value = pixel_data[y * width + x]
                img_data[x-9, y-9] = pixel_value
        return img_data
     
    def out_img(self, img_name, img_pix):
        image = Image.open('1.png').convert('L')
        width, height = image.size

        new_image = Image.new('L', (width, height))
        new_pixel_data = new_image.load()

        for x in range(width-9):
            for y in range(height-9):
                new_pixel_data[x, y] = img_pix[y*(width-9)+x]
        new_image.save(img_name, compress_level=0)
        print("New 8-bit color PNG image generated successfully.")

        return 0
