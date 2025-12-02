import torch
from PIL import Image

class dataGenerator:
    '''
    func: 需要拟合的函数；
    [start, end]：生成数据样本的区间；
    step：生成数据的步长
    example：
    输入：func: f(x), start=-1, end=1, step=0.5,
    输出：x就是tensor[-1, -0.5, 0, 0.5, 1], y就是 tensor[f(-1), f(-0.5), f(0), f(0.5), f(1)]
    '''
    def __init__(self):
        self.func = 0

    def generate_data(self):
        image = Image.open('func_img/approx1/1.png').convert('L')
        pixel_data = list(image.getdata())
        width, height = image.size
        img_data = torch.zeros([width-9, height-9])
        for x in range(9, width):
            for y in range(9, height):
                pixel_value = pixel_data[y * width + x]
                img_data[x-9, y-9] = pixel_value
        return img_data
     
    def out_img(self, img_name, img_pix):
        image = Image.open('func_img/approx1/1.png').convert('L')
        width, height = image.size

        new_image = Image.new('L', (width, height))
        new_pixel_data = new_image.load()

        for x in range(width-9):
            for y in range(height-9):
                new_pixel_data[x, y] = img_pix[y*(width-9)+x]
        new_image.save(img_name, compress_level=0)
        print("New 8-bit color PNG image generated successfully.")

        return 0
