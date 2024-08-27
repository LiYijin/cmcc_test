from PIL import Image
import os
# 打开图像文件

for root, dirs, files in os.walk("./ch4_test_images"):
    for file in files:
        file_name = os.path.join(root, file)
        image = Image.open('./ch4_test_images/img_1.jpg')
        width, height = image.size
        print(f"File: {file_name}, Width: {width}, Height: {height}")

