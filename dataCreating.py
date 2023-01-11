from PIL import Image
import os
maindir = r"C:\Users\Acer\Desktop\Faults"
folders = os.listdir(r"C:\Users\Acer\Desktop\Faults")
for folder in folders:
    files = os.listdir(fr"{maindir}\{folder}")
    for each in files:
        image = Image.open(fr"{maindir}\{folder}\{each}")
        imageResized = image.resize(size=(224,224))
        print(files.index(each))
        if files.index(each) < 400:
            imageResized.save(fr"C:\Users\Acer\Desktop\faultsdata\{folder}\{files.index(each)+1}.png")
        else:
            pass