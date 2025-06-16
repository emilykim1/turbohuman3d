import os
from PIL import Image

folder = 'ava-outputs-0.1-renamed'

save_to = 'cropped-gp'

images = os.listdir(folder)
images = [image for image in images if 'png' in image]
images.sort()

index = 0
os.makedirs(folder + "/" + save_to, exist_ok=True)

for im in images:
    img = Image.open(folder + '/' + im)
    # img1 = img.crop([5*256, 0, 6*256, 256])
    # img2 = img.crop([6*256, 0, 7*256, 256])
    img1 = img.crop([3*256, 0, 4*256, 256])
    img2 = img.crop([4*256, 0, 5*256, 256])

    img1.save(folder + '/' + save_to + f'/{index:02d}.png')
    index += 1
    img2.save(folder + '/' + save_to + f'/{index:02d}.png')
    index += 1

