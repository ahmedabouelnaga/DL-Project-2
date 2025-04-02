import shutil
import glob
import cv2



img_dir = "Enter Directory of all images"
files = glob.glob(img_dir)
data = []
for f1 in files:
img = cv2.imread(f1)
data.append(img)




for (i, filename) in enumerate(glob.glob('../face_arb/*.jpg')):
  shutil.copyfile(filename, 'image%05d.jpg'%i)

