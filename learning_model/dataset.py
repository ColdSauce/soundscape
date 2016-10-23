from PIL import Image
import numpy as np
import os
import random

emotes = ['angry','grand','happy','neutral','sad','scary']

def convertImage(image, grayscale=False):
	if grayscale:
		return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32,copy=False).T.flatten();
	else:	
		return image.astype(np.float32,copy=False).T.flatten();

def getData(size = 32):
	input_data = []
	out_vecs = []
	filenames = []
	for dirname in emotes:
		files = os.listdir("emotes/"+dirname)
		for i in range(len(files)):
			files[i] = "emotes/" + dirname + "/"+ files[i]
		for nm in files:	
			filenames.append(nm)
	#random.shuffle(filenames)
	for mfile in filenames:
		image = Image.open(mfile)
		image = np.asarray(image.resize((size,size)))
		image = convertImage(image)
		classname = np.array([0]*len(emotes))
		cn = mfile.split("/")[1]
		classname[emotes.index(cn)] = 1
		input_data.append(image)
		out_vecs.append(classname)
	return [np.array(input_data),np.array(out_vecs)]
