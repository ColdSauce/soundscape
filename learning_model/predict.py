import tensorflow as tf
import cnn
import dataset as d
import numpy as np
from PIL import Image
size = 50
print "Building network model..."
scapeNet = cnn.CNN(size,6,rgb=True,convLayers=1)
scapeNet.restore("scapenet")

def convertImage(image, grayscale=False):
	if grayscale:
		return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32,copy=False).T.flatten();
	else:	
		return image.astype(np.float32,copy=False).T.flatten();


def predict(image):
	with tf.Session() as sess:
		y = scapeNet.tensors['y']
		x = scapeNet.tensors['x']
		keep_prob = scapeNet.tensors['keep_prob']
		return y.eval(feed_dict={x:image,y:np.zeros((1,6)),keep_prob:1})

image = Image.open("emotes/angry/181_volcano.jpg")
image = np.asarray(image.resize((size,size)))
image = convertImage(image)
image = image-128
print predict([image])
