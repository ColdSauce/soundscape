import tensorflow as tf
import cnn
import dataset as d
size = 50
print "Building network model..."
scapeNet = cnn.CNN(size,6,rgb=True,convLayers=1)
print "Gathering dataset..."
data = d.getData(size=size)
print "Begin training..."
scapeNet.train(0.002,10,data,data,print_freq=1)
scapeNet.save("scapenet")
