import tensorflow as tf
import subprocess
import datetime

def weight_variable(shape,name=None):
	initial = tf.truncated_normal(shape,stddev=0.01)
	return tf.Variable(initial,name=name)

def bias_variable(shape,name=None):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial,name=name)

def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')

def max_pool(x,dim=2):
	return tf.nn.max_pool(x,ksize=[1,dim,dim,1],strides=[1,dim,dim,1],padding='SAME')

class CNN:

	def convLayer(self,input_tensor,features,dimension=5,channels=1,pool=True):
		M_conv = weight_variable([dimension,dimension,channels,features],name='M_conv' + str(self.convlayers))
		b_conv = bias_variable([features],name ='b_conv' + str(self.convlayers))
		h_conv = tf.nn.relu(conv2d(input_tensor,M_conv)+b_conv)
		self.tensors.update({
			'M_conv'+str(self.convlayers): M_conv,
			'b_conv'+str(self.convlayers): b_conv,
			'h_conv'+str(self.convlayers): h_conv
		})
		self.convlayers+=1
		if pool:
			h_pool = max_pool(h_conv)
			return h_pool
		return h_conv

	def fullyConnectedLayer(self,input_tensor,features,activation=''):
		M_fcl = weight_variable([int(input_tensor.get_shape()[1]),features],name="M_fcl"+str(self.fclayers))
		b_fcl = bias_variable([features],name="b_fcl"+str(self.fclayers))
		h_fcl = None
		if activation == 'relu':
			h_fcl = tf.nn.relu(tf.matmul(input_tensor,M_fcl)+b_fcl)
		elif activation == 'softmax':
			h_fcl = tf.nn.softmax(tf.matmul(input_tensor,M_fcl)+b_fcl)
		else:
			h_fcl = tf.nn.sigmoid(tf.matmul(input_tensor,M_fcl)+b_fcl)
		self.tensors.update({
			'M_fcl' + str(self.fclayers): M_fcl,
			'b_fcl' + str(self.fclayers): b_fcl,
			'h_fcl' + str(self.fclayers): h_fcl
		})
		self.fclayers += 1
		return h_fcl

	def train(self,alpha,iterations,train_data,test_data,print_freq=10):
		keep_prob = self.tensors['keep_prob']
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.tensors['y_conv'],self.tensors['y'])
		train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(self.tensors['y_conv'],1),tf.argmax(self.tensors['y'],1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		session = self.session
		session.run(tf.initialize_all_variables())
		address = "/tmp/log/convnet/" + str(datetime.datetime.now()).replace(' ','')
		train_writer = tf.train.SummaryWriter(address,session.graph)
		proc = subprocess.Popen(["tensorboard","--logdir="+address])
		acc_summary = tf.scalar_summary("Accuracy",accuracy)
		x = self.tensors['x']
		y = self.tensors['y']		
		for i in range(iterations+1):
			if i%print_freq == 0:	
				train_accuracy = accuracy.eval(feed_dict={x:train_data[0],y:train_data[1],keep_prob:1.0})
				summary = acc_summary.eval(feed_dict={x:train_data[0],y:train_data[1],keep_prob:1.0})
				print("Step %d training accuracy: %g"%(i,train_accuracy))
				train_writer.add_summary(summary,i)
			train_step.run(feed_dict={x:train_data[0],y:train_data[1],keep_prob:0.5})
		print("Final test accuracy %g"%accuracy.eval(feed_dict={x:test_data[0],y:test_data[1],keep_prob:1.0}))
		proc.kill()

	def save(self,filename="my_convnet_model.cpkt"):
		if self.session:
			saver = tf.train.Saver()
			saver.save(self.session,filename)
			self.session.close()
	
	def restore(self,filename):
		if self.session:
			saver = tf.train.Saver()
			saver.restore(self.session,filename)

	def __init__(self,inputSize,outputSize,convFeatures=128,convLayers=2,rgb=False,dtype = tf.float32):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.tensors = {}
		self.fclayers = 0
		self.convlayers = 0
		self.session = None
		channels =  3 if rgb else 1
		#Create input and output variables, add to dict		
		x = tf.placeholder(dtype,shape=[None,inputSize*inputSize*channels])
		self.tensors.update({'x':x})
		y = tf.placeholder(dtype,shape=[None,outputSize])
		self.tensors.update({'y':y})
		#Create initial weights and biases, add to dict
		M = weight_variable([inputSize*inputSize*channels,outputSize])
		b = bias_variable([outputSize])
		self.tensors.update({'M':M})
		self.tensors.update({'b':b})
		#Create first convolutional layer
		x_image = tf.reshape(x,[-1,inputSize,inputSize,channels])
		h_conv = None
		input_tensor = x_image
		input_chan = channels
		for i in range(convLayers):
			h_conv = self.convLayer(input_tensor,convFeatures,dimension=3,channels=input_chan)
			input_tensor =h_conv
			input_chan = convFeatures			
		#Create second convolutional layer
		poolshape = h_conv.get_shape()
		size = int(poolshape[1] * poolshape[2] * poolshape[3])
		h_pool_flat = tf.reshape(h_conv,[-1,size])
		#First Fully Connected layer
		features = 1024
		h_fcl = self.fullyConnectedLayer(h_pool_flat,features,activation='relu')
		#Applying drouput
		keep_prob = tf.placeholder(tf.float32,name='keep_prob')
		self.tensors.update({'keep_prob':keep_prob})
		h_fcl_drop = tf.nn.dropout(h_fcl,keep_prob)
		#Fully connected layer 2
		y_conv = self.fullyConnectedLayer(h_fcl_drop,outputSize,activation='softmax')
		self.tensors.update({'y_conv':y_conv})
		session = tf.InteractiveSession()
		self.session = session
		
