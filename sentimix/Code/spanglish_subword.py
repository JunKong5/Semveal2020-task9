import numpy as np
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn import metrics
import pandas as pd
from sklearn.metrics import f1_score
from keras.utils import np_utils
from MyNormalizer import token


Masterdir = 'E:\sentimix/'

train_file = 'dataprocess/spanlish/Data/train_user_hashtag_hindi_nochar.tsv'
dev_file = 'dataprocess/spanlish/Data/dev_user_hashtag_hindi_nochar.tsv'
test_file = 'dataprocess/spanlish/Data/test_user_hashtag_hindi_nochar.tsv'

#Data I/O formatting
SEPERATOR = '\t'
DATA_COLUMN = 0
LABEL_COLUMN = 1
DATAtest_COLUMN = 1
id_COLUMN = 0
LABELS = ['negative','neutral','positive'] # 0 -> Negative, 1-> Neutral, 2-> Neutral
mapping_char2num = {}
mapping_num2char = {}
MAXLEN = 200

#LSTM Model Parameters
#Embedding
MAX_FEATURES = 0
embedding_size = 128
# Convolution
filter_length = 3
nb_filter = 128
pool_length = 3
# LSTM
lstm_output_size = 128
# Training
batch_size = 128
number_of_epochs = 5
numclasses = 3
########################################################

def parse(Masterdir,filename,seperator,datacol,labelcol,labels):
	"""
	Purpose -> Data I/O
	Input   -> Data file containing sentences and labels along with the global variables
	Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
	"""
	#Reads the files and splits data into individual lines
	f=open(Masterdir+filename,'r', encoding='UTF-8')
	lines = f.read().lower()
	lines = lines.lower().split('\n')[:-1]

	X_train = []
	Y_train = []
	
	#Processes individual lines
	for line in lines:
		# Seperator for the current dataset. Currently '\t'. 
		line = line.split(seperator)
		#Token is the function which implements basic preprocessing as mentioned in our paper
		tokenized_lines = token(line[datacol])
		
		#Creates character lists
		char_list = []
		for words in tokenized_lines:
			for char in words:
				char_list.append(char)
			char_list.append(' ')
		#print(char_list) - Debugs the character list created
		X_train.append(char_list)

		#Appends labels
		# print(line[labelcol])
		if line[labelcol] == labels[0]:
			Y_train.append(0)
		if line[labelcol] == labels[1]:
			Y_train.append(1)
		if line[labelcol] == labels[2]:
			Y_train.append(2)
	
	#Converts Y_train to a numpy array
	# print(len(X_train))
	# print(Y_train)
	Y_train = np.asarray(Y_train)
	# print(Y_train.shape[0])
	assert(len(X_train) == Y_train.shape[0])

	return [X_train,Y_train]

def parsetest(Masterdir, filename, seperator, datacol, idlcol):
	f = open(Masterdir + filename, 'r', encoding='UTF-8')
	lines = f.read().lower()
	lines = lines.lower().split('\n')[:-1]
	X_test = []
	id_test = []
	for line in lines:
		line = line.split(seperator)
		tokenized_lines = token(line[datacol])
		char_list = []
		for words in tokenized_lines:
			for char in words:
				char_list.append(char)
			char_list.append(' ')
		X_test.append(char_list)
		id_test.append(line[idlcol])
	return [X_test, id_test]



def convert_char2num(mapping_n2c,mapping_c2n,trainwords,maxlen):
	"""
	Purpose -> Convert characters to integers, a unique value for every character
	Input   -> Training data (In list of lists format) along with global variables
	Output  -> Converted training data along with global variables
	"""
	allchars = []
	errors = 0

	#Creates a list of all characters present in the dataset
	for line in trainwords:
		try:
			allchars = set(allchars+line)
			allchars = list(allchars)
		except:
			errors += 1

	#print(errors) #Debugging
	#print(allchars) #Debugging 

	#Creates character dictionaries for the characters
	charno = 0
	for char in allchars:
		mapping_char2num[char] = charno
		mapping_num2char[charno] = char
		charno += 1

	assert(len(allchars)==charno) #Checks

	#Converts the data from characters to numbers using dictionaries 
	X_train = []
	for line in trainwords:
		char_list=[]
		for letter in line:
			char_list.append(mapping_char2num[letter])
		#print(no) -- Debugs the number mappings
		X_train.append(char_list)
	print(mapping_char2num)
	print(mapping_num2char)
	#Pads the X_train to get a uniform vector
	#TODO: Automate the selection instead of manual input
	X_train = sequence.pad_sequences(X_train[:], maxlen=maxlen)
	return [X_train,mapping_num2char,mapping_char2num,charno]

def RNN(X_train,y_train,X_dev,y_dev,args):
	"""
	Purpose -> Define and train the proposed LSTM network
	Input   -> Data, Labels and model hyperparameters
	Output  -> Trained LSTM network
	"""
	#Sets the model hyperparameters
	#Embedding hyperparameters
	max_features = args[0]
	maxlen = args[1]
	embedding_size = args[2]
	# Convolution hyperparameters
	filter_length = args[3]
	nb_filter = args[4]
	pool_length = args[5]
	# LSTM hyperparameters
	lstm_output_size = args[6]
	# Training hyperparameters
	batch_size = args[7]
	nb_epoch = args[8]
	numclasses = args[9]


	#Format conversion for y_train for compatibility with Keras
	y_train = np_utils.to_categorical(y_train, numclasses)
	y_dev = np_utils.to_categorical(y_dev, numclasses)
	
	#Build the sequential model
	# Model Architecture is:
	# Input -> Embedding -> Conv1D+Maxpool1D -> LSTM -> LSTM -> FC-1 -> Softmaxloss
	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, embedding_size, input_length=maxlen))
	model.add(Convolution1D(nb_filter=nb_filter,
							filter_length=filter_length,
							border_mode='valid',
							activation='relu',
							subsample_length=1))
	model.add(MaxPooling1D(pool_length=pool_length))
	# model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
	model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
	model.add(Dense(numclasses))
	model.add(Activation('softmax'))

	# Optimizer is Adamax along with categorical crossentropy loss
	model.compile(loss='categorical_crossentropy',
			  	optimizer='adamax',
			  	metrics=['accuracy'])

	es = EarlyStopping(monitor='val_loss', patience=2)
	print('Train...')
	#Trains model for 50 epochs with shuffling after every epoch for training data and validates on validation data
	model.fit(X_train, y_train, 
			  batch_size=batch_size, 
			  shuffle=True, 
			  nb_epoch=nb_epoch,
			  validation_data=(X_dev, y_dev),
			  callbacks=[es]
	)
	return model



def change(labels):
	f = open("E:\sentimix\label/test_data_labels_spanglish.txt", 'r', encoding='UTF-8')
	lines = f.read().lower()
	lines = lines.lower().split('\n')[:-1]
	print(labels)
	labelcol=1
	Y_test=[]
	for line in lines:
		line = line.split(',')
		if line[labelcol] == labels[0]:
			Y_test.append(0)
		if line[labelcol] == labels[1]:
			Y_test.append(1)
		if line[labelcol] == labels[2]:
			Y_test.append(2)

	return Y_test




if __name__ == '__main__':

	label = change(LABELS)
	print(len(label))
	train = parse(Masterdir, train_file, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
	dev = parse(Masterdir, dev_file, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
	test = parsetest(Masterdir, test_file, SEPERATOR, DATAtest_COLUMN, id_COLUMN)
	print(train)
	X_train = train[0]
	X_train = np.asarray(X_train)
	y_train = train[1]

	X_dev = dev[0]
	X_dev = np.asarray(X_dev)
	y_dev = dev[1]

	X_test = test[0]
	X_test = np.asarray(X_test)
	y_test = test[1]
	print('Creating character dictionaries and format conversion in progess...')
	out = convert_char2num(mapping_num2char,mapping_char2num,X_train,MAXLEN)
	mapping_num2char = out[1]
	mapping_char2num = out[2]
	MAX_FEATURES = out[3]
	X_train = np.asarray(out[0])
	y_train = np.asarray(y_train).flatten()

	out = convert_char2num(mapping_num2char,mapping_char2num,X_dev,MAXLEN)
	mapping_num2char = out[1]
	mapping_char2num = out[2]
	X_dev = np.asarray(out[0])
	y_dev = np.asarray(y_dev).flatten()

	out = convert_char2num(mapping_num2char, mapping_char2num, X_test, MAXLEN)
	mapping_num2char = out[1]
	mapping_char2num = out[2]
	X_test = np.asarray(out[0])
	y_test = np.asarray(y_test).flatten()

	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
	
	print('Creating LSTM Network...')
	model = RNN(deepcopy(X_train),deepcopy(y_train),deepcopy(X_dev),deepcopy(y_dev),[MAX_FEATURES, MAXLEN, embedding_size,\
			     filter_length, nb_filter, pool_length, lstm_output_size, batch_size, \
			     number_of_epochs, numclasses])

	
	y_pre_probability = model.predict(X_test, batch_size=batch_size)
	y_pred = np.argmax(y_pre_probability, axis=1)
	print("F1 score is: " + str(f1_score(label, y_pred, average='weighted')))
	target_names = ['class 0', 'class 1','class 2']
	print(metrics.classification_report(label,y_pred , target_names=target_names))