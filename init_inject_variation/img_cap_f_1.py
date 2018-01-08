import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Embedding,TimeDistributed
from keras.layers import Bidirectional,RepeatVector,Dropout
from keras.layers.merge import add,concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import keras.backend as K
from IPython.display import clear_output
f = open("output.txt","w+")
# i=0

# def on_epoch_end():
#    with open("output.txt","a") as f:
#    	global i
#    	i+=1
#    	f.write("Epoch ")

#    	f.write(str(i))
#    f.close()

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    

    def __init__(self):
        Callback.__init__(self)

#    def on_train_begin(self, logs={}):
 #       self.i = 0
  #      self.x = []
   #     self.losses = []
    #    self.val_losses = []
     #   self.acc = []
      #  self.val_acc = []
       # self.fig = plt.figure()
        
       # self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.iteritems()))
        print(msg)
        with open("output.txt","a") as f:
        	f.write(msg + "\n")
        f.close()
#        self.logs.append(logs)
 #       self.x.append(self.i)
  #      self.losses.append(logs.get('loss'))
   #     self.val_losses.append(logs.get('val_loss'))
    #    self.acc.append(logs.get('acc'))
     #   self.val_acc.append(logs.get('val_acc'))
      #  self.i += 1
       # f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
       # 
       # clear_output(wait=True)
        
       # ax1.set_yscale('log')
       # ax1.plot(self.x, self.losses, label="loss")
       # ax1.plot(self.x, self.val_losses, label="val_loss")
       # ax1.legend()
       # 
       # ax2.plot(self.x, self.acc, label="accuracy")
       # ax2.plot(self.x, self.val_acc, label="validation accuracy")
       # ax2.legend()
        
       # plt.savefig('accuracy_loss_model.png')


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(512, activation='relu')(fe1)
        #fe2 = RepeatVector(max_length)(fe2)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = Bidirectional(LSTM(units = 256,dropout = 0.5,recurrent_dropout = 0.4,return_sequences=True))(se2)
        #se3 = TimeDistributed(Dense(512))(se3)
	# decoder model
        fe3 = np.asarray(fe2)
	hidden_state = fe3
  	#print(fe.shape)
	#decoder1 = concatenate([fe2, se3],axis = -1)
        #decoder1 = Dense(512,activation = 'relu')(decoder1)
	decoder2 = GRU(256)(se3)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
#	model1 = multi_gpu_model(model,gpus=2)
#	model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	# summarize model
	#print(model1.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        model.layers[4].states[0] = hidden_state        
        print(model.summary())
        print(model.layers) 
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# train dataset

# load training dataset (6K)
filename = 'Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# dev dataset

# load test set
filename = 'Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

model = define_model(vocab_size, max_length)
# define checkpoint callback
#filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
# fit model
early = EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 4,verbose = 1,mode = 'auto')
history = model.fit([X1train, X2train], ytrain, epochs=50, verbose=1, batch_size = 64,callbacks=[LoggingCallback(),early], validation_data=([X1test, X2test], ytest))

#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model weights to disk")

#from keras.models import model_from_json
#model = model_from_json(json_string)

#model.save('/others/guest2/aarif/Flickr8kImageCaptioning/model1.best.hdf5')
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
#with open("output.txt") as f:
#	f.write("done with epoch")

#json_string = model.to_json()
#model.save_weights('my_model_weights.h5')

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_model.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_model.png')
