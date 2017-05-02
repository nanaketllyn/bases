from sklearn.metrics import classification_report
import numpy as np
import cv2
import cPickle
import gzip

# import the necessary packages
#from sklearn import datasets
from nolearn.dbn import DBN

# grab the MNIST dataset (if this is the first time you are running
# this script, this make take a minute -- the 55mb MNIST digit dataset
# will be downloaded)
#print "[X] downloading data..."
#dataset = datasets.fetch_mldata("MNIST Original")



train_set, test_set = cPickle.load(gzip.open('base2.pkl.gz', 'rb'))
(trainX, trainY), (testX, testY) = train_set, test_set

trainX = trainX.reshape(trainX.shape[0], 2704L)
testX = testX.reshape(testX.shape[0], 2704L)
trainX = trainX.astype('float')
testX = testX.astype('float')
trainX /= 255
testX /= 255

#trainY = np_utils.to_categorical(trainY, 2)
#testY = np_utils.to_categorical(testY,2)

# scale the data to the range [0, 1] and then construct the training
# and testing splits
#(trainX, testX, trainY, testY) = train_test_split(
#	dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)

# train the Deep Belief Network with 784 input units (the flattened,
#  28x28 grayscale image), 800 hidden units in the 1st hidden layer,
# 800 hidden nodes in the 2nd hidden layer, and 10 output units (one
# for each possible output classification, which are the digits 1-10)
dbn = DBN(
	[trainX.shape[1], 30, 2],
	learn_rates = 0.1,
	learn_rate_decays = 0.9,
	epochs = 15,
	verbose = 1)
dbn.fit(trainX, trainY)
	
listPredict = []

listPredict = dbn.predict(testX)

vp, fp, vn, fn = 0, 0, 0, 0

for i in xrange(testX.shape[0]):
    if testY[i] == 0 and listPredict[i] == 0:
        vn += 1
    elif testY[i] == 0 and listPredict[i] == 1:
        fp += 1
    elif testY[i] == 1 and listPredict[i] == 1:
        vp += 1
    else:
        fn += 1

sen = float((vp * 1.0) / (vp + fn))

esp = float((vn * 1.0) / (vn + fp))

acu = float(((vp + vn) * 1.0)/ (vn + vp + fp + fn))
##print (sen, esp)

print ("\nSensitivity: " + str(sen) + " Specificity: " + str(esp) + " Acuracy: " + str(acu))

##print (sen, esp)

print ("\n\n")

print ("vp: " + str(vp) + " vn: " + str(vn) + " fn: " + str(fn) + " fp: " + str(fp))
# compute the predictions for the test data and show a classification
# report
preds = dbn.predict(testX)
print classification_report(testY, preds)

# randomly select a few of the test instances
#for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
	# classify the digit
#	pred = dbn.predict(np.atleast_2d(testX[i]))
 
	# reshape the feature vector to be a 28x28 pixel image, then change
	# the data type to be an unsigned 8-bit integer
#	image = (testX[i] * 255).reshape((52, 52)).astype("uint8")
 
	# show the image and prediction
#	print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
#	cv2.imshow("Digit", image)
#	cv2.waitKey(0)