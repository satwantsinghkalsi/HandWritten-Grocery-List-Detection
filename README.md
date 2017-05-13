# HandWritten-Grocery-List-Detection
Comparison between SVM and KNN classifier

Data Set: The chars 74K dataset http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz
Project Idea :   The objective of this project is to design and implement an order generation application that detects a hand written grocery shopping list and converts it to a digital item list using linear SVM algorithm coupled with HOG feature extraction. The data set contains different 62 different characters (English alphabets capital and small, Digits 0-9) written in 55 different styles. We will apply the tools of machine learning to detect the handwritten characters. We will use segmentation, HOG feature extraction and linear SVM model to obtain better performance from any of the constituent machine learning algorithms.
The input here is the images of characters written in 55 different styles. The final output, the digitalized form of the recognized characters, will be obtained by extracting its HOG features and detecting the characters by matching the HOG features from the trained linear SVM model.

Requirements:
			Opencv3.0
			Python 3.0 
			scikit-learn
			numpy
Extract the zip file and inside the folder Handwritten-Grocery-List-Detection type the command:

	sh handWritingRecog.sh <Classifier type> <input image name>
	
Example :
	
	sh handWritingRecog.sh SVM input1

The output is generated in the same folder in file hft.out

Note:
	-The input images are in inputimage folder and it also contains the a text file of the same name as the image conatingn the correct labels to compute accuracy. Make sure to have a similar file for new inputs.
	-The second argument of the above command should only contain the name of the image.
	-The first argument of the command can be either SVM or KNN.


