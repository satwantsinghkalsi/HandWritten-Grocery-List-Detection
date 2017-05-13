Readme:
======================================================
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
