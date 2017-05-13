#!/bin/sh
clf=$1
echo $1
cd /mnt/c/Users/satwa/Documents/CS6375ML/FinalProject/Handwritten-Grocery-List-Detection/tst
#cmake .
#make
rm -r *.jpg
./Segmentation /mnt/c/Users/satwa/Documents/CS6375ML/FinalProject/Handwritten-Grocery-List-Detection/inputimage/$2.png
cd /mnt/c/Users/satwa/Documents/CS6375ML/FinalProject/Handwritten-Grocery-List-Detection/svmTrainTest/
#echo "Work on cv.."
#workon cv
#echo "Extract features and train for characters..."
#python hogForChar.py $1 > hfc.out
echo $1$2.txt
python hogForTest.py $1 $2.txt > ../hft.out

