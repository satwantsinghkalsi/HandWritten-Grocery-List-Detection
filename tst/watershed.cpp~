#include <opencv2/opencv.hpp>
#include <string.h>
#include <string> 
#include <stdlib.h>
#include <sstream>
using namespace cv;
using namespace std;

int main(char* argv)
{
   if(argv[1] == ""){
	cout<<"Provide input image";
        return 0;
   }

    // Read image
    Mat3b img = imread(argv[1]);

    // Binarize image. Text is white, background is black
    Mat1b bin;
    cvtColor(img, bin, COLOR_BGR2GRAY);
    bin = bin < 200;

    // Find all white pixels
    /*vector<Point> pts;
    findNonZero(bin, pts);
    // Get rotated rect of white pixels
    RotatedRect box = minAreaRect(pts);
    if (box.size.width > box.size.height)
    {
        int w = box.size.width;
        box.size.width = box.size.height;
        box.size.height = w;
        box.angle += CV_PI / 2.0;
    }
    Point2f vertices[4];
    box.points(vertices);*/
    // Rotate the image according to the found angle

    Mat1b rotated;
    bin.copyTo(rotated);

   // Mat M = getRotationMatrix2D(box.center, box.angle, 1.0);    //warpAffine(bin, rotated, M, bin.size());

    // Compute horizontal projections
    Mat1f horProj;
    reduce(rotated, horProj, 1, CV_REDUCE_AVG);

    // Remove noise in histogram. White bins identify space lines, black bins identify text lines
    float th = 0;
    Mat1b hist = horProj <= th;
	
    // Get mean coordinate of white white pixels groups
    vector<int> ycoords;
    int y = 0;
    int count = 0;
    bool isSpace = false;
    for (int i = 0; i < rotated.rows; ++i)
    {
        if (!isSpace)
        {
            if (hist(i))
            {
                isSpace = true;
                count = 1;
                y = i;
            }
        }
        else
        {
            if (!hist(i))
            {
                isSpace = false;
                ycoords.push_back(y / count);
            }
            else
            {
                y += i;
                count++;
            }
        }
    }
    // Draw line as final result
    Mat3b result;
    cvtColor(rotated, result, COLOR_GRAY2BGR);

if(ycoords.size()>0){
    for (int i = 0; i < ycoords.size()-1; i++)
    {
	Rect rect1;
	rect1.x = 0;
	rect1.y = ycoords[i];
	rect1.width = result.size().width;
	rect1.height = ycoords[i+1]-ycoords[i];
        if(rect1.height > 30){
		Mat Image1 = result(rect1);
		imshow("Display Image", Image1); 
		string name = "";
		std::stringstream ss; 
		ss << i;
		name = "Image"+ss.str()+".jpg";
		imwrite(name,Image1);
		waitKey(0);
        }
	if(i == ycoords.size()-2){
		Rect rect1;
		rect1.x = 0;
		rect1.y = ycoords[i+1];
		rect1.width = result.size().width;
		rect1.height = result.size().height-ycoords[i+1];
		if(rect1.height > 30){
			Mat Image1 = result(rect1);
			imshow("Display Image", Image1); 
			string name = "";
			std::stringstream ss; 
			ss << i+1;
			name = "Image"+ss.str()+".jpg";
			imwrite(name,Image1);
			waitKey(0);
		}	
	}
    }
}
else
   cout<<"No coordinates formed";
   return 0;
}
