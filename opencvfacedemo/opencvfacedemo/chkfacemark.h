#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

class chkfacemark
{
private:
	string _lfbmodel;
	Ptr<Facemark> _facemark;
public:
	chkfacemark();
	chkfacemark(string lfmodel);

	~chkfacemark();

	//ÈËÁ³¹Ø¼üµã¼ì²â
	bool facemarkdetector(Mat src, vector<Rect> faces, vector<vector<Point2f>> &facemarks);
	bool facemarkdetector(Mat src, Mat face, vector<vector<Point2f>>& facemarks);
};

