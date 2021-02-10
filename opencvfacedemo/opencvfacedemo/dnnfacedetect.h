#pragma once

#include<opencv2/opencv.hpp>
#include<opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

class dnnfacedetect
{
private:
	string _modelbinary, _modeldesc;
	dnn::Net _net;
public:
	//构造函数 传入模型文件
	dnnfacedetect(string modelBinary, string modelDesc);

	~dnnfacedetect();
	//置信阈值
	float confidenceThreshold;
	double inScaleFactor;
	size_t inWidth;
	size_t inHeight;
	Scalar meanVal;

	//初始化DNN网络
	bool initdnnNet();

	//人脸检测
	vector<Mat> detect(Mat frame);

	bool detectRect(Mat frame, vector<Rect> &rects);
};

