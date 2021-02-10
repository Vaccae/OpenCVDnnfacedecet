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
	//���캯�� ����ģ���ļ�
	dnnfacedetect(string modelBinary, string modelDesc);

	~dnnfacedetect();
	//������ֵ
	float confidenceThreshold;
	double inScaleFactor;
	size_t inWidth;
	size_t inHeight;
	Scalar meanVal;

	//��ʼ��DNN����
	bool initdnnNet();

	//�������
	vector<Mat> detect(Mat frame);

	bool detectRect(Mat frame, vector<Rect> &rects);
};

