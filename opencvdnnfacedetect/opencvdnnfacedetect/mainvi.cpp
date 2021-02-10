#include<opencv2/opencv.hpp>
#include<iostream>
#include <direct.h>
#include "dnnfacedetect.h"

using namespace std;
using namespace cv;

dnnfacedetect fdetect;
void detectface(Mat frame);

int mainvi(int argc, char* argv) {
	//获取程序目录
	char filepath[256];
	_getcwd(filepath, sizeof(filepath));

	cout << filepath << endl;
	//定义模型文件
	string ModelBinary = (string)filepath + "/opencv_face_detector_uint8.pb";
	string ModelDesc = (string)filepath + "/opencv_face_detector.pbtxt";

	//视频文件
	string videodesc = (string)filepath + "/test.mp4";

	cout << ModelBinary << endl;
	cout << ModelDesc << endl;

	int startfps = 0;
	int detectfps = 3;

	//初始化
	fdetect = dnnfacedetect(ModelBinary, ModelDesc);
	if (!fdetect.initdnnNet())
	{
		cout << "初始化DNN人脸检测失败！" << endl;
		return -1;
	}

	//加载视频
	Mat frame;
	VideoCapture video;
	video.open(videodesc);
	if (!video.isOpened())
	{
		cout << "视频加载失败！" << endl;
		return -1;
	}


	try
	{
		//读取图像每一帧
		while (video.read(frame))
		{
			if (startfps % detectfps == 0) {
				double t = (double)getTickCount();
				//旋转90度
				rotate(frame, frame, 0);
				//缩放图片
				resize(frame, frame, cv::Size(0, 0), 0.6, 0.6);
				//人脸检测
				detectface(frame);

				imshow("src", frame);

				t = ((double)getTickCount() - t) / getTickFrequency();
				cout << "执行时间(秒): " << t << endl;
			}
			startfps++;
			char c = waitKey(1);
			if (c == 27)
			{
				break;
			}
		}
		cout << "检测完成" << endl;
	}
	catch (const std::exception & ex)
	{
		cout << ex.what() << endl;
	}

	video.release();
	waitKey(0);
	return 0;
}

void detectface(Mat frame)
{
	if (!frame.empty()) {
		vector<Mat> dst = fdetect.detect(frame);
		if (!dst.empty()) {
			for (int i = 0; i < dst.size(); i++) {
				string title = "detectface" + i;
				imshow(title, dst[i]);
			}
		}
	}
}
