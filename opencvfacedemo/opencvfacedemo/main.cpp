#include<opencv2/opencv.hpp>
#include<iostream>
#include <direct.h>
#include "dnnfacedetect.h"
#include "chkfacemark.h"

using namespace std;
using namespace cv;

//计算图像缩放
void MatResize(Mat& frame, int maxwidth = 800, int maxheight = 600);

int main(int argc, char* argv) {
	//获取程序目录
	char filepath[256];
	_getcwd(filepath, sizeof(filepath));

	cout << filepath << endl;
	//定义模型文件
	string ModelBinary = (string)filepath + "/opencv_face_detector_uint8.pb";
	string ModelDesc = (string)filepath + "/opencv_face_detector.pbtxt";

	//人脸关键点模型
	string LBFModel = (string)filepath + "/lbfmodel.yaml";

	cout << ModelBinary << endl;
	cout << ModelDesc << endl;

	//加载多张图片
	string picdesc = "E:/DCIM/person/";
	vector<string> filenames;
	cv::glob(picdesc, filenames);

	try
	{
		//初始化
		dnnfacedetect fdetect = dnnfacedetect(ModelBinary, ModelDesc);
		if (!fdetect.initdnnNet())
		{
			cout << "初始化DNN人脸检测失败！" << endl;
			return -1;
		}
		//初始化人脸关键点
		double time = static_cast<double>(getTickCount());
		chkfacemark facemarkdetect = chkfacemark(LBFModel);
		// 计算时间差
		time = (static_cast<double>(getTickCount()) - time) / getTickFrequency();
		// 输出运行时间
		cout << "加载关键点模型时间：" << time << "秒" << endl;

		for (int i = 0; i < filenames.size(); ++i) {
			// 记录起始的时钟周期数
			time = static_cast<double>(getTickCount());

			//读取图像
			Mat frame = imread(filenames[i]);
			//计算图像大小进行缩放
			MatResize(frame);
			cout << "cols:" << frame.cols << "  rows:" << frame.rows << endl;
			if (!frame.empty()) {
				vector<Rect> rects;
				bool blcheck = fdetect.detectRect(frame, rects);
				if (blcheck) {
					for (int i = 0; i < rects.size(); i++) {
						rectangle(frame, rects[i], Scalar(0, 0, 255));
					}
					//imshow("src2", tmpsrc);

					//检测人脸关键点
					vector<vector<Point2f>> facemarks;
					blcheck = facemarkdetect.facemarkdetector(frame, rects, facemarks);
					if (blcheck) {
						for (int i = 0; i < facemarks.size(); i++) {
							drawFacemarks(frame, facemarks[i]);
						}
					}
					imshow("src3", frame);
				}
			}
			// 计算时间差
			time = (static_cast<double>(getTickCount()) - time) / getTickFrequency();
			// 输出运行时间
			cout << "运行时间：" << time << "秒" << endl;

			waitKey(0);
		}

	}
	catch (const std::exception& ex)
	{
		cout << ex.what() << endl;
	}

	waitKey(0);
	return 0;
}

//计算图像缩放
void MatResize(Mat& frame, int maxwidth, int maxheight)
{
	double scale;
	//判断图像是水平还是垂直
	bool isHorizontal = frame.cols > frame.rows ? true : false;

	//根据水平还是垂直计算缩放
	if (isHorizontal) {
		if (frame.cols > maxwidth) {
			scale = (double)maxwidth / frame.cols;
			resize(frame, frame, Size(0, 0), scale, scale);
	    }
	}
	else {
		if (frame.rows > maxheight) {
			scale = (double)maxheight / frame.rows;
			resize(frame, frame, Size(0, 0), scale, scale);
		}
	}
}
