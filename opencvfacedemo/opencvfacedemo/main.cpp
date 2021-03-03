#include<opencv2/opencv.hpp>
#include<iostream>
#include <direct.h>
#include "dnnfacedetect.h"
#include "chkfacemark.h"
#include "DelaunayCore.h"

using namespace std;
using namespace cv;

//计算图像缩放
void MatResize(Mat& frame, int maxwidth = 800, int maxheight = 600);

//初始化模版人脸
void InitFaceMarkModel(Mat& frame, Rect rect, Mat& dst, vector<Point2f>& dstfacemarkmodel);

//仿射变换图像
bool WarpAffineFaceMat(Mat& src, vector<vector<Point2f>> srcpoints, Mat& dst, vector<vector<Point2f>>& dstpoints);

Mat srcmodel;
vector<Point2f> facemarkmodel;
chkfacemark facemarkdetect;
vector<Point> contourhull;

vector<vector<Point2f>> srcdelaunay;
vector<vector<Point2f>> dstdelaunay;


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
		facemarkdetect = chkfacemark(LBFModel);
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
			Mat tmpframe;
			frame.copyTo(tmpframe);
			cout << "cols:" << frame.cols << "  rows:" << frame.rows << endl;
			if (!frame.empty()) {
				//换了图片后清除目标
				dstdelaunay.clear();

				vector<Rect> rects;
				bool blcheck = fdetect.detectRect(frame, rects);
				if (blcheck) {
					for (int i = 0; i < rects.size(); i++) {
						if (srcmodel.empty()) {
							InitFaceMarkModel(tmpframe, rects[i], srcmodel, facemarkmodel);

							//加入矩形点
							//DelaunayCore::InsertRectPoint(facemarkmodel, rects[i]);

							//人脸三角形检测
							srcdelaunay = DelaunayCore::GetTriangleList(srcmodel, contourhull, "srcdelaunay");
						}
						rectangle(frame, rects[i], Scalar(0, 0, 255));
						
					}

					//检测人脸关键点
					vector<vector<Point2f>> facemarks;
					blcheck = facemarkdetect.facemarkdetector(tmpframe, rects, facemarks);
					if (blcheck) {
						for (int i = 0; i < facemarks.size(); i++) {
							cout << "特征点个数：" << facemarks[i].size() << endl;
							drawFacemarks(frame, facemarks[i]);

							//人脸三角形检测
							if (dstdelaunay.empty()) {
								Rect rect;
								for (Rect tmprect : rects) {
									if (tmprect.contains(facemarks[i][0])) {
										rect = tmprect;
										break;
									}
								}
								
								vector<Point> tmphull;
								vector<Point> tmpcontour = CvUtils::Vecpt2fToVecpt(facemarks[i]);
								convexHull(tmpcontour, tmphull);
								//加入矩形点
								// DelaunayCore::InsertRectPoint(facemarks[i], rect);

								dstdelaunay = DelaunayCore::GetTriangleList(frame, tmphull, "dstdelaunay");
							}
						}
						CvUtils::SetShowWindow(frame, "src3", 200, 20);
						imshow("src3", frame);
						WarpAffineFaceMat(srcmodel, srcdelaunay, frame, dstdelaunay);
					}

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
		cout << "Error:" << ex.what() << endl;
	}
	catch (...) {
		cout << "未知错误" << endl;
	}

	waitKey(0);
	facemarkdetect.~chkfacemark();
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
//初始化模版人脸
void InitFaceMarkModel(Mat& frame, Rect rect, Mat& dst, vector<Point2f>& dstfacemarkmodel)
{
	vector<int> faceid = { 30,27,8,36,39,45,42,66,48,54,0,16 };

	dst = Mat(frame, rect);

	Mat tmpmodel;
	dst.copyTo(tmpmodel);

	Rect tmprect = Rect(0, 0, dst.cols, dst.rows);
	vector<Rect> tmprects;
	tmprects.push_back(tmprect);
	vector<vector<Point2f>> tmpfacemodels;
	facemarkdetect.facemarkdetector(dst, tmprects, tmpfacemodels);
	Mat tmpdst = Mat::zeros(dst.size(), CV_8UC1);
	if (tmpfacemodels.size() > 0) {
		facemarkmodel = tmpfacemodels[0];

		//将vector<Point2f>转为vector<Point>用于做凸包检测
		vector<Point> facemarkcontour;
		facemarkcontour = CvUtils::Vecpt2fToVecpt(facemarkmodel);

		//凸包检测	
		convexHull(facemarkcontour, contourhull);

		//存入二维数组vector<vector<Point>>中用于画轮廓处理
		vector<vector<Point>> contours;
		contours.push_back(contourhull);
		//画轮廓并添充
		drawContours(tmpdst, contours, -1, Scalar(255, 255, 255), -1);
	}

	dst = Mat::zeros(tmpmodel.size(), CV_8UC3);
	tmpmodel.copyTo(dst, tmpdst);
	imshow("dst", dst);

	imshow("srcmodel", tmpmodel);
	imshow("tmpdst", tmpdst);
}

bool WarpAffineFaceMat(Mat& src, vector<vector<Point2f>> srcpoints, Mat& dst, vector<vector<Point2f>>& dstpoints)
{
	Mat tmpdst = Mat::zeros(dst.size(), CV_8UC3);
	Rect rect = Rect(0, 0, src.cols, src.rows);
	src.copyTo(tmpdst(rect));

	DelaunayCore::WarpAffineFaceMark(src, tmpdst, srcpoints, dstpoints);

	imshow("tmpfacemat", tmpdst);

	return false;
}


