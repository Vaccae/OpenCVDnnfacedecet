#include<opencv2/opencv.hpp>
#include<iostream>
#include <direct.h>
#include "dnnfacedetect.h"
#include "chkfacemark.h"
#include "DelaunayCore.h"

using namespace std;
using namespace cv;

//����ͼ������
void MatResize(Mat& frame, int maxwidth = 800, int maxheight = 600);

//��ʼ��ģ������
void InitFaceMarkModel(Mat& frame, Rect rect, Mat& dst, vector<Point2f>& dstfacemarkmodel);

//����任ͼ��
bool WarpAffineFaceMat(Mat& src, vector<vector<Point2f>> srcpoints, Mat& dst, vector<vector<Point2f>>& dstpoints);

Mat srcmodel;
vector<Point2f> facemarkmodel;
chkfacemark facemarkdetect;
vector<Point> contourhull;

vector<vector<Point2f>> srcdelaunay;
vector<vector<Point2f>> dstdelaunay;


int main(int argc, char* argv) {
	//��ȡ����Ŀ¼
	char filepath[256];
	_getcwd(filepath, sizeof(filepath));

	cout << filepath << endl;
	//����ģ���ļ�
	string ModelBinary = (string)filepath + "/opencv_face_detector_uint8.pb";
	string ModelDesc = (string)filepath + "/opencv_face_detector.pbtxt";

	//�����ؼ���ģ��
	string LBFModel = (string)filepath + "/lbfmodel.yaml";

	cout << ModelBinary << endl;
	cout << ModelDesc << endl;

	//���ض���ͼƬ
	string picdesc = "E:/DCIM/person/";
	vector<string> filenames;
	cv::glob(picdesc, filenames);



	try
	{
		//��ʼ��
		dnnfacedetect fdetect = dnnfacedetect(ModelBinary, ModelDesc);
		if (!fdetect.initdnnNet())
		{
			cout << "��ʼ��DNN�������ʧ�ܣ�" << endl;
			return -1;
		}
		//��ʼ�������ؼ���
		double time = static_cast<double>(getTickCount());
		facemarkdetect = chkfacemark(LBFModel);
		// ����ʱ���
		time = (static_cast<double>(getTickCount()) - time) / getTickFrequency();
		// �������ʱ��
		cout << "���عؼ���ģ��ʱ�䣺" << time << "��" << endl;

		for (int i = 0; i < filenames.size(); ++i) {
			// ��¼��ʼ��ʱ��������
			time = static_cast<double>(getTickCount());

			//��ȡͼ��
			Mat frame = imread(filenames[i]);
			//����ͼ���С��������
			MatResize(frame);
			Mat tmpframe;
			frame.copyTo(tmpframe);
			cout << "cols:" << frame.cols << "  rows:" << frame.rows << endl;
			if (!frame.empty()) {
				//����ͼƬ�����Ŀ��
				dstdelaunay.clear();

				vector<Rect> rects;
				bool blcheck = fdetect.detectRect(frame, rects);
				if (blcheck) {
					for (int i = 0; i < rects.size(); i++) {
						if (srcmodel.empty()) {
							InitFaceMarkModel(tmpframe, rects[i], srcmodel, facemarkmodel);

							//������ε�
							//DelaunayCore::InsertRectPoint(facemarkmodel, rects[i]);

							//���������μ��
							srcdelaunay = DelaunayCore::GetTriangleList(srcmodel, contourhull, "srcdelaunay");
						}
						rectangle(frame, rects[i], Scalar(0, 0, 255));
						
					}

					//��������ؼ���
					vector<vector<Point2f>> facemarks;
					blcheck = facemarkdetect.facemarkdetector(tmpframe, rects, facemarks);
					if (blcheck) {
						for (int i = 0; i < facemarks.size(); i++) {
							cout << "�����������" << facemarks[i].size() << endl;
							drawFacemarks(frame, facemarks[i]);

							//���������μ��
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
								//������ε�
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
			// ����ʱ���
			time = (static_cast<double>(getTickCount()) - time) / getTickFrequency();
			// �������ʱ��
			cout << "����ʱ�䣺" << time << "��" << endl;

			waitKey(0);
		}

	}
	catch (const std::exception& ex)
	{
		cout << "Error:" << ex.what() << endl;
	}
	catch (...) {
		cout << "δ֪����" << endl;
	}

	waitKey(0);
	facemarkdetect.~chkfacemark();
	return 0;
}

//����ͼ������
void MatResize(Mat& frame, int maxwidth, int maxheight)
{
	double scale;
	//�ж�ͼ����ˮƽ���Ǵ�ֱ
	bool isHorizontal = frame.cols > frame.rows ? true : false;

	//����ˮƽ���Ǵ�ֱ��������
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
//��ʼ��ģ������
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

		//��vector<Point2f>תΪvector<Point>������͹�����
		vector<Point> facemarkcontour;
		facemarkcontour = CvUtils::Vecpt2fToVecpt(facemarkmodel);

		//͹�����	
		convexHull(facemarkcontour, contourhull);

		//�����ά����vector<vector<Point>>�����ڻ���������
		vector<vector<Point>> contours;
		contours.push_back(contourhull);
		//�����������
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


