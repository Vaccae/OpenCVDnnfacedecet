#include<opencv2/opencv.hpp>
#include<iostream>
#include <direct.h>
#include "dnnfacedetect.h"
#include "chkfacemark.h"

using namespace std;
using namespace cv;

//����ͼ������
void MatResize(Mat& frame, int maxwidth = 800, int maxheight = 600);

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
		chkfacemark facemarkdetect = chkfacemark(LBFModel);
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
			cout << "cols:" << frame.cols << "  rows:" << frame.rows << endl;
			if (!frame.empty()) {
				vector<Rect> rects;
				bool blcheck = fdetect.detectRect(frame, rects);
				if (blcheck) {
					for (int i = 0; i < rects.size(); i++) {
						rectangle(frame, rects[i], Scalar(0, 0, 255));
					}
					//imshow("src2", tmpsrc);

					//��������ؼ���
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
			// ����ʱ���
			time = (static_cast<double>(getTickCount()) - time) / getTickFrequency();
			// �������ʱ��
			cout << "����ʱ�䣺" << time << "��" << endl;

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
