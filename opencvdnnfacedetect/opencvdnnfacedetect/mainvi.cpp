#include<opencv2/opencv.hpp>
#include<iostream>
#include <direct.h>
#include "dnnfacedetect.h"

using namespace std;
using namespace cv;

dnnfacedetect fdetect;
void detectface(Mat frame);

int mainvi(int argc, char* argv) {
	//��ȡ����Ŀ¼
	char filepath[256];
	_getcwd(filepath, sizeof(filepath));

	cout << filepath << endl;
	//����ģ���ļ�
	string ModelBinary = (string)filepath + "/opencv_face_detector_uint8.pb";
	string ModelDesc = (string)filepath + "/opencv_face_detector.pbtxt";

	//��Ƶ�ļ�
	string videodesc = (string)filepath + "/test.mp4";

	cout << ModelBinary << endl;
	cout << ModelDesc << endl;

	int startfps = 0;
	int detectfps = 3;

	//��ʼ��
	fdetect = dnnfacedetect(ModelBinary, ModelDesc);
	if (!fdetect.initdnnNet())
	{
		cout << "��ʼ��DNN�������ʧ�ܣ�" << endl;
		return -1;
	}

	//������Ƶ
	Mat frame;
	VideoCapture video;
	video.open(videodesc);
	if (!video.isOpened())
	{
		cout << "��Ƶ����ʧ�ܣ�" << endl;
		return -1;
	}


	try
	{
		//��ȡͼ��ÿһ֡
		while (video.read(frame))
		{
			if (startfps % detectfps == 0) {
				double t = (double)getTickCount();
				//��ת90��
				rotate(frame, frame, 0);
				//����ͼƬ
				resize(frame, frame, cv::Size(0, 0), 0.6, 0.6);
				//�������
				detectface(frame);

				imshow("src", frame);

				t = ((double)getTickCount() - t) / getTickFrequency();
				cout << "ִ��ʱ��(��): " << t << endl;
			}
			startfps++;
			char c = waitKey(1);
			if (c == 27)
			{
				break;
			}
		}
		cout << "������" << endl;
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
