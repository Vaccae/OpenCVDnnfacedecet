#include<opencv2/opencv.hpp>
#include<iostream>
#include <direct.h>
#include "dnnfacedetect.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv) {
	//��ȡ����Ŀ¼
	char filepath[256];
	_getcwd(filepath, sizeof(filepath));

	cout << filepath << endl;
	//����ģ���ļ�
	string ModelBinary = (string)filepath + "/opencv_face_detector_uint8.pb";
	string ModelDesc = (string)filepath + "/opencv_face_detector.pbtxt";

	//ͼƬ�ļ�
	string picdesc = (string)filepath + "/lena.jpg";

	cout << ModelBinary << endl;
	cout << ModelDesc << endl;

	//����ͼƬ
	Mat frame = imread(picdesc);
	imshow("src", frame);

	try
	{
		//��ʼ��
		dnnfacedetect fdetect = dnnfacedetect(ModelBinary, ModelDesc);
		if (!fdetect.initdnnNet())
		{
			cout << "��ʼ��DNN�������ʧ�ܣ�" << endl;
			return -1;
		}

		if (!frame.empty()) {
			vector<Mat> dst = fdetect.detect(frame);
			if (!dst.empty()) {
				for (int i = 0; i < dst.size(); i++) {
					string title = "dst" + i;
					imshow(title, dst[i]);
				}
				imshow("src2", frame);
			}
		}
	}
	catch (const std::exception & ex)
	{
		cout << ex.what() << endl;
	}

	waitKey(0);
	return 0;
}

