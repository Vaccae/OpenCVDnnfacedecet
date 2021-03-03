#include "dnnfacedetect.h"




//���캯��
dnnfacedetect::dnnfacedetect(string modelBinary, string modelDesc)
{
	_modelbinary = modelBinary;
	_modeldesc = modelDesc;

	//��ʼ��������ֵ
	confidenceThreshold = 0.5;
	inScaleFactor = 0.5;
	inWidth = 300;
	inHeight = 300;
	meanVal = Scalar(104.0, 177.0, 123.0);
}

dnnfacedetect::~dnnfacedetect()
{
	_net.~Net();
}

//��ʼ��dnnnet
bool dnnfacedetect::initdnnNet()
{
	try
	{
		_net = dnn::readNetFromTensorflow(_modelbinary, _modeldesc);
		_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
		_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
	}
	catch (const std::exception& ex)
	{
		throw ex.what();
	}

	return !_net.empty();
}

//�������
vector<Mat> dnnfacedetect::detect(Mat frame)
{
	Mat tmpsrc = frame;
	vector<Mat> dsts = vector<Mat>();
	// �޸�ͨ����
	if (tmpsrc.channels() == 4)
		cvtColor(tmpsrc, tmpsrc, COLOR_BGRA2BGR);
	// �������ݵ���
	Mat inputBlob = dnn::blobFromImage(tmpsrc, inScaleFactor,
		Size(inWidth, inHeight), meanVal, false, false);
	_net.setInput(inputBlob, "data");

	//�������
	Mat detection = _net.forward("detection_out");

	Mat detectionMat(detection.size[2], detection.size[3],
		CV_32F, detection.ptr<float>());

	//�����Ľ�����л��ƺʹ�ŵ�dsts��
	for (int i = 0; i < detectionMat.rows; i++) {
		//��ֵ�Ȼ�ȡ
		float confidence = detectionMat.at<float>(i, 2);
		//���������ֵ˵����⵽����
		if (confidence > confidenceThreshold) {
			//�������
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * tmpsrc.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * tmpsrc.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * tmpsrc.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * tmpsrc.rows);
			//���ɾ���
			Rect rect((int)xLeftBottom, (int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));

			//�س�ͼ���δ�ŵ�dsts������
			Mat tmp = tmpsrc(rect);
			dsts.push_back(tmp);

			//��ԭͼ���ú�򻭳�����
			rectangle(frame, rect, Scalar(0, 0, 255));
		}
	}

	return dsts;
}

bool dnnfacedetect::detectRect(Mat frame, vector<Rect> &rects)
{
	Mat tmpsrc = frame;

	// �޸�ͨ����
	if (tmpsrc.channels() == 4)
		cvtColor(tmpsrc, tmpsrc, COLOR_BGRA2BGR);
	// �������ݵ���
	Mat inputBlob = dnn::blobFromImage(tmpsrc, inScaleFactor,
		Size(inWidth, inHeight), meanVal, false, false);
	_net.setInput(inputBlob, "data");

	//�������
	Mat detection = _net.forward("detection_out");

	Mat detectionMat(detection.size[2], detection.size[3],
		CV_32F, detection.ptr<float>());

	if (detectionMat.rows <= 0) return false;

	//�����Ľ�����л��ƺʹ�ŵ�dsts��
	for (int i = 0; i < detectionMat.rows; i++) {
		//��ֵ�Ȼ�ȡ
		float confidence = detectionMat.at<float>(i, 2);
		//���������ֵ˵����⵽����
		if (confidence > confidenceThreshold) {
			//�������
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * tmpsrc.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * tmpsrc.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * tmpsrc.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * tmpsrc.rows);
			//���ɾ��δ������������
			Rect rect((int)xLeftBottom-5, (int)yLeftBottom-5,
				(int)(xRightTop - xLeftBottom)+10,
				(int)(yRightTop - yLeftBottom)+10);

			rects.push_back(rect);
		}
	}

	return true;
}
