#include "chkfacemark.h"


chkfacemark::chkfacemark()
{
	//��������
	_facemark = FacemarkLBF::create();
}

chkfacemark::chkfacemark(string lfmodel)
{
	_lfbmodel = lfmodel;
	//��������
	_facemark = FacemarkLBF::create();
	//����ģ��
	_facemark->loadModel(_lfbmodel);
}

chkfacemark::~chkfacemark()
{
	_facemark.release();
}

// ����68��������
// �Ǽ� 30  
// �Ǹ� 27  
// �°� 8  
// ������� 36  
// �����ڽ� 39  
// ������� 45  
// �����ڽ� 42  
// ������   66  
// �����   48  
// ���ҽ�   54  
// �������� 0  
// �������� 16 
bool chkfacemark::facemarkdetector(Mat src, vector<Rect> faces, vector<vector<Point2f>> &facemarks)
{
	return _facemark->fit(src, faces, facemarks);
}

bool chkfacemark::facemarkdetector(Mat src, Mat face, vector<vector<Point2f>>& facemarks)
{
	return _facemark->fit(src, face, facemarks);
}
