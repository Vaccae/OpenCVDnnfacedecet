#include "chkfacemark.h"


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

bool chkfacemark::facemarkdetector(Mat src, vector<Rect> faces, vector<vector<Point2f>> &facemarks)
{
	return _facemark->fit(src, faces, facemarks);
}
