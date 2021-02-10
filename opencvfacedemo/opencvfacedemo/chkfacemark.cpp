#include "chkfacemark.h"


chkfacemark::chkfacemark(string lfmodel)
{
	_lfbmodel = lfmodel;
	//创建对象
	_facemark = FacemarkLBF::create();
	//加载模型
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
