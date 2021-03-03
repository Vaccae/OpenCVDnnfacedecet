#include "chkfacemark.h"


chkfacemark::chkfacemark()
{
	//创建对象
	_facemark = FacemarkLBF::create();
}

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

// 人脸68个特征点
// 鼻尖 30  
// 鼻根 27  
// 下巴 8  
// 左眼外角 36  
// 左眼内角 39  
// 右眼外角 45  
// 右眼内角 42  
// 嘴中心   66  
// 嘴左角   48  
// 嘴右角   54  
// 左脸最外 0  
// 右脸最外 16 
bool chkfacemark::facemarkdetector(Mat src, vector<Rect> faces, vector<vector<Point2f>> &facemarks)
{
	return _facemark->fit(src, faces, facemarks);
}

bool chkfacemark::facemarkdetector(Mat src, Mat face, vector<vector<Point2f>>& facemarks)
{
	return _facemark->fit(src, face, facemarks);
}
