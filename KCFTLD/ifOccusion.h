/*******************************************************************
*Copyright (c) 2016 XXXX Corporation
*All Rights Reserved.
*
*Project Name          :   OpenTLD_V1.0
*File Name             :   track_V1.0.c
*Abstract Description  :   track the object from current frame
*class Name            :   LKtracker

*Create Date           :   2016/07/30
*Author                :   Uwen
*
*--------------------------Revision History---------------------------
*No.  Version    Date          Revised By     Description
*01   V1.0       2016/08/06    Uwen           the first version
*********************************************************************/
#include <iostream>
#include <opencv.hpp>


using namespace std;
using namespace cv;

class BoundingBox : public cv::Rect
{
public:
	BoundingBox()
	{};

	BoundingBox(cv::Rect r) : cv::Rect(r)
	{};

	~BoundingBox()
	{};

	float overlap;          //Overlap with current Bounding Box
	int   sidx;             //scale index 
};

class LKtracker
{
public:
	LKtracker();
	~LKtracker();

	void throwPoint_v(vector<Point2f>& CurrPt_cv32P, const BoundingBox bb);

	bool getPredictPt(const Mat& CurrImg_con_r_cvM, const Mat& NextImg_con_r_cvM,
		vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32);

	void getCorrelateMedian_v(const Mat& CurrImg_con_r_cvM, const Mat& NextImg_con_r_cvM,
		vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32);

	bool filterPts(vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32);

	void PredictObj_v(const vector<Point2f>& CurrPoints_r_vt_cvP32, const vector<Point2f>& NextPoints_r_vt_cvP32,
		const BoundingBox& CurrBox_con_r_st, BoundingBox& NextBox_r_st);

	float median(vector<float> num);

	float mGetBackwardErrMedian() { return mBackwardErrMedian_f; }

private:
	vector<Point2f> mPointsFB_vt_cvP32;

	vector<uchar> mForwardStatus_b_vt;    //ǰ��Ԥ��ʱ�����Ӧ�����Ĺ��������֣�����
	//�е�ÿ��Ԫ�ض�������Ϊ1����������Ϊ0

	vector<uchar> mBackwardStatus_b_vt;//����Ԥ��ʱ��status

	vector<float> mForwardDistErr_f_vt;//ǰ��Ԥ��ʱԤ��ĵ���ԭ���ĵ�ľ���

	vector<float> mBackwardDistErr_f_vt;//����Ԥ��ʱԤ��ĵ���ԭ���ĵ�ľ���

	float mForwardErrMedian_f;
	float mBackwardErrMedian_f;
};