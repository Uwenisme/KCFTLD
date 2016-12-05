#include "ifOccusion.h"

LKtracker::LKtracker()
{

}

LKtracker::~LKtracker()
{

}


//网格均匀撒点，box共10*10=100个特征点  
void LKtracker::throwPoint_v(vector<Point2f>& CurrPt_cv32P, const BoundingBox bb)
{
	int max_pts = 10;
	int margin_h = 0;//bb.width / 4;//采样边界  
	int margin_v = 0;//bb.height/4;
	int stepx = ceil(double((bb.width - 2 * margin_h) / max_pts));
	int stepy = ceil(double((bb.height - 2 * margin_v) / max_pts));

	//网格均匀撒点，box共10*10=100个特征点. 
	for (int y = bb.y + margin_v; y<bb.y + bb.height - margin_v; y += stepy){
		for (int x = bb.x + margin_h; x<bb.x + bb.width - margin_h; x += stepx){
			CurrPt_cv32P.push_back(Point2f(x, y));
		}
	}
}

bool LKtracker::getPredictPt(const Mat& CurrImg_con_r_cvM, const Mat& NextImg_con_r_cvM,
	vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32)

{
	//前向预测
	calcOpticalFlowPyrLK(CurrImg_con_r_cvM, NextImg_con_r_cvM, CurrPoints_r_vt_cvP32,
		NextPoints_r_vt_cvP32, mForwardStatus_b_vt, mForwardDistErr_f_vt, Size(4, 4), 5, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03), 0.5, 0);
	//后向预测
	calcOpticalFlowPyrLK(NextImg_con_r_cvM, CurrImg_con_r_cvM, NextPoints_r_vt_cvP32, mPointsFB_vt_cvP32,
		mBackwardStatus_b_vt, mBackwardDistErr_f_vt, Size(4, 4), 5, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03), 0.5, 0);
	//CUDA!!

	for (int i = 0; i<mPointsFB_vt_cvP32.size(); i++)
	{
		//后向预测的点与原点的欧氏距离
		mBackwardDistErr_f_vt[i] = norm(mPointsFB_vt_cvP32[i] - CurrPoints_r_vt_cvP32[i]);
	}

	//获得前向预测到的点周围图像与原点图像相似度的中位数
	getCorrelateMedian_v(CurrImg_con_r_cvM, NextImg_con_r_cvM,
		CurrPoints_r_vt_cvP32, NextPoints_r_vt_cvP32);
	//选取满足 1.前向预测大于CorrelateMedian的点 2.后向预测小于欧氏距离中位数的点
	return filterPts(CurrPoints_r_vt_cvP32, NextPoints_r_vt_cvP32);
}
/*在前向预测中，得到前面的点周围与后面的点周围相似度，以此获得相似度中位数，用于下面除去相似度较小的一半点*/
void LKtracker::getCorrelateMedian_v(const Mat& CurrImg_con_r_cvM, const Mat& NextImg_con_r_cvM,
	vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32)

{
	Mat CurrSubPix_cvM(10, 10, CV_8U);
	Mat NextSubPix_cvM(10, 10, CV_8U);
	Mat MatchResult_cvM(1, 1, CV_32F);


	for (int i = 0; i < CurrPoints_r_vt_cvP32.size(); i++)
	{
		if (mForwardStatus_b_vt[i])//如果发现前向预测能够预测得到
		{
			//以一点为中心得到大小为10*10为亚像素精度的图像
			getRectSubPix(CurrImg_con_r_cvM, Size(10, 10), CurrPoints_r_vt_cvP32[i], CurrSubPix_cvM);
			getRectSubPix(NextImg_con_r_cvM, Size(10, 10), NextPoints_r_vt_cvP32[i], NextSubPix_cvM);

			//前向预测的与后向预测的点周围图像的相似度
			matchTemplate(CurrSubPix_cvM, NextSubPix_cvM, MatchResult_cvM, CV_TM_CCOEFF_NORMED);

			mForwardDistErr_f_vt[i] = ((float *)MatchResult_cvM.data)[0];
		}
		else
		{
			mForwardDistErr_f_vt[i] = 0.0;
		}
	}

	/*求中位数*/
	mForwardErrMedian_f = median(mForwardDistErr_f_vt);
}
/*除去前向预测点周围图像相似度较小一半和后向预测距离较大一半的点*/
bool LKtracker::filterPts(vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32)
{
	int PassPts = 0;

	for (int i = 0; i < NextPoints_r_vt_cvP32.size(); i++)
	{
		if (mForwardDistErr_f_vt[i]>mForwardErrMedian_f)//选出前向预测大于CorrelateMedian的点
		{
			CurrPoints_r_vt_cvP32[PassPts] = CurrPoints_r_vt_cvP32[i];
			NextPoints_r_vt_cvP32[PassPts] = NextPoints_r_vt_cvP32[i];
			mBackwardDistErr_f_vt[PassPts] = mBackwardDistErr_f_vt[i];
			PassPts++;
		}
	}

	if (PassPts == 0)
		return false;
	CurrPoints_r_vt_cvP32.resize(PassPts);
	NextPoints_r_vt_cvP32.resize(PassPts);
	mBackwardDistErr_f_vt.resize(PassPts);

	mBackwardErrMedian_f = median(mBackwardDistErr_f_vt);

	PassPts = 0;

	for (int i = 0; i < NextPoints_r_vt_cvP32.size(); i++)//选出后向预测小于欧氏距离中位数的点
	{
		if (!mForwardStatus_b_vt[i])
			continue;
		if (mBackwardDistErr_f_vt[i] <= mBackwardErrMedian_f)
		{
			CurrPoints_r_vt_cvP32[PassPts] = CurrPoints_r_vt_cvP32[i];
			NextPoints_r_vt_cvP32[PassPts] = NextPoints_r_vt_cvP32[i];
			PassPts++;
		}
	}

	CurrPoints_r_vt_cvP32.resize(PassPts);
	NextPoints_r_vt_cvP32.resize(PassPts);
	printf("%d\n", PassPts);
	if (PassPts > 0)
		return true;
	else
		return false;
}

float LKtracker::median(vector<float> num)
{
	int n = floor(double(num.size() / 2));
	nth_element(num.begin(), num.begin() + n, num.end());
	return num[n];
}

void LKtracker::PredictObj_v(const vector<Point2f>& CurrPoints_r_vt_cvP32, const vector<Point2f>& NextPoints_r_vt_cvP32,
	const BoundingBox& CurrBox_con_r_st, BoundingBox& NextBox_r_st)

{
	int nPoint_i = CurrPoints_r_vt_cvP32.size();
	vector<float> xoff_vt_f(nPoint_i);//用来计算x方向位移
	vector<float> yoff_vt_f(nPoint_i);//用来计算y方向位移

	printf("Track points: %d\n", nPoint_i);

	for (int i = 0; i < nPoint_i; i++)
	{
		xoff_vt_f[i] = NextPoints_r_vt_cvP32[i].x - CurrPoints_r_vt_cvP32[i].x;
		yoff_vt_f[i] = NextPoints_r_vt_cvP32[i].y - CurrPoints_r_vt_cvP32[i].y;
	}

	float dx = median(xoff_vt_f);
	float dy = median(yoff_vt_f);

	float scale_f;
	if (nPoint_i>1)
	{
		vector<float> d(nPoint_i);
		d.reserve(nPoint_i*(nPoint_i - 1) / 2); //等差数列求和：1+2+...+(npoints-1),求得最终d的数量，减少push进d时间
		for (int i = 0; i < nPoint_i; i++)
		{
			for (int j = i + 1; j < nPoint_i; j++)
			{
				//前后两图中各图点与点之间比值大小，以此确定缩放大小
				d.push_back(norm(NextPoints_r_vt_cvP32[i] - NextPoints_r_vt_cvP32[j])
					/ norm(CurrPoints_r_vt_cvP32[i] - CurrPoints_r_vt_cvP32[j]));
			}
		}
		scale_f = median(d);
	}
	else
	{
		scale_f = 1;
	}

	float sx = 0.5*(scale_f - 1)*CurrBox_con_r_st.width;
	float sy = 0.5*(scale_f - 1)*CurrBox_con_r_st.height;

	NextBox_r_st.x = round(CurrBox_con_r_st.x + dx - sx);
	NextBox_r_st.y = round(CurrBox_con_r_st.y + dy - sy);
	NextBox_r_st.width = round(CurrBox_con_r_st.width*scale_f);
	NextBox_r_st.height = round(CurrBox_con_r_st.height*scale_f);

}