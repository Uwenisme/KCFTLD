#include "ifOccusion.h"

LKtracker::LKtracker()
{

}

LKtracker::~LKtracker()
{

}


//����������㣬box��10*10=100��������  
void LKtracker::throwPoint_v(vector<Point2f>& CurrPt_cv32P, const BoundingBox bb)
{
	int max_pts = 10;
	int margin_h = 0;//bb.width / 4;//�����߽�  
	int margin_v = 0;//bb.height/4;
	int stepx = ceil(double((bb.width - 2 * margin_h) / max_pts));
	int stepy = ceil(double((bb.height - 2 * margin_v) / max_pts));

	//����������㣬box��10*10=100��������. 
	for (int y = bb.y + margin_v; y<bb.y + bb.height - margin_v; y += stepy){
		for (int x = bb.x + margin_h; x<bb.x + bb.width - margin_h; x += stepx){
			CurrPt_cv32P.push_back(Point2f(x, y));
		}
	}
}

bool LKtracker::getPredictPt(const Mat& CurrImg_con_r_cvM, const Mat& NextImg_con_r_cvM,
	vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32)

{
	//ǰ��Ԥ��
	calcOpticalFlowPyrLK(CurrImg_con_r_cvM, NextImg_con_r_cvM, CurrPoints_r_vt_cvP32,
		NextPoints_r_vt_cvP32, mForwardStatus_b_vt, mForwardDistErr_f_vt, Size(4, 4), 5, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03), 0.5, 0);
	//����Ԥ��
	calcOpticalFlowPyrLK(NextImg_con_r_cvM, CurrImg_con_r_cvM, NextPoints_r_vt_cvP32, mPointsFB_vt_cvP32,
		mBackwardStatus_b_vt, mBackwardDistErr_f_vt, Size(4, 4), 5, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03), 0.5, 0);
	//CUDA!!

	for (int i = 0; i<mPointsFB_vt_cvP32.size(); i++)
	{
		//����Ԥ��ĵ���ԭ���ŷ�Ͼ���
		mBackwardDistErr_f_vt[i] = norm(mPointsFB_vt_cvP32[i] - CurrPoints_r_vt_cvP32[i]);
	}

	//���ǰ��Ԥ�⵽�ĵ���Χͼ����ԭ��ͼ�����ƶȵ���λ��
	getCorrelateMedian_v(CurrImg_con_r_cvM, NextImg_con_r_cvM,
		CurrPoints_r_vt_cvP32, NextPoints_r_vt_cvP32);
	//ѡȡ���� 1.ǰ��Ԥ�����CorrelateMedian�ĵ� 2.����Ԥ��С��ŷ�Ͼ�����λ���ĵ�
	return filterPts(CurrPoints_r_vt_cvP32, NextPoints_r_vt_cvP32);
}
/*��ǰ��Ԥ���У��õ�ǰ��ĵ���Χ�����ĵ���Χ���ƶȣ��Դ˻�����ƶ���λ�������������ȥ���ƶȽ�С��һ���*/
void LKtracker::getCorrelateMedian_v(const Mat& CurrImg_con_r_cvM, const Mat& NextImg_con_r_cvM,
	vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32)

{
	Mat CurrSubPix_cvM(10, 10, CV_8U);
	Mat NextSubPix_cvM(10, 10, CV_8U);
	Mat MatchResult_cvM(1, 1, CV_32F);


	for (int i = 0; i < CurrPoints_r_vt_cvP32.size(); i++)
	{
		if (mForwardStatus_b_vt[i])//�������ǰ��Ԥ���ܹ�Ԥ��õ�
		{
			//��һ��Ϊ���ĵõ���СΪ10*10Ϊ�����ؾ��ȵ�ͼ��
			getRectSubPix(CurrImg_con_r_cvM, Size(10, 10), CurrPoints_r_vt_cvP32[i], CurrSubPix_cvM);
			getRectSubPix(NextImg_con_r_cvM, Size(10, 10), NextPoints_r_vt_cvP32[i], NextSubPix_cvM);

			//ǰ��Ԥ��������Ԥ��ĵ���Χͼ������ƶ�
			matchTemplate(CurrSubPix_cvM, NextSubPix_cvM, MatchResult_cvM, CV_TM_CCOEFF_NORMED);

			mForwardDistErr_f_vt[i] = ((float *)MatchResult_cvM.data)[0];
		}
		else
		{
			mForwardDistErr_f_vt[i] = 0.0;
		}
	}

	/*����λ��*/
	mForwardErrMedian_f = median(mForwardDistErr_f_vt);
}
/*��ȥǰ��Ԥ�����Χͼ�����ƶȽ�Сһ��ͺ���Ԥ�����ϴ�һ��ĵ�*/
bool LKtracker::filterPts(vector<Point2f>& CurrPoints_r_vt_cvP32, vector<Point2f>& NextPoints_r_vt_cvP32)
{
	int PassPts = 0;

	for (int i = 0; i < NextPoints_r_vt_cvP32.size(); i++)
	{
		if (mForwardDistErr_f_vt[i]>mForwardErrMedian_f)//ѡ��ǰ��Ԥ�����CorrelateMedian�ĵ�
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

	for (int i = 0; i < NextPoints_r_vt_cvP32.size(); i++)//ѡ������Ԥ��С��ŷ�Ͼ�����λ���ĵ�
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
	vector<float> xoff_vt_f(nPoint_i);//��������x����λ��
	vector<float> yoff_vt_f(nPoint_i);//��������y����λ��

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
		d.reserve(nPoint_i*(nPoint_i - 1) / 2); //�Ȳ�������ͣ�1+2+...+(npoints-1),�������d������������push��dʱ��
		for (int i = 0; i < nPoint_i; i++)
		{
			for (int j = i + 1; j < nPoint_i; j++)
			{
				//ǰ����ͼ�и�ͼ�����֮���ֵ��С���Դ�ȷ�����Ŵ�С
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