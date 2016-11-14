/*******************************************************************
*Copyright (c) 2016 XXXX Corporation
*All Rights Reserved.
*
*Project Name          :   OpenTLD_V1.0
*File Name             :   TLD_V1.0.c
*Abstract Description  :   the implementation of TLD algorithm
*class Name            :   TLD

*Create Date           :   2016/08/06
*Author                :   Uwen
*
*--------------------------Revision History---------------------------
*No.  Version    Date          Revised By     Description
*01   V1.0       2016/08/06    Uwen           the first version
*02   V1.1       2016/08/23    Uwen           ��OComparator��DetComparator��struct��Ϊclass �����������ŵ�TLD���캯��ִ��
*********************************************************************/


//#include "track_V1.0.h"
#include "Fernclassifier_V1.0.h"
#include "kcftracker.hpp"
#include "NNclassifier_V1.0.h"

#include "time.h"


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

/*������ż��ʱFern�����ƶȵȵȱ���*/
struct DetectVar
{
	vector<int> bbidx_i_vt;            //����ͨ��fern�������box��ţ�����ѧϰʱ�õ�
	vector<Mat> pattern_vt_cvM;        //����ͨ��fern�������box�Ĺ�һ��ͼ�񣬻���ѧϰʱѵ��NExģ���õ�

};

class OComparator  //�Ƚ����ߵ��غ϶�,����ѡȡ�غ϶Ƚϴ��box
{
public:
	OComparator(BoundingBox* _grid) :grid(_grid)
	{
	}
	BoundingBox* grid;
	bool operator()(int idx1, int idx2)
	{
		return grid[idx1].overlap > grid[idx2].overlap;
	}
};

class DetComparator     //���ʱ����ѡȡfern posterior�ϴ��box
{
public:
	DetComparator(const vector<float>& _FernPost) :FernPost(_FernPost)
	{}
	vector<float> FernPost;
	bool operator()(int idx1, int idx2)
	{
		return FernPost[idx1] > FernPost[idx2];
	}
};

struct GridFernPosterior//������ż��ʱͨ��������������box��fern��fern posterior
{
	vector<vector<int> > Fern;
	vector<float> Posterior;
};

class TLD
{
public:
	TLD();
	TLD(const FileNode& file);
	~TLD();

	void init_v(const Mat& FirstFrame_cvM, const Rect& box, const Mat Frame_cvM);

	void mEvaluate();

	//void mtrack_v(const Mat& CurrFrame_con_cvM, const Mat& NextFrame_con_cvM);

	void mdetect_v(const Mat& NextFrame_con_cvM);

	void mlearn_v(const Mat& NextFrame_con_cvM, const Mat& Frame_con_cvM, bool& lastboxFound);

	void processFrame(const Mat& CurrFrame_con_cvM, const Mat& NextFrame_con_cvM, BoundingBox& Nextbb, bool& lastboxFound, const Mat Frame_con_cvM);

	void mbuildgrid_v(const Mat& FirstFrame, const Rect& box);

	float mGetbbOverlap(const BoundingBox bb1, const BoundingBox bb2);

	void mGetGoodBadbb_v();

	void mGetGoodbbHull_v();

	void mGetCurrFernModel_v(const Mat& frame_cvM, bool isUpdate);

	double mGetVariance(const BoundingBox& bb);

	void mGetPattern_v(const Mat& Img, Mat& pattern, Scalar& StdDev);

	void mGetNNModel_v(const Mat& frame_cvM);

	void mCalIntegralImgVariance_v(const Mat& FirstFrame_cvM);

	void mCluster(const vector<BoundingBox>& Detectbb, const vector<float>& DetectbbCconf, vector<BoundingBox>& Clusterbb, vector<float>& ClusterbbCconf);

	void mDetelteGrid_ptr();
private:

	bool mIsLastValid_b;//�����ж��Ƿ�ѵ��
	bool mIsTrackValid_b;//�����Ƿ���Ч�����mIsLastValid_bʹ��
	bool mIsTracked_b = true;//�Ƿ���ٳɹ�
	bool mIsDetected_b;//�Ƿ���ɹ�
	//bool mIskcflearn = true;

	int mMinGridSize;//��Сgrid��size 15*15
	int mWarpNuminit_i;//���ڳ�ʼ��fernʱ��goodbox����任����
	int mWarpNumupdate_i;//����ѵ��ʱ��goodbox����任����
	int mMaxBadbbNum_i;//���badbox���� 100
	int mPatternSize_i;//ͼ���һ��size 15*15
	int mMaxGoodbbNum_i;//���goodbox���� 10
	int mGridSize_i;

	float mthrIsNExpert_f;//�õ�ѵ������ʱ�����ж��Ƿ�ΪNר��
	float mthrGoodOverlap_f;//�ж�Ϊgoodbox��overlap��ֵ
	float mthrBadOverlap_f;//�ж�Ϊbadbox��overlap��ֵ
	float mTrackedCconf;//���ٵ���box��NNmodel�ı������ƶ�
	float mthrTrackValid;//�����Ƿ���Ч����ֵ
	float mNoiseinit_f;//��ʼ��ʱ����任��������
	float mScaleinit_f;//��ʼ��ʱ����任�߶Ȳ���
	float mAngleinit_f;//��ʼ��ʱ����任�ǶȲ���
	float mNoiseUpdate_f;//����ʱ����任��������  ����û���������ʱ����
	float mScsleUpdate_f;//����ʱ����任��������  ����û���������ʱ����
	float mAngleUpdate_f;//����ʱ����任��������  ����û���������ʱ����
	float mFernPosterior_f;//����Ҫ��fern��10�������Ӧ�ĸ����ܺͣ�����������Posterior����ָ�ܺ�


	double mBestbbVariance_d;//��õ�box��ͼ��ķ�����ڷ��������

	vector<Size> mScales;//����ÿ��gird�ĳ߶�
	//vector<BoundingBox> mGrid_vt;//gird
	vector<int> mGoodbb_i_vt;//goodbox�Ķ�Ӧgrid���
	vector<int> mBadbb_i_vt;//badbox�Ķ�Ӧgrid���
	vector<BoundingBox> mDetectedbb;//��⵽��box
	vector<float> mDetectCconf;//��⵽box�ı������ƶ�
	vector<Point2f> CurrPoints_vt_cvP32;//���������ٵ�ǰ���ĵ�
	vector<Point2f> NextPoints_vt_cvP32;//Ԥ��ĵ�
	vector<BoundingBox> mClusterbb;//��⵽��box�ľ���box
	vector<float> mClusterCconf;//ÿ������box�ı������ƶ�
	vector<vector<int>> mFernTest;//���ڲ��Ըı���ֵ��fern
	vector<Mat> mNNTest;//���ڲ��Ըı���ֵ�õ�NN
	vector<pair<vector<int>, bool>> mCurrFern_vt;//ÿ��ͼƬ��ǰ��fern
	vector<Mat> mNExpert_vt_cvM;//ÿ��ͼƬ��ǰ����Ϊ��Nר��

	Rect mGoodbbHull;//�ܰ�Χ����goodbox�Ŀ�
	BoundingBox mBestbb;
	BoundingBox mTrackbb;
	BoundingBox mLastbb;//���ȷ����box
	BoundingBox* mGrid_ptr;

	Mat mPExpert_cvM;//ÿ��ͼƬ��ǰ����Ϊ��Pר��
	Mat mIntegralImg_cvM;//����ͼ
	Mat mIntegralSqImg_cvM;//ƽ������ͼ E(X^2)

	PatchGenerator generator;//���ڷ���任

//	LKtracker tracker;//���ٵ���
	Fernclassifie mFernModel_cls;//fern����������
	NNclassifier mNNModel_cls;//NN����������

	DetectVar mDetectvar_st;//������ʱ��ʱ�ı�������Щ����ѧϰ���õ�
	GridFernPosterior FernPosterior_st;//������ż��ʱͨ��������������box��fern��fern posterior
	//���ں���ѧϰʱѡȡѵ�������������ظ�����

	KCFTracker tracker;
};








