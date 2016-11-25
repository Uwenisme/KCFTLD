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
*02   V1.1       2016/08/23    Uwen           将OComparator和DetComparator由struct改为class ，将读参数放到TLD构造函数执行
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

/*用来存放检测时Fern和相似度等等变量*/
struct DetectVar
{
	vector<int> bbidx_i_vt;            //保存通过fern检测器的box序号，会在学习时用到
	vector<Mat> pattern_vt_cvM;        //保存通过fern检测器的box的归一化图像，会在学习时训练NEx模型用到

};

class OComparator  //比较两者的重合度,用来选取重合度较大的box
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

class DetComparator     //检测时用来选取fern posterior较大的box
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

struct GridFernPosterior//用来存放检测时通过方差分类器后的box的fern和fern posterior
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

	bool mIsLastValid_b;//用来判断是否训练
	bool mIsTrackValid_b;//跟踪是否有效，配合mIsLastValid_b使用
	bool mIsTracked_b = true;//是否跟踪成功
	bool mIsDetected_b;//是否检测成功
	//bool mIskcflearn = true;

	int mMinGridSize;//最小grid的size 15*15
	int mWarpNuminit_i;//用于初始化fern时，goodbox仿射变换次数
	int mWarpNumupdate_i;//用于训练时，goodbox仿射变换次数
	int mMaxBadbbNum_i;//最大badbox个数 100
	int mPatternSize_i;//图像归一化size 15*15
	int mMaxGoodbbNum_i;//最大goodbox个数 10
	int mGridSize_i;

	float mthrIsNExpert_f;//得到训练样本时用于判断是否为N专家
	float mthrGoodOverlap_f;//判断为goodbox的overlap阈值
	float mthrBadOverlap_f;//判断为badbox的overlap阈值
	float mTrackedCconf;//跟踪到的box与NNmodel的保守相似度
	float mthrTrackValid;//跟踪是否有效的阈值
	float mNoiseinit_f;//初始化时仿射变换噪声参数
	float mScaleinit_f;//初始化时仿射变换尺度参数
	float mAngleinit_f;//初始化时仿射变换角度参数
	float mNoiseUpdate_f;//更新时仿射变换噪声参数  这里没用这个，暂时保留
	float mScsleUpdate_f;//更新时仿射变换噪声参数  这里没用这个，暂时保留
	float mAngleUpdate_f;//更新时仿射变换噪声参数  这里没用这个，暂时保留
	float mFernPosterior_f;//这里要是fern中10个编码对应的概率总和，程序里所有Posterior都是指总和


	double mBestbbVariance_d;//最好的box的图像的方差，用于方差分类器

	vector<Size> mScales;//保留每个gird的尺度
	//vector<BoundingBox> mGrid_vt;//gird
	vector<int> mGoodbb_i_vt;//goodbox的对应grid序号
	vector<int> mBadbb_i_vt;//badbox的对应grid序号
	vector<BoundingBox> mDetectedbb;//检测到的box
	vector<float> mDetectCconf;//检测到box的保守相似度
	vector<Point2f> CurrPoints_vt_cvP32;//光流法跟踪当前撒的点
	vector<Point2f> NextPoints_vt_cvP32;//预测的点
	vector<BoundingBox> mClusterbb;//检测到的box的聚类box
	vector<float> mClusterCconf;//每个聚类box的保守相似度
	vector<vector<int>> mFernTest;//用于测试改变阈值的fern
	vector<Mat> mNNTest;//用于测试改变阈值得到NN
	vector<pair<vector<int>, bool>> mCurrFern_vt;//每张图片当前的fern
	vector<Mat> mNExpert_vt_cvM;//每张图片当前所认为的N专家

	Rect mGoodbbHull;//能包围所有goodbox的框
	BoundingBox mBestbb;
	BoundingBox mTrackbb;
	BoundingBox mLastbb;//最后确定的box
	BoundingBox* mGrid_ptr;

	Mat mPExpert_cvM;//每张图片当前所认为的P专家
	Mat mIntegralImg_cvM;//积分图
	Mat mIntegralSqImg_cvM;//平方积分图 E(X^2)

	PatchGenerator generator;//用于仿射变换

//	LKtracker tracker;//跟踪的类
	Fernclassifie mFernModel_cls;//fern分类器的类
	NNclassifier mNNModel_cls;//NN分类器的类

	DetectVar mDetectvar_st;//保存检测时临时的变量，有些后面学习会用到
	GridFernPosterior FernPosterior_st;//用来存放检测时通过方差分类器后的box的fern和fern posterior
	//用于后面学习时选取训练样本，避免重复工作

	KCFTracker tracker;
	

};








