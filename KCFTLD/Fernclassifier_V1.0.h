/*******************************************************************
*Copyright (c) 2016 XXXX Corporation
*All Rights Reserved.
*
*Project Name          :   OpenTLD_V1.0
*File Name             :   Fernclassifier_V1.0.c
*Abstract Description  :   to arrange and use FernModel to classify 
*class Name            :   Fernclassifie

*Create Date           :   2016/08/06
*Author                :   Uwen
*
*--------------------------Revision History---------------------------
*No.  Version    Date          Revised By     Description
*01   V1.0       2016/08/06    Uwen           the first version
*********************************************************************/

#include <iostream>
#include <opencv.hpp>
#include <legacy\legacy.hpp>

using namespace std;
using namespace cv;

class Feature
{
public:
	Feature() : mX1(0), mY1(0), mX2(0), mY2(0){};
	Feature(uchar x1, uchar y1, uchar x2, uchar y2)
		: mX1(x1), mY1(y1), mX2(x2), mY2(y2) {};
	bool getCode(const Mat& patch) const
	{
		//两个对应随机点像素值比较，返回1或0
		return patch.at<uchar>(mY1, mX1) > patch.at<uchar>(mY2, mX2);
	}
private:
	uchar mX1, mY1, mX2, mY2;
};

class Fernclassifie
{
public:
	Fernclassifie();
	~Fernclassifie();

	void  PrepareRandomPoints_v(const vector<Size>& scale_si);

	void  GetFern_v(const Mat& patch, vector<int>& fern, int scale_index);

	double GetFernPosterior(const vector<int>& fern);

	void  UpdateFernModel(const vector <pair<vector<int>,bool>>& fern);

	void read(const FileNode& file);

	int mGetFernNum();

	float mthrP;
private:
	int mNFern_i;
	int mFernSize_i;
	float mthrN;
	vector<vector<Feature>> mFeature_vt_cls; //Ferns features (one std::vector for each scale)
	vector<vector<int>> mNCounter_vt_i; //negative counter
	vector<vector<int>> mPCounter_vt_i; //positive counter
	vector<vector<float>> mPosteriors_vt_f; //Ferns posteriors
};