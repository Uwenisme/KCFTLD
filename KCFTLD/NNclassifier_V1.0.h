/*******************************************************************
*Copyright (c) 2016 XXXX Corporation
*All Rights Reserved.
*
*Project Name          :   OpenTLD_V1.0
*File Name             :   NNclassifier_V1.0.c
*Abstract Description  :   to arrange and use NNModel to classify 
*class Name            :   NNclassifier

*Create Date           :   2016/08/06
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

class NNclassifier
{
public:
	NNclassifier();
	~NNclassifier();

	void GetNNConf(const Mat& example, bool& isSimilar2PEx, bool& isSimilar2NEx, float& rconf, float& cconf);

	void UpdateNNmodel(const Mat& pPatch_con_cvM, const vector<Mat>& nPatch_con_cvM);

	void read(const FileNode& file);

	float mthrUpdatePEx;
//private:
	vector<Mat> mPExpert_vt_cvM;
	vector<Mat> mNExpert_vt_cvM;
	float mnccthrSame_f;
	
};