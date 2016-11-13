#include "NNclassifier_V1.0.h"

NNclassifier::NNclassifier()
{

}

NNclassifier::~NNclassifier()
{

}

void NNclassifier::read(const FileNode& file)
{
	mnccthrSame_f = (float)file["ncc_thesame"];
	mthrUpdatePEx = (float)file["thr_nn"];
}

void NNclassifier::GetNNConf(const Mat& example, bool& isSimilar2PEx, bool& isSimilar2NEx, float& rconf, float& cconf)
{
	if (mPExpert_vt_cvM.empty())
	{
		rconf = 0;
		cconf = 0;
		return;
	}
	if (mNExpert_vt_cvM.empty())
	{
		rconf = 1;
		cconf = 1;
		return;
	}

	isSimilar2PEx = false;
	isSimilar2NEx = false;

	float ncc_f;//临时保存每个样本的ncc模板匹配值
	float maxN_f = 0;//与N专家匹配最大值
	float maxP_f = 0;//与P专家匹配最大值
	float cmaxP_f = 0;//与P专家匹配最大保守值

	int nPex_i = mPExpert_vt_cvM.size();
	const int PmidIndex_con_i = ceil(nPex_i*0.5f);//取p专家前一半计算最大相似度，用此计算cconf

	Mat nccResult_cvM(1, 1, CV_32F);

	for (int i = 0; i < nPex_i; i++)
	{
		//计算送入图像片与所有p专家最大的相似值
		matchTemplate(mPExpert_vt_cvM[i], example, nccResult_cvM, CV_TM_CCORR_NORMED);
		ncc_f = (((float*)nccResult_cvM.data)[0] + 1)*0.5;
		if (ncc_f>maxP_f)
		{
			isSimilar2PEx = true;
			maxP_f = ncc_f;
			if (i < PmidIndex_con_i)
			{
				cmaxP_f = ncc_f;
			}
		}
	}

	int nNex_i = mNExpert_vt_cvM.size();

	for (int i = 0; i < nNex_i; i++)
	{
		//计算送入图像片与所有n专家最大的相似值
		matchTemplate(mNExpert_vt_cvM[i], example, nccResult_cvM, CV_TM_CCORR_NORMED);
		ncc_f = (((float*)nccResult_cvM.data)[0] + 1)*0.5;
		if (ncc_f>maxN_f)
		{
			maxN_f = ncc_f;
		}
	}

	if (maxN_f > mnccthrSame_f)
		isSimilar2NEx = true;

	float dP = 1 - maxP_f;
	float cdP = 1 - cmaxP_f;
	float dN = 1 - maxN_f;


	rconf = dN / (dN + dP);
	cconf = dN / (dN + cdP);

}

void NNclassifier::UpdateNNmodel(const Mat& pPatch_con_cvM, const vector<Mat>& nPatch_con_cvM)
{

	float rconf_f;
	float cconf_f;
	bool isSimilar_b = false, dummy;

	GetNNConf(pPatch_con_cvM, isSimilar_b, dummy, rconf_f, cconf_f);
	//当检测到认为可能是P专家与模型相似度过小时，若与P专家相似，则保存，否则用此重新初始化PExpert
	if (rconf_f <= mthrUpdatePEx)//mthrUpdatePEx=0.65
	{
		if (isSimilar_b)
		{
			mPExpert_vt_cvM.push_back(pPatch_con_cvM);
		}
		else
		{
			mPExpert_vt_cvM = vector<Mat>(1, pPatch_con_cvM);
		}
	}

	for (int i = 0; i < nPatch_con_cvM.size(); i++)
	{
		GetNNConf(nPatch_con_cvM[i], isSimilar_b, dummy, rconf_f, cconf_f);
		//当检测到认为可能是N专家与模型相似度较大时，则保存
		if (rconf_f>0.5)
		{
			mNExpert_vt_cvM.push_back(nPatch_con_cvM[i]);
		}
	}
}