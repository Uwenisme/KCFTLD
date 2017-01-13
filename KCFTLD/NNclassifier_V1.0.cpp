#include "NNclassifier_V1.0.h"

NNclassifier::NNclassifier()
{

}

NNclassifier::~NNclassifier()
{

}

void NNclassifier::read(const FileNode& file)
{
	mnccthrSame_f = (float)file["ncc_thesame"]+0.05;
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

	float ncc_f;//��ʱ����ÿ��������nccģ��ƥ��ֵ
	float maxN_f = 0;//��Nר��ƥ�����ֵ
	float maxP_f = 0;//��Pר��ƥ�����ֵ
	float cmaxP_f = 0;//��Pר��ƥ�������ֵ

	int nPex_i = mPExpert_vt_cvM.size();
	const int PmidIndex_con_i = ceil(nPex_i*0.5f);//ȡpר��ǰһ�����������ƶȣ��ô˼���cconf

	Mat nccResult_cvM(1, 1, CV_32F);

	for (int i = 0; i < nPex_i; i++)
	{
		//��������ͼ��Ƭ������pר����������ֵ
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
		//��������ͼ��Ƭ������nר����������ֵ
		matchTemplate(mNExpert_vt_cvM[i], example, nccResult_cvM, CV_TM_CCORR_NORMED);
		ncc_f = (((float*)nccResult_cvM.data)[0] + 1)*0.5;
		if (ncc_f>maxN_f)
		{
			maxN_f = ncc_f;
		}
	}

	if (maxN_f > mnccthrSame_f)
	{	
		//printf("%f %f\n", maxN_f, mnccthrSame_f);
		isSimilar2NEx = true;
	}
		

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
	//����⵽��Ϊ������Pר����ģ�����ƶȹ�Сʱ������Pר�����ƣ��򱣴棬�����ô����³�ʼ��PExpert
	if (rconf_f <= mthrUpdatePEx+0.05)//mthrUpdatePEx=0.65
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
		//����⵽��Ϊ������Nר����ģ�����ƶȽϴ�ʱ���򱣴�
		if (rconf_f>0.52)
		{
			mNExpert_vt_cvM.push_back(nPatch_con_cvM[i]);
		}
	}
}