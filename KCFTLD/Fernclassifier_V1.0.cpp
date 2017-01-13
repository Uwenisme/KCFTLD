#include "Fernclassifier_V1.0.h"

Fernclassifie::Fernclassifie()
{

}

Fernclassifie::~Fernclassifie()
{

}

void Fernclassifie::read(const FileNode& file)
{
	mthrP = 6.0f;
	mthrN = 5.0f;
	mNFern_i = (int)file["num_trees"];
	mFernSize_i = (int)file["num_features"];
}

void Fernclassifie::PrepareRandomPoints_v(const vector<Size>& scale_si)
{
	int totalfeature = mNFern_i*mFernSize_i;
	int scaleSize_i = scale_si.size();
	mFeature_vt_cls = vector<vector<Feature>>(scaleSize_i, vector<Feature>(totalfeature));

	RNG& rng = theRNG();

	float x1f, y1f, x2f, y2f;
	int x1, y1, x2, y2;

	for (int i = 0; i < totalfeature; i++)
	{
		x1f = (float)rng;//0<rng<1
		y1f = (float)rng;
		x2f = (float)rng;
		y2f = (float)rng;//������Կ����ò�ͬ���������������Ҳ�����Ƿ����е㶼Ϊ��ͬ����㣬������ͬһ�߶���ͬ

		for (int s = 0; s < scaleSize_i; s++)
		{
			x1 = scale_si[s].width*x1f;
			y1 = scale_si[s].height*y1f;
			x2 = scale_si[s].width*x2f;
			y2 = scale_si[s].height*y2f;

			mFeature_vt_cls[s][i] = Feature(x1, y1, x2, y2);
		}
	}
	mNCounter_vt_i = vector<vector<int>>(mNFern_i, vector<int>(pow(2, mFernSize_i), 0));//����fern������ÿ����������
	mPCounter_vt_i = vector<vector<int>>(mNFern_i, vector<int>(pow(2, mFernSize_i), 0));//����fern������ÿ����������
	mPosteriors_vt_f = vector<vector<float>>(mNFern_i, vector<float>(pow(2, mFernSize_i), 0));//ÿ�������Ӧ�ĸ��ʣ�#P/(#N+#P)
}

void Fernclassifie::GetFern_v(const Mat& patch, vector<int>& fern, int scale_index)
{
	int leaf;
	for (int i = 0; i < mNFern_i; i++)
	{
		leaf = 0;
		for (int j = 0; j < mFernSize_i; j++)
		{
			leaf = (leaf << 1) + mFeature_vt_cls[scale_index][i*mNFern_i + j].getCode(patch);
			//��Ϊÿ��grid�ĵ��λ���Ѿ�ȷ����ͨ���ȶԸ�ͼƬ���ڵ��������ֵ����ȷ��13������Ԫ�����ɵõ�0-2^13��Χ�ı���
		}
		fern[i] = leaf;//ÿ��fern���ܹ���10�������ı���
	}
}

double Fernclassifie::GetFernPosterior(const vector<int>& fern)
{
	float Posterior = 0.0f;
	for (int i = 0; i < mNFern_i; i++)
	{
		Posterior += mPosteriors_vt_f[i][fern[i]];
	}
	return Posterior;
}

void Fernclassifie::UpdateFernModel(const vector <pair<vector<int>, bool>>& fern)
{
	int fernSize = fern.size();
	for (int i = 0; i < fernSize; i++)
	{
		if (fern[i].second)//����Ǻõ�box��fern
		{
			if (GetFernPosterior(fern[i].first) <= mthrP-0.038)//mthrP = 6
			{
				vector<int> fern_vt = fern[i].first;
				for (int j = 0; j < mNFern_i; j++)
				{
					mPCounter_vt_i[j][fern_vt[j]]++;//��������Ӧ����Ԫ�ص���Ŀ��һ
					//���º������		
					mPosteriors_vt_f[j][fern_vt[j]] = ((float)(mPCounter_vt_i[j][fern_vt[j]]) / (mPCounter_vt_i[j][fern_vt[j]] + mNCounter_vt_i[j][fern_vt[j]]));
				}
			}
		}//end of if (fern[i].second)
		else
		{
			if (GetFernPosterior(fern[i].first) >= mthrN)
			{
				vector<int> fern_vt = fern[i].first;
				for (int j = 0; j < mNFern_i; j++)
				{
					mNCounter_vt_i[j][fern_vt[j]]++;//��������Ӧ����Ԫ�ص���Ŀ��һ
					if (0 == mPCounter_vt_i[j][fern_vt[j]]) {
						mPosteriors_vt_f[j][fern_vt[j]] = 0;
					}
					else
					{
						mPosteriors_vt_f[j][fern_vt[j]] = ((float)(mPCounter_vt_i[j][fern_vt[j]]) / (mPCounter_vt_i[j][fern_vt[j]] + mNCounter_vt_i[j][fern_vt[j]]));
					}
				}
			}


		}//end of else
	}
}

int Fernclassifie::mGetFernNum()
{
	return mNFern_i;
}