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
		y2f = (float)rng;//这里可以考虑用不同的随机函数，可以也考虑是否所有点都为不同随机点，而不是同一尺度相同

		for (int s = 0; s < scaleSize_i; s++)
		{
			x1 = scale_si[s].width*x1f;
			y1 = scale_si[s].height*y1f;
			x2 = scale_si[s].width*x2f;
			y2 = scale_si[s].height*y2f;

			mFeature_vt_cls[s][i] = Feature(x1, y1, x2, y2);
		}
	}
	mNCounter_vt_i = vector<vector<int>>(mNFern_i, vector<int>(pow(2, mFernSize_i), 0));//放置fern正样本每个编码数量
	mPCounter_vt_i = vector<vector<int>>(mNFern_i, vector<int>(pow(2, mFernSize_i), 0));//放置fern负样本每个编码数量
	mPosteriors_vt_f = vector<vector<float>>(mNFern_i, vector<float>(pow(2, mFernSize_i), 0));//每个编码对应的概率，#P/(#N+#P)
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
			//因为每个grid的点对位置已经确定，通过比对该图片所在点对中像素值，可确定13个编码元，即可得到0-2^13范围的编码
		}
		fern[i] = leaf;//每个fern中总共有10个这样的编码
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
		if (fern[i].second)//如果是好的box的fern
		{
			if (GetFernPosterior(fern[i].first) <= mthrP)//mthrP = 6
			{
				vector<int> fern_vt = fern[i].first;
				for (int j = 0; j < mNFern_i; j++)
				{
					mPCounter_vt_i[j][fern_vt[j]]++;//正样本对应编码元素的数目加一
					//更新后验概率		
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
					mNCounter_vt_i[j][fern_vt[j]]++;//负样本对应编码元素的数目加一
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