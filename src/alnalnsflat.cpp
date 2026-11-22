#include "muscle.h"
#include "mpcflat.h"

float CalcAlnFlat(const float *Post, uint LX, uint LY,
  float *DPRows, char *TB, string &Path);

MultiSequence *MPCFlat::AlignAlns(const MultiSequence &MSA1,
  const MultiSequence &MSA2, float *ptrScore)
	{
	return AlignAlns(MSA1, MSA2, ptrScore, false);
	}

MultiSequence *AlignRefinementScored(MPCFlat &M, const MultiSequence &MSA1,
  const MultiSequence &MSA2, float *ptrScore, bool UseRowPosts, const string *OldPath, double *Gain);

MultiSequence *MPCFlat::AlignAlns(const MultiSequence &MSA1,
  const MultiSequence &MSA2, float *ptrScore, bool UseRowPosts)
	{
	return AlignRefinementScored(*this,MSA1,MSA2,ptrScore,UseRowPosts,nullptr,nullptr);
	}

MultiSequence *AlignRefinementScored(MPCFlat &M, const MultiSequence &MSA1,
  const MultiSequence &MSA2, float *ptrScore, bool UseRowPosts, const string *OldPath, double *Gain)
	{
	const uint SeqCount1 = MSA1.GetSeqCount();
	const uint SeqCount2 = MSA2.GetSeqCount();

	const uint ColCount1 = MSA1.GetColCount();
	const uint ColCount2 = MSA2.GetColCount();

	// BuildPost uses m_Weights per sequence (see MPCFlat::Run: ClustalWeights, currently
	// overridden to 1). Paths such as -profalign skip Run() and never fill m_Weights.
	const uint SeqCount = M.GetSeqCount();
	if (SIZE(M.m_Weights) != SeqCount)
		M.m_Weights.assign(SeqCount, 1.0f);

	float *Post = AllocPost(ColCount1, ColCount2);
	if (UseRowPosts)
		M.BuildRefinementPost(MSA1, MSA2, Post);
	else
		M.BuildPost(MSA1, MSA2, Post);

	float *DPRows = AllocDPRows(ColCount1, ColCount2);
	char *TB = AllocTB(ColCount1, ColCount2);

	string Path;
	float Score = CalcAlnFlat(Post, ColCount1, ColCount2, DPRows, TB, Path);
	if (OldPath)
		{
		auto crossScore = [&](const string &merge)
			{
			uint i=0,j=0;
			double sum=0;
			for (char step : merge)
				{
				if (step=='B') sum+=Post[size_t(i)*ColCount2+j];
				if (step!='Y') ++i;
				if (step!='X') ++j;
				}
			asserta(i==ColCount1 && j==ColCount2);
			return sum;
			};
		const double before=crossScore(*OldPath),after=crossScore(Path);
		*Gain=std::fabs(after-before)/max(1.0,before);
		}
	if (ptrScore != 0)
		*ptrScore = Score;
	myfree(Post);
	myfree(DPRows);
	myfree(TB);

	MultiSequence *result = new MultiSequence();
	for (uint SeqIndex1 = 0; SeqIndex1 < SeqCount1; ++SeqIndex1)
		{
		const Sequence *InputRow = MSA1.GetSequence(SeqIndex1);
		Sequence *AlignedRow = InputRow->AddGapsPath(Path, 'X');
		result->AddSequence(AlignedRow, true);
		}

	for (uint SeqIndex2 = 0; SeqIndex2 < SeqCount2; ++SeqIndex2)
		{
		const Sequence *InputRow = MSA2.GetSequence(SeqIndex2);
		Sequence *AlignedRow = InputRow->AddGapsPath(Path, 'Y');
		result->AddSequence(AlignedRow, true);
		}

	return result;
	}
