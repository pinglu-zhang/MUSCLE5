#include "muscle.h"
#include "mpcflat.h"

void MPCFlat::RefineIter()
	{
	RefineIter(false);
	}

MultiSequence *AlignRefinementScored(MPCFlat &M, const MultiSequence &MSA1,
  const MultiSequence &MSA2, float *ptrScore, bool UseRowPosts, const string *OldPath, double *Gain);

double RunMeasuredRefinement(MPCFlat &model,bool UseRowPosts,bool track);

void MPCFlat::RefineIter(bool UseRowPosts)
	{
	RunMeasuredRefinement(*this,UseRowPosts,false);
	}

double RunMeasuredRefinement(MPCFlat &model,bool UseRowPosts,bool track)
	{
	set<int> SeqIndexes1, SeqIndexes2;

	const uint SeqCount = model.GetSeqCount();
	asserta(model.m_MSA != 0);
	asserta(model.m_MSA->GetSeqCount() == SeqCount);

	// create two separate groups
	for (uint SeqIndex = 0; SeqIndex < SeqCount; SeqIndex++)
		if (rand()%2 == 0)
			SeqIndexes1.insert(SeqIndex);
		else
			SeqIndexes2.insert(SeqIndex);

	if (SeqIndexes1.empty() || SeqIndexes2.empty())
		return 0.0;

	string merged;
	if (track)
		{
		const uint cols=model.m_MSA->GetColCount();
		vector<byte> presence(cols,0);
		auto mark = [&](const set<int> &ids,byte group)
			{
			for (int row : ids)
				{
				const char *s=model.m_MSA->GetSequence(row)->GetCharPtr();
				for (uint c=0;c<cols;++c) presence[c]|=s[c]=='-' ? 0 : group;
				}
			};
		mark(SeqIndexes1,1); mark(SeqIndexes2,2);
		merged.reserve(cols);
		for (byte mask : presence)
			if (mask) merged.push_back(mask==3 ? 'B' : mask==1 ? 'X' : 'Y');
		}
	double gain=0;
	const MultiSequence *MSA1 = model.m_MSA->Project(SeqIndexes1);
	const MultiSequence *MSA2 = model.m_MSA->Project(SeqIndexes2);
	delete model.m_MSA;

	MultiSequence *MSA12 = AlignRefinementScored(model,*MSA1,*MSA2,nullptr,UseRowPosts,track?&merged:nullptr,&gain);
	model.m_MSA = MSA12;

	delete MSA1;
	delete MSA2;
	return gain;
	}
