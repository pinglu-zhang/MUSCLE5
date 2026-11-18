#include "muscle.h"
#include "mpcflat.h"

#define TRACE	0

// Builds a posterior probability matrix needed to align a pair
// of alignments.  Mathematically, the returned matrix M is
// defined as follows:
//    M[i,j] =     sum          sum      f(s,t,i,j)
//             s in align1  t in align2
// where
//                  [  P(s[i] <--> t[j])
//                  [       if s[i] is a letter in the ith column of align1 and
//                  [          t[j] is a letter in the jth column of align2
//    f(s,t,i,j) =  [
//                  [  0    otherwise
//
void MPCFlat::BuildPost(const MultiSequence &MSA1, const MultiSequence &MSA2,
  float *Post)
	{
	const uint SeqCount1 = MSA1.GetSeqCount();
	const uint SeqCount2 = MSA2.GetSeqCount();

	const uint ColCount1 = MSA1.GetColCount();
	const uint ColCount2 = MSA2.GetColCount();

	uint Ix = 0;
	for (uint i = 0; i < ColCount1; ++i)
		for (uint j = 0; j < ColCount2; ++j)
			Post[Ix++] = 0;

// for each s in MSA1
	vector<uint> PosToCol1;
	vector<vector<uint> > ColumnMaps2(SeqCount2);
	vector<uint> InputIndexes2(SeqCount2);
	for (uint s = 0; s < SeqCount2; ++s)
		{
		const Sequence *Seq2 = MSA2.GetSequence(s);
		InputIndexes2[s] = GetMyInputSeqIndex(Seq2->m_Label);
		Seq2->GetPosToCol(ColumnMaps2[s]);
		}
	for (uint SeqIndex1 = 0; SeqIndex1 < SeqCount1; ++SeqIndex1)
		{
		const Sequence *Seq1 = MSA1.GetSequence(SeqIndex1);
		uint SMI_1 = GetMyInputSeqIndex(Seq1->m_Label);
		asserta(SMI_1 != UINT_MAX);
		const float w1 = m_Weights[SeqIndex1];

		Seq1->GetPosToCol(PosToCol1);

	// for each t in MSA2
		for (uint SeqIndex2 = 0; SeqIndex2 < SeqCount2; SeqIndex2++)
			{
			const vector<uint> &PosToCol2 = ColumnMaps2[SeqIndex2];
			uint SMI_2 = InputIndexes2[SeqIndex2];
			asserta(SMI_2 != UINT_MAX);
			asserta(SMI_1 != SMI_2);
			const float w2 = m_Weights[SeqIndex2];


			if (SMI_1 < SMI_2)
				{
				uint PairIndex = GetPairIndex(SMI_1, SMI_2);
				const MySparseMx &Mx = GetSparsePost(PairIndex);
				const uint LX = Mx.GetLX();
				const uint LY = Mx.GetLY();
				assert(SIZE(PosToCol1) == LX);
				assert(SIZE(PosToCol2) == LY);
				for (uint i = 0; i < LX; ++i)
					{
					uint Col1 = PosToCol1[i];
					uint Offset = Mx.m_Offsets[i];
					uint Size = (Mx.m_Offsets[i+1] - Mx.m_Offsets[i]);
					for (uint k = 0; k < Size; ++k)
						{
						float P = Mx.GetProb_Offset(Offset);
						uint j = Mx.GetCol_Offset(Offset);
						++Offset;
						uint Col2 = PosToCol2[j];
						Post[Col1*ColCount2 + Col2] += w1*w2*P;
						}
					}
				}
			else
				{
				uint PairIndex = GetPairIndex(SMI_2, SMI_1);
				const MySparseMx &Mx = GetSparsePost(PairIndex);
				const uint LX = Mx.GetLX();
				const uint LY = Mx.GetLY();
				assert(SIZE(PosToCol2) == LX);
				assert(SIZE(PosToCol1) == LY);
				for (uint i = 0; i < LX; ++i)
					{
					uint Col2 = PosToCol2[i];
					uint Offset = Mx.m_Offsets[i];
					uint Size = (Mx.m_Offsets[i+1] - Mx.m_Offsets[i]);
					for (uint k = 0; k < Size; ++k)
						{
						float P = Mx.GetProb_Offset(Offset);
						uint j = Mx.GetCol_Offset(Offset);
						++Offset;
						uint Col1 = PosToCol1[j];
						Post[Col1*ColCount2 + Col2] += w1*w2*P;
						}
					}
				}
			}
		}
#if 0//TRACE
	LogFlatMx("MSAPost", Post, ColCount1, ColCount2);
#endif
	}
