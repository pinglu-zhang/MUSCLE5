#include "muscle.h"
#include "mpcflat.h"
#include "sparseposttranspose.h"

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

bool MPCFlat::PrepareRefinementPosts()
	{
	// Amortize the transpose over repeated, large profile comparisons.
	// Consistency has finished, so its inactive value arrays can be reused.
	if (m_RefineIterCount < 8 || GetSeqCount() < 32 || GetRequestedThreadCount() < 2 ||
	  m_ConsistencyIterCount == 0 || omp_in_parallel())
		return false;
	for (float Weight : m_Weights)
		if (Weight != 1.0f)
			return false;
	const auto &Active = *m_ptrSparsePosts;
	auto &Spare = *m_ptrUpdatedSparsePosts;
	const size_t Pairs = m_Pairs.size();
	if (Active.size() < Pairs || Spare.size() < Pairs)
		return false;
	uint64 Entries = 0, ExtraBytes = 0;
	for (size_t p = 0; p < Pairs; ++p)
		{
		if (Active[p] == nullptr || Spare[p] == nullptr)
			return false;
		const auto &Mx = *Active[p];
		const uint Count = Mx.m_Offsets[Mx.GetLX()];
		if (Spare[p]->m_MaxVecSize < Count)
			return false;
		Entries += Count;
		if (Mx.GetLY() > Spare[p]->m_MaxLX)
			ExtraBytes += (uint64(Mx.GetLY()) - Spare[p]->m_MaxLX)*sizeof(uint);
		}
	if (Entries < 500000 || ExtraBytes > 4u*1024*1024)
		return false;
	// Allocate serially: the allocator also updates process-wide statistics.
	// Parallel workers below only write to already reserved, disjoint arrays.
	for (size_t p = 0; p < Pairs; ++p)
		ReserveSparseRows(*Spare[p], Active[p]->GetLY());
	const uint Threads = GetRequestedThreadCount();
#pragma omp parallel for num_threads(Threads) schedule(static)
	for (int64_t p = 0; p < int64_t(Pairs); ++p)
		TransposeSparsePost(*Active[p], *Spare[p]);
	return true;
	}

void MPCFlat::BuildRefinementPost(const MultiSequence &MSA1, const MultiSequence &MSA2,
  float *Post)
	{
	const uint N1 = MSA1.GetSeqCount(), N2 = MSA2.GetSeqCount();
	const uint Cols1 = MSA1.GetColCount(), Cols2 = MSA2.GetColCount();
	vector<vector<uint> > RowStarts(N1), Map2(N2);
	vector<uint> Id2(N2);
	for (uint j = 0; j < N2; ++j)
		{
		const Sequence *s = MSA2.GetSequence(j);
		s->GetPosToCol(Map2[j]);
		Id2[j] = GetMyInputSeqIndex(s->m_Label);
		}
	vector<const MySparseMx *> Matrices(size_t(N1)*N2);
	for (uint i = 0; i < N1; ++i)
		{
		const Sequence *s = MSA1.GetSequence(i);
		s->GetPosToCol(RowStarts[i]);
		// AllocPost checks that the number of output cells fits in uint.
		for (uint &Column : RowStarts[i])
			Column *= Cols2;
		const uint Id1 = GetMyInputSeqIndex(s->m_Label);
		for (uint j = 0; j < N2; ++j)
			{
			const uint Other = Id2[j];
			const uint Pair = GetPairIndex(min(Id1, Other), max(Id1, Other));
			Matrices[size_t(i)*N2+j] = Id1 < Other ? (*m_ptrSparsePosts)[Pair] :
			  (*m_ptrUpdatedSparsePosts)[Pair];
			}
		}
	// A worker owns complete output rows. Both sparse orientations expose
	// those rows directly. Every cell sees the original (i,j) pair order.
	const uint64 PairRows = uint64(N1)*N2*min(Cols1, Cols2);
	const bool BalanceRows = N1 + N2 >= 128 && PairRows >= 2000000;
	const uint Limit = BalanceRows ? 32u : (PairRows >= 2000000 ? 8u : 4u);
	const uint Stripes = min(Limit, GetRequestedThreadCount());
	vector<uint> Cuts(Stripes+1);
	if (BalanceRows)
		{
		// Gappy profile columns vary greatly in cost. Count occupied residues
		// before cutting the output into disjoint stripes; each cell still
		// receives its contributions in the original sequence-pair order.
		vector<uint64> Work(Cols1+1, 1);
		Work[0] = 0;
		for (const auto &Starts : RowStarts)
			for (uint Offset : Starts)
				++Work[Offset/Cols2+1];
		for (uint r = 0; r < Cols1; ++r)
			Work[r+1] += Work[r];
		for (uint t = 1; t < Stripes; ++t)
			Cuts[t] = uint(std::lower_bound(Work.begin(),Work.end(), Work.back()*t/Stripes)-Work.begin());
		}
	else
		for (uint t = 1; t < Stripes; ++t)
			Cuts[t] = uint(uint64(Cols1)*t/Stripes);
	Cuts[Stripes] = Cols1;
#pragma omp parallel for num_threads(Stripes) schedule(static)
	for (int Stripe = 0; Stripe < int(Stripes); ++Stripe)
		{
		const uint Lo = Cuts[Stripe];
		const uint Hi = Cuts[Stripe+1];
		std::fill(Post + size_t(Lo)*Cols2, Post + size_t(Hi)*Cols2, 0.0f);
		for (uint i = 0; i < N1; ++i)
			{
			const auto &Starts = RowStarts[i];
			const auto First = std::lower_bound(Starts.begin(), Starts.end(), size_t(Lo)*Cols2);
			const uint Begin = uint(First-Starts.begin());
			const uint End = uint(std::lower_bound(First, Starts.end(), size_t(Hi)*Cols2)-Starts.begin());
			for (uint j = 0; j < N2; ++j)
				{
				const MySparseMx &Mx = *Matrices[size_t(i)*N2+j];
				const auto &YMap = Map2[j];
				for (uint x = Begin; x < End; ++x)
					{
					float *Row = Post + Starts[x];
					for (uint k = Mx.m_Offsets[x]; k < Mx.m_Offsets[x+1]; ++k)
						Row[YMap[Mx.GetCol_Offset(k)]] += Mx.GetProb_Offset(k);
					}
				}
			}
		}
	}
