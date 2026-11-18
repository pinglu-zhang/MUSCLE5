#include "muscle.h"
#include "mega.h"
#include "hmmrolling.h"
#include "hmmsimd.h"
#include <limits>

static float *AllocRollingFB(uint64 MatchCells, uint64 InsertFloats)
	{
	const uint64 Count = MatchCells + InsertFloats;
	if (Count > std::numeric_limits<size_t>::max()/sizeof(float))
		Die("HMM rolling buffer exceeds size_t");
	return myalloc(float, size_t(Count));
	}

float *CalcPost(const string &LabelX, const string &LabelY)
	{
	uint LX = GetSeqLengthByGlobalLabel(LabelX);
	uint LY = GetSeqLengthByGlobalLabel(LabelY);
	if (double(LX)*double(LY)*5 + 100 > double(INT_MAX))
		Die("HMM overflow, sequence lengths %u, %u (max ~21k)", LX, LY);

	float *Fwd;
	float *Bwd;
	float *Post;
	if (Mega::m_Loaded)
		{
		Fwd = AllocFB(LX, LY);
		Bwd = AllocFB(LX, LY);
		const vector<vector<byte> > &ProfileX = *Mega::GetProfileByLabel(LabelX);
		const vector<vector<byte> > &ProfileY = *Mega::GetProfileByLabel(LabelY);
		asserta(SIZE(ProfileX) == LX);
		asserta(SIZE(ProfileY) == LY);
		Mega::CalcFwdFlat_mega(ProfileX, ProfileY, Fwd);
		Mega::CalcBwdFlat_mega(ProfileX, ProfileY, Bwd);
		Post = AllocPost(LX, LY);
		CalcPostFlat(Fwd, Bwd, LX, LY, Post);
		}
	else
		{
		const byte *X = GetGlobalByteSeqByLabel(LabelX);
		const byte *Y = GetGlobalByteSeqByLabel(LabelY);
		const uint64 Width = uint64(LY) + 1;
		const uint64 Cells = (uint64(LX) + 1)*Width;
		Fwd = AllocRollingFB(Cells, Width*FWD_INSERT_FLOATS_PER_COLUMN);
#if MUSCLE_HMM_SIMD
		float *Work = Fwd + Cells;
		float FwdEnd[HMMSTATE_COUNT];
		CalcFwdRolling(X, LX, Y, LY, Fwd, Work, FwdEnd);

		float BwdEnd[HMMSTATE_COUNT];
		GetBwdEnd(BwdEnd);
		const float Total = CalcTotalProbFlat(FwdEnd, BwdEnd, 0, 0);
		Post = AllocPost(LX, LY);
		static_assert(BWD_POST_FLOATS_PER_COLUMN <= FWD_INSERT_FLOATS_PER_COLUMN,
		  "Forward insertion rows must hold the backward workspace");
		CalcBwdPost(X, LX, Y, LY, Fwd, BwdEnd, Total, Work, Post);
		myfree(Fwd);
		return Post;
#else
		Bwd = AllocFB(LX, LY);
		float FwdEnd[HMMSTATE_COUNT];
		CalcFwdRolling(X, LX, Y, LY, Fwd, Fwd + Cells, FwdEnd);
		CalcBwdFlat(X, LX, Y, LY, Bwd);
		const float *BwdEnd = Bwd + (Cells - 1)*HMMSTATE_COUNT;
		// The contiguous exit states form a single cell of the flat layout.
		const float Total = CalcTotalProbFlat(FwdEnd, BwdEnd, 0, 0);
		Post = AllocPost(LX, LY);
		CalcPostFromMatches<1, HMMSTATE_COUNT>(Fwd, Bwd, LX, LY, Total, Post);
#endif
		}

	myfree(Fwd);
	myfree(Bwd);
	return Post;
	}
