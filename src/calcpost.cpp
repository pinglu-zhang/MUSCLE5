#include "muscle.h"
#include "mega.h"
#include "hmmrolling.h"
#include "hmmsimd.h"
#include <limits>
#if MUSCLE_HMM_SIMD && defined(__AVX2__)
#include "hmmdiagonal.h"
#endif

static float *AllocRollingFB(uint64 MatchCells, uint64 InsertFloats)
	{
	const uint64 Count = MatchCells + InsertFloats;
	if (Count > std::numeric_limits<size_t>::max()/sizeof(float))
		Die("HMM rolling buffer exceeds size_t");
	return myalloc(float, size_t(Count));
	}

#if MUSCLE_HMM_SIMD
static void CalcRollingPostWithPolicy(const byte *X, uint LX, const byte *Y, uint LY,
  float *Fwd,bool guarded)
	{
#if MUSCLE_HMM_SIMD && defined(__AVX2__)
	if (LX > 0 && LY > 0 && LX <= diagonal_hmm::ShortLimit && LY <= diagonal_hmm::ShortLimit)
		{
		diagonal_hmm::calculateShort(X, LX, Y, LY, Fwd,guarded);
		return;
		}
	if (diagonal_hmm::tryMedium(X, LX, Y, LY, Fwd,guarded))
		return;
#endif
	const size_t Cells = (size_t(LX) + 1)*(size_t(LY) + 1);
	float *Work = Fwd + Cells;
	float FwdEnd[HMMSTATE_COUNT];
	CalcFwdRolling(X, LX, Y, LY, Fwd, Work, FwdEnd);

	float BwdEnd[HMMSTATE_COUNT];
	GetBwdEnd(BwdEnd);
	const float Total = CalcTotalProbFlat(FwdEnd, BwdEnd, 0, 0);
	static_assert(BWD_POST_FLOATS_PER_COLUMN <= FWD_INSERT_FLOATS_PER_COLUMN,
	  "Forward insertion rows must hold the backward workspace");
	// The backward recurrence reads a forward match only when decoding that cell.
	// Write the posterior to the very same cell, keeping both leading axes.
	const size_t Width = size_t(LY) + 1;
	CalcBwdPostStrided(X, LX, Y, LY, Fwd, BwdEnd, Total, Work, Fwd + Width + 1, Width);
	// Other stages expect contiguous rows without either leading axis.
	for (uint i = 0; i < LX; ++i)
		memmove(Fwd + size_t(i)*LY, Fwd + (size_t(i)+1)*Width+1, size_t(LY)*sizeof(float));
	}
void CalcRollingPostInPlace(const byte *X, uint LX, const byte *Y, uint LY,
  float *Fwd)
	{ CalcRollingPostWithPolicy(X,LX,Y,LY,Fwd,false); }
#endif

static float *CalcPostWithPolicy(const string &LabelX, const string &LabelY,bool guarded)
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
		CalcRollingPostWithPolicy(X, LX, Y, LY, Fwd,guarded);
		return Fwd;
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

float *CalcPost(const string &LabelX, const string &LabelY)
	{ return CalcPostWithPolicy(LabelX,LabelY,false); }

float *CalcProfilePost(const string &LabelX,const string &LabelY)
	{
	const bool guarded=optset_super5 && g_Alpha==ALPHA_Amino && !optset_hmmin &&
	  (!optset_perturb || opt_perturb==0);
	return CalcPostWithPolicy(LabelX,LabelY,guarded);
	}
