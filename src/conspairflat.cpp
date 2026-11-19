#include "muscle.h"
#include "mpcflat.h"

void RelaxOnSupport(const MySparseMx &XZ, const MySparseMx &YZ,
  const MySparseMx &XY, float *Post);

#if defined(__GNUC__) && !defined(__clang__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#include <new>

namespace {
constexpr uint Lanes = 8;
constexpr uint MaxLength = 1024;
constexpr size_t ScratchLimit = 512*1024;

struct OutputGroup
	{
	uint Column;
	uint Mask;
	float Sum[Lanes];
	};

struct BridgePair
	{
	const MySparseMx *XZ;
	const MySparseMx *YZ;
	};

// For Z > Y both inputs are stored as rows ending in Z. Group neighboring
// X residues that contribute to the same output column. Each SIMD lane is
// one independent (x,y) sum; bridges and their residues stay in order.
bool RelaxLaterBridges(MPCFlat &M, uint X, uint Y, const MySparseMx &XY, float *Post)
	try
	{
	const uint N = M.GetSeqCount(), LX = XY.GetLX(), LY = XY.GetLY();
	const uint BridgeCount = N-Y-1;
	if (BridgeCount < 8 || LX < Lanes || LX > MaxLength || LY == 0 || LY > MaxLength)
		return false;
	const uint Tiles = (LX+Lanes-1)/Lanes;
	const uint Entries = XY.m_Offsets[LX];
	const uint64 Bytes = uint64(BridgeCount)*sizeof(BridgePair) +
	  uint64(Entries)*sizeof(OutputGroup) + size_t(LY+Tiles+1)*sizeof(uint) +
	  size_t(MaxLength)*Lanes*sizeof(float);
	if (Bytes > ScratchLimit)
		return false;

	vector<BridgePair> Bridges;
	Bridges.reserve(BridgeCount);
	uint Longest = 0;
	for (uint Z = Y+1; Z < N; ++Z)
		{
		const MySparseMx &XZ = M.GetSparsePost(M.GetPairIndex(X,Z));
		if (XZ.GetLY() > MaxLength)
			return false;
		Longest = max(Longest, XZ.GetLY());
		Bridges.push_back({&XZ, &M.GetSparsePost(M.GetPairIndex(Y,Z))});
		}

	vector<OutputGroup> Groups;
	Groups.reserve(Entries);
	vector<uint> Starts(Tiles+1), Slot(LY,0);
	for (uint t = 0; t < Tiles; ++t)
		{
		const uint Base = t*Lanes, End = min(LX,Base+Lanes);
		Starts[t] = uint(Groups.size());
		for (uint x = Base; x < End; ++x)
			for (uint k = XY.m_Offsets[x]; k < XY.m_Offsets[x+1]; ++k)
				{
				const uint y = XY.GetCol_Offset(k);
				// Slots are one-based, so a slot at or below Starts[t]
				// belongs to an earlier tile or has never been assigned.
				if (Slot[y] <= Starts[t])
					{
					Groups.push_back(OutputGroup{y,0,{}});
					Slot[y] = uint(Groups.size());
					}
				OutputGroup &G = Groups[Slot[y]-1];
				G.Mask |= 1u << (x-Base);
				G.Sum[x-Base] = Post[size_t(x)*LY+y];
				}
		}
	Starts[Tiles] = uint(Groups.size());

	alignas(32) float ByZ[MaxLength*Lanes];
	std::fill_n(ByZ, size_t(Longest)*Lanes, 0.0f);
	// Limit the number of source matrices revisited for each tile. Missing
	// left entries multiply the nonnegative posterior probabilities by zero.
	for (size_t First = 0; First < Bridges.size(); First += 16)
		{
		const size_t Last = min(First+16, Bridges.size());
		for (uint t = 0; t < Tiles; ++t)
			{
			const uint Base = t*Lanes, End = min(LX,Base+Lanes);
			for (size_t j = First; j < Last; ++j)
				{
				const MySparseMx &A = *Bridges[j].XZ, &B = *Bridges[j].YZ;
				if (A.m_Offsets[Base] == A.m_Offsets[End])
					continue;
				for (uint x = Base; x < End; ++x)
					for (uint k = A.m_Offsets[x]; k < A.m_Offsets[x+1]; ++k)
						ByZ[A.GetCol_Offset(k)*Lanes + x-Base] = A.GetProb_Offset(k);
				for (uint g = Starts[t]; g < Starts[t+1]; ++g)
					{
					OutputGroup &G = Groups[g];
					__m256 Sum = _mm256_loadu_ps(G.Sum);
					for (uint k = B.m_Offsets[G.Column]; k < B.m_Offsets[G.Column+1]; ++k)
						{
						const __m256 Left = _mm256_load_ps(ByZ+B.GetCol_Offset(k)*Lanes);
						const __m256 Right = _mm256_set1_ps(B.GetProb_Offset(k));
						Sum += Left*Right;
						// Use the scalar recurrence's contraction setting, then retain its rounding.
						asm("" : "+x"(Sum));
						}
					_mm256_storeu_ps(G.Sum,Sum);
					}
				for (uint x = Base; x < End; ++x)
					for (uint k = A.m_Offsets[x]; k < A.m_Offsets[x+1]; ++k)
						ByZ[A.GetCol_Offset(k)*Lanes + x-Base] = 0.0f;
				}
			}
		}

	for (uint t = 0; t < Tiles; ++t)
		for (uint g = Starts[t]; g < Starts[t+1]; ++g)
			{
			const OutputGroup &G = Groups[g];
			for (uint lane = 0; lane < Lanes; ++lane)
				if (G.Mask & (1u << lane))
					Post[size_t(t*Lanes+lane)*LY + G.Column] = G.Sum[lane];
			}
	return true;
	}
	catch (const std::bad_alloc &)
	{
	// The caller's matrix is untouched until scratch preparation succeeds.
	return false;
	}
} // namespace
#endif

#if 0//TRACE
const byte *g_X;
const byte *g_Y;
const byte *g_Z;
#endif

void MPCFlat::ConsPair(uint PairIndex)
	{
	const pair<uint, uint> &Pair = GetPair(PairIndex);

	uint SeqIndexX = Pair.first;
	uint SeqIndexY = Pair.second;

	const MySparseMx &SparsePostXY = GetSparsePost(PairIndex);

	uint LX = GetSeqLength(SeqIndexX);
	uint LY = GetSeqLength(SeqIndexY);

	asserta(SparsePostXY.GetLX() == LX);
	asserta(SparsePostXY.GetLY() == LY);

	float *Post = AllocPost(LX, LY);
	SparsePostXY.ToPost(Post);

// Account for Z=X and Z=Y (hence the factor 2)
	for (uint k = 0; k < LX*LY; ++k)
		Post[k] *= 2;
#if 0//TRACE
	LogFlatMx("ConsPair Z=X Z=Y", Post, LX, LY);
#endif

	const uint SeqCount = GetSeqCount();
	asserta(SeqIndexX < SeqIndexY); // because convention for pairs
	for (uint SeqIndexZ = 0; SeqIndexZ < SeqCount; ++SeqIndexZ)
		{
		if (SeqIndexZ == SeqIndexX || SeqIndexZ == SeqIndexY)
			continue;
		float wZ = m_Weights[SeqIndexZ];
		wZ = 1.0f;

#if 0//TRACE
		g_X = GetSequence(SeqIndexX)->GetBytePtr();
		g_Y = GetSequence(SeqIndexY)->GetBytePtr();
		g_Z = GetSequence(SeqIndexZ)->GetBytePtr();
#endif
		if (SeqIndexZ < SeqIndexX)
			{
			asserta(SeqIndexZ < SeqIndexY); // because SeqIndexX < SeqIndexY

			uint PairIndexZX = GetPairIndex(SeqIndexZ, SeqIndexX);
			uint PairIndexZY = GetPairIndex(SeqIndexZ, SeqIndexY);

			const MySparseMx &ZX = GetSparsePost(PairIndexZX);
			const MySparseMx &ZY = GetSparsePost(PairIndexZY);

			RelaxFlat_ZX_ZY(ZX, ZY, wZ, Post);
#if 0//TRACE
		LogFlatMx("ConsPair after RelaxFlat_ZX_ZY", Post, LX, LY);
#endif
			}
		else if (SeqIndexZ > SeqIndexX && SeqIndexZ < SeqIndexY)
			{
			uint PairIndexXZ = GetPairIndex(SeqIndexX, SeqIndexZ);
			uint PairIndexZY = GetPairIndex(SeqIndexZ, SeqIndexY);

			const MySparseMx &XZ = GetSparsePost(PairIndexXZ);
			const MySparseMx &ZY = GetSparsePost(PairIndexZY);

			RelaxFlat_XZ_ZY(XZ, ZY, wZ, Post);
#if 0//TRACE
			LogFlatMx("ConsPair after RelaxFlat_XZ_ZY", Post, LX, LY);
#endif
			}
		else if (SeqIndexZ > SeqIndexX && SeqIndexZ > SeqIndexY)
			{
#if defined(__GNUC__) && !defined(__clang__) && defined(__AVX2__) && defined(__FMA__)
			if (SeqIndexZ == SeqIndexY+1 &&
			  RelaxLaterBridges(*this, SeqIndexX, SeqIndexY, SparsePostXY, Post))
				break;
#endif
			uint PairIndexXZ = GetPairIndex(SeqIndexX, SeqIndexZ);
			uint PairIndexYZ = GetPairIndex(SeqIndexY, SeqIndexZ);

			const MySparseMx &XZ = GetSparsePost(PairIndexXZ);
			const MySparseMx &YZ = GetSparsePost(PairIndexYZ);

			RelaxOnSupport(XZ, YZ, SparsePostXY, Post);
#if 0//TRACE
			LogFlatMx("ConsPair after RelaxFlat_XZ_YZ", Post, LX, LY);
#endif
			}
		else
			asserta(false);
		}

	MySparseMx &UpdatedSparsePostXY = GetUpdatedSparsePost(PairIndex);

#if 0//TRACE
	LogFlatMx("Final post before update", Post, LX, LY);
#endif
	UpdatedSparsePostXY.UpdateFromPost(SparsePostXY, Post, SeqCount);
	UpdatedSparsePostXY.m_X = SparsePostXY.m_X;
	UpdatedSparsePostXY.m_Y = SparsePostXY.m_Y;
	myfree(Post);

#if 0//TRACE
	Log("\nBefore:");
	SparsePostXY.LogMe();
	Log("\nUpdated:");
	UpdatedSparsePostXY.LogMe();
#endif
	}
