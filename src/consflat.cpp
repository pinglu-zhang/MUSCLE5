#include "muscle.h"
#include "mpcflat.h"
#include "locallock.h"

#if defined(__GNUC__) && !defined(__clang__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#include <new>
#include "sparseposttranspose.h"

namespace {
constexpr uint Lanes = 8, BridgeBlock = 24, TargetBlock = 16;
constexpr uint MaxSeqLength = 1024;
constexpr size_t RowBudget = 42u*1024*1024;
constexpr size_t WorkerBudget = 4u*1024*1024;

// Keys are offsets into the aligned block of left-hand vectors.
static_assert(BridgeBlock*MaxSeqLength*Lanes <= uint64(UINT_MAX), "joined key exceeds offset range");

struct JoinedRows
	{
	vector<uint> Offsets;
	struct Entry
		{
		uint Offset;
		float Probability;
		};
	vector<Entry> Entries;

	// Concatenate the sparse rows in bridge order. A key identifies one
	// residue in this block, so no bridge-index lookup is needed in the sum.
	void Build(MPCFlat &M, uint Y, uint Begin, uint End, const vector<uint> &Base)
		{
		const uint LY = M.GetSeqLength(Y);
		Offsets.assign(LY+1,0);
		const MySparseMx *Matrices[BridgeBlock] = {};
		for (uint Z = Begin; Z < End; ++Z)
			{
			if (Z == Y)
				continue;
			const MySparseMx &A = Z < Y ? M.GetUpdatedSparsePost(M.GetPairIndex(Z,Y))
			  : M.GetSparsePost(M.GetPairIndex(Y,Z));
			Matrices[Z-Begin] = &A;
			for (uint y = 0; y < LY; ++y)
				Offsets[y+1] += A.m_Offsets[y+1]-A.m_Offsets[y];
			}
		for (uint y = 0; y < LY; ++y)
			Offsets[y+1] += Offsets[y];
		Entries.resize(Offsets[LY]);
		for (uint y = 0; y < LY; ++y)
			{
			uint out = Offsets[y];
			for (uint Z = Begin; Z < End; ++Z)
				{
				const MySparseMx *A = Matrices[Z-Begin];
				if (!A)
					continue;
				for (uint k = A->m_Offsets[y]; k < A->m_Offsets[y+1]; ++k)
					{
					Entries[out++] = {(Base[Z]+A->GetCol_Offset(k))*Lanes, A->GetProb_Offset(k)};
					}
				}
			}
		}
	};

struct SumRow
	{
	uint Column;
	float Sum[Lanes];
	};

struct PairTiles
	{
	vector<uint> Starts, Slots;
	vector<SumRow> Groups;
	const MySparseMx *Source = nullptr;

	void Init(const MySparseMx &XY)
		{
		Source = &XY;
		const uint Tiles = (XY.m_LX+Lanes-1)/Lanes;
		Starts.resize(Tiles+1);
		Slots.resize(XY.m_Offsets[XY.m_LX]);
		Groups.reserve(Slots.size());
		vector<uint> Last(XY.m_LY,0);
		for (uint t = 0; t < Tiles; ++t)
			{
			Starts[t] = uint(Groups.size());
			for (uint x = t*Lanes; x < min((t+1)*Lanes,XY.m_LX); ++x)
				for (uint k = XY.m_Offsets[x]; k < XY.m_Offsets[x+1]; ++k)
					{
					const uint y = XY.GetCol_Offset(k);
					if (Last[y] <= Starts[t])
						{
						Groups.push_back({y,{}});
						Last[y] = uint(Groups.size());
						}
					Slots[k] = Last[y]-1;
					Groups[Slots[k]].Sum[x%Lanes] = 2*XY.GetProb_Offset(k);
					}
			}
		Starts[Tiles] = uint(Groups.size());
		}


	void Update(uint Tile, const JoinedRows &Right, const float *Weights)
		{
		const auto *Entries = Right.Entries.data();
		for (uint g = Starts[Tile]; g < Starts[Tile+1]; ++g)
			{
			auto &Group = Groups[g];
			uint k = Right.Offsets[Group.Column];
			const uint End = Right.Offsets[Group.Column+1];
			__m256 Sum0 = _mm256_loadu_ps(Group.Sum);
			auto Accumulate = [&](uint Index, __m256 Sum)
				{
				const auto &Entry = Entries[Index];
				return _mm256_fmadd_ps(_mm256_load_ps(Weights+Entry.Offset),
				  _mm256_set1_ps(Entry.Probability), Sum);
				};

			// Independent partial sums hide FMA latency on long sparse rows.
			// Every original term is included. The final tree regroups additions.
			if (End-k >= 16)
				{
				__m256 Sum1 = _mm256_setzero_ps(), Sum2 = Sum1, Sum3 = Sum1;
				__m256 Sum4 = Sum1, Sum5 = Sum1, Sum6 = Sum1, Sum7 = Sum1;
				for (; End-k >= 8; k += 8)
					{
					Sum0 = Accumulate(k,   Sum0);
					Sum1 = Accumulate(k+1, Sum1);
					Sum2 = Accumulate(k+2, Sum2);
					Sum3 = Accumulate(k+3, Sum3);
					Sum4 = Accumulate(k+4, Sum4);
					Sum5 = Accumulate(k+5, Sum5);
					Sum6 = Accumulate(k+6, Sum6);
					Sum7 = Accumulate(k+7, Sum7);
					asm("" : "+x"(Sum0), "+x"(Sum1), "+x"(Sum2), "+x"(Sum3),
					  "+x"(Sum4), "+x"(Sum5), "+x"(Sum6), "+x"(Sum7));
					}
				Sum0 = _mm256_add_ps(
				  _mm256_add_ps(_mm256_add_ps(Sum0,Sum1), _mm256_add_ps(Sum2,Sum3)),
				  _mm256_add_ps(_mm256_add_ps(Sum4,Sum5), _mm256_add_ps(Sum6,Sum7)));
				}
			for (; k < End; ++k)
				{
				Sum0 = Accumulate(k,Sum0);
				asm("" : "+x"(Sum0));
				}
			_mm256_storeu_ps(Group.Sum,Sum0);
			}
		}

	void Finish(MySparseMx &Out, uint N)
		{
		const MySparseMx &XY = *Source;
		// Reuse the destination storage, writing only the original sparse coordinates.
		Out.AllocLX(XY.m_LX);
		Out.AllocVec(XY.m_VecSize);
		Out.m_LX = XY.m_LX;
		Out.m_LY = XY.m_LY;
		memcpy(Out.m_Offsets,XY.m_Offsets,(XY.m_LX+1)*sizeof(uint));
		for (uint x = 0; x < XY.m_LX; ++x)
			for (uint k = XY.m_Offsets[x]; k < XY.m_Offsets[x+1]; ++k)
				{
				Out.SetCol_Offset(k,XY.GetCol_Offset(k));
				Out.SetProb_Offset(k,Groups[Slots[k]].Sum[x%Lanes]/N);
				}
		Out.m_X = XY.m_X;
		Out.m_Y = XY.m_Y;
		}
	};

bool UpdateTargets(MPCFlat &M, uint X, uint Lo, uint Hi, const vector<JoinedRows> &Right,
  const vector<uint> &Base, uint MaxKeys)
	{
	const uint N = M.GetSeqCount(), LX = M.GetSeqLength(X);
	const uint Blocks = (N+BridgeBlock-1)/BridgeBlock;
	// Dense or unusual inputs can use the existing pair kernel independently.
	size_t Bytes = (size_t(MaxKeys)*Lanes+7)*sizeof(float);
	for (uint Y = Lo; Y < Hi; ++Y)
		{
		const MySparseMx &XY = M.GetSparsePost(M.GetPairIndex(X,Y));
		Bytes += size_t(XY.m_Offsets[LX])*(sizeof(SumRow)+sizeof(uint))
		  + size_t((LX+Lanes-1)/Lanes+1+XY.m_LY)*sizeof(uint);
		}
	if (Bytes > WorkerBudget)
		return false;
	PairTiles Pairs[TargetBlock];
	for (uint Y = Lo; Y < Hi; ++Y)
		Pairs[Y-Lo].Init(M.GetSparsePost(M.GetPairIndex(X,Y)));
	// A lane belongs to one X residue. Missing entries have probability zero.
	// The lane mapping is reused across all target sequences in this block.
	vector<float> Storage(size_t(MaxKeys)*Lanes+7,0.0f);
	float *Weights = reinterpret_cast<float *>((uintptr_t(Storage.data())+31)&~uintptr_t(31));
	for (uint b = 0; b < Blocks; ++b)
		{
		const uint Begin = b*BridgeBlock, End = min(Begin+BridgeBlock,N);
		const MySparseMx *Left[BridgeBlock] = {};
		for (uint Z = Begin; Z < End; ++Z)
			{
			if (Z != X)
				Left[Z-Begin] = Z < X ? &M.GetUpdatedSparsePost(M.GetPairIndex(Z,X))
				  : &M.GetSparsePost(M.GetPairIndex(X,Z));
			}
		for (uint x0 = 0; x0 < LX; x0 += Lanes)
			{
			const uint x1 = min(x0+Lanes,LX);
			uint Low[BridgeBlock], High[BridgeBlock];
			for (uint Z = Begin; Z < End; ++Z)
				{
				const MySparseMx *A = Left[Z-Begin];
				if (!A)
					continue;
				float *Rows = Weights+Base[Z]*Lanes;
				uint ColLo = A->m_LY, ColEnd = 0;
				for (uint x = x0; x < x1; ++x)
					{
					const uint First = A->m_Offsets[x], End = A->m_Offsets[x+1];
					if (First == End)
						continue;
					ColLo = min(ColLo,A->GetCol_Offset(First));
					ColEnd = max(ColEnd,A->GetCol_Offset(End-1)+1);
					for (uint k = First; k < End; ++k)
						Rows[A->GetCol_Offset(k)*Lanes+x-x0] = A->GetProb_Offset(k);
					}
				Low[Z-Begin] = ColLo;
				High[Z-Begin] = ColEnd;
				}
			for (uint Y = Lo; Y < Hi; ++Y)
				Pairs[Y-Lo].Update(x0/Lanes,Right[(Y-Lo)*Blocks+b],Weights);
			for (uint Z = Begin; Z < End; ++Z)
				{
				const MySparseMx *A = Left[Z-Begin];
				if (!A)
					continue;
				float *Rows = Weights+Base[Z]*Lanes;
				const uint ColLo = Low[Z-Begin], ColEnd = High[Z-Begin];
				if (ColEnd <= ColLo)
					continue;
				// A contiguous clear is cheaper for a compact span. Widely
				// separated entries retain the sparse clear.
				if (ColEnd-ColLo <= (A->m_Offsets[x1]-A->m_Offsets[x0])*2)
					std::fill(Rows+ColLo*Lanes,Rows+ColEnd*Lanes,0.0f);
				else
					for (uint x = x0; x < x1; ++x)
						for (uint k = A->m_Offsets[x]; k < A->m_Offsets[x+1]; ++k)
							Rows[A->GetCol_Offset(k)*Lanes+x-x0] = 0.0f;
				}
			}
		}
	for (uint Y = Lo; Y < Hi; ++Y)
		Pairs[Y-Lo].Finish(M.GetUpdatedSparsePost(M.GetPairIndex(X,Y)),N);
	return true;
	}

bool UpdateConsistencyBlocks(MPCFlat &M, uint Threads, uint Iter)
	try
	{
	const uint N = M.GetSeqCount(), Pairs = uint(M.m_Pairs.size());
	if (Threads < 2 || N < 48 || N > 1024 || omp_in_parallel())
		return false;
	const uint Blocks = (N+BridgeBlock-1)/BridgeBlock;
	vector<uint64> Need(N);
	for (uint Y = 0; Y < N; ++Y)
		Need[Y] = uint64(Blocks)*(M.GetSeqLength(Y)+1)*sizeof(uint);
	uint64 Extra = 0, SparseBytes = 0;
	for (uint p = 0; p < Pairs; ++p)
		{
		const MySparseMx &A = M.GetSparsePost(p);
		if (!A.m_LX || !A.m_LY || A.m_LX > MaxSeqLength || A.m_LY > MaxSeqLength)
			return false;
		const auto &Pair = M.GetPair(p);
		const uint64 Bytes = uint64(A.m_Offsets[A.m_LX])*sizeof(JoinedRows::Entry);
		SparseBytes += 2*uint64(A.m_Offsets[A.m_LX])*(sizeof(float)+sizeof(uint));
		Need[Pair.first] += Bytes;
		Need[Pair.second] += Bytes;
		const MySparseMx *Out = (*M.m_ptrUpdatedSparsePosts)[p];
		const uint Capacity = max(A.m_LX, Out ? Out->m_MaxLX : 0u);
		if (A.m_LY > Capacity)
			Extra += size_t(A.m_LY-Capacity)*sizeof(uint);
		}
	// Adapt only the cap on row offsets to the already resident sparse matrices.
	const uint64 OffsetCap = min(uint64(64u*1024*1024),
	  max(uint64(16u*1024*1024),SparseBytes/50));
	if (Extra > OffsetCap)
		return false;

	for (uint p = 0; p < Pairs; ++p)
		{
		const MySparseMx &A = M.GetSparsePost(p);
		MySparseMx &T = M.GetUpdatedSparsePost(p);
		ReserveSparseRows(T,max(A.m_LX,A.m_LY));
		T.AllocVec(A.m_Offsets[A.m_LX]);
		}
#pragma omp parallel for num_threads(Threads) schedule(static)
	for (int p = 0; p < int(Pairs); ++p)
		{
		MySparseMx &Out = M.GetUpdatedSparsePost(p);
		const uint SavedSize = Out.m_VecSize;
		TransposeSparsePost(M.GetSparsePost(p), Out);
		Out.m_VecSize = SavedSize;
		}

	vector<uint> Base(N);
	uint Keys = 0, MaxKeys = 0;
	for (uint Z = 0; Z < N; ++Z)
		{
		if (Z%BridgeBlock == 0)
			Keys = 0;
		Base[Z] = Keys;
		Keys += M.GetSeqLength(Z);
		MaxKeys = max(MaxKeys,Keys);
		}
	uint Completed = 0;
	ProgressStep(0,Pairs,"Consistency (%u/%u)",Iter+1,M.m_ConsistencyIterCount);
	vector<uint> Pending;
	// Descending target blocks keep every required transpose (Z,X) readable:
	// its larger endpoint X is below Lo, while outputs have a target at or above Lo.
	for (uint Hi = N; Hi > 1;)
		{
		uint Lo = Hi-1;
		uint64 Bytes = Need[Lo];
		while (Lo > 1 && Hi-Lo < TargetBlock && Bytes+Need[Lo-1] <= RowBudget)
			Bytes += Need[--Lo];
		vector<JoinedRows> Right;
		int Ready = Bytes <= RowBudget;
		if (Ready)
			{
			Right.resize((Hi-Lo)*Blocks);
#pragma omp parallel for num_threads(Threads) schedule(dynamic) reduction(&:Ready)
			for (int j = 0; j < int(Right.size()); ++j)
				{
				const uint b = j%Blocks;
				try
					{
					Right[j].Build(M,Lo+j/Blocks,b*BridgeBlock,min((b+1)*BridgeBlock,N),Base);
					}
				catch (const std::bad_alloc &)
					{
					Ready = 0;
					}
				}
			}
#pragma omp parallel for num_threads(Threads) schedule(dynamic)
		for (int x = 0; x < int(Lo); ++x)
			{
			if (Ready)
				{
				try
					{
					if (UpdateTargets(M,x,Lo,Hi,Right,Base,MaxKeys))
						continue;
					}
				catch (const std::bad_alloc &)
					{
					// Active inputs still hold the entire original iteration.
					}
				}
			for (uint Y = Lo; Y < Hi; ++Y)
				M.ConsPair(M.GetPairIndex(x,Y));
			}
		// Pairs inside a target block depend on one another's input orientation.
		// Defer them and evaluate them together on the original active matrices.
		for (uint X = Lo; X < Hi; ++X)
			for (uint Y = X+1; Y < Hi; ++Y)
				Pending.push_back(M.GetPairIndex(X,Y));
		Completed += (Hi-Lo)*Lo;
		ProgressStep(Completed-1,Pairs,"Consistency (%u/%u)",Iter+1,M.m_ConsistencyIterCount);
		Hi = Lo;
		}
#pragma omp parallel for num_threads(Threads) schedule(dynamic,2)
	for (int j = 0; j < int(Pending.size()); ++j)
		M.ConsPair(Pending[j]);
	if (!Pending.empty())
		ProgressStep(Pairs-1,Pairs,"Consistency (%u/%u)",Iter+1,M.m_ConsistencyIterCount);
	return true;
	}
	catch (const std::bad_alloc &)
	{
	return false;
	}
} // namespace
#endif

void MPCFlat::ConsIter(uint Iter)
	{
	uint PairCount = SIZE(m_Pairs);
	asserta(PairCount > 0);
	unsigned ThreadCount = GetRequestedThreadCount();
#if defined(__GNUC__) && !defined(__clang__) && defined(__AVX2__) && defined(__FMA__)
	if (UpdateConsistencyBlocks(*this, ThreadCount, Iter))
		{
		swap(m_ptrSparsePosts, m_ptrUpdatedSparsePosts);
		return;
		}
#endif
	uint PairCounter = 0;
#pragma omp parallel for num_threads(ThreadCount) schedule(dynamic, 4)
	for (int PairIndex = 0; PairIndex < (int) PairCount; ++PairIndex)
		{
		Lock();
		ProgressStep(PairCounter++, PairCount, "Consistency (%u/%u)",
		  Iter+1, m_ConsistencyIterCount);
		Unlock();

		ConsPair(PairIndex);
		}

	swap(m_ptrSparsePosts, m_ptrUpdatedSparsePosts);
	}
