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

// Pool very similar bridge sequences in ordinary protein alignments.
// All endpoint pairs and their sparse output coordinates are still evaluated.
struct BridgePlan
	{
	vector<uint> Weight, Mate, Active;
	float Step = 1.0f;
	vector<uint> Owner;
	vector<vector<uint>> Members;

	// Select centers far apart under the existing HMM expected-accuracy
	// similarity. Each assignment must clear the same similarity floor as
	// ordinary bridge pairs; no reference alignment enters the selection.
	void Cover(const MPCFlat &M, uint Iter)
		{
		const uint n=M.GetSeqCount(), budget=(n+3)/4;
		vector<float> nearest(n, -1.0f);
		vector<uint> centers;
		Owner.assign(n, UINT_MAX);
		Members.resize(n);
		uint next=0;
		for (;;)
			{
			centers.push_back(next);
			Owner[next]=next;
			Members[next].push_back(next);
			uint far=UINT_MAX;
			for (uint i=0;i<n;++i)
				if (Owner[i]==UINT_MAX)
					{
					nearest[i]=max(nearest[i],M.m_DistMx[next][i]);
					if (far==UINT_MAX || nearest[i]<nearest[far]) far=i;
					}
			if (far==UINT_MAX || (centers.size()>=budget && nearest[far]>=0.80f)) break;
			next=far;
			}
		vector<uint> order;
		for (uint i=0;i<n;++i) if (Owner[i]==UINT_MAX) order.push_back(i);
		std::sort(order.begin(), order.end(), [&](uint a,uint b)
			{ return nearest[a]!=nearest[b] ? nearest[a]>nearest[b] : a<b; });
		for (uint i:order)
			{
			uint best=UINT_MAX;
			for (uint c:centers)
				if (Members[c].size()<8 && M.m_DistMx[c][i]>=0.80f &&
				  (best==UINT_MAX || M.m_DistMx[c][i]>M.m_DistMx[best][i]))
					best=c;
			if (best==UINT_MAX) { best=i; centers.push_back(i); }
			Owner[i]=best;
			Members[best].push_back(i);
			}
		Weight.assign(n,0);
		for (uint c:centers)
			{
			auto &g=Members[c];
			// Rank alternate representatives by their mean similarity to the group.
			vector<float> similarity(n,0.0f);
			for (uint a:g) for (uint b:g)
				if (a!=b) similarity[a]+=M.m_DistMx[a][b];
			std::sort(g.begin(),g.end(),[&](uint a,uint b)
				{ return similarity[a]!=similarity[b] ? similarity[a]>similarity[b] : a<b; });
			std::rotate(g.begin(),g.begin()+Iter%g.size(),g.end());
			Weight[g.front()]=uint(g.size());
			}
		for (uint i=0;i<n;++i) if (Weight[i]) Active.push_back(i);
		}

	uint AvailableRepresentative(uint group,uint x,uint y) const
		{
		for (uint z:Members[group]) if (z!=x && z!=y) return z;
		return UINT_MAX;
		}

	BridgePlan(const MPCFlat &M, uint Iter)
		{
		const uint N = M.GetSeqCount();
		if (optset_align && g_Alpha==ALPHA_Amino && !optset_hmmin &&
		  (!optset_perturb || opt_perturb==0) && !opt(stratified) && !opt(diversified) &&
		  !optset_replicates && !optset_consiters && Iter==1 && N>=128)
			{
			const BridgePlan first(M,0);
			Cover(M,Iter);
			uint64 firstSquares=0, secondSquares=0;
			for (uint w:first.Weight) firstSquares+=uint64(w)*w;
			for (uint w:Weight) secondSquares+=uint64(w)*w;
			// Both plans have total weight N. Their Kish effective sample
			// size ratio is therefore the inverse ratio of squared weights.
			Step=float(min(1.0,double(firstSquares)/double(secondSquares)));
			Progress("Bridge effective sample sizes: %.1f -> %.1f, step %.5f",
			  double(N)*N/firstSquares,double(N)*N/secondSquares,double(Step));
			return;
			}
		Weight.assign(N, 1);
		Mate.assign(N, UINT_MAX);
		if (optset_align && g_Alpha == ALPHA_Amino && !optset_hmmin &&
		  (!optset_perturb || opt_perturb == 0) && N >= 128)
			{
			struct Edge { uint A,B; float Score; };
			vector<Edge> Edges;
			for (uint a = 0; a < N; ++a)
				for (uint b = a+1; b < N; ++b)
					if (M.m_DistMx[a][b] >= 0.80f)
						Edges.push_back({a,b,M.m_DistMx[a][b]});
			std::sort(Edges.begin(),Edges.end(),[](const Edge &u,const Edge &v)
				{
				if (u.Score != v.Score) return u.Score > v.Score;
				if (u.A != v.A) return u.A < v.A;
				return u.B < v.B;
				});
			for (const Edge &e : Edges)
				{
				if (Mate[e.A] != UINT_MAX || Mate[e.B] != UINT_MAX) continue;
				Mate[e.A] = e.B;
				Mate[e.B] = e.A;
				Weight[e.A] = Iter%2 ? 0 : 2;
				Weight[e.B] = 2 - Weight[e.A];
				}
			}
		for (uint z = 0; z < N; ++z)
			if (Weight[z]) Active.push_back(z);
		}

	uint Count(uint x,uint y,uint z) const
		{
		if (!Owner.empty())
			{
			const uint c=Owner[z];
			if (c!=Owner[x] && c!=Owner[y]) return Weight[z];
			return z==AvailableRepresentative(c,x,y) ?
			  uint(Members[c].size())-uint(Owner[x]==c)-uint(Owner[y]==c) : 0;
			}
		if (z == Mate[x] || z == Mate[y]) return 1;
		return Weight[z];
		}
	};

// Original matrices remain read-only throughout the pass.
static const MySparseMx &RowsEndingIn(MPCFlat &M,uint seq,uint z,MySparseMx &T)
	{
	const auto &A=M.GetSparsePost(M.GetPairIndex(min(seq,z),max(seq,z)));
	if (seq < z) return A;
	ReserveSparseRows(T,A.m_LY);
	T.AllocVec(A.m_Offsets[A.m_LX]);
	TransposeSparsePost(A,T);
	return T;
	}

// Both passes retain the same coordinates in their original order.
static void DampCover(const MySparseMx &input, MySparseMx &updated, float alpha)
	{
	if (alpha>=1.0f) return;
	const uint used=input.m_Offsets[input.m_LX];
	for (uint k=0;k<used;++k)
		{
		const float first=input.GetProb_Offset(k);
		const float second=updated.GetProb_Offset(k);
		updated.SetProb_Offset(k,first+alpha*(second-first));
		}
	}

static void PooledPair(MPCFlat &M,uint p,const BridgePlan &Plan)
	{
	if (Plan.Active.size()==Plan.Weight.size()) { M.ConsPair(p); return; }
	const auto &pair=M.GetPair(p);
	const uint X=pair.first,Y=pair.second,N=M.GetSeqCount();
	const MySparseMx &XY=M.GetSparsePost(p);
	vector<float> Post(size_t(XY.m_LX)*XY.m_LY);
	XY.ToPost(Post.data());
	for (float &a:Post) a*=2;
	MySparseMx T;
	for (uint z=0;z<N;++z)
		{
		const uint weight=Plan.Count(X,Y,z);
		if (z==X || z==Y || !weight) continue;
		const auto &A=M.GetSparsePost(M.GetPairIndex(min(X,z),max(X,z)));
		const auto &B=M.GetSparsePost(M.GetPairIndex(min(Y,z),max(Y,z)));
		if (z<X) RelaxFlat_ZX_ZY(A,B,float(weight),Post.data());
		else if (z<Y) RelaxFlat_XZ_ZY(A,B,float(weight),Post.data());
		else
			{
			ReserveSparseRows(T,B.m_LY);
			T.AllocVec(B.m_Offsets[B.m_LX]);
			TransposeSparsePost(B,T);
			RelaxFlat_XZ_ZY(A,T,float(weight),Post.data());
			}
		}
	auto &Out=M.GetUpdatedSparsePost(p);
	Out.UpdateFromPost(XY,Post.data(),N);
	DampCover(XY,Out,Plan.Step);
	Out.m_X=XY.m_X; Out.m_Y=XY.m_Y;
	}

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
	void Build(MPCFlat &M, uint Y, uint Begin, uint End, const vector<uint> &Base, const BridgePlan &Plan)
		{
		const uint LY = M.GetSeqLength(Y);
		Offsets.assign(LY+1,0);
		const MySparseMx *Matrices[BridgeBlock] = {};
		for (uint j=Begin; j<End; ++j)
			{
			const uint Z=Plan.Active[j];
			if (Z == Y)
				continue;
			const MySparseMx &A = Z < Y ? M.GetUpdatedSparsePost(M.GetPairIndex(Z,Y))
			  : M.GetSparsePost(M.GetPairIndex(Y,Z));
			Matrices[j-Begin] = &A;
			for (uint y = 0; y < LY; ++y)
				Offsets[y+1] += A.m_Offsets[y+1]-A.m_Offsets[y];
			}
		for (uint y = 0; y < LY; ++y)
			Offsets[y+1] += Offsets[y];
		Entries.resize(Offsets[LY]);
		for (uint y = 0; y < LY; ++y)
			{
			uint out = Offsets[y];
			for (uint j=Begin; j<End; ++j)
				{
				const uint Z=Plan.Active[j];
				const MySparseMx *A = Matrices[j-Begin];
				if (!A)
					continue;
				for (uint k = A->m_Offsets[y]; k < A->m_Offsets[y+1]; ++k)
					{
					Entries[out++] = {(Base[Z]+A->GetCol_Offset(k))*Lanes, A->GetProb_Offset(k)*Plan.Weight[Z]};
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

		// The mate of either endpoint contributes its own evidence exactly once.
	// This also preserves the two identity contributions and denominator N.
	void RestoreEndpointMate(MPCFlat &M,uint X,uint Y,uint Z,const BridgePlan &Plan)
		{
		if (Z==UINT_MAX || Z==X || Z==Y) return;
		MySparseMx ta,tb;
		const auto &A=RowsEndingIn(M,X,Z,ta);
		const auto &B=RowsEndingIn(M,Y,Z,tb);
		const auto &XY=*Source;
		const float adjustment=float(Plan.Count(X,Y,Z))-float(Plan.Weight[Z]);
		if (adjustment==0) return;
		for (uint x=0;x<XY.m_LX;++x)
			for (uint k=XY.m_Offsets[x];k<XY.m_Offsets[x+1];++k)
				{
				const uint y=XY.GetCol_Offset(k);
				uint i=A.m_Offsets[x],j=B.m_Offsets[y];
				float contribution=0;
				while (i<A.m_Offsets[x+1] && j<B.m_Offsets[y+1])
					{
					const uint a=A.GetCol_Offset(i),b=B.GetCol_Offset(j);
					if (a<b) ++i;
					else if (b<a) ++j;
					else { contribution+=A.GetProb_Offset(i)*B.GetProb_Offset(j); ++i; ++j; }
					}
				Groups[Slots[k]].Sum[x%Lanes]+=adjustment*contribution;
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
  const vector<uint> &Base, uint MaxKeys, const BridgePlan &Plan)
	{
	const uint N = M.GetSeqCount(), LX = M.GetSeqLength(X);
	const uint Blocks = (uint(Plan.Active.size())+BridgeBlock-1)/BridgeBlock;
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
		const uint Begin = b*BridgeBlock, End = min(Begin+BridgeBlock,uint(Plan.Active.size()));
		const MySparseMx *Left[BridgeBlock] = {};
		for (uint j=Begin; j<End; ++j)
			{
			const uint Z=Plan.Active[j];
			if (Z != X)
				Left[j-Begin] = Z < X ? &M.GetUpdatedSparsePost(M.GetPairIndex(Z,X))
				  : &M.GetSparsePost(M.GetPairIndex(X,Z));
			}
		for (uint x0 = 0; x0 < LX; x0 += Lanes)
			{
			const uint x1 = min(x0+Lanes,LX);
			uint Low[BridgeBlock], High[BridgeBlock];
			for (uint j=Begin; j<End; ++j)
				{
				const uint Z=Plan.Active[j];
				const MySparseMx *A = Left[j-Begin];
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
				Low[j-Begin] = ColLo;
				High[j-Begin] = ColEnd;
				}
			for (uint Y = Lo; Y < Hi; ++Y)
				Pairs[Y-Lo].Update(x0/Lanes,Right[(Y-Lo)*Blocks+b],Weights);
			for (uint j=Begin; j<End; ++j)
				{
				const uint Z=Plan.Active[j];
				const MySparseMx *A = Left[j-Begin];
				if (!A)
					continue;
				float *Rows = Weights+Base[Z]*Lanes;
				const uint ColLo = Low[j-Begin], ColEnd = High[j-Begin];
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
		{
		auto &P=Pairs[Y-Lo];
		if (Plan.Owner.empty())
			{
			P.RestoreEndpointMate(M,X,Y,Plan.Mate[X],Plan);
			P.RestoreEndpointMate(M,X,Y,Plan.Mate[Y],Plan);
			}
		else
			{
			for (uint endpoint : {X,Y})
				{
				const uint c=Plan.Owner[endpoint];
				if (endpoint==Y && c==Plan.Owner[X]) continue;
				const uint z=Plan.AvailableRepresentative(c,X,Y);
				P.RestoreEndpointMate(M,X,Y,z,Plan);
				}
			}
		MySparseMx &out=M.GetUpdatedSparsePost(M.GetPairIndex(X,Y));
		P.Finish(out,N);
		DampCover(*P.Source,out,Plan.Step);
		}
	return true;
	}

bool UpdateConsistencyBlocks(MPCFlat &M, uint Threads, uint Iter)
	try
	{
	const uint N = M.GetSeqCount(), Pairs = uint(M.m_Pairs.size());
	if (Threads < 2 || N < 48 || N > 1024 || omp_in_parallel())
		return false;
	const BridgePlan Plan(M,Iter);
	const uint Blocks = (uint(Plan.Active.size())+BridgeBlock-1)/BridgeBlock;
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
		if (Plan.Weight[Pair.second]) Need[Pair.first] += Bytes;
		if (Plan.Weight[Pair.first]) Need[Pair.second] += Bytes;
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
	for (uint j=0; j<Plan.Active.size(); ++j)
		{
		const uint Z=Plan.Active[j];
		if (j%BridgeBlock == 0)
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
					Right[j].Build(M,Lo+j/Blocks,b*BridgeBlock,min((b+1)*BridgeBlock,uint(Plan.Active.size())),Base,Plan);
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
					if (UpdateTargets(M,x,Lo,Hi,Right,Base,MaxKeys,Plan))
						continue;
					}
				catch (const std::bad_alloc &)
					{
					// Active inputs still hold the entire original iteration.
					}
				}
			for (uint Y = Lo; Y < Hi; ++Y)
				PooledPair(M,M.GetPairIndex(x,Y),Plan);
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
		PooledPair(M,Pending[j],Plan);
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
