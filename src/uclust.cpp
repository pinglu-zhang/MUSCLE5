#include "muscle.h"
#include "uclust.h"
#include "pwpath.h"
#include "hmmrolling.h"
#include "hmmsimd.h"
#include "mega.h"
#include <memory>
#ifdef __linux__
#include <sys/mman.h>
#endif

float UClust::AlignSeqPair(uint SeqIndex1, uint SeqIndex2, string &Path)
	{
	Path.clear();
	const Sequence *Seq1 = m_InputSeqs->GetSequence(SeqIndex1);
	const Sequence *Seq2 = m_InputSeqs->GetSequence(SeqIndex2);
	const string &Label1 = Seq1->m_Label;
	const string &Label2 = Seq2->m_Label;
	asserta(Seq1 != 0);
	asserta(Seq2 != 0);
	float EA = AlignPairFlat(Label1, Label2, Path);
	return EA;
	}

void UClust::AddSeqToIndex(uint SeqIndex)
	{
	const Sequence *Seq = m_InputSeqs->GetSequence(SeqIndex);
	const byte *ByteSeq = Seq->GetBytePtr();
	const uint L = Seq->GetLength();
	m_US.AddSeq(ByteSeq, L, SeqIndex);
	}


static void RankCentroids(UClust &U, uint Query, uint Limit, vector<uint> &Ranked)
	{
	uint L;
	const byte *Seq = U.m_InputSeqs->GetByteSeq(Query, L);
	vector<uint> Counts;
	U.m_US.SearchSeq(Seq, L, Ranked, Counts, Limit);
	asserta(Ranked.size() == Counts.size());
	if (Ranked.size() > Limit)
		Ranked.resize(Limit);
	}

#if MUSCLE_HMM_SIMD && defined(__linux__)
static float ScoreCentroid(const Sequence &X, const Sequence &Y, float *Fwd, string &Path)
	{
	const uint LX = X.GetLength();
	const uint LY = Y.GetLength();
	const size_t Width = size_t(LY) + 1;
	float *Rows = Fwd + (size_t(LX)+1)*Width;
	char *TB = reinterpret_cast<char *>(Rows + Width*FWD_INSERT_FLOATS_PER_COLUMN);
	CalcRollingPostInPlace(X.GetBytePtr(), LX, Y.GetBytePtr(), LY, Fwd);
	return CalcAlnFlat(Fwd, LX, LY, Rows, TB, Path)/min(LX, LY);
	}

// Evaluate a wave against one fixed centroid index. Only the serial caller
// decides the first passing rank and publishes its path.
static bool SearchCentroidWave(UClust &U, uint Query, const vector<uint> &Ranked,
  uint Count, uint Requested, uint &Hit, string &Path)
	{
	if (Count < 2 || Requested < 2 || Mega::m_Loaded)
		return false;
	const Sequence &X = GetGlobalInputSeqByLabel(U.m_InputSeqs->GetLabelStr(Query));
	const uint LX = X.GetLength();
	uint MaxLY = 0;
	for (uint k = 0; k < Count; ++k)
		{
		const string &Label = U.m_InputSeqs->GetLabelStr(Ranked[k]);
		MaxLY = max(MaxLY, GetSeqLengthByGlobalLabel(Label));
		}

	// Limit total HMM workspace for the wave. Long pairs use the serial path.
	const size_t Budget = (3u*1024*1024)/sizeof(float);
	if (LX == 0 || MaxLY == 0 || LX > Budget || MaxLY > Budget)
		return false;
	const uint64 Width = uint64(MaxLY) + 1;
	const uint64 Stride64 = (LX + 1 + FWD_INSERT_FLOATS_PER_COLUMN)*Width + ((uint64(LX)+1)*Width + sizeof(float)-1)/sizeof(float);
	if (Stride64 > Budget/2)
		return false;
	const size_t Stride = size_t(Stride64);
	// Keep the OpenMP team size fixed across queries of different hit counts.
	const uint Team = min(8u, Requested);
	const uint Workers = min(min(Count, Team), uint(Budget/Stride));
	// A wave owns its pages. Return them after use without growing the
	// allocator arenas used by later Super5 stages.
	const size_t Bytes = Workers*Stride*sizeof(float);
	void *Pages = mmap(nullptr, Bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	if (Pages == MAP_FAILED)
		return false;
	auto Release = [Bytes](float *p) { munmap(p, Bytes); };
	std::unique_ptr<float, decltype(Release)> Scratch(static_cast<float *>(Pages), Release);
	vector<float> Scores(Workers);
	vector<string> Paths(Workers);
	for (string &p : Paths)
		p.reserve(size_t(LX) + MaxLY);
	for (uint First = 0; First < Count; First += Workers)
		{
		const int N = int(min(Count - First, Workers));
#pragma omp parallel for num_threads(Team) if(N > 1) schedule(static)
		for (int i = 0; i < N; ++i)
			{
			const string &Label = U.m_InputSeqs->GetLabelStr(Ranked[First+i]);
			const Sequence &Y = GetGlobalInputSeqByLabel(Label);
			Scores[i] = ScoreCentroid(X, Y, Scratch.get() + i*Stride, Paths[i]);
			}

		for (int i = 0; i < N; ++i)
			if (Scores[i] >= U.m_MinEA)
				{
				Hit = Ranked[First+i];
				Path.swap(Paths[i]);
				return true;
				}
		if (First + N == Count)
			Path.swap(Paths[N-1]);
		}
	Hit = UINT_MAX;
	return true;
	}

// Pair scores and paths depend only on the two sequences. Rank candidates
// against a read-only index in parallel, then publish queries in input order.
class CentroidQueryBatch
	{
	struct Trial
		{
		vector<uint> Tried;
		uint Hit;
		string Path;
		};

	UClust &U;
	const vector<uint> &Order;
	const uint Limit;
	vector<const Sequence *> Seqs;
	vector<Trial> Trials;
	uint Threads = 0;
	uint Start = 0, End = 0, IndexSize = 0;
	size_t Stride = 0, Bytes = 0;
	float *Pages = nullptr;

	void Prepare(uint First)
		{
		Start = First;
		const uint Count = min(uint(Trials.size()), uint(Order.size())-First);
		End = Start + Count;
		IndexSize = SIZE(U.m_CentroidSeqIndexes);
		// The word sorter uses MUSCLE's process-wide allocation accounting.
		// Keep ranking on the caller and parallelize only the pair scoring.
		for (uint j = 0; j < Count; ++j)
			RankCentroids(U, Order[First+j], Limit, Trials[j].Tried);
#pragma omp parallel for num_threads(Threads) schedule(dynamic, 1)
		for (int j = 0; j < (int) Count; ++j)
			{
			const uint Q = Order[First+j];
			Trial &T = Trials[j];
			T.Hit = UINT_MAX;
			float *Work = Pages + size_t(omp_get_thread_num())*Stride;
			for (uint k = 0; k < T.Tried.size(); ++k)
				{
				const uint Target = T.Tried[k];
				if (ScoreCentroid(*Seqs[Q], *Seqs[Target], Work, T.Path) >= U.m_MinEA)
					{
					T.Hit = Target;
					T.Tried.resize(k+1);
					break;
					}
				}
			}
		}

public:
	CentroidQueryBatch(UClust &Owner, const vector<uint> &Queries, uint MaxRejects)
	  : U(Owner), Order(Queries), Limit(MaxRejects)
		{
		const uint Requested = GetRequestedThreadCount();
		if (Requested < 2 || Order.size() < 2*size_t(Requested) || Mega::m_Loaded)
			return;
		uint Longest = 0;
		Seqs.resize(Order.size());
		for (uint i = 0; i < Seqs.size(); ++i)
			{
			Seqs[i] = &GetGlobalInputSeqByLabel(U.m_InputSeqs->GetLabelStr(i));
			const uint L = Seqs[i]->GetLength();
			if (L == 0)
				return;
			Longest = max(Longest, L);
			}

		const size_t Budget = (64u*1024*1024)/sizeof(float);
		const uint64 Width = uint64(Longest)+1;
		if (Width > Budget/Width)
			return;
		const size_t Cells = size_t(Width*Width);
		Stride = Cells + size_t(Width)*FWD_INSERT_FLOATS_PER_COLUMN
		  + (Cells+sizeof(float)-1)/sizeof(float);
		Threads = min(Requested, uint(Budget/Stride));
		if (Threads < 2)
			return;
		Trials.resize(2*size_t(Threads));
		Bytes = Threads*Stride*sizeof(float);
		void *p = mmap(nullptr, Bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
		if (p != MAP_FAILED)
			Pages = static_cast<float *>(p);
		}
	CentroidQueryBatch(const CentroidQueryBatch &) = delete;
	CentroidQueryBatch &operator=(const CentroidQueryBatch &) = delete;
	~CentroidQueryBatch() { if (Pages) munmap(Pages, Bytes); }
	bool Available() const { return Pages != nullptr; }

	uint Search(uint k, string &Path)
		{
		if (k == End)
			Prepare(k);
		Trial &T = Trials[k-Start];
		if (SIZE(U.m_CentroidSeqIndexes) == IndexSize)
			{
			Path.swap(T.Path);
			return T.Hit;
			}
		// An earlier query added a centroid. Repeat the original k-mer
		// ranking, including its ties and rejection limit. Never assume
		// the old candidates kept their ranks.
		const uint Q = Order[k];
		vector<uint> Ranked;
		RankCentroids(U, Q, Limit, Ranked);
		for (uint Target : Ranked)
			{
			if (Target == T.Hit)
				{
				Path.swap(T.Path);
				return Target;
				}
			if (find(T.Tried.begin(), T.Tried.end(), Target) == T.Tried.end() &&
			  ScoreCentroid(*Seqs[Q], *Seqs[Target], Pages, Path) >= U.m_MinEA)
				return Target;
			}
		return UINT_MAX;
		}
	};

#endif

uint UClust::Search(uint SeqIndex, string &Path)
	{
	vector<uint> TopSeqIndexes;
	RankCentroids(*this, SeqIndex, MAX_REJECTS, TopSeqIndexes);
	const uint TopCount = SIZE(TopSeqIndexes);
	if (TopCount == 0)
		return UINT_MAX;
	uint ThreadCount = GetRequestedThreadCount();

	uint CentroidSeqIndex = UINT_MAX;
#if MUSCLE_HMM_SIMD && defined(__linux__)
	if (SearchCentroidWave(*this, SeqIndex, TopSeqIndexes, TopCount, ThreadCount, CentroidSeqIndex, Path))
		return CentroidSeqIndex;
#endif
	for (int TopIndex = 0; TopIndex < (int) TopCount; ++TopIndex)
		{
		uint TopSeqIndex = TopSeqIndexes[TopIndex];
		float EA = AlignSeqPair(SeqIndex, TopSeqIndex, Path);
		if (EA >= m_MinEA)
			{
			CentroidSeqIndex = TopSeqIndex;
			break;
			}
		}
	return CentroidSeqIndex;
	}

void UClust::Run(MultiSequence &InputSeqs, float MinEA)
	{
	m_InputSeqs = &InputSeqs;
	m_MinEA = MinEA;
	m_US.Init();

	const uint InputSeqCount = m_InputSeqs->GetSeqCount();
	const uint GSICount = GetGlobalMSSeqCount();
	vector<uint> GSIToInputSeqIndex(GSICount, UINT_MAX);
	for (uint SeqIndex = 0; SeqIndex < InputSeqCount; ++SeqIndex)
		{
		const string Label = string(m_InputSeqs->GetLabel(SeqIndex));
		uint GSI = GetGSIByLabel(Label);
		asserta(GSI < GSICount);
		asserta(GSIToInputSeqIndex[GSI] == UINT_MAX);
		GSIToInputSeqIndex[GSI] = SeqIndex;
		}

	uint CentroidCount = 0;
	uint MemberCount = 0;

	m_SeqIndexToCentroidSeqIndex.clear();
	m_SeqIndexToPath.clear();

	m_SeqIndexToCentroidSeqIndex.resize(InputSeqCount, UINT_MAX);
	m_SeqIndexToPath.resize(InputSeqCount);

#if DEBUG
	vector<bool> Done(InputSeqCount, false);
#endif
	vector<uint> Order;
	InputSeqs.GetLengthOrder(Order);
#if MUSCLE_HMM_SIMD && defined(__linux__)
	CentroidQueryBatch Batch(*this, Order, MAX_REJECTS);
#endif
	uint LastLength = UINT_MAX;
	const float MinEE = (1 - m_MinEA);
	for (uint k = 0; k < InputSeqCount; ++k)
		{
		uint SeqIndex = Order[k];
		asserta(SeqIndex < InputSeqCount);
#if DEBUG
		asserta(Done[SeqIndex] == false);
		Done[SeqIndex] = true;
#endif
		const uint L = (uint) InputSeqs.GetSeqLength(SeqIndex);
		asserta(L <= LastLength);
		LastLength = L;

		ProgressStep(k, InputSeqCount,
		  "UCLUST %u seqs EE<%.2f, %u centroids, %u members",
		  InputSeqCount, MinEE, CentroidCount, MemberCount);

		string &Path = m_SeqIndexToPath[SeqIndex];
		uint RepSeqIndex;
#if MUSCLE_HMM_SIMD && defined(__linux__)
		if (Batch.Available())
			RepSeqIndex = Batch.Search(k, Path);
		else
#endif
			RepSeqIndex = Search(SeqIndex, Path);
		if (RepSeqIndex == UINT_MAX)
			{
			m_CentroidSeqIndexes.push_back(SeqIndex);
			AddSeqToIndex(SeqIndex);
			++CentroidCount;
			RepSeqIndex = SeqIndex;
			Path.clear();
			}
		else
			++MemberCount;
		  
		m_SeqIndexToCentroidSeqIndex[SeqIndex] = RepSeqIndex;
		}
	}

void UClust::GetCentroidSeqs(MultiSequence &CentroidSeqs) const
	{
	const uint CentroidCount = SIZE(m_CentroidSeqIndexes);
	for (uint i = 0; i < CentroidCount; ++i)
		{
		uint SeqIndex = m_CentroidSeqIndexes[i];
		const Sequence *Seq = m_InputSeqs->GetSequence(SeqIndex);
		CentroidSeqs.AddSequence(Seq, false);
		}
	AssertSameLabels(CentroidSeqs);
	}

void UClust::GetGSIs(
  vector<uint> &CentroidGSIs,
  vector<uint> &MemberGSIs,
  vector<uint> &MemberCentroidGSIs,
  vector<string> &GSIToMemberCentroidPath) const
	{
	CentroidGSIs.clear();
	MemberGSIs.clear();
	MemberCentroidGSIs.clear();
	GSIToMemberCentroidPath.clear();

	const uint InputSeqCount = m_InputSeqs->GetSeqCount();
	const uint GSICount = GetGSICount();

	GSIToMemberCentroidPath.resize(GSICount);

	const uint ClusterCount = SIZE(m_CentroidSeqIndexes);
	for (uint ClusterIndex = 0; ClusterIndex < ClusterCount; ++ClusterIndex)
		{
		uint CentroidSeqIndex = m_CentroidSeqIndexes[ClusterIndex];
		const Sequence *Seq = m_InputSeqs->GetSequence(CentroidSeqIndex);
		uint CentroidGSI = GetGSIByLabel(Seq->m_Label);
		CentroidGSIs.push_back(CentroidGSI);
		}

	asserta(SIZE(m_SeqIndexToCentroidSeqIndex) == InputSeqCount);
	for (uint MemberSeqIndex = 0; MemberSeqIndex < InputSeqCount; ++MemberSeqIndex)
		{
		uint CentroidSeqIndex = m_SeqIndexToCentroidSeqIndex[MemberSeqIndex];
		if (CentroidSeqIndex == MemberSeqIndex)
			continue;

		const string &Path = m_SeqIndexToPath[MemberSeqIndex];
		const Sequence *MemberSeq = m_InputSeqs->GetSequence(MemberSeqIndex);
		const Sequence *CentroidSeq = m_InputSeqs->GetSequence(CentroidSeqIndex);

		uint MemberGSI = GetGSIByLabel(MemberSeq->m_Label);
		uint MemberCentroidGSI = GetGSIByLabel(CentroidSeq->m_Label);

		MemberGSIs.push_back(MemberGSI);
		MemberCentroidGSIs.push_back(MemberCentroidGSI);

		asserta(!Path.empty());
		GSIToMemberCentroidPath[MemberGSI] = Path;
		}
	}

void cmd_uclust()
	{
	const string &InputFileName = opt(uclust);
	const string &OutputFileName = opt(output);
	const float MinEA = (float) optd(minea, 0.9);

	//LoadGlobalInputMS(InputFileName);
	//MultiSequence &InputSeqs = GetGlobalInputMS();
	MultiSequence InputSeqs;
	LoadInput(InputSeqs);

	bool IsNucleo = InputSeqs.GuessIsNucleo();
	if (IsNucleo)
		SetAlpha(ALPHA_Nucleo);
	else
		SetAlpha(ALPHA_Amino);

	UClust U;
	U.Run(InputSeqs, MinEA);

	MultiSequence *CentroidSeqs = new MultiSequence;
	U.GetCentroidSeqs(*CentroidSeqs);
	CentroidSeqs->WriteMFA(OutputFileName);
	}
