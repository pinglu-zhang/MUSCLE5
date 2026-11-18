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


#if MUSCLE_HMM_SIMD && defined(__linux__)
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
			const uint LY = Y.GetLength();
			const size_t Cells = (size_t(LX) + 1)*(size_t(LY) + 1);
			float *Fwd = Scratch.get() + i*Stride;
			float *Rows = Fwd + Cells;
			CalcRollingPostInPlace(X.GetBytePtr(), LX, Y.GetBytePtr(), LY, Fwd);
			// Reuse the retired HMM insertion rows for the decoder's score rows.
			char *TB = reinterpret_cast<char *>(Rows + (size_t(LY)+1)*FWD_INSERT_FLOATS_PER_COLUMN);
			float Score = CalcAlnFlat(Fwd, LX, LY, Rows, TB, Paths[i]);
			Scores[i] = Score/min(LX, LY);
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
#endif

uint UClust::Search(uint SeqIndex, string &Path)
	{
	const Sequence *Seq = m_InputSeqs->GetSequence(SeqIndex);
	const byte *ByteSeq = Seq->GetBytePtr();
	const uint L = Seq->GetLength();

	vector<uint> TopSeqIndexes;
	vector<uint> TopWordCounts;
	m_US.SearchSeq(ByteSeq, L, TopSeqIndexes, TopWordCounts);
	uint TopCount = SIZE(TopSeqIndexes);
	asserta(SIZE(TopWordCounts) == TopCount);
	if (TopCount == 0)
		return UINT_MAX;
	if (TopCount > MAX_REJECTS)
		TopCount = MAX_REJECTS;
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
		uint RepSeqIndex = Search(SeqIndex, Path);
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
