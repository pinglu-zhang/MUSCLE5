#include "muscle.h"
#include "eacluster.h"
#include <queue>

void MakeReplicateFileName_N(const string &Pattern, uint N, string &FileName)
	{
	FileName.clear();
	bool Found = false;
	for (uint i = 0; i < SIZE(Pattern); ++i)
		{
		char c = Pattern[i];
		if (c == '@')
			{
			string s;
			Ps(s, "%u", N);
			FileName += s;
			Found = true;
			}
		else
			FileName += c;
		}
	if (!Found)
		{
		string s;
		Ps(s, "%u", N);
		FileName += s;
		}
	}

void EACluster::Clear()
	{
	m_CentroidSeqIndexes.clear();
	m_CentroidIndexToSeqIndexes.clear();
	m_SeqIndexToCentroidIndex.clear();
	m_ClusterMFAs.clear();
	}

void EACluster::Run(MultiSequence &InputSeqs, float MinEA)
	{
	AssertSameLabels(InputSeqs);

	Clear();

	m_US.Init();

	m_InputSeqs = &InputSeqs;
	const uint InputSeqCount = InputSeqs.GetSeqCount();
	asserta(InputSeqCount > 0);
	m_SeqIndexToCentroidIndex.clear();
	m_SeqIndexToCentroidIndex.resize(InputSeqCount, UINT_MAX);
	const float MinEE = (1 - MinEA);
	uint ClusterCount = 0;
	uint MemberCount = 0;
	for (uint SeqIndex = 0; SeqIndex < InputSeqCount; ++SeqIndex)
		{
		ProgressStep(SeqIndex, InputSeqCount,
		  "UCLUST %u seqs EE<%.2f, %u centroids, %u members",
		  InputSeqCount, MinEE, ClusterCount, MemberCount);

		const char *Label = m_InputSeqs->GetSequence(SeqIndex)->m_Label.c_str();
		float BestEA;
		uint CentroidIndex = GetBestCentroid(SeqIndex, MinEA, BestEA);
		m_SeqIndexToCentroidIndex[SeqIndex] = CentroidIndex;
		if (CentroidIndex == UINT_MAX)
			{
			uint ClusterIndex = ClusterCount;
			++ClusterCount;
			m_SeqIndexToCentroidIndex[SeqIndex] = ClusterIndex;
			m_CentroidSeqIndexes.push_back(SeqIndex);
			vector<uint> v;
			v.push_back(SeqIndex);
			m_CentroidIndexToSeqIndexes.push_back(v);

			uint L;
			const byte *ByteSeq = m_InputSeqs->GetByteSeq(SeqIndex, L);
			m_US.AddSeq(ByteSeq, L, SeqIndex);
			}
		else
			{
			++MemberCount;
			asserta(CentroidIndex < SIZE(m_CentroidIndexToSeqIndexes));
			asserta(CentroidIndex < SIZE(m_CentroidSeqIndexes));
			uint CentroidSeqIndex = m_CentroidSeqIndexes[CentroidIndex];
			const char *CentroidLabel = m_InputSeqs->GetSequence(CentroidSeqIndex)->m_Label.c_str();
			m_CentroidIndexToSeqIndexes[CentroidIndex].push_back(SeqIndex);
			}

		Validate();
		}
	MakeClusterMFAs();
	}

namespace
{
struct CentroidCompletion
	{
	uint64 Work;
	uint Rank;
	uint End;
	uint Worker;

	bool operator<(const CentroidCompletion &Other) const
		{
		if (Work != Other.Work)
			return Work > Other.Work;
		return Rank > Other.Rank;
		}
	};
}

uint EACluster::GetBestCentroid(uint SeqIndex, float MinEA, float &BestEA)
	{
	uint CentroidCount = SIZE(m_CentroidSeqIndexes);
	if (CentroidCount == 0)
		return UINT_MAX;

	uint L;
	const byte *ByteSeq = m_InputSeqs->GetByteSeq(SeqIndex, L);

	vector<uint> TopSeqIndexes;
	vector<uint> TopWordCounts;
	m_US.SearchSeq(ByteSeq, L, TopSeqIndexes, TopWordCounts);
	const uint TopCount = SIZE(TopSeqIndexes);
	asserta(SIZE(TopWordCounts) == TopCount);
	if (TopCount == 0)
		return UINT_MAX;

	const uint ThreadCount = GetRequestedThreadCount();
	const uint Workers = min(ThreadCount, TopCount);
	const string &Label = m_InputSeqs->GetLabelStr(SeqIndex);
	BestEA = 0;
	uint BestCentroidIndex = UINT_MAX;
	uint BestRank = UINT_MAX;
	auto Consider = [&](uint Rank, float EA)
		{
		if (EA > MinEA && (EA > BestEA || (EA == BestEA && Rank < BestRank)))
			{
			BestEA = EA;
			BestRank = Rank;
			BestCentroidIndex = m_SeqIndexToCentroidIndex[TopSeqIndexes[Rank]];
			asserta(BestCentroidIndex < CentroidCount);
			}
		return (BestEA >= MinEA && (BestEA > 0.9 || BestEA - EA > 0.3)) ||
		  (BestEA < MinEA - 0.3 && Rank > 20);
		};
	auto Score = [&](uint Rank)
		{
		return AlignSeqPair(Label, m_InputSeqs->GetLabelStr(TopSeqIndexes[Rank]));
		};
	if (Workers == 1)
		{
		for (uint Rank = 0; Rank < TopCount; ++Rank)
			if (Consider(Rank, Score(Rank)))
				break;
		return BestCentroidIndex;
		}
	vector<float> Scores(Workers);
	if (TopCount == Workers)
		{
#pragma omp parallel for num_threads(ThreadCount) schedule(static)
		for (int Rank = 0; Rank < (int) TopCount; ++Rank)
			Scores[Rank] = Score(Rank);
		for (uint Rank = 0; Rank < TopCount; ++Rank)
			Consider(Rank, Scores[Rank]);
		return BestCentroidIndex;
		}

	// Static contiguous rank ranges reproduce the set of initial trials.
	// Order their completions by a deterministic work estimate. A stop
	// prevents new trials, while every already-started trial remains eligible.
	auto TargetLength = [&](uint Rank)
		{
		return m_InputSeqs->GetSequence(TopSeqIndexes[Rank])->GetLength();
		};
	priority_queue<CentroidCompletion> Pending;
	vector<CentroidCompletion> ToScore;
	ToScore.reserve(Workers);
	vector<byte> Ready(Workers, 0);
	auto Start = [&](const CentroidCompletion &C)
		{
		Pending.push(C);
		ToScore.push_back(C);
		Ready[C.Worker] = 0;
		};
	const uint Base = TopCount / Workers;
	const uint Extra = TopCount % Workers;
	uint First = 0;
	for (uint Worker = 0; Worker < Workers; ++Worker)
		{
		const uint End = First + Base + (Worker < Extra);
		const uint LY = TargetLength(First);
		// For a fixed query, matrix work scales with target length in each
		// of the two HMM sweeps. Give earlier k-mer ranges a small head start,
		// reduced in proportion to the shorter/longer length ratio.
		const uint64 Offset = uint64(Worker)*min(L, LY)/max(1u, max(L, LY));
		Start({2*uint64(LY) + Offset, First, End, Worker});
		First = End;
		}
	bool Stop = false;
	while (!Pending.empty())
		{
		const CentroidCompletion C = Pending.top();
		Pending.pop();
		if (!Ready[C.Worker])
			{
			const int Count = (int) ToScore.size();
#pragma omp parallel for num_threads(ThreadCount) if(Count > 1) schedule(static)
			for (int i = 0; i < Count; ++i)
				{
				const CentroidCompletion &Trial = ToScore[i];
				Scores[Trial.Worker] = Score(Trial.Rank);
				Ready[Trial.Worker] = 1;
				}
			ToScore.clear();
			}
		if (Consider(C.Rank, Scores[C.Worker]))
			Stop = true;
		const uint Next = C.Rank + 1;
		if (!Stop && Next < C.End)
			Start({C.Work + 2*uint64(TargetLength(Next)), Next, C.End, C.Worker});
		}
	return BestCentroidIndex;
	}

void EACluster::GetClusterMFAs(vector<MultiSequence *> &MFAs) const
	{
	const uint N = SIZE(m_ClusterMFAs);
	MFAs.clear();
	for (uint i = 0; i < N; ++i)
		{
		MultiSequence *ClusterMFA = m_ClusterMFAs[i];
		AssertSameLabels(*ClusterMFA);
		MFAs.push_back(ClusterMFA);
		}
	}

void EACluster::WriteMFAs(const string &FileNamePattern) const
	{
	const uint CentroidCount = SIZE(m_ClusterMFAs);
	for (uint CentroidIndex = 0; CentroidIndex < CentroidCount; ++CentroidIndex)
		{
		ProgressStep(CentroidIndex, CentroidCount, "Write cluster MFAs");

		const MultiSequence *MFA = m_ClusterMFAs[CentroidIndex];
		asserta(MFA != 0);

		string FileName;
		MakeReplicateFileName_N(FileNamePattern, CentroidIndex+1, FileName);

		MFA->WriteMFA(FileName);
		}
	}

void EACluster::MakeClusterMFAs()
	{
	const uint CentroidCount = SIZE(m_CentroidSeqIndexes);
	m_ClusterMFAs.clear();
	for (uint CentroidIndex = 0; CentroidIndex < CentroidCount; ++CentroidIndex)
		{
		ProgressStep(CentroidIndex, CentroidCount, "Make cluster MFAs");

		MultiSequence *ClusterMFA = new MultiSequence;
		asserta(ClusterMFA != 0);

		const vector<uint> &SeqIndexes =
		  m_CentroidIndexToSeqIndexes[CentroidIndex];
		const uint MemberCount = SIZE(SeqIndexes);
		for (uint i = 0; i < MemberCount; ++i)
			{
			uint SeqIndex = SeqIndexes[i];
			const Sequence *seq = m_InputSeqs->GetSequence(SeqIndex);
			ClusterMFA->AddSequence(seq, false);
			}
		AssertSameLabels(*ClusterMFA);
		m_ClusterMFAs.push_back(ClusterMFA);
		}
	AssertSameSeqsVec(*m_InputSeqs, m_ClusterMFAs);
	}

float EACluster::AlignSeqPair(const string &Label1, const string &Label2)
	{
	//const Sequence *Seq1 = m_InputSeqs->GetSequence(SeqIndex1);
	//const Sequence *Seq2 = m_InputSeqs->GetSequence(SeqIndex2);
	const Sequence *Seq1 = GetSequenceByGlobalLabel(Label1);
	const Sequence *Seq2 = GetSequenceByGlobalLabel(Label2);

	string Path;
	float EA = AlignPairFlat(Label1, Label2, Path);
	return EA;
	}

void EACluster::Validate() const
	{
	const uint SeqCount = m_InputSeqs->GetSeqCount();
	const uint CentroidCount = SIZE(m_CentroidSeqIndexes);
	asserta(SIZE(m_CentroidIndexToSeqIndexes) == CentroidCount);
	for (uint CentroidIndex = 0; CentroidIndex < CentroidCount; ++CentroidIndex)
		{
		uint CentroidSeqIndex = m_CentroidSeqIndexes[CentroidIndex];
		asserta(CentroidSeqIndex < SeqCount);
		const vector<uint> &MemberSeqIndexes = m_CentroidIndexToSeqIndexes[CentroidIndex];
		const uint MemberCount = SIZE(MemberSeqIndexes);
		for (uint MemberIndex = 0; MemberIndex < MemberCount; ++MemberIndex)
			{
			uint MemberSeqIndex = MemberSeqIndexes[MemberIndex];
			asserta(MemberSeqIndex < SeqCount);
			uint CentroidIndex2 = m_SeqIndexToCentroidIndex[MemberSeqIndex];
			asserta(CentroidIndex2 == CentroidIndex);
			}
		}
	}

void cmd_eacluster()
	{
	//LoadGlobalInputMS(opt(eacluster));

	MultiSequence InputSeqs;
	LoadInput(InputSeqs);

	const float MinEA = (float) optd(minea, 0.9);
	string OutputFileNamePattern = optd(output, "cluster%.afa");

	bool Nucleo = false;
	if (opt(nt))
		Nucleo = true;
	else if (opt(amino))
		Nucleo = false;
	else
		Nucleo = InputSeqs.GuessIsNucleo();

	SetAlpha(Nucleo ? ALPHA_Nucleo : ALPHA_Amino);
	InitProbcons();

	EACluster EC;
	EC.Run(InputSeqs, MinEA);
	EC.WriteMFAs(OutputFileNamePattern);
	}
