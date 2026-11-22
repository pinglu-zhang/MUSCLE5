#include "muscle.h"

MultiSequence* MultiSequence::Project(const vector<uint> &SeqIndexes)
	{
	set<int> IndexSet;
	for (uint i = 0; i < SIZE(SeqIndexes); ++i)
		{
		int Index = (int) SeqIndexes[i];
		IndexSet.insert(Index);
		}
	return Project(IndexSet);
	}

// Create a projected MSA in input-index order, dropping exactly the columns
// containing '-' in every selected row. Scan individual rows contiguously.
MultiSequence* MultiSequence::Project(const set<int>& SeqIndexes)
	{
	MultiSequence *out=new MultiSequence;
	if (SeqIndexes.empty()) return out;
	const uint cols=GetSequence(*SeqIndexes.begin())->GetLength();
	vector<byte> keep(cols,0);
	for (int i : SeqIndexes)
		{
		const Sequence *s=GetSequence(i);
		asserta(s->GetLength()==cols);
		const char *letters=s->GetCharPtr();
		for (uint c=0; c<cols; ++c)
			keep[c] |= byte(letters[c]!='-');
		}
	// Coalesce adjacent kept columns once for all output rows.
	vector<pair<uint,uint> > spans;
	uint kept=0;
	for (uint c=0; c<cols;)
		{
		if (!keep[c]) { ++c; continue; }
		const uint first=c++;
		while (c<cols && keep[c]) ++c;
		spans.emplace_back(first,c-first);
		kept+=c-first;
		}
	for (int i : SeqIndexes)
		{
		const Sequence *s=GetSequence(i);
		Sequence *row=NewSequence();
		row->m_Label=s->m_Label;
		row->m_CharVec.resize(kept);
		uint at=0;
		for (const auto &span : spans)
			{
			memcpy(row->m_CharVec.data()+at,s->GetCharPtr()+span.first,span.second);
			at+=span.second;
			}
		out->AddSequence(row,true);
		}
	return out;
	}
