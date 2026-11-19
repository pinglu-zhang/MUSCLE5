#pragma once

// A transposed sparse matrix needs one offset per target residue and a
// final sentinel. Its row count is fixed while the buffer is in use.
inline void ReserveSparseRows(MySparseMx &Mx, uint Rows)
	{
	if (Rows <= Mx.m_MaxLX)
		return;
	uint *Offsets = myalloc(uint, size_t(Rows)+1);
	myfree(Mx.m_Offsets);
	Mx.m_Offsets = Offsets;
	Mx.m_MaxLX = Rows;
	}

// The caller reserves the row and value arrays before parallel transposition.
inline void TransposeSparsePost(const MySparseMx &From, MySparseMx &To)
	{
	const uint NX = From.GetLX(), NY = From.GetLY();
	const uint Count = From.m_Offsets[NX];
	uint *Offsets = To.m_Offsets;
	std::fill(Offsets, Offsets + NY, 0u);
	for (uint k = 0; k < Count; ++k)
		++Offsets[From.GetCol_Offset(k)];
	uint End = 0;
	for (uint j = 0; j < NY; ++j)
		{
		End += Offsets[j];
		Offsets[j] = End;
		}
	Offsets[NY] = Count;
	// Insert backwards from each row's end. After the last insertion each
	// cursor is the row start and the entries are in increasing X order.
	for (uint i = NX; i-- > 0;)
		for (uint k = From.m_Offsets[i+1]; k > From.m_Offsets[i]; --k)
			{
			const uint At = --Offsets[From.GetCol_Offset(k-1)];
			To.SetCol_Offset(At, i);
			To.SetProb_Offset(At, From.GetProb_Offset(k-1));
			}
	To.m_LX = NY;
	To.m_LY = NX;
	To.m_VecSize = Count;
	To.m_X = From.m_Y;
	To.m_Y = From.m_X;
	}
