#pragma once

#include "muscle.h"

const uint FWD_INSERT_FLOATS_PER_COLUMN = 2*(HMMSTATE_COUNT - 1);
const uint BWD_POST_ROW_STATES = 4;
const uint BWD_POST_FLOATS_PER_COLUMN = 2*BWD_POST_ROW_STATES;

void CalcFwdRolling(const byte *X, uint LX, const byte *Y, uint LY,
  float *Match, float *Insert, float *End);

inline void GetBwdEnd(float *End)
	{
	const float *Start = PairHMM::m_StartScore;
	End[HMMSTATE_M] = Start[HMMSTATE_M];
	End[HMMSTATE_IX] = End[HMMSTATE_IY] = Start[HMMSTATE_IX];
	End[HMMSTATE_JX] = End[HMMSTATE_JY] = Start[HMMSTATE_JX];
	}

// Work holds two rows of four floats per column, including column zero.
void CalcBwdPost(const byte *X, uint LX, const byte *Y, uint LY,
  const float *Fwd, const float *End, float Total, float *Work, float *Post);

// The match state is the first float of each cell in either layout.
template<uint FwdStride, uint BwdStride, bool HasLeadingAxes = true>
#if defined(__GNUC__) && !defined(__clang__)
// Keep the flat build's (Fwd + Bwd) - Total and scalar expf under -ffast-math.
__attribute__((optimize("no-tree-vectorize", "no-associative-math")))
#endif
void CalcPostFromMatches(const float *Fwd, const float *Bwd,
  uint LX, uint LY, float Total, float *Post)
	{
	size_t Cell = HasLeadingAxes ? size_t(LY) + 2 : 0;
	for (uint i = 0; i < LX; ++i)
		{
		for (uint j = 0; j < LY; ++j, ++Cell)
			{
			float Score = Fwd[Cell*FwdStride] + Bwd[Cell*BwdStride] - Total;
			if (Score < MIN_SPARSE_SCORE)
				*Post++ = 0;
			else
				*Post++ = (Score >= LOG_ONE ? 1.0f : expf(Score));
			}
		Cell += HasLeadingAxes ? 1 : 0;
		}
	}
