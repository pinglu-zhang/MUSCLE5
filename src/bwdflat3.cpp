#include "muscle.h"
#include "hmmrolling.h"
#include "hmmsimd.h"
#include "mega.h"

/***
Bwd[s][i][j] =
	probability of starting in state s and aligning
	last (LX-i) letters of X to
	last (LY-j) letters of Y.
***/

void CalcBwdFlat(const byte *X, uint LX, const byte *Y, uint LY, float *Flat)
	{
	asserta(!Mega::m_Loaded);

#include "hmmscores.h"
	if (double(LX)*double(LY)*5 + 100 > double(INT_MAX))
		Die("HMM overflow, sequence lengths %u, %u (max ~21k)", LX, LY);

	const size_t RightOffset = size_t(LY)*HMMSTATE_COUNT;
	const size_t Stride = RightOffset + HMMSTATE_COUNT;
	float *Row = Flat + LX*Stride;
	float *Cell = Row + RightOffset;
	Cell[HMMSTATE_M] = tSM;
	Cell[HMMSTATE_IX] = tSI;
	Cell[HMMSTATE_IY] = tSI;
	Cell[HMMSTATE_JX] = tSJ;
	Cell[HMMSTATE_JY] = tSJ;

	// The last row permits only transitions consuming Y.
	for (uint j = LY; j > 1; )
		{
		--j;
		const char y = Y[j];
		const float Emit_y = InsScore[y];
		const float NextIY = Cell[HMMSTATE_IY] + Emit_y;
		const float NextJY = Cell[HMMSTATE_JY] + Emit_y;
		const float M_IY = tMI + NextIY;
		const float M_JY = tMJ + NextJY;
		Cell -= HMMSTATE_COUNT;
		Cell[HMMSTATE_M] = LOG_ADD(M_IY, M_JY);
		Cell[HMMSTATE_IX] = LOG_ZERO;
		Cell[HMMSTATE_JX] = LOG_ZERO;
		Cell[HMMSTATE_IY] = tII + NextIY;
		Cell[HMMSTATE_JY] = tJJ + NextJY;
		}
	if (LY != 0)
		std::fill_n(Row, HMMSTATE_COUNT, LOG_ZERO);
	if (LX == 0)
		return;

	for (uint i = LX - 1; i > 0; --i)
		{
		const float *Below = Row;
		Row -= Stride;
		const char x = X[i];
		const float Emit_x = InsScore[x];
		const float *MatchRow = MatchScore[x];

		// The last column permits only transitions consuming X.
		Cell = Row + RightOffset;
		const float *BelowCell = Below + RightOffset;
		const float RightIX = BelowCell[HMMSTATE_IX] + Emit_x;
		const float RightJX = BelowCell[HMMSTATE_JX] + Emit_x;
		const float RightM_IX = tMI + RightIX;
		const float RightM_JX = tMJ + RightJX;
		Cell[HMMSTATE_M] = LOG_ADD(RightM_IX, RightM_JX);
		Cell[HMMSTATE_IX] = tII + RightIX;
		Cell[HMMSTATE_JX] = tJJ + RightJX;
		Cell[HMMSTATE_IY] = LOG_ZERO;
		Cell[HMMSTATE_JY] = LOG_ZERO;

		float RightIY = LOG_ZERO;
		float RightJY = LOG_ZERO;
		for (uint j = LY; j > 1; )
			{
			--j;
			const char y = Y[j];
			const float Emit_y = InsScore[y];
			const float NextM = BelowCell[HMMSTATE_M] + MatchRow[y];
			Cell -= HMMSTATE_COUNT;
			BelowCell -= HMMSTATE_COUNT;
			const float NextIX = BelowCell[HMMSTATE_IX] + Emit_x;
			const float NextJX = BelowCell[HMMSTATE_JX] + Emit_x;
			const float NextIY = RightIY + Emit_y;
			const float NextJY = RightJY + Emit_y;

			const float M_M = tMM + NextM;
			const float M_IX = tMI + NextIX;
			const float M_JX = tMJ + NextJX;
			const float M_IY = tMI + NextIY;
			const float M_JY = tMJ + NextJY;
			Cell[HMMSTATE_M] = LOG_ADD(M_M, M_IX, M_JX, M_IY, M_JY);

			const float I_M = tIM + NextM;
			const float J_M = tJM + NextM;
			const float IX_IX = tII + NextIX;
			Cell[HMMSTATE_IX] = LOG_ADD(IX_IX, I_M);
			const float JX_JX = tJJ + NextJX;
			Cell[HMMSTATE_JX] = LOG_ADD(JX_JX, J_M);
			const float IY_IY = tII + NextIY;
			RightIY = LOG_ADD(IY_IY, I_M);
			Cell[HMMSTATE_IY] = RightIY;
			const float JY_JY = tJJ + NextJY;
			RightJY = LOG_ADD(JY_JY, J_M);
			Cell[HMMSTATE_JY] = RightJY;
			}

		if (LY != 0)
			{
			// At j == 0, only X insertions can start from this cell.
			const char y = Y[0];
			const float NextM = Below[HMMSTATE_COUNT + HMMSTATE_M] + MatchRow[y];
			// Round the emission sum before adding the transition, as in the interior.
			const volatile float NextIX = Below[HMMSTATE_IX] + Emit_x;
			const volatile float NextJX = Below[HMMSTATE_JX] + Emit_x;
			const float IX_IX = tII + NextIX;
			const float IX_M = tIM + NextM;
			const float JX_JX = tJJ + NextJX;
			const float JX_M = tJM + NextM;
			Row[HMMSTATE_M] = LOG_ZERO;
			Row[HMMSTATE_IX] = LOG_ADD(IX_IX, IX_M);
			Row[HMMSTATE_JX] = LOG_ADD(JX_JX, JX_M);
			Row[HMMSTATE_IY] = LOG_ZERO;
			Row[HMMSTATE_JY] = LOG_ZERO;
			}
		}

	// At i == 0, only Y insertions can start from an interior column.
	Cell = Flat + RightOffset;
	std::fill_n(Cell, HMMSTATE_COUNT, LOG_ZERO);
	if (LY == 0)
		return;
	const char x = X[0];
	const float *MatchRow = MatchScore[x];
	const float *BelowCell = Row + RightOffset;
	for (uint j = LY; j > 1; )
		{
		--j;
		const char y = Y[j];
		const float Emit_y = InsScore[y];
		const float NextM = BelowCell[HMMSTATE_M] + MatchRow[y];
		const volatile float NextIY = Cell[HMMSTATE_IY] + Emit_y;
		const volatile float NextJY = Cell[HMMSTATE_JY] + Emit_y;
		const float IY_IY = tII + NextIY;
		const float IY_M = tIM + NextM;
		const float JY_JY = tJJ + NextJY;
		const float JY_M = tJM + NextM;
		Cell -= HMMSTATE_COUNT;
		BelowCell -= HMMSTATE_COUNT;
		Cell[HMMSTATE_M] = LOG_ZERO;
		Cell[HMMSTATE_IX] = LOG_ZERO;
		Cell[HMMSTATE_JX] = LOG_ZERO;
		Cell[HMMSTATE_IY] = LOG_ADD(IY_IY, IY_M);
		Cell[HMMSTATE_JY] = LOG_ADD(JY_JY, JY_M);
		}
	std::fill_n(Flat, HMMSTATE_COUNT, LOG_ZERO);
	}

#if MUSCLE_HMM_SIMD
// Reuse the forward scratch as two rows of [M, IX, IY, JX]. Only M, IX and
// JX are read from the row below. IY and JY for the current row stay in RightInsert.
void CalcBwdPostStrided(const byte *X, uint LX, const byte *Y, uint LY,
  const float *Fwd, const float *End, float Total, float *Work, float *Post, size_t PostStride)
	{
	asserta(!Mega::m_Loaded && LX > 0 && LY > 0);
#include "hmmscores.h"
	const uint M = HMMSTATE_M;
	const uint IX = HMMSTATE_IX;
	const uint IY = HMMSTATE_IY;
	const uint JX = HMMSTATE_JX;
	const uint Step = BWD_POST_ROW_STATES;
	const size_t Width = size_t(LY) + 1;
	const size_t LastColumn = size_t(LY)*Step;
	float *Row = Work;
	float *Below = Work + Width*Step;
	const float *FwdRow = Fwd + size_t(LX)*Width;
	float *PostRow = Post + size_t(LX - 1)*PostStride;

	float *Cell = Row + LastColumn;
	_mm_storeu_ps(Cell, _mm_loadu_ps(End));
	float RightIY = End[HMMSTATE_IY];
	float RightJY = End[HMMSTATE_JY];
	for (uint j = LY; j > 1; )
		{
		--j;
		Cell -= Step;
		const float Emit_y = InsScore[char(Y[j])];
		const float NextIY = RightIY + Emit_y;
		const float NextJY = RightJY + Emit_y;
		const float M_IY = tMI + NextIY;
		const float M_JY = tMJ + NextJY;
		Cell[M] = LOG_ADD(M_IY, M_JY);
		Cell[IX] = LOG_ZERO;
		Cell[IY] = LOG_ZERO;
		Cell[JX] = LOG_ZERO;
		RightIY = tII + NextIY;
		RightJY = tJJ + NextJY;
		}
	CalcPostFromMatches<1, Step, false>(FwdRow + 1, Row + Step, 1, LY, Total, PostRow);

	const __m128 OpenInsert = _mm_setr_ps(tMI, tMI, tMJ, tMJ);
	const __m128 ExtendInsert = _mm_setr_ps(tII, tII, tJJ, tJJ);
	const __m128 CloseInsert = _mm_setr_ps(tIM, tIM, tJM, tJM);
	for (uint i = LX; i > 1; )
		{
		--i;
		std::swap(Row, Below);
		FwdRow -= Width;
		PostRow -= PostStride;
		const char x = X[i];
		const float Emit_x = InsScore[x];
		const float *MatchScores = MatchScore[x];

		Cell = Row + LastColumn;
		const float *BelowCell = Below + LastColumn;
		const float RightIX = BelowCell[IX] + Emit_x;
		const float RightJX = BelowCell[JX] + Emit_x;
		Cell[M] = LOG_ADD(tMI + RightIX, tMJ + RightJX);
		Cell[IX] = tII + RightIX;
		Cell[IY] = LOG_ZERO;
		Cell[JX] = tJJ + RightJX;

		__m128 RightInsert = _mm_set1_ps(LOG_ZERO);
		for (uint j = LY; j > 1; )
			{
			--j;
			const char y = Y[j];
			const float Emit_y = InsScore[y];
			const float EmittedMatch = HMMRound(BelowCell[M] + MatchScores[y]);
			Cell -= Step;
			BelowCell -= Step;
			const __m128 BelowStates = _mm_loadu_ps(BelowCell);
			const __m128 Vertical = _mm_permute_ps(BelowStates, _MM_SHUFFLE(JX, JX, IX, IX));
			const __m128 Emit = _mm_setr_ps(Emit_x, Emit_y, Emit_x, Emit_y);
			const __m128 Next = HMMRound(_mm_add_ps(_mm_blend_ps(Vertical, RightInsert, 0xA), Emit));
			const __m128 MatchToInsert = _mm_add_ps(Next, OpenInsert);
			const float Match = LOG_ADD(tMM + EmittedMatch, HMMInsertLane<0>(MatchToInsert),
			  HMMInsertLane<2>(MatchToInsert), HMMInsertLane<1>(MatchToInsert),
			  HMMInsertLane<3>(MatchToInsert));
			const __m128 Extend = _mm_add_ps(Next, ExtendInsert);
			const __m128 Close = _mm_add_ps(_mm_set1_ps(EmittedMatch), CloseInsert);
			RightInsert = HMMLogAdd4(Extend, Close);
			const __m128 Packed = _mm_permute_ps(RightInsert, _MM_SHUFFLE(2, 1, 0, 0));
			_mm_storeu_ps(Cell, _mm_move_ss(Packed, _mm_set_ss(Match)));
			}
		CalcPostFromMatches<1, Step, false>(FwdRow + 1, Row + Step, 1, LY, Total, PostRow);
		}
	}

void CalcBwdPost(const byte *X, uint LX, const byte *Y, uint LY,
  const float *Fwd, const float *End, float Total, float *Work, float *Post)
	{
	CalcBwdPostStrided(X, LX, Y, LY, Fwd, End, Total, Work, Post, LY);
	}
#endif
