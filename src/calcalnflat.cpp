#include "muscle.h"
#include "best3.h"
#include "alnrowavx2.h"

void TraceBackFlat(const char *TB, uint LX, uint LY, string &Path);

float CalcAlnFlat(const float *Post, uint LX, uint LY,
  float *DPRows, char *TB, string &Path)
	{
	Path.clear();
#ifdef __AVX2__
	if (LY >= 8)
		{
		const float Score = alnrow::Score<true>(Post, LX, LY, DPRows, TB);
		TraceBackFlat(TB, LX, LY, Path);
		return Score;
		}
#endif

	float *OldRow = DPRows;
	float *NewRow = DPRows + (LY+1);

	char *TBPtr = TB;
	for (uint j = 0; j <= LY; ++j)
		{
		OldRow[j] = 0;
		*TBPtr++ = 'Y';
		}

	const float *PostPtr = Post;
	for (uint i = 1; i <= LX; ++i)
		{
		uint64 k = TBPtr - TB;
		*TBPtr++ = 'X';
		NewRow[0] = 0;

		for (uint j = 1; j <= LY; ++j)
			{
			float B = OldRow[j-1] + *PostPtr++;
			float X = OldRow[j];
			float Y = NewRow[j-1];

			float Best;
			char TBChar;
			Best3(B, X, Y, 'B', 'X', 'Y', &Best, &TBChar);
			NewRow[j] = Best;
			*TBPtr++ = TBChar;
			}

		swap(OldRow, NewRow);
		}
	float Score = OldRow[LY];
	TraceBackFlat(TB, LX, LY, Path);
	return Score;
	}
