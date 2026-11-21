#pragma once

#ifdef __AVX2__
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include "best3.h"

namespace alnrow
{
// The finite posterior/profile scores are accumulated in exactly one diagonal
// addition per cell. All remaining horizontal dependencies are prefix maxima.
template<bool Trace>
inline float Score(const float *Post, unsigned NX, unsigned NY, float *Row, char *TB)
	{
	const size_t Width = size_t(NY) + 1;
	std::fill(Row, Row + Width, 0.0f);
	if (Trace)
		std::memset(TB, 'Y', Width);
	const __m256i Shift = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
	for (unsigned i = 0; i < NX; ++i, Post += NY)
		{
		if (Trace)
			{
			TB += Width;
			TB[0] = 'X';
			}
		__m256 PrevUp = _mm256_setzero_ps();
		__m256 Left = _mm256_setzero_ps();
		unsigned j = 0;
		for (; NY - j >= 8; j += 8)
			{
			const __m256 Up = _mm256_loadu_ps(Row + j + 1);
			const __m256 Previous = _mm256_blend_ps(_mm256_permutevar8x32_ps(Up, Shift), PrevUp, 1);
			PrevUp = _mm256_permutevar8x32_ps(Up, _mm256_set1_epi32(7));
			const __m256 Diag = _mm256_add_ps(Previous, _mm256_loadu_ps(Post + j));
			__m256 V = _mm256_max_ps(Diag, Up);
			V = _mm256_max_ps(V, _mm256_permute_ps(V, _MM_SHUFFLE(2,1,0,0)));
			V = _mm256_max_ps(V, _mm256_permute_ps(V, _MM_SHUFFLE(1,0,0,0)));
			const __m256 Half = _mm256_permutevar8x32_ps(V, _mm256_set1_epi32(3));
			V = _mm256_blend_ps(V, _mm256_max_ps(V, Half), 0xf0);
			V = _mm256_max_ps(V, Left);
			Left = _mm256_permutevar8x32_ps(V, _mm256_set1_epi32(7));
			_mm256_storeu_ps(Row + j + 1, V);
			if (Trace)
				{
				// Best3 resolves equal scores in diagonal, up, left order.
				__m256i Dir = _mm256_blendv_epi8(_mm256_set1_epi32('Y'), _mm256_set1_epi32('X'),
				  _mm256_castps_si256(_mm256_cmp_ps(Up, V, _CMP_EQ_OQ)));
				Dir = _mm256_blendv_epi8(Dir, _mm256_set1_epi32('B'),
				  _mm256_castps_si256(_mm256_cmp_ps(Diag, V, _CMP_EQ_OQ)));
				const __m128i Shorts = _mm_packs_epi32(_mm256_castsi256_si128(Dir), _mm256_extracti128_si256(Dir, 1));
				_mm_storel_epi64((__m128i *)(TB + j + 1), _mm_packus_epi16(Shorts, Shorts));
				}
			}
		float Old = _mm_cvtss_f32(_mm256_castps256_ps128(PrevUp));
		float Best = _mm_cvtss_f32(_mm256_castps256_ps128(Left));
		for (; j < NY; ++j)
			{
			const float Up = Row[j+1];
			const float Diag = Old + Post[j];
			if (Trace)
				Best3(Diag, Up, Best, 'B', 'X', 'Y', &Best, TB + j + 1);
			else
				Best3(Diag, Up, Best, &Best);
			Row[j+1] = Best;
			Old = Up;
			}
		}
	return Row[NY];
	}
}
#endif
