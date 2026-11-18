#pragma once

#if defined(__GNUC__) && !defined(__clang__) && defined(__FAST_MATH__) && defined(__AVX__) && defined(__FMA__)
#include <immintrin.h>

#define MUSCLE_HMM_SIMD 1

// The lanes are IX, IY, JX, JY, as in a flat HMM cell.
static_assert(HMMSTATE_IX == 1 && HMMSTATE_IY == 2 &&
  HMMSTATE_JX == 3 && HMMSTATE_JY == 4 && HMMSTATE_COUNT == 5, "HMM insertion state layout");

template<int Lane>
static inline float HMMInsertLane(__m128 States)
	{
	return _mm_cvtss_f32(_mm_shuffle_ps(States, States, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
	}

// Retain the rounded sum in an SIMD register before adding the transition.
// The compiler barrier keeps -ffast-math from reassociating those two adds.
template<class Score>
static inline Score HMMRound(Score Value)
	{
	asm("" : "+x"(Value));
	return Value;
	}

// Evaluate four independent LOG_ADD calls. Column k holds the coefficients
// from interval k of LOGEXP1, in descending order of polynomial degree.
static inline __m128 HMMLogAdd4(__m128 X, __m128 Y)
	{
	const __m128 High = _mm_max_ps(Y, X);
	const __m128 Low = _mm_min_ps(X, Y);
	const __m128 Limit = _mm_set1_ps(LOG_UNDERFLOW_THRESHOLD);
	const __m128 Diff = _mm_sub_ps(High, Low);
	const __m128 KeepHigh = _mm_or_ps(_mm_cmpge_ps(Diff, Limit),
	  _mm_cmpeq_ps(Low, _mm_set1_ps(LOG_ZERO)));
	const __m128 D = _mm_min_ps(Diff, Limit);
	const __m128i Over1 = _mm_castps_si128(_mm_cmpgt_ps(D, _mm_set1_ps(1.0f)));
	const __m128i Over2 = _mm_castps_si128(_mm_cmpgt_ps(D, _mm_set1_ps(2.5f)));
	const __m128i Over3 = _mm_castps_si128(_mm_cmpgt_ps(D, _mm_set1_ps(4.5f)));
	const __m128i Segment = _mm_sub_epi32(_mm_sub_epi32(_mm_setzero_si128(), Over1),
	  _mm_add_epi32(Over2, Over3));
	const __m128 C3 = _mm_permutevar_ps(_mm_setr_ps(
	  -0.009350833524763f, -0.014532321752540f, -0.004605031767994f, -0.000458661602210f), Segment);
	const __m128 C2 = _mm_permutevar_ps(_mm_setr_ps(
	  0.130659527668286f, 0.139942324101744f, 0.063427417320019f, 0.009695946122598f), Segment);
	const __m128 C1 = _mm_permutevar_ps(_mm_setr_ps(
	  0.498799810682272f, 0.495635523139337f, 0.695956496475118f, 0.930734667215156f), Segment);
	const __m128 C0 = _mm_permutevar_ps(_mm_setr_ps(
	  0.693203116424741f, 0.692140569840976f, 0.514272634594009f, 0.168037164329057f), Segment);
	const __m128 P2 = _mm_fmadd_ps(C3, D, C2);
	const __m128 P1 = _mm_fmadd_ps(P2, D, C1);
	const __m128 Value = _mm_add_ps(_mm_fmadd_ps(P1, D, C0), Low);
	return _mm_blendv_ps(Value, High, KeepHigh);
	}

#else
#define MUSCLE_HMM_SIMD 0
#endif
