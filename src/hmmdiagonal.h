#pragma once
#include <memory>
#include <new>

// Independent cells on i+j=d use only diagonals d-1 and d-2 (or d+1 and d+2).
// Keep lanes in increasing column order. The two scalar axes use the original
// recurrences; padding belongs only to scratch, never to the returned posterior.
namespace diagonal_hmm {
using V = __m256;
const unsigned ShortLimit = 256;
const unsigned MediumLimit = 1024;
enum { M, IX, IY, JX, JY, States };
template<unsigned Capacity>
struct Diagonal { float s[States][Capacity + 16]; };
inline V load(const float *p) { return _mm256_loadu_ps(p); }
inline void put(float *p, V x) { _mm256_storeu_ps(p,x); }
inline V splat(float x) { return _mm256_set1_ps(x); }
inline V rounded(V x) { asm("" : "+x"(x)); return x; }

inline V logadd(V x, V y)
	{
	V hi = _mm256_max_ps(y,x), lo = _mm256_min_ps(x,y);
	V delta = rounded(hi-lo);
	// LOG_ZERO is -2e20f. A finite live partner is separated from it by
	// more than the cutoff. With two sentinels, adding the d=0 correction
	// rounds back to LOG_ZERO in binary32.
	V skip = _mm256_cmp_ps(delta,splat(LOG_UNDERFLOW_THRESHOLD),_CMP_GE_OQ);
	if (_mm256_movemask_ps(skip) == 255)
		return hi;
	V d = _mm256_min_ps(delta,splat(LOG_UNDERFLOW_THRESHOLD));
	auto gt = [d](float edge) { return _mm256_castps_si256(_mm256_cmp_ps(d,splat(edge),_CMP_GT_OQ)); };
	__m256i idx = _mm256_sub_epi32(_mm256_setzero_si256(),_mm256_add_epi32(gt(1),_mm256_add_epi32(gt(2.5f),gt(4.5f))));
	auto coeff = [idx](float a,float b,float c,float e) {
		return _mm256_permutevar_ps(_mm256_setr_ps(a,b,c,e,a,b,c,e),idx); };
	V p = coeff(-0.009350833524763f,-0.014532321752540f,-0.004605031767994f,-0.000458661602210f);
	p = _mm256_fmadd_ps(p,d,coeff(0.130659527668286f,0.139942324101744f,0.063427417320019f,0.009695946122598f));
	p = _mm256_fmadd_ps(p,d,coeff(0.498799810682272f,0.495635523139337f,0.695956496475118f,0.930734667215156f));
	p = _mm256_fmadd_ps(p,d,coeff(0.693203116424741f,0.692140569840976f,0.514272634594009f,0.168037164329057f));
	return _mm256_blendv_ps(p+lo,hi,skip);
	}

template<unsigned Capacity> struct Emissions
	{
	static constexpr unsigned Pitch = Capacity + 16;
	int x[Pitch] = {}, y[Pitch] = {};
	float ex[Pitch] = {}, ey[Pitch] = {};
	Emissions(const byte *X,int nx,const byte *Y,int ny)
		{
		for (int i=0;i<nx;++i)
			{ byte a=X[nx-1-i]; x[i]=int(a)*256; ex[i]=PairHMM::m_InsScore[a]; }
		for (int j=0;j<ny;++j)
			{ byte a=Y[j]; y[j]=a; ey[j]=PairHMM::m_InsScore[a]; }
		}
	V match(int i,int j) const
		{
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x+i));
		__m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y+j));
		return _mm256_i32gather_ps(&PairHMM::m_MatchScore[0][0],_mm256_add_epi32(a,b),4);
		}
	};

template<unsigned Capacity> struct Workspace
	{
	Emissions<Capacity> emissions;
	Diagonal<Capacity> ring[3] = {};

	Workspace(const byte *x, int nx, const byte *y, int ny)
	  : emissions(x, nx, y, ny) {}
	};

template<unsigned Capacity>
inline void zeroCell(Diagonal<Capacity> &a,int j)
	{ for (int k=0;k<States;++k) a.s[k][j]=LOG_ZERO; }

__attribute__((optimize("no-tree-vectorize","no-associative-math")))
static void decode(float *out, const float *matches, int d,int ny,int lo,int hi,float total)
	{
	for (int j=lo;j<=hi;++j)
		{
		float &v=out[(d-j-1)*ny+j-1];
		const float score=(v+matches[j])-total;
		v=score<MIN_SPARSE_SCORE ? 0 : (score>=LOG_ONE ? 1 : expf(score));
		}
	}

template<unsigned Capacity, int Radius = 0>
static bool calculate(const byte *X,int nx,const byte *Y,int ny,float *out,
  Workspace<Capacity> &work)
	{
#include "hmmscores.h"
	const auto &e = work.emissions;
	auto *old = &work.ring[0], *prev = &work.ring[1], *cur = &work.ring[2];
	// Save the five forward scores on each artificial edge. A supported
	// escape through any match or insertion state retries the full matrix.
	float boundary[Radius ? 2*Capacity+1 : 1][2][States];
	auto center = [nx,ny](int d) { return (d*ny+(nx+ny)/2)/(nx+ny); };
	if (Radius)
		std::fill_n(out,size_t(nx)*ny,0.0f);
	zeroCell(*old,0);
	zeroCell(*prev,0);
	zeroCell(*prev,1);
	prev->s[IX][0]=tSI+InsScore[X[0]]; prev->s[JX][0]=tSJ+InsScore[X[0]];
	prev->s[IY][1]=tSI+InsScore[Y[0]]; prev->s[JY][1]=tSJ+InsScore[Y[0]];
	for (int d=2;d<=nx+ny;++d)
		{
		const int mid = Radius ? center(d) : 0;
		const int lo=Radius ? max(max(1,d-nx),mid-Radius) : max(1,d-nx);
		const int hi=Radius ? min(min(ny,d-1),mid+Radius) : min(ny,d-1);
		for (int j=lo;j<=hi;j+=8)
			{
			const int x=nx-(d-j), y=j-1;
			V sum=logadd(load(old->s[IY]+j-1)+splat(tIM),load(old->s[JY]+j-1)+splat(tJM));
			sum=logadd(load(old->s[JX]+j-1)+splat(tJM),sum);
			sum=logadd(load(old->s[IX]+j-1)+splat(tIM),sum);
			sum=logadd(load(old->s[M]+j-1)+splat(tMM),sum);
			put(cur->s[M]+j,sum+e.match(x,y));
			V above=load(prev->s[M]+j), left=load(prev->s[M]+j-1);
			V ex=load(e.ex+x), ey=load(e.ey+y);
			put(cur->s[IX]+j,logadd(load(prev->s[IX]+j)+splat(tII),above+splat(tMI))+ex);
			put(cur->s[JX]+j,logadd(load(prev->s[JX]+j)+splat(tJJ),above+splat(tMJ))+ex);
			put(cur->s[IY]+j,logadd(load(prev->s[IY]+j-1)+splat(tII),left+splat(tMI))+ey);
			put(cur->s[JY]+j,logadd(load(prev->s[JY]+j-1)+splat(tJJ),left+splat(tMJ))+ey);
			}
		if (d==2) cur->s[M][1]=tSM+MatchScore[X[0]][Y[0]];
		for (int j=lo;j<=hi;++j) out[(d-j-1)*ny+j-1]=cur->s[M][j];
		if (d<=nx && (!Radius || mid<=Radius))
			{
			zeroCell(*cur,0);
			volatile float a=tII+InsScore[X[d-1]], b=tJJ+InsScore[X[d-1]];
			cur->s[IX][0]=a+prev->s[IX][0]; cur->s[JX][0]=b+prev->s[JX][0];
			}
		if (d<=ny && (!Radius || d<=mid+Radius))
			{
			zeroCell(*cur,d);
			volatile float a=tII+InsScore[Y[d-1]], b=tJJ+InsScore[Y[d-1]];
			cur->s[IY][d]=a+prev->s[IY][d-1]; cur->s[JY][d]=b+prev->s[JY][d-1];
			}
		if (Radius)
			{
			if (lo==mid-Radius)
				{
				zeroCell(*cur,lo-1);
				for (int k=0; k<States; ++k) boundary[d][0][k]=cur->s[k][lo];
				}
			if (hi==mid+Radius)
				{
				zeroCell(*cur,hi+1);
				for (int k=0; k<States; ++k) boundary[d][1][k]=cur->s[k][hi];
				}
			}
		std::swap(old,prev); std::swap(prev,cur);
		}
	float fend[States], bend[States];
	GetBwdEnd(bend);
	for (int k=0;k<States;++k) fend[k]=prev->s[k][ny];
	float total=CalcTotalProbFlat(fend,bend,0,0);
	for (int d=nx+ny;d>=2;--d)
		{
		const int mid = Radius ? center(d) : 0;
		const int lo=Radius ? max(max(1,d-nx),mid-Radius) : max(1,d-nx);
		const int hi=Radius ? min(min(ny,d-1),mid+Radius) : min(ny,d-1);
		const int innerLo=max(lo,d-nx+1), innerHi=min(hi,ny-1);
		for (int j=innerLo;j<=innerHi;j+=8)
			{
			const int x=nx-(d-j)-1,y=j;
			V next=rounded(load(old->s[M]+j+1)+e.match(x,y));
			V ex=load(e.ex+x), ey=load(e.ey+y);
			V ix=rounded(load(prev->s[IX]+j)+ex), jx=rounded(load(prev->s[JX]+j)+ex);
			V iy=rounded(load(prev->s[IY]+j+1)+ey), jy=rounded(load(prev->s[JY]+j+1)+ey);
			V sum=logadd(iy+splat(tMI),jy+splat(tMJ));
			sum=logadd(jx+splat(tMJ),sum);
			sum=logadd(ix+splat(tMI),sum);
			put(cur->s[M]+j,logadd(next+splat(tMM),sum));
			put(cur->s[IX]+j,logadd(ix+splat(tII),next+splat(tIM)));
			put(cur->s[IY]+j,logadd(iy+splat(tII),next+splat(tIM)));
			put(cur->s[JX]+j,logadd(jx+splat(tJJ),next+splat(tJM)));
			put(cur->s[JY]+j,logadd(jy+splat(tJJ),next+splat(tJM)));
			}
		if (d==nx+ny)
			for (int k=0;k<States;++k) cur->s[k][ny]=bend[k];
		else
			{
			int j=d-nx;
			if (j>=lo && j<=hi)
				{
				float iy=prev->s[IY][j+1]+InsScore[Y[j]], jy=prev->s[JY][j+1]+InsScore[Y[j]];
				zeroCell(*cur,j); cur->s[M][j]=LOG_ADD(tMI+iy,tMJ+jy);
				cur->s[IY][j]=tII+iy; cur->s[JY][j]=tJJ+jy;
				}
			int i=d-ny;
			if (i>=1 && hi==ny)
				{
				float ix=prev->s[IX][ny]+InsScore[X[i]], jx=prev->s[JX][ny]+InsScore[X[i]];
				zeroCell(*cur,ny); cur->s[M][ny]=LOG_ADD(tMI+ix,tMJ+jx);
				cur->s[IX][ny]=tII+ix; cur->s[JX][ny]=tJJ+jx;
				}
			}
		if (Radius)
			{
			auto supported = [&](int side,int col)
				{
				for (int k=0; k<States; ++k)
					if (boundary[d][side][k]+cur->s[k][col] >= total-18.0f)
						return true;
				return false;
				};
			if (lo==mid-Radius)
				{
				if (supported(0,lo)) return false;
				zeroCell(*cur,lo-1);
				}
			if (hi==mid+Radius)
				{
				if (supported(1,hi)) return false;
				zeroCell(*cur,hi+1);
				}
			}
		decode(out,cur->s[M],d,ny,lo,hi,total);
		std::swap(old,prev); std::swap(prev,cur);
		}
	return true;
	}

template<unsigned Capacity>
static void calculateGuarded(const byte *x,int nx,const byte *y,int ny,float *out,
  Workspace<Capacity> &work)
	{
	const int shorter=min(nx,ny), longer=max(nx,ny);
	if (shorter>=160 && 5*shorter>=3*longer &&
	  calculate<Capacity,48>(x,nx,y,ny,out,work))
		return;
	calculate<Capacity,0>(x,nx,y,ny,out,work);
	}

static void calculateShort(const byte *x, int nx, const byte *y, int ny, float *out,bool guarded)
	{
	Workspace<ShortLimit> work(x, nx, y, ny);
	if (guarded) calculateGuarded(x, nx, y, ny, out, work);
	else calculate<ShortLimit,0>(x,nx,y,ny,out,work);
	}

static bool tryMedium(const byte *x, unsigned nx, const byte *y, unsigned ny, float *out,bool guarded)
	{
	// Very narrow diagonals leave most SIMD lanes idle. Use the rolling
	// implementation for these shapes and for pairs above the workspace cap.
	if (nx < 32 || ny < 32 || nx > MediumLimit || ny > MediumLimit)
		return false;
	// Keep the larger workspace off OpenMP worker stacks. A failed allocation
	// leaves the caller's original rolling implementation available.
	std::unique_ptr<Workspace<MediumLimit>> work(
	  new (std::nothrow) Workspace<MediumLimit>(x, nx, y, ny));
	if (!work)
		return false;
	if (guarded) calculateGuarded(x, nx, y, ny, out, *work);
	else calculate<MediumLimit,0>(x,nx,y,ny,out,*work);
	return true;
	}
} // namespace diagonal_hmm
