#include "muscle.h"
#include "mpcflat.h"

void RelaxFlat_XZ_ZY(const MySparseMx &XZ, const MySparseMx &ZY,
  float WeightZ, float *Post)
	{
	const uint LX = XZ.GetLX();
	const uint LZ = XZ.GetLY();
	const uint LY = ZY.GetLY();
	asserta(ZY.GetLX() == LZ);

	for (uint PosX = 0; PosX < LX; ++PosX)
		{
		uint Offset_XZ = XZ.m_Offsets[PosX];
		uint Size_XZ = (XZ.m_Offsets[PosX+1] - XZ.m_Offsets[PosX]);
		for (uint k = 0; k < Size_XZ; ++k)
			{
			float P_XZ = XZ.GetProb_Offset(Offset_XZ + k);
			uint PosZ = XZ.GetCol_Offset(Offset_XZ + k);

			uint Offset_ZY = ZY.m_Offsets[PosZ];
			uint Size_ZY = (ZY.m_Offsets[PosZ+1] - ZY.m_Offsets[PosZ]);
			for (uint m = 0; m < Size_ZY; ++m)
				{
				float P_ZY = ZY.GetProb_Offset(Offset_ZY + m);
				uint PosY = ZY.GetCol_Offset(Offset_ZY + m);
				Post[PosX*LY + PosY] += WeightZ*P_XZ*P_ZY;
				}
			}
		}
	}

void RelaxFlat_ZX_ZY(const MySparseMx &ZX, const MySparseMx &ZY,
  float WeightZ, float *Post)
	{
	const uint LZ = ZX.GetLX();
	const uint LX = ZX.GetLY();
	const uint LY = ZY.GetLY();
	asserta(ZY.GetLX() == LZ);

	for (uint PosZ = 0; PosZ < LZ; ++PosZ)
		{
		uint Offset_ZX = ZX.m_Offsets[PosZ];
		uint Size_ZX = (ZX.m_Offsets[PosZ+1] - ZX.m_Offsets[PosZ]);
		for (uint k = 0; k < Size_ZX; ++k)
			{
			float P_ZX = ZX.GetProb_Offset(Offset_ZX + k);
			uint PosX = ZX.GetCol_Offset(Offset_ZX + k);

			uint Offset_ZY = ZY.m_Offsets[PosZ];
			uint Size_ZY = (ZY.m_Offsets[PosZ+1] - ZY.m_Offsets[PosZ]);
			for (uint m = 0; m < Size_ZY; ++m)
				{
				float P_ZY = ZY.GetProb_Offset(Offset_ZY + m);
				uint PosY = ZY.GetCol_Offset(Offset_ZY + m);
				Post[PosX*LY + PosY] += WeightZ*P_ZX*P_ZY;
				}
			}
		}
	}

void RelaxFlat_XZ_YZ(const MySparseMx &XZ, const MySparseMx &YZ,
  float WeightZ, float *Post)
	{
	const uint LX = XZ.GetLX();
	const uint LZ = XZ.GetLY();
	const uint LY = YZ.GetLX();
	asserta(YZ.GetLY() == LZ);

	vector<uint> PosZToLoPosY;
	vector<uint> PosZToHiPosY;
	YZ.GetColToRowLoHi(PosZToLoPosY, PosZToHiPosY);

	for (uint PosX = 0; PosX < LX; ++PosX)
		{
		uint Offset_XZ = XZ.m_Offsets[PosX];
		uint Size_XZ = (XZ.m_Offsets[PosX+1] - XZ.m_Offsets[PosX]);
		for (uint k = 0; k < Size_XZ; ++k)
			{
			float P_XZ = XZ.GetProb_Offset(Offset_XZ + k);
			uint PosZ = XZ.GetCol_Offset(Offset_XZ + k);

			uint LoPosY = PosZToLoPosY[PosZ];
			uint HiPosY = PosZToHiPosY[PosZ];
			if (LoPosY == UINT_MAX)
				continue;
			for (uint PosY = LoPosY; PosY <= HiPosY; ++PosY)
				{
				float P_YZ = YZ.GetProb(PosY, PosZ);
				Post[PosX*LY + PosY] += WeightZ*P_XZ*P_YZ;
				}
			}
		}
	}

// UpdateFromPost will read only the coordinates already present in XY.
// Intersect two sorted Z rows, accumulating each output in increasing Z order.
void RelaxOnSupport(const MySparseMx &XZ, const MySparseMx &YZ,
  const MySparseMx &XY, float *Post)
	{
	const uint LX = XY.GetLX();
	const uint LY = XY.GetLY();
	assert(XZ.GetLX() == LX && YZ.GetLX() == LY && XZ.GetLY() == YZ.GetLY());
	const uint *XOffsets = XZ.m_Offsets;
	const uint *YOffsets = YZ.m_Offsets;
	const uint *OutputOffsets = XY.m_Offsets;
	for (uint PosX = 0; PosX < LX; ++PosX)
		{
		const uint XBegin = XOffsets[PosX];
		const uint XEnd = XOffsets[PosX+1];
		if (XBegin == XEnd)
			continue;
		float *Row = Post + size_t(PosX)*LY;
		for (uint k = OutputOffsets[PosX]; k < OutputOffsets[PosX+1]; ++k)
			{
			const uint PosY = XY.GetCol_Offset(k);
			uint i = XBegin;
			uint j = YOffsets[PosY];
			const uint YEnd = YOffsets[PosY+1];
			float P = Row[PosY];
			while (i < XEnd && j < YEnd)
				{
				const uint a = XZ.GetCol_Offset(i);
				const uint b = YZ.GetCol_Offset(j);
				if (a < b)
					++i;
				else if (a > b)
					++j;
				else
					{
					P += XZ.GetProb_Offset(i)*YZ.GetProb_Offset(j);
					++i;
					++j;
					}
				}
			Row[PosY] = P;
			}
		}
	}
