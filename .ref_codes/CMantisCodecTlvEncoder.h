#ifndef __MANTISCODECTLV_ENCODER_H__
#define __MANTISCODECTLV_ENCODER_H__

#pragma once

struct TlvEncodeDescriptor_t {
	FrameHeader_t xFrameHeader;
	TlHeader_t xTlvHeader;
	BOOL bEnableCrc;
	BYTE *Buffer;
};

class CMantisCodecTlvEncoder
{
public:
	/// <summary>
	/// constructor
	/// </summary>
	/// <param name=""></param>
	CMantisCodecTlvEncoder( void );

	/// <summary>
	/// destructor
	/// </summary>
	/// <param name=""></param>
	~CMantisCodecTlvEncoder( void );

	/// <summary>
	/// reset descriptor info; fill in unDeviceId and unFrameNum
	/// </summary>
	/// <param name=""></param>
	void initDescriptor( TlvEncodeDescriptor_t &xDescriptor /*Output*/, UINT32 bufferSize = 10240 );

	void releaseDescriptor( TlvEncodeDescriptor_t &xDescriptor );

	void resetDescriptor( TlvEncodeDescriptor_t &descriptor /*Output*/,
						  const UINT32 &unDeviceId = 0, const UINT32 &unFrameNum = 0,
						  const BOOL &bEnableCrc = TRUE );
	/// <summary>
	/// insert data into the buffer, by specifying Type, Length, Value
	/// </summary>
	/// <param name=""></param>
	void fillTlvPacket( TlvEncodeDescriptor_t &descriptor,
						const UINT32 &unTlvType, const UINT32 &unTlvLen, const void *pDataAddr );

	/// <summary>
	/// fill in padding bytes, total packet length, and CRC result
	/// </summary>
	/// <param name=""></param>
	UINT32 wrapBuffer( TlvEncodeDescriptor_t &descriptor /*Output*/ );
};

#endif // end of !__MANTISCODECTLV_ENCODER_H__