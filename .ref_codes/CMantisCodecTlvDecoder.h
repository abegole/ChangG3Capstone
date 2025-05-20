#ifndef __MANTISCODECTLV_DECODER_H__
#define __MANTISCODECTLV_DECODER_H__

#pragma once

enum class EFrameDecodeState
{
	Init = 0,
	MagicNum,
	TotalPacketLen,
	WaitFullFrame,
	ParseFrame
};

enum class EDecodeErrorCode
{
	NoError = 0,
	CrcError,
	TotalPacketLenError,
	TlvPacketLenError,
	BufferOutOfIndex
};

struct TotalPacketLen_t {
	union {
		UINT32 data;
		BYTE payload[ sizeof( UINT32 ) ];
	};
};

struct TlvDecodeDescriptor_t
{
	// for manipulating payload buffer
	EFrameDecodeState eDecodeState = EFrameDecodeState::Init;;

	// error code
	EDecodeErrorCode eErrorCode = EDecodeErrorCode::NoError;

	// for manipulating payload buffer
	UINT32 unOfst;

	// Total Packet Length
	TotalPacketLen_t unTotalPacketLen;

	// storing header information
	FrameHeader_t xFrameHeader;
	
	// storing TLV header information
	TlHeader_t xTlvHeader;

	// storing TLV packet information
	TlvPacket_t xTlvPacket;

	// buffer in case an incomplete data frame is added
	BYTE *Buffer = nullptr;
	
	// buffer maximum length
	UINT32 unBufferLength;

	// buffer index
	UINT32 unBufferPacketLen;
};


class CMantisCodecTlvDecoder
{
public:
	typedef std::function<void( const EDecodeErrorCode &, const FrameHeader_t &, const TlvPacket_t & )> decodeCallbackFunc_t;

	/// <summary>
	/// constructor
	/// </summary>
	/// <param name=""></param>
	CMantisCodecTlvDecoder();

	/// <summary>
	/// destructor
	/// </summary>
	/// <param name=""></param>
	~CMantisCodecTlvDecoder();

	/// <summary>
	/// reset descriptor info; fill in unDeviceId and unFrameNum
	/// </summary>
	/// <param name=""></param>
	void initDescriptor( TlvDecodeDescriptor_t &xDescriptor /*Output*/, UINT32 bufferSize = 10240 );

	void releaseDescriptor( TlvDecodeDescriptor_t &xDescriptor );

	void parseBuffer( TlvDecodeDescriptor_t &xDescriptor /*Output*/,
					  const BYTE inputBuffer[], const UINT32 &unPacketLen,
					  decodeCallbackFunc_t callbackHandler = nullptr,
					  BOOL bEnableCrc=TRUE );
private:
	/// <summary>
	/// parse one single packet; update its internal state inside the descriptor
	/// </summary>
	void decodePacket( TlvDecodeDescriptor_t &xDescriptor, BYTE Packet, decodeCallbackFunc_t callbackHandler = nullptr, BOOL bEnableCrc = TRUE );

	/// <summary>
	/// Reset descriptor info; set eDecodeState to init
	/// </summary>
	void resetDescriptor( TlvDecodeDescriptor_t &xDescriptor /*Output*/ );

	void parseFrame( TlvDecodeDescriptor_t &xDescriptor, decodeCallbackFunc_t callbackHandler = nullptr, BOOL bEnableCrc = TRUE );
};

#endif // end of !__MANTISCODECTLV_DECODER_H__