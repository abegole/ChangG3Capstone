#include "stdafx.h"
#include "MantisCodecTlv.h"
#include "CMantisCodecTlvEncoder.h"
#include <iostream>

CMantisCodecTlvEncoder::CMantisCodecTlvEncoder( void )
{
}

CMantisCodecTlvEncoder::~CMantisCodecTlvEncoder( void )
{
}

void CMantisCodecTlvEncoder::initDescriptor( TlvEncodeDescriptor_t &xDescriptor, UINT32 bufferSize )
{
	SAFENEW( xDescriptor.Buffer, BYTE[ bufferSize ] );
	resetDescriptor( xDescriptor );
}

void CMantisCodecTlvEncoder::releaseDescriptor( TlvEncodeDescriptor_t &xDescriptor )
{
	SAFEDELETE_ARRAY( xDescriptor.Buffer );
}

void CMantisCodecTlvEncoder::resetDescriptor( TlvEncodeDescriptor_t &descriptor /*Output*/,
											  const UINT32 &unDeviceId, const UINT32 &unFrameNum,
											  const BOOL &bEnableCrc )
{
	// reset xFrameHeader; fill in magicNum, device ID, and FrameNum
	memcpy( descriptor.xFrameHeader.unMagicNum, FRAME_HEADER_MAGIC_NUM, 8 );
	descriptor.xFrameHeader.unTotalPacketLen = sizeof( descriptor.xFrameHeader ); // offset for the header
	descriptor.xFrameHeader.unCrc32 = 0;
	descriptor.xFrameHeader.unDeviceId = unDeviceId;
	descriptor.xFrameHeader.unNumTlvs = 0;
	descriptor.xFrameHeader.unFrameNum = unFrameNum;

	// setup crc option
	descriptor.bEnableCrc = bEnableCrc;

	// reset xTlvHeader
	descriptor.xTlvHeader.unTlvType = TLV_TYPE__INIT;
	descriptor.xTlvHeader.unTlvLen = 0;
}

void CMantisCodecTlvEncoder::fillTlvPacket( TlvEncodeDescriptor_t &descriptor,
											const UINT32 &unTlvType, const UINT32 &unTlvLen, const void *pDataAddr )
{
	// prepare in TlvHeader
	descriptor.xTlvHeader.unTlvType = unTlvType;
	descriptor.xTlvHeader.unTlvLen = unTlvLen;

	// fill in TlvHeader
	memcpy( descriptor.Buffer + descriptor.xFrameHeader.unTotalPacketLen,
			( const BYTE * )&descriptor.xTlvHeader, sizeof( TlHeader_t ) );
	descriptor.xFrameHeader.unTotalPacketLen += static_cast< UINT32 >( sizeof( TlHeader_t ) );

	// fill in data
	memcpy( descriptor.Buffer + descriptor.xFrameHeader.unTotalPacketLen,
			( const BYTE * )pDataAddr, unTlvLen );
	descriptor.xFrameHeader.unTotalPacketLen += unTlvLen;

	// increase unNumTlvs by 1
	descriptor.xFrameHeader.unNumTlvs += 1;
}

UINT32 CMantisCodecTlvEncoder::wrapBuffer( TlvEncodeDescriptor_t &descriptor /*Output*/ )
{
	assert( descriptor.xFrameHeader.unTotalPacketLen >= 28 );

	// fill in padding bytes as 0, update the final totalPacketLen
	memset( descriptor.Buffer + descriptor.xFrameHeader.unTotalPacketLen,
			0, NUM_PADDING_BYTES( descriptor.xFrameHeader.unTotalPacketLen ) );
	descriptor.xFrameHeader.unTotalPacketLen += NUM_PADDING_BYTES( descriptor.xFrameHeader.unTotalPacketLen );

	// fill in header (fill in empty CRC)
	memcpy( descriptor.Buffer, ( const BYTE * )&descriptor.xFrameHeader, sizeof( descriptor.xFrameHeader ) );

	// calculate and fill in CRC
	if( descriptor.bEnableCrc ) {
		descriptor.xFrameHeader.unCrc32 = CRC32( descriptor.Buffer + CRC32_START_OFFSET,
												 descriptor.xFrameHeader.unTotalPacketLen - CRC32_START_OFFSET );
	}
	else {
		descriptor.xFrameHeader.unCrc32 = 0;
	}

	memcpy( descriptor.Buffer + offsetof( FrameHeader_t, unCrc32 ), &descriptor.xFrameHeader.unCrc32, sizeof( UINT32 ) );

	return descriptor.xFrameHeader.unTotalPacketLen;
}


