#include <iostream>
#include "stdafx.h"
#include "MantisCodecTlv.h"
#include "CMantisCodecTlvDecoder.h"

CMantisCodecTlvDecoder::CMantisCodecTlvDecoder()
{}

CMantisCodecTlvDecoder::~CMantisCodecTlvDecoder()
{}

void CMantisCodecTlvDecoder::initDescriptor( TlvDecodeDescriptor_t &xDescriptor, UINT32 bufferSize )
{
	SAFENEW( xDescriptor.Buffer, BYTE[ bufferSize ] );
	xDescriptor.unBufferLength = bufferSize;

	resetDescriptor( xDescriptor );
}

void CMantisCodecTlvDecoder::releaseDescriptor( TlvDecodeDescriptor_t &xDescriptor )
{
	SAFEDELETE_ARRAY( xDescriptor.Buffer );
}

void CMantisCodecTlvDecoder::parseBuffer( TlvDecodeDescriptor_t &xDescriptor, const BYTE inputBuffer[],
										  const UINT32 &unPacketLen, decodeCallbackFunc_t callbackHandler, BOOL bEnableCrc )
{
	for( UINT32 i = 0; i < unPacketLen; ++i ) {
		decodePacket( xDescriptor, inputBuffer[ i ], callbackHandler, bEnableCrc );
	}
}

void CMantisCodecTlvDecoder::decodePacket( TlvDecodeDescriptor_t &xDescriptor, BYTE Packet, decodeCallbackFunc_t callbackHandler, BOOL bEnableCrc )
{

	switch( xDescriptor.eDecodeState ) {
	case EFrameDecodeState::Init: // wait for the MagicNum [0]
		if( Packet == FRAME_HEADER_MAGIC_NUM[ 0 ] ) {
			xDescriptor.Buffer[ xDescriptor.unBufferPacketLen++ ] = Packet;
			xDescriptor.unOfst = 1;
			xDescriptor.eDecodeState = EFrameDecodeState::MagicNum;
		}
		break;

	case EFrameDecodeState::MagicNum: // try matching magicNum [1:7]
		if( Packet == FRAME_HEADER_MAGIC_NUM[ xDescriptor.unOfst++ ] ) {
			xDescriptor.Buffer[ xDescriptor.unBufferPacketLen++ ] = Packet;
			if( xDescriptor.unOfst >= 8 ) {
				xDescriptor.unOfst = 0;
				xDescriptor.eDecodeState = EFrameDecodeState::TotalPacketLen;
			}
		}
		else {
			resetDescriptor( xDescriptor ); // also changes the state to init
		}
		break;
	case EFrameDecodeState::TotalPacketLen:
		xDescriptor.unTotalPacketLen.payload[ xDescriptor.unOfst++ ] = Packet;
		xDescriptor.Buffer[ xDescriptor.unBufferPacketLen++ ] = Packet;

		if( xDescriptor.unOfst >= sizeof( TotalPacketLen_t ) ) {

			if( xDescriptor.unTotalPacketLen.data > xDescriptor.unBufferLength
				|| xDescriptor.unTotalPacketLen.data < xDescriptor.unBufferPacketLen ) {
				// send result with error 
				if( callbackHandler != nullptr ) {
					xDescriptor.eErrorCode = EDecodeErrorCode::TotalPacketLenError;
					callbackHandler( xDescriptor.eErrorCode, xDescriptor.xFrameHeader, xDescriptor.xTlvPacket );
				}
				resetDescriptor( xDescriptor ); // also changes the state to init
			}
			else if( xDescriptor.unTotalPacketLen.data == xDescriptor.unBufferPacketLen ) {
				// this is a TLV packet without payload
				xDescriptor.unOfst = 0;
				parseFrame( xDescriptor, callbackHandler, bEnableCrc );
				resetDescriptor( xDescriptor ); // also changes the state to init
			}
			else {
				xDescriptor.eDecodeState = EFrameDecodeState::WaitFullFrame;
			}
		}
		break;
	case EFrameDecodeState::WaitFullFrame:
		xDescriptor.Buffer[ xDescriptor.unBufferPacketLen++ ] = Packet;

		if( xDescriptor.unBufferPacketLen >= xDescriptor.unTotalPacketLen.data ) {
			xDescriptor.unOfst = 0;
			parseFrame( xDescriptor, callbackHandler, bEnableCrc );
			resetDescriptor( xDescriptor ); // also changes the state to init
		}
		break;
	default:
		assert( false );
		resetDescriptor( xDescriptor );
		break;
	}
}

void CMantisCodecTlvDecoder::resetDescriptor( TlvDecodeDescriptor_t &xDescriptor )
{
	xDescriptor.eDecodeState = EFrameDecodeState::Init;

	// for manipulating payload buffer
	xDescriptor.eErrorCode = EDecodeErrorCode::NoError;

	// for manipulating payload buffer
	xDescriptor.unOfst = 0;

	// reset Frame header information to all zero
	memset( &xDescriptor.xFrameHeader, 0, sizeof( FrameHeader_t ) );

	// clear buffer (reset buffer Index)
	xDescriptor.unBufferPacketLen = 0;
}

void CMantisCodecTlvDecoder::parseFrame( TlvDecodeDescriptor_t &xDescriptor, decodeCallbackFunc_t callbackHandler, BOOL bEnableCrc )
{
	assert( xDescriptor.unBufferPacketLen == xDescriptor.unTotalPacketLen.data );

	xDescriptor.xFrameHeader = *( FrameHeader_t * )( xDescriptor.Buffer + xDescriptor.unOfst ); // parse all info for a single frame header
	xDescriptor.unOfst += static_cast< UINT32 >( sizeof( FrameHeader_t ) );


	UINT32 validCrc32 = CRC32( xDescriptor.Buffer + CRC32_START_OFFSET,
							   xDescriptor.unTotalPacketLen.data - CRC32_START_OFFSET );

	//Check CRC32 error
	if( bEnableCrc && xDescriptor.xFrameHeader.unCrc32 != validCrc32 ) {

		xDescriptor.eErrorCode = EDecodeErrorCode::CrcError;
		if( callbackHandler != nullptr ) {
			callbackHandler( xDescriptor.eErrorCode, xDescriptor.xFrameHeader, xDescriptor.xTlvPacket );
		}
		return;
	}

	// parse TLVs
	for( UINT32 i = 0; i < xDescriptor.xFrameHeader.unNumTlvs; ++i ) {
		// TL header (Type, Length)
		xDescriptor.xTlvHeader = *( TlHeader_t * )( xDescriptor.Buffer + xDescriptor.unOfst );
		xDescriptor.unOfst += static_cast< UINT32 >( sizeof( TlHeader_t ) );

		// add data pointer
		xDescriptor.xTlvPacket.unType = xDescriptor.xTlvHeader.unTlvType;
		xDescriptor.xTlvPacket.unLength = xDescriptor.xTlvHeader.unTlvLen;
		xDescriptor.xTlvPacket.pValue = xDescriptor.Buffer + xDescriptor.unOfst;

		// send result
		if( callbackHandler != nullptr ) {
			if( xDescriptor.xTlvHeader.unTlvLen > xDescriptor.unTotalPacketLen.data - xDescriptor.unOfst ) {
				xDescriptor.eErrorCode = EDecodeErrorCode::TlvPacketLenError;
			}
			else {
				xDescriptor.eErrorCode = EDecodeErrorCode::NoError;
			}
			callbackHandler( xDescriptor.eErrorCode, xDescriptor.xFrameHeader, xDescriptor.xTlvPacket );
		}
		xDescriptor.unOfst += xDescriptor.xTlvHeader.unTlvLen;
	}
}
