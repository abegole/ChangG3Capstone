import ctypes
import struct
from enum import Enum
import binascii # for crc32

FRAME_HEADER_MAGIC_NUM = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MATIS_CODEC_TLV_MSG_SEGMENT_LEN = 32

def NUM_PADDING_BYTES(unTotalPacketLen):
    return MATIS_CODEC_TLV_MSG_SEGMENT_LEN - (unTotalPacketLen & (MATIS_CODEC_TLV_MSG_SEGMENT_LEN - 1))

class EFrameDecodeState(Enum):
    Init = 0
    MagicNum = 1
    TotalPacketLen = 2
    WaitFullFrame = 3


class EDecodeErrorCode(Enum):
    NoError = 0
    CrcError = 1
    BufferOutOfIndex = 2


class TotalPacketLen_t(ctypes.Union):
    _fields_ = [("data", ctypes.c_uint32),
                ("payload", ctypes.ARRAY(ctypes.c_byte, 4))]


class TlHeader_t(ctypes.Structure):
    _fields_ = [("unTlvType", ctypes.c_uint32),
                ("unTlvLen", ctypes.c_uint32)]


class FrameHeader_t(ctypes.Structure):
    _fields_ = [("unMagicNum", ctypes.ARRAY(ctypes.c_byte, 8)),
                ("unTotalPacketLen", ctypes.c_uint32),
                ("unCrc32", ctypes.c_uint32),
                ("unDeviceId", ctypes.c_uint32),
                ("unNumTlvs", ctypes.c_uint32),
                ("unFrameNum", ctypes.c_uint32)]


class TlvPacket_t(ctypes.Structure):
    _fields_ = [("tlv_header", ctypes.c_uint32),
                ("tlv_data", ctypes.POINTER(ctypes.c_byte))]

class TlvEncodeDescriptor_t:
    def __init__(self, bufferLen=10240) -> None:
        self.xFrameHeader = FrameHeader_t()
        self.xTlvHeader = TlHeader_t()
        self.bEnableCrc = ctypes.c_bool()
        self.Buffer = bytearray(bufferLen)

class TlvDecodeDescriptor_t:
    def __init__(self, bufferLen=10240):
        self.eDecodeState = EFrameDecodeState.Init
        self.eErrorCode = EDecodeErrorCode.NoError
        self.unOfst = 0
        self.unTotalPacketLen = TotalPacketLen_t()
        self.xFrameHeader = FrameHeader_t()
        self.xTlvHeader = TlHeader_t()
        self.xTlvPacket = TlvPacket_t()
        self.Buffer = bytearray(bufferLen)
        self.unBufferPacketLen = 0

class EFrameDecodeState(Enum):
    Init = 0
    MagicNum = 1
    TotalPacketLen = 2
    WaitFullFrame = 3
    ParseFrame = 4


class EDecodeErrorCode(Enum):
    NoError = 0
    CrcError = 1
    BufferOutOfIndex = 2

class CMantisCodecTlvEncoder:
    def __init__(self) -> None:
        self.descriptor = TlvEncodeDescriptor_t()
        # init Magic Number for the frame header
        for i in range(len(FRAME_HEADER_MAGIC_NUM)):
            self.descriptor.xFrameHeader.unMagicNum[i] = FRAME_HEADER_MAGIC_NUM[i]

    def resetDescriptor(self, bEnableCrc=True):
        self.descriptor.xFrameHeader.unTotalPacketLen = ctypes.sizeof(self.descriptor.xFrameHeader)
        self.descriptor.xFrameHeader.unCrc32 = 0
        self.descriptor.xFrameHeader.unNumTlvs = 0
        self.descriptor.bEnableCrc = bEnableCrc

    def fillTlvPacket(self, unTlvType, unTlvLen, pDataAddr):
        self.descriptor.xTlvHeader.unTlvType = unTlvType
        self.descriptor.xTlvHeader.unTlvLen = unTlvLen

        # copy tlv header to the descriptor buffer
        index = self.descriptor.xFrameHeader.unTotalPacketLen
        length = ctypes.sizeof(self.descriptor.xTlvHeader)
        self.descriptor.Buffer[index:index+length] = ctypes.string_at(ctypes.addressof(self.descriptor.xTlvHeader), length)
        self.descriptor.xFrameHeader.unTotalPacketLen += length

        # fill in data
        index = self.descriptor.xFrameHeader.unTotalPacketLen
        length = unTlvLen
        self.descriptor.Buffer[index:index+length] = ctypes.string_at(pDataAddr, length)
        self.descriptor.xFrameHeader.unTotalPacketLen += length

        # update number of tlvs
        self.descriptor.xFrameHeader.unNumTlvs += 1

    def wrapupBuffer(self, deviceId, frameNum):
        # add zeos to the end of the buffer
        index = self.descriptor.xFrameHeader.unTotalPacketLen
        length = NUM_PADDING_BYTES(self.descriptor.xFrameHeader.unTotalPacketLen)
        self.descriptor.Buffer[index:index+length] = b'\x00' * length

        # update total packet length
        self.descriptor.xFrameHeader.unTotalPacketLen += length

        # update frame header
        self.descriptor.xFrameHeader.unDeviceId = deviceId
        self.descriptor.xFrameHeader.unFrameNum = frameNum

        # calculate crc32 value for the whole buffer except for the magic number and total packet length in the frame header
        if (self.descriptor.bEnableCrc == True):
            crc32 = binascii.crc32(self.descriptor.Buffer[16:self.descriptor.xFrameHeader.unTotalPacketLen])
            self.descriptor.xFrameHeader.unCrc32 = crc32
        else:
            self.descriptor.xFrameHeader.unCrc32 = 0
        
        # copy frame header to the descriptor buffer
        index = 0
        length = ctypes.sizeof(self.descriptor.xFrameHeader)
        self.descriptor.Buffer[index:index+length] = ctypes.string_at(ctypes.addressof(self.descriptor.xFrameHeader), length)

class CMantisCodecTlvDecoder:
    def __init__(self, callbackHandler, bufferLen=10240):
        self.callbackHandler = callbackHandler
        self.descriptor = TlvDecodeDescriptor_t(bufferLen=bufferLen)

    def __del__(self):
        pass

    def parseBuffer(self, inputBuffer):
        i = 0
        retval = None
        while i < len(inputBuffer):
            self.decodePacket(inputBuffer[i])
            i = i+1


    def resetDescriptor(self):
        self.descriptor.eDecodeState = EFrameDecodeState.Init
        self.descriptor.eErrorCode = EDecodeErrorCode.NoError
        self.descriptor.unOfst = 0
        self.descriptor.xFrameHeader = FrameHeader_t()
        self.descriptor.unBufferPacketLen = 0

    def decodePacket(self, Packet: bytes):
        if self.descriptor.eDecodeState == EFrameDecodeState.Init:
            if Packet == FRAME_HEADER_MAGIC_NUM[0]:
                self.descriptor.Buffer[self.descriptor.unBufferPacketLen] = Packet
                self.descriptor.unBufferPacketLen += 1
                self.descriptor.unOfst = 1
                self.descriptor.eDecodeState = EFrameDecodeState.MagicNum
        elif self.descriptor.eDecodeState == EFrameDecodeState.MagicNum:
            if Packet == FRAME_HEADER_MAGIC_NUM[self.descriptor.unOfst]:
                self.descriptor.Buffer[self.descriptor.unBufferPacketLen] = Packet
                self.descriptor.unBufferPacketLen += 1
                self.descriptor.unOfst += 1
                if self.descriptor.unOfst >= len(FRAME_HEADER_MAGIC_NUM):
                    self.descriptor.unOfst = 0
                    self.descriptor.eDecodeState = EFrameDecodeState.TotalPacketLen
            else:
                self.resetDescriptor()
        elif self.descriptor.eDecodeState == EFrameDecodeState.TotalPacketLen:
            self.descriptor.unTotalPacketLen.payload[self.descriptor.unOfst] = Packet
            self.descriptor.unBufferPacketLen += 1
            self.descriptor.unOfst += 1
            if self.descriptor.unOfst >= ctypes.sizeof(self.descriptor.unTotalPacketLen):
                self.descriptor.unOfst = 0
                if (self.descriptor.unBufferPacketLen >= self.descriptor.unTotalPacketLen.data):
                    self.descriptor.unOfst = 0
                    self.parseFrame()
                    self.resetDescriptor()
                else:
                    self.descriptor.eDecodeState = EFrameDecodeState.WaitFullFrame
        elif self.descriptor.eDecodeState == EFrameDecodeState.WaitFullFrame:
            self.descriptor.Buffer[self.descriptor.unBufferPacketLen] = Packet
            self.descriptor.unBufferPacketLen += 1
            if self.descriptor.unBufferPacketLen >= self.descriptor.unTotalPacketLen.data:
                self.parseFrame()
                self.resetDescriptor()
                # return self.outputDict

    def parseFrame(self):
        headerStruct = 'Q5I'
        tlvHeaderStruct = '2I'
        frameHeaderLen = struct.calcsize(headerStruct)
        tlvHeaderLength = struct.calcsize(tlvHeaderStruct)

        self.outputDict = {}
        self.outputDict['error'] = 0
        frameData = self.descriptor.Buffer[:self.descriptor.unBufferPacketLen]
        # TODO: add CRC32 Check
        
        try:
            # Read in frame Header
            magic, totalPacketLen, crc32, deviceId, numTlvs, frameNum = struct.unpack(
                headerStruct, frameData[:frameHeaderLen])
            
        except:
            print('Error: Could not read frame header')
            self.outputDict['error'] = 1
            return self.outputDict

        # Save frame number to output
        self.outputDict['crc32'] = crc32
        self.outputDict['deviceId'] = deviceId
        self.outputDict['numTlvs'] = numTlvs
        self.outputDict['frameNum'] = frameNum

        tlvs = {"tlvType": [], "tlvLength":[], "tlvValue":[]}
        # Move frameData ptr to start of 1st TLV
        frameData = frameData[frameHeaderLen:]
        for i in range(numTlvs):
            try:
                tlvType, tlvLength = struct.unpack(
                    tlvHeaderStruct, frameData[:tlvHeaderLength])
                tlvValue = frameData[tlvHeaderLength:tlvHeaderLength+tlvLength]
                # Move frameData ptr to start of the next TLV
                frameData = frameData[tlvHeaderLength+tlvLength:]
            except:
                print(f'TLV Header #{i} Parsing Failed')
                self.outputDict['error'] = 2
            try:
                tlvs["tlvType"].append(tlvType)
                tlvs["tlvLength"].append(tlvLength)
                tlvs["tlvValue"].append(tlvValue)
            except Exception as e: 
                print(f'TLV #{i} callback error!')
                print(e)
                self.outputDict['error'] = 3
        self.callbackHandler(self.outputDict, tlvs)
