# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:30:55 2021

@author: weslay
"""

import numpy as np
import struct
import os ## Only For Test!!

def _LZG_LENGTH_ENCODE_LUT(idx):
    if idx<=29:
        return idx
    elif idx>29 and idx<=34:
        return 29
    elif idx>34 and idx<=47:
        return 30
    elif idx>47 and idx<=71:
        return 31
    elif idx>71 and idx<=127:
        return 32
    else:
        return 33
    
def _LZG_LENGTH_QUANT_LUT(idx):
    if idx<=29:
        return idx
    elif idx>29 and idx<=34:
        return 29
    elif idx>34 and idx<=47:
        return 35
    elif idx>47 and idx<=71:
        return 48
    elif idx>71 and idx<=127:
        return 72
    else:
        return 128
    
def _LZG_DetermineMarkers(in_data,in_size):
    hist = np.zeros((256,2),dtype="int")
    for i in range(0,256):
        hist[i][0] = 0
        hist[i][1] = i
    for i in range(0, in_size):
        hist[in_data[i]][0] += 1
    sort_idx =np.lexsort((hist[:,1],hist[:,0]))
    # for i in range(0,256):
    #     print(hist[i][1],hist[i][0])
    # print (sort_idx)
    M1 = hist[sort_idx[0]][1]
    M2 = hist[sort_idx[1]][1]
    M3 = hist[sort_idx[2]][1]
    M4 = hist[sort_idx[3]][1]
    return M1,M2,M3,M4

def _LZG_UpdateLastPos(fast,windowMask,
                       sa_tab,
                       sa_last,
                       in_data,
                       in_size,
                       first,
                       pos):
    if pos-first+2 >= in_size:
        return 0
    else:
        if fast==1:
            lIdx = (in_data[pos] << 16) | (in_data[pos+1] << 8) | in_data[pos+2]
        else:
            lIdx = (in_data[pos] << 8) | in_data[pos+1]
        sa_tab[(pos - first) & windowMask] = sa_last[lIdx]
        # print ("***sa_tab[%d] = %d" % ((pos - first) & windowMask,sa_last[lIdx]),end='')
        sa_last[lIdx] = pos
        # print (" | sa_last[0x%06X] = %d" % (lIdx,pos))
        return 1


def _LZG_CalcChecksum(in_data,in_size,first):
    a = 1
    b = 0
    a = np.uint16(a)
    b = np.uint16(b)
    ptr = first
    size = in_size
    sizediv8 = int(size/8)
    size8    = sizediv8*8
    end      = ptr + size8
    while ptr < end:
        for i in range(0,8):
            a += in_data[ptr]
            a = np.uint16(a)
            ptr += 1
            b += a
            b = np.uint16(b)
    size -= size8
    while size>0:
        size -= 1
        a += in_data[ptr]
        a = np.uint16(a)
        ptr += 1
        b += a
        b = np.uint16(b)
    # print ("%04X,%04X" % (a,b))
    return np.uint32(b << 16) | a


def _LZG_FindMatch( fast,
                    p_window,
                    p_maxMatches,
                    p_goodLength,
                    windowMask,
                    symbolCost,
                    sa_tab,
                    sa_last,
                    in_data,
                    in_size,
                    first,
                    pos):
    _LZG_MAX_RUN_LENGTH = 128
    offset = 0
    win    = 0
    bestWin = 0
    bestLength = 2
    if (pos - first) >=  p_window:
        minPos = pos - p_window
    else:
        minPos = first
    endStr = pos + _LZG_MAX_RUN_LENGTH
    if endStr > first+in_size:
        endStr = first+in_size
    pos2 = sa_tab[(pos - first) & windowMask]
    if fast==1:
        preMatch = 3
    else:
        preMatch = 2
    maxMatches = p_maxMatches
    
    while (pos2!=0 and pos2>minPos and maxMatches>0):
        maxMatches -= 1
        # print(" ----find Match",pos2,maxMatches)
        if in_data[pos+bestLength]==in_data[pos2+bestLength]:
            cmp1 = pos + preMatch
            cmp2 = pos2 + preMatch
            while (cmp1 < endStr and in_data[cmp1]==in_data[cmp2]):
                cmp1 += 1
                cmp2 += 1
            length = cmp1 - pos
            length = _LZG_LENGTH_QUANT_LUT(length)
            if length > bestLength:
                dist = pos - pos2
                if dist <= 8 or (length <= 6 and dist <= 71):
                    win = length + symbolCost - 3
                else:
                    win = length + symbolCost - 4
                    if dist >= 2056:
                        win -= 1
                if win > bestWin:
                    bestWin     = win
                    offset      = dist
                    bestLength  = length
                    if length >= p_goodLength or cmp1 >= endStr:
                        break
        pos2 = sa_tab[(pos2 - first) & windowMask]
        # print (sa_tab[(pos2-first-3):(pos2-first+3)])

    if bestWin > 0:
        return bestLength,offset
    else:
        return 0,0

def LZG_EncodeFull(in_data,in_size,level):
    fast = 1
    LZG_HEADER_SIZE = 16
    LZG_METHOD_LZG1 = 1
    isMarkerSymbol = 0
    
    if level==1:
        p_window        =   2048
        p_maxMatches    =   30
        p_goodLength    =   35
        windowMask      =   0x7FF
    elif level==2:
        p_window        =   4096
        p_maxMatches    =   40
        p_goodLength    =   48
        windowMask      =   0xFFF
    elif level==3:
        p_window        =   8192
        p_maxMatches    =   50
        p_goodLength    =   72
        windowMask      =   0x1FFF
    elif level==4:
        p_window        =   16384
        p_maxMatches    =   60
        p_goodLength    =   72
        windowMask      =   0x3FFF
    elif level==5:
        p_window        =   32768
        p_maxMatches    =   70
        p_goodLength    =   72
        windowMask      =   0x7FFF
    elif level==6:
        p_window        =   65536
        p_maxMatches    =   80
        p_goodLength    =   72
        windowMask      =   0xFFFF
    elif level==7:
        p_window        =   131072
        p_maxMatches    =   150
        p_goodLength    =   128
        windowMask      =   0x1FFFF
    elif level==8:
        p_window        =   262144
        p_maxMatches    =   250
        p_goodLength    =   128
        windowMask      =   0x3FFFF
    else:
        p_window        =   524288
        p_maxMatches    =   524288
        p_goodLength    =   128
        windowMask      =   0x7FFFF
    
    if fast==1:
        sa_last  = np.zeros(2**24,dtype="int")
    else:
        sa_last  = np.zeros(2**16,dtype="int")
    sa_tab = np.zeros(p_window,dtype="int")
    out_data = np.zeros(in_size+LZG_HEADER_SIZE*2,dtype=np.uint8)
    
    M1,M2,M3,M4 = _LZG_DetermineMarkers(in_data, in_size)
    print(M1,M2,M3,M4)
    
    init_rd_pos = 0
    init_wr_pos = 0
    src     =    init_rd_pos
    inEnd   =    init_rd_pos + in_size
    dst     =    init_wr_pos + LZG_HEADER_SIZE

    out_data[dst] = M1
    dst += 1
    out_data[dst] = M2
    dst += 1
    out_data[dst] = M3
    dst += 1
    out_data[dst] = M4
    dst += 1
    
    isMarkerSymbolLUT = np.zeros(256,dtype="int")
    for i in range(0,256):
        isMarkerSymbolLUT[i] = 0
    isMarkerSymbolLUT[M1] = 1
    isMarkerSymbolLUT[M2] = 1
    isMarkerSymbolLUT[M3] = 1
    isMarkerSymbolLUT[M4] = 1

    while src < inEnd:
        symbol = in_data[src]
        isMarkerSymbol = isMarkerSymbolLUT[symbol]
        symbolCost = 2 if isMarkerSymbol==1 else 1

        _LZG_UpdateLastPos(fast,windowMask,
                           sa_tab,
                           sa_last,
                           in_data,
                           in_size,
                           init_rd_pos,
                           src)
        
        length,offset = _LZG_FindMatch( fast,
                                        p_window,
                                        p_maxMatches,
                                        p_goodLength,
                                        windowMask,
                                        symbolCost,
                                        sa_tab,
                                        sa_last,
                                        in_data,
                                        in_size,
                                        init_rd_pos,
                                        src)
            
        if length > 0:
            # print ("length=",length,"dist=",offset)
            if (length <= 6) and (offset >= 9) and (offset <= 71):
                out_data[dst] = M3
                dst += 1
                out_data[dst] = ((length - 3) << 6) | (offset - 8)
                dst += 1
            elif offset <= 8:
                lengthEnc = _LZG_LENGTH_ENCODE_LUT(length)
                out_data[dst] = M4
                dst += 1
                out_data[dst] = ((offset - 1) << 5) | (lengthEnc - 2)
                dst += 1
            elif offset >= 2056:
                lengthEnc = _LZG_LENGTH_ENCODE_LUT(length)
                offset -= 2056
                out_data[dst] = M1
                dst += 1
                out_data[dst] = ((offset >> 11) & 0xE0) | (lengthEnc - 2)
                dst += 1
                out_data[dst] = (offset >> 8)
                dst += 1
                out_data[dst] = offset
                dst += 1
            else:
                lengthEnc = _LZG_LENGTH_ENCODE_LUT(length)
                offset -= 8
                out_data[dst] = M2
                dst += 1
                out_data[dst] = ((offset >> 3) & 0xe0) | (lengthEnc - 2)
                dst += 1
                out_data[dst] = offset
                dst += 1

            for i in range(1,length):  # 注意，这里是从1开始
                _LZG_UpdateLastPos(fast,windowMask,
                                   sa_tab,
                                   sa_last,
                                   in_data,
                                   in_size,
                                   init_rd_pos,
                                   src+i)
            src += length
        else:
            # print("Plain:%02X" % symbol)
            out_data[dst] = symbol
            dst += 1
            src += 1

            if isMarkerSymbol==1:
                out_data[dst] = 0
                dst += 1


    hdr_method = LZG_METHOD_LZG1
    hdr_encodedSize = dst - init_wr_pos - LZG_HEADER_SIZE
    hdr_decodedSize = in_size
    hdr_checksum = _LZG_CalcChecksum(out_data,hdr_encodedSize,LZG_HEADER_SIZE)
    
    out_data[0] = 0x4C # 'L'
    out_data[1] = 0x5A # 'Z'
    out_data[2] = 0x47 # 'G'
    out_data[3] = hdr_decodedSize >> 24
    out_data[4] = hdr_decodedSize >> 16
    out_data[5] = hdr_decodedSize >> 8
    out_data[6] = hdr_decodedSize
    out_data[7]  = hdr_encodedSize >> 24
    out_data[8]  = hdr_encodedSize >> 16
    out_data[9]  = hdr_encodedSize >> 8
    out_data[10] = hdr_encodedSize
    out_data[11] = hdr_checksum >> 24
    out_data[12] = hdr_checksum >> 16
    out_data[13] = hdr_checksum >> 8
    out_data[14] = hdr_checksum & 0xFF
    out_data[15] = hdr_method
    
    return out_data,hdr_encodedSize+LZG_HEADER_SIZE



if __name__=="__main__":
    SRC_PATH = "obj1"
    DST_PATH = "obj1.hwlzg"
    fp_src = open(SRC_PATH,'rb')
    fp_dst = open(DST_PATH,'wb')
    in_data = fp_src.read()
    
    # print (_LZG_DetermineMarkers(in_data,len(in_data)))
    out_data,out_size = LZG_EncodeFull(in_data,len(in_data),5)

    for i in range(0,out_size):
        fp_dst.write(struct.pack("B",out_data[i] & 0xFF))
    print ("Src Size: %d ====> Dst Size: %d Bytes\n" % (len(in_data),out_size))
    
    fp_src.close()
    fp_dst.close()