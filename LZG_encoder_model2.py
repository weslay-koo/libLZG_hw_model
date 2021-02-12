# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:30:55 2021

@author: weslay
"""

import numpy as np
import struct
import os,filecmp ## Only For Test!!

MAX_SYMBOL_LEN  = 128
MAX_SYMBOL_MASK = MAX_SYMBOL_LEN-1

def LZG_LENGTH_ENCODE_LUT(idx):
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
    
def LZG_LENGTH_QUANT_LUT(idx):
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

def Out_Encoder(length,offset,symbol,M1,M2,M3,M4):
    out_code = np.zeros(4,dtype=np.uint8)
    out_len  = 0
    if(length>0):
        if (length <= 6) and (offset >= 9) and (offset <= 71):
            out_code[0] = M3
            out_code[1] = ((length-3)<<6) | (offset-8)
            out_len     = 2
        elif (offset <= 8):
            lengthEnc = LZG_LENGTH_ENCODE_LUT(length)
            out_code[0] = M4
            out_code[1] = ((offset-1)<<5) | (lengthEnc-2)
            out_len     = 2
        elif (offset >= 2056):
            lengthEnc = LZG_LENGTH_ENCODE_LUT(length)
            offset -= 2056
            out_code[0] = M1
            out_code[1] = ((offset>>11)&0xE0) | (lengthEnc-2)
            out_code[2] = (offset>>8)
            out_code[3] = offset
            out_len     = 4
        else:
            lengthEnc = LZG_LENGTH_ENCODE_LUT(length)
            offset -= 8
            out_code[0] = M2
            out_code[1] = ((offset>>3) & 0xe0) | (lengthEnc-2)
            out_code[2] = offset & 0xFF
            out_len     = 3
    else:
        if (symbol==M1) or (symbol==M2) or (symbol==M3) or (symbol==M4):
            out_code[0] = symbol
            out_code[1] = 0x00
            out_len     = 2
        else:
            out_code[0] = symbol
            out_len     = 1
    return out_code,out_len

def LZG_level_set(level):
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
    return p_window,p_maxMatches,p_goodLength,windowMask

def LZG_CalcChecksum(in_data,in_size,first):
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

def Unit_UpdateLastPos(windowMask,
                       sa_tab,
                       sa_last,
                       current_byte,
                       next_byte,
                       slide_window,
                       pos):
    lIdx = (current_byte<<8) | next_byte
    sa_tab[pos & windowMask] = sa_last[lIdx]
    sa_last[lIdx] = pos
    slide_window[(pos-1)&windowMask] = current_byte
    slide_window[pos & windowMask]   = next_byte
    # print ("****sa_last[%04X]=%x, slide[%x]=0x%02X" % (lIdx,pos,pos,next_byte))


def Unit_FindMatch( windowMask,
                    p_window,
                    p_maxMatches,
                    sa_tab,
                    pos2_list,
                    dist_list,
                    pos):
    pos2 = sa_tab[pos&windowMask]
    i = 0
    while(pos2!=0 and pos2>pos-p_window and i<p_maxMatches):
        print("----Find Match: pos=%x, pos2=%x" % (pos,pos2))
        dist_list[i] = pos-pos2
        pos2_list[i] = pos2
        pos2         = sa_tab[pos2&windowMask]
        i+=1
    if i>0:
        # print(pos2_list)
        return 1
    else:
        return 0
    
def Unit_ExtendMatch(   windowMask,
                        next_byte,
                        slide_window,
                        pos2_list,
                        dist_list,
                        length_cnt):
    valid_count = 0
    valid_dist  = 0
    valid_pos   = 0
    # print("^^^^Extend Match: pos=0x%02X" % next_byte)
    # print(pos2_list)
    for i in range(0,len(pos2_list)):
        next_pos2 = pos2_list[i]+length_cnt-1
        if (dist_list[i]!=0):
            if (slide_window[next_pos2&windowMask]!=next_byte):
                # print ("++++ slide_window[%x]=0x%02X, next_byte=0x%02X" % (next_pos2,slide_window[next_pos2&windowMask],next_byte))
                dist_list[i] = 0
            else:
                valid_count += 1
                if (valid_dist==0):  # 初始化匹配距离和匹配位置
                    valid_dist = dist_list[i]
                    valid_pos  = pos2_list[i]
                elif (valid_dist>=dist_list[i]): # 选择距离最近的匹配
                    valid_dist = dist_list[i]
                    valid_pos  = pos2_list[i]
    # print("^"*80)
    return valid_count,valid_dist,valid_pos




def Unit_StringMatch_slow(  p_window,
                            p_maxMatches,
                            p_goodLength,
                            windowMask,
                            sa_tab,
                            sa_last,
                            in_data,
                            slide_window,
                            byte_list,
                            list_head_addr,
                            list_tail_addr,
                            pos):
    LZG_MAX_RUN_LENGTH = 128
    bestLength = 2
    length_cnt = 0
    extend_cnt = 0
    dist_d2  = 0
    dist_d1  = 0
    dist     = 0
    match_ok = 0
    head_addr = list_head_addr
    tail_addr = list_tail_addr
    valid_pos = 0
    pos2_list = np.zeros(p_maxMatches,dtype="int")
    dist_list = np.zeros(p_maxMatches,dtype="int")
    
    if (pos==0):
        byte_list[0] = in_data.pop(0)
        head_addr    = 0
        tail_addr    = 0
        return 0,0,pos+1,head_addr,tail_addr,0
    elif (pos==1):
        byte_list[1] = in_data.pop(0)
        head_addr    = 0
        tail_addr    = 1
        return 0,0,pos+1,head_addr,tail_addr,0
    else:
        if (head_addr+1>=tail_addr): # 读入新字符并尝试匹配
            if (in_data!=[]):
                tail_addr += 1
                byte_list[tail_addr&MAX_SYMBOL_MASK] = in_data.pop(0)
                Unit_UpdateLastPos(windowMask,
                                   sa_tab,
                                   sa_last,
                                   byte_list[(tail_addr-1)&MAX_SYMBOL_MASK],
                                   byte_list[tail_addr&MAX_SYMBOL_MASK],
                                   slide_window,
                                   pos)
            else:
                print ("FIFO Empty!!!\n")
                
        match_ok = Unit_FindMatch(windowMask,
                                  p_window,
                                  p_maxMatches,
                                  sa_tab,
                                  pos2_list,
                                  dist_list,
                                  pos)
        
        if (match_ok!=0 and in_data!=[]): #扩展匹配
            extend_length = 0
            extend_pos    = 0
            extend_dist   = 0
            extend_term   = 0
            length_cnt   += bestLength
            head_addr    += 1
            while (match_ok>=1):
                if (head_addr+1>=tail_addr):
                    tail_addr += 1
                    byte_list[tail_addr&MAX_SYMBOL_MASK] = in_data.pop(0)
                    Unit_UpdateLastPos(windowMask,
                                       sa_tab,
                                       sa_last,
                                       byte_list[(tail_addr-1)&MAX_SYMBOL_MASK],
                                       byte_list[tail_addr&MAX_SYMBOL_MASK],
                                       slide_window,
                                       pos+length_cnt-1)
                dist_d2 = dist_d1
                dist_d1 = dist
                match_ok,dist,valid_pos = Unit_ExtendMatch(windowMask,
                                                           byte_list[(head_addr+2)&MAX_SYMBOL_MASK],
                                                           slide_window,
                                                           pos2_list,
                                                           dist_list,
                                                           length_cnt)
                head_addr += 1
                if (dist_d1!=0 and dist_d1<=71 and dist>71 and length_cnt<=6):
                    # print ("%"*80)
                    extend_cnt = 1
                if (match_ok>=1):
                    extend_dist   = dist
                    extend_length = length_cnt+1
                    extend_pos    = pos+length_cnt+1
                    if (extend_cnt>0):
                        extend_cnt += 1
                else:
                    if (length_cnt<=2):
                        return -1,0,pos+length_cnt,head_addr,tail_addr,0
                    elif (extend_cnt==2 and dist_d2 <= 71):
                        extend_dist   = dist_d2
                        extend_length = length_cnt-1
                        extend_pos    = pos+length_cnt
                        extend_term   = 1
                    elif (dist==0 and dist_d1>=2056 and dist_d2<2056 and dist_d2!=0 and length_cnt>4):
                        extend_dist   = dist_d2
                        extend_length = length_cnt-1
                        extend_pos    = pos+length_cnt
                        extend_term   = 1
                        
                length_cnt += 1
                if (in_data == []):
                    if (match_ok>=1):
                        length_cnt += 1
                    break
                if (length_cnt>=LZG_MAX_RUN_LENGTH+2):
                    head_addr += 1
                    break
                
            if (extend_dist>=2056 and length_cnt-1==4):
                return -3,0,pos+length_cnt-1,head_addr,tail_addr,0
            elif (extend_dist>71 and length_cnt-1==3):
                return -2,0,pos+length_cnt-1,head_addr,tail_addr,0
            else: # 正常匹配
                if (length_cnt>=LZG_MAX_RUN_LENGTH):
                    print ("&"*80)
                return extend_length,extend_dist,extend_pos,head_addr,tail_addr,extend_term
            
        else: # 未找到匹配
            if (in_data==[]):
                print ("FIFO empty ***")
            head_addr += 1
            return 0,0,pos+1,head_addr,tail_addr,0
        
            


def LZG_EncodeFull_slow(in_data,in_size,level):
    LZG_HEADER_SIZE = 16
    LZG_METHOD_LZG1 = 1
    # LZG_METHOD_COPY = 0
    
    p_window,p_maxMatches,p_goodLength,windowMask = LZG_level_set(level)
    M1,M2,M3,M4 = _LZG_DetermineMarkers(in_data, in_size)
    print(M1,M2,M3,M4)
    
    symbol_list = np.zeros(MAX_SYMBOL_LEN,dtype=np.uint8)
    sa_last     = np.zeros(2**16,dtype="int")
    sa_tab      = np.zeros(p_window,dtype="int")
    out_data    = np.zeros(in_size+LZG_HEADER_SIZE*2,dtype=np.uint8)
    slide_window= np.zeros(p_window,dtype=np.uint8)
    
    head_addr = 0
    tail_addr = 0
    length    = 0
    offset    = 0
    extend_term = 0
    
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
    while (src < inEnd):
        length_diff = 0
        length,\
        offset,\
        src,\
        head_addr,\
        tail_addr,\
        extend_term = Unit_StringMatch_slow(p_window,
                                            p_maxMatches,
                                            p_goodLength,
                                            windowMask,
                                            sa_tab,
                                            sa_last,
                                            in_data,
                                            slide_window,
                                            symbol_list,
                                            head_addr,
                                            tail_addr,
                                            src)
        print ("length=%d, offset=%d, pos=%x, head_addr=%x, tail_addr=%x,extend_term=%d\n" % (length,offset,src,head_addr,tail_addr,extend_term))
        
        if (src>1):
            # print (src,extend_term)
            length_diff = length-LZG_LENGTH_QUANT_LUT(length)
            if (length_diff!=0): # 匹配长度量化回退
                print ("p"*80)
                print ("length_diff=%d" % length_diff)
                out_code,out_len = Out_Encoder( length, offset,
                                                symbol_list[tail_addr&MAX_SYMBOL_MASK],
                                                M1, M2, M3, M4)
                out_data[dst:dst+out_len] = out_code[0:out_len]
                dst += out_len
                head_addr -= length_diff
                src -= length_diff
            elif (length<0): # 其他情况回退
                head_addr += length
                src       += length
                out_code,out_len = Out_Encoder( 0, 0,
                                                symbol_list[head_addr&MAX_SYMBOL_MASK],
                                                M1, M2, M3, M4)
                out_data[dst:dst+out_len] = out_code[0:out_len]
                dst += out_len
            else:
                out_code,out_len = Out_Encoder( length, offset,
                                                symbol_list[head_addr&MAX_SYMBOL_MASK],
                                                M1, M2, M3, M4)
                out_data[dst:dst+out_len] = out_code[0:out_len]
                dst += out_len
                if (src>inEnd):
                    break
                elif(src==inEnd):
                    out_code,out_len = Out_Encoder( 0, 0,
                                                    symbol_list[tail_addr&MAX_SYMBOL_MASK],
                                                    M1, M2, M3, M4)
                    out_data[dst:dst+out_len] = out_code[0:out_len]
                    dst += out_len
                    break
                if (extend_term==1):
                    # print ("$"*80)
                    head_addr -= 1
                    src -= 1

    hdr_method = LZG_METHOD_LZG1
    hdr_encodedSize = dst - init_wr_pos - LZG_HEADER_SIZE
    hdr_decodedSize = in_size
    hdr_checksum = LZG_CalcChecksum(out_data,hdr_encodedSize,LZG_HEADER_SIZE)
    
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
    SRC_PATH  = "./"
    DST_PATH  = "./"
    file_list = ["xargs.1"]
    # file_list = ["tt"]
    error_list = []
    
    for file in file_list:
        fp_src = open(SRC_PATH+file,'rb')
        fp_dst = open(DST_PATH+file+'.hwlzg','wb')
        in_data = fp_src.read()
        in_stream = []
        for d in in_data:
            in_stream.append(d)
        print (len(in_stream),type(in_stream))
        
        out_data,out_size = LZG_EncodeFull_slow(in_stream,len(in_stream),5)
    
        for i in range(0,out_size):
            fp_dst.write(struct.pack("B",out_data[i] & 0xFF))
        print ("Src Size: %d ====> Dst Size: %d Bytes\n" % (len(in_data),out_size))
        
        fp_src.close()
        fp_dst.close()
        
        #比较解压后的文件
        if os.path.exists(DST_PATH+file+".hwdecomp"):
            os.remove(DST_PATH+file+".hwdecomp")
        os.system("unlzg %s %s" % (DST_PATH+file+".hwlzg",DST_PATH+file+".hwdecomp"))
        
        if filecmp.cmp(file,DST_PATH+file+".hwdecomp"):
            print ("%s file compress OK\n" % file)
        else:
            print ("%s file compress error!!\n" % file)
            error_list.append(file)
        
    print (error_list)