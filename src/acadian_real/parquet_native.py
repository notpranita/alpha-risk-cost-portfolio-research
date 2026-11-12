"""Minimal, read-only Parquet decoder for this flat SNAPPY-encoded public dataset.
Prefer pyarrow in user environment; this decoder is for offline reproducibility only.
"""
import ctypes, struct, numpy as np, pandas as pd
from .parquet_probe import Tr
SN=ctypes.CDLL('libsnappy.so.1');SN.snappy_uncompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]

def uncompress(src,size):
 buf=ctypes.create_string_buffer(size);sz=ctypes.c_size_t(size);ret=SN.snappy_uncompress(src,len(src),buf,ctypes.byref(sz))
 if ret:raise RuntimeError(('snappy decompression failed',ret))
 return buf.raw[:sz.value]
def uint(buf,pos):
 out=0;shift=0
 while True:
  b=buf[pos];pos+=1;out|=(b&127)<<shift
  if b<128:return out,pos
  shift+=7

def hybrid(data,bitwidth,n):
 if bitwidth==0:return np.zeros(n,dtype=np.int32)
 out=[];pos=0;bytewidth=(bitwidth+7)//8
 while len(out)<n:
  hdr,pos=uint(data,pos)
  if hdr&1==0:
   count=hdr>>1; val=int.from_bytes(data[pos:pos+bytewidth],'little');pos+=bytewidth
   out.extend([val]*min(count,n-len(out)))
  else:
   groups=hdr>>1; nb=groups*bitwidth
   bits=np.unpackbits(np.frombuffer(data[pos:pos+nb],dtype=np.uint8),bitorder='little');pos+=nb
   ints=np.zeros(groups*8,dtype=np.int32)
   for j in range(bitwidth):ints += bits[j::bitwidth][:len(ints)].astype(np.int32)<<j
   out.extend(ints[:n-len(out)].tolist())
 return np.array(out[:n],dtype=np.int32)
def plain(buf,typ,n):
 if typ==5:return np.frombuffer(buf,dtype='<f8',count=n).copy()
 if typ==6:
  p=0;out=[]
  for _ in range(n):
   l=struct.unpack_from('<I',buf,p)[0];p+=4;out.append(buf[p:p+l].decode('utf8'));p+=l
  return np.array(out,dtype=object)
 raise ValueError(typ)
def read_col(f,chunk,nrows):
 meta=chunk[3];typ=meta[1];codec=meta[4];start=meta.get(11,meta[9]);end=meta[9]+meta[7];f.seek(start)
 dictionary=None;results=[];nseen=0
 while nseen<nrows:
  startpos=f.tell();hdr,used=Tr(f.read(200)).read();f.seek(startpos+used)
  data=f.read(hdr[3]);
  if hdr[1]==2:
   raw=uncompress(data,hdr[2]) if codec==1 else data
   dictionary=plain(raw,typ,hdr[7][1]);continue
  if hdr[1]==3:
   ph=hdr[8];num=ph[1];rep=ph[6];deff=ph[5]
   level=data[rep:rep+deff];max_def=1
   mask=hybrid(level,1,num).astype(bool) if deff else np.ones(num,bool)
   body=data[rep+deff:]; raw=uncompress(body,hdr[2]-rep-deff) if (codec==1 and ph.get(7,True)) else body
   encoding=ph[4]
  elif hdr[1]==0:
   ph=hdr[5];num=ph[1];raw=uncompress(data,hdr[2]) if codec==1 else data
   l=struct.unpack_from('<I',raw,0)[0];mask=hybrid(raw[4:4+l],1,num).astype(bool);raw=raw[4+l:];encoding=ph[2]
  else:raise ValueError(('page type',hdr[1]))
  num_non=int(mask.sum())
  if encoding in (2,8):
   w=raw[0];indices=hybrid(raw[1:],w,num_non); values=dictionary[indices]
  elif encoding==0:values=plain(raw,typ,num_non)
  else:raise ValueError(('encoding',encoding))
  if num_non==num:result=values
  else:
   result=np.full(num,np.nan if typ==5 else None,dtype=float if typ==5 else object);result[mask]=values
  results.append(result);nseen+=num
 return np.concatenate(results)[:nrows]
def read(path,columns=None):
 with open(path,'rb') as f:
  f.seek(-8,2);n=struct.unpack('<I',f.read(4))[0];f.seek(-8-n,2);meta,_=Tr(f.read(n)).read()
  names=[c[4].decode() for c in meta[2][1:]];columns=columns or names
  out={k:[] for k in columns}
  for ri,rg in enumerate(meta[4]):
   for name,c in zip(names,rg[1]):
    if name in out:out[name].append(read_col(f,c,rg[3]))
  return pd.DataFrame({k:np.concatenate(v) for k,v in out.items()})
if __name__=='__main__':
 import sys,time
 t=__import__('time').time();d=read(sys.argv[1]);print('elapsed',__import__('time').time()-t,d.shape);print(d.head());print(d.tail());print(d.isna().sum());d.to_pickle('/mnt/data/sp500-data.pkl')
