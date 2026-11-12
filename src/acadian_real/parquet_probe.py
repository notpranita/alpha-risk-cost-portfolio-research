from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from thrift.Thrift import TType
import struct

def value(p,typ):
 if typ==TType.BOOL:return p.readBool()
 if typ==TType.BYTE:return p.readByte()
 if typ==TType.I16:return p.readI16()
 if typ==TType.I32:return p.readI32()
 if typ==TType.I64:return p.readI64()
 if typ==TType.DOUBLE:return p.readDouble()
 if typ==TType.STRING:return p.readBinary()
 if typ==TType.STRUCT:return obj(p)
 if typ in (TType.LIST,TType.SET):
  t,n=p.readListBegin();r=[value(p,t) for _ in range(n)];p.readListEnd();return r
 if typ==TType.MAP:
  kt,vt,n=p.readMapBegin();r={value(p,kt):value(p,vt) for _ in range(n)};p.readMapEnd();return r
 raise ValueError(typ)
def obj(p):
 p.readStructBegin();o={}
 while True:
  _,t,i=p.readFieldBegin()
  if t==TType.STOP:break
  o[i]=value(p,t);p.readFieldEnd()
 p.readStructEnd();return o
class Tr:
 def __init__(self,data):self.b=TTransport.TMemoryBuffer(data);self.p=TCompactProtocol.TCompactProtocol(self.b)
 def read(self):return obj(self.p),self.b._buffer.tell()

if __name__=='__main__':
 f=open('/mnt/data/sp500-stock-prices.parquet','rb'); f.seek(-8,2);n=struct.unpack('<I',f.read(4))[0];f.seek(-8-n,2);meta,_=Tr(f.read(n)).read(); print('keys',meta.keys(),'rows',meta[3]);
 print('schema',meta[2]); print('rowgroups',len(meta[4])); r=meta[4][0];print('rowgroup keys',r.keys(),r[3]);
 for c in r[1]: print(c[3].get(3), 'colkeys',c.keys(),'meta', {k:v for k,v in c[3].items() if k not in (2,7,8)})
