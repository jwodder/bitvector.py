import sys
from   bitvector import bitvector

def showBV(bv):
    print
    print bv
    i=0
    for b in bv:
        if i % 8 == 0 and i != 0:
            sys.stdout.write('.')
        sys.stdout.write('1' if b else '0')
        i += 1
    sys.stdout.write("\n")

#bv = bitvector(0b1010110111011111011111111)
bv = bitvector(0x15BBEFF)
showBV(bv)

a = bv.copy()
# 1  0101 1011  10[11 1110  111]1 1111
del a[5:14]  # a = bitvector(0b1010110111011111) = bitvector(0xADDF)
showBV(a)

b = bv.copy()
del b[5:9]  # b = bitvector(0b101011011101111111111) = bitvector(0x15BBFF)
showBV(b)

c = bv.copy()
del c[5]
showBV(c)

d = bv.copy()
del d[5:17]  # d = bitvector(0b1010110111111) = bitvector(0x15BF)
showBV(d)

bv = a
bv >>= 3
showBV(bv)
bv <<= 4
showBV(bv)
