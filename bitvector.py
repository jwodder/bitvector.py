# Bits are stored in ascending order.

# Remember: As bitvector is a mutable type, a __hash__ method is not
# appropriate.

from array       import array
from collections import defaultdict

__all__ = ["bitvector"]

class bitvector(object):
    def __init__(self, obj=None, width=None, fill=None):
	if fill is None:
	    fill = obj if isinstance(obj, bool) else False
	if width is not None and width <= 0:
	    self._blob = array('B')
	    self._size = 0
	elif isinstance(obj, bool):
	# This needs to go before the `obj == 0` case because `False == 0`.
	    if width is None:
		self._blob = array('B', [int(obj)])
		self._size = 1
	    else:
		self.__init__(None, width=width, fill=fill)
		if fill != obj: self._blob[0] ^= 1
	elif obj is None or obj == 0:
	    if width is None:
		self._blob = array('B')
		self._size = 0
	    else:
		(bytes, extra) = divmod(width, 8)
		self._blob = array('B', [255 if fill else 0] * bytes)
		if extra != 0:
		    self._blob.append((1 << extra) - 1 if fill else 0)
		self._size = width
	elif isinstance(obj, bitvector):
	    self._blob = obj._blob[:]
	    self._size = len(obj)
	    if width is not None:
		self.setWidth(width, fill)
	elif isinstance(obj, str):
	    self._blob = array('B', map(ord, obj))
	    self._size = len(obj) * 8
	    if width is not None:
		self.setWidth(width, fill)
	elif isinstance(obj, (int, long)):
	    ### TODO: This ignores `fill` - should it?
	    bytes = []
	    if width is None:
		if obj == -1:
		    (bytes, self._size) = ([1], 1)
		else:
		    cutoff = 0 if obj >= 0 else -1
		    while obj != cutoff:
			(obj, b) = divmod(obj, 256)
			bytes.append(b)
		    last = bytes[-1]
		    if cutoff == -1: last ^= 255
		    unused = 8
		    while last > 0:
			last >>= 1
			unused -= 1
		    self._size = len(bytes) * 8 - unused
		    if cutoff == -1:
			if unused == 0:
			    bytes.append(1)
			else:
			    bytes[-1] &= (1 << (9-unused)) - 1
			self._size += 1
	    else:
		for i in xrange((width+7) // 8):
		    (obj, b) = divmod(obj, 256)
		    bytes.append(b)
		if obj < 0 and width % 8 != 0:
		    bytes[-1] &= (1 << (width % 8)) - 1
		self._size = width
	    self._blob = array('B', bytes)
	else:
	    obj = list(obj)
	    if width is not None:
		if len(obj) <= width:
		    obj = obj + [fill] * (width - len(obj))
		else:
		    obj = obj[0:width]
	    bytes = []
	    for i in xrange(0, len(obj), 8):
		b=0
		for j, bit in zip(xrange(8), obj[i:i+8]):
		    if bit:
			b |= 1 << j
		bytes.append(b)
	    self._blob = array('B', bytes)
	    self._size = len(obj)

    def __getitem__(self, i):
	if isinstance(i, slice):
	    (start, stop, step) = i.indices(len(self))
	    if start >= stop: return bitvector()
	    if step == 1:
		new = self >> start
		stop -= start
		new.setWidth(stop)
		return new
	    else:
		return bitvector([self[j] for j in range(start, stop, step)])
	else:
	    if i < 0:
		i += self._size
	    if not (0 <= i < self._size):
		raise IndexError('bitvector index out of range')
	    (byte, offset) = divmod(i, 8)
	    return bool(self._blob[byte] & (1 << offset))

    def __setitem__(self, i, new):
    ### TODO: Is this supposed to return something?
	if isinstance(i, slice):
	    (start, stop, step) = i.indices(len(self))
	    if start > stop:
		### TODO: Is there a better error or way to handle this?  Lists
		### seem to handle this situation oddly.
		raise ValueError('bitvector.__setitem__: slice start cannot be'
				 ' after stop')
	    if step == 1:
		### TODO: Make this part more efficient.
		new = bitvector(new)
		# It's best to flush out construction errors before changing
		# `self`'s width.
		beyond = self[stop:]
		self.setWidth(start)
		self.extend(new)
		self.extend(beyond)
	    else:
		### TODO: What types should be permitted for `new` here?  This
		### currently only works for iterable types with __len__
		### methods, `list` and `bitvector` being the only such types
		### that are also accepted by the `bitvector` constructor.
		indices = range(start, stop, step)
		if len(indices) != len(new):
		    raise ValueError('attempt to assign sequence of size %d'
				     ' to extended slice of size %d'
				      % (len(new), len(indices)))
		    # If it's good enough for the `list` type, it's good enough
		    # for me!
		for (j,b) in zip(indices, new):
		    (byte, offset) = divmod(j, 8)
		    if b: self._blob[byte] |= 1 << offset
		    else: self._blob[byte] &= ~(1 << offset)
	else:
	    if i < 0:
		i += self._size
	    if not (0 <= i < self._size):
		raise IndexError('bitvector index out of range')
	    (byte, offset) = divmod(i, 8)
	    if new: self._blob[byte] |= 1 << offset
	    else: self._blob[byte] &= ~(1 << offset)

    def __delitem__(self, i):
	if isinstance(i, slice):
	    (start, stop, step) = i.indices(len(self))
	    if start >= stop: return
	    if stop >= len(self):
		self.setWidth(start)
	    elif step == 1:
		(byte1, offset1) = divmod(start, 8)
		(byte2, offset2) = divmod(stop, 8)
		if byte1 != byte2:
		    inter = (stop - start - (8 - offset1)) // 8
		    if inter > 0:
			del self._blob[byte1+1 : byte1+1+inter]
			byte2 -= inter
			self._size -= inter * 8
		    self._blob[byte1] &= (1 << offset1) - 1
		    self._blob[byte2] &= 255 << offset2
		else:
		    b = self._blob[byte1]
		    above = b & (255 << offset2)
		    self._blob[byte1] = (b & (1 << offset1) - 1) \
				      | (above >> (offset2-offset1))
		    offset1 = 8 - (offset2 - offset1)
		    (byte2, offset2) = (byte2+1, 0)
		    if byte2 * 8 >= len(self):
			self._size -= stop - start
			return
		shiftBy = 8 - offset1 + offset2
		carry = 0
		for i in xrange(len(self._blob)-1, byte1, -1):
		    (self._blob[i], carry) \
			= divmod(self._blob[i] | (carry << 8), 1 << shiftBy)
		self._blob[byte1] |= (carry >> offset2) << offset1
		if 0 < self._size % 8 <= shiftBy:
		    self._blob.pop()
		self._size -= shiftBy
	    else:
		delled = 0
		for j in xrange(start, stop, step):
		    del self[j-delled]
		    if step > 0:
			delled += 1
	else:
	    if i < 0:
		i += self._size
	    if not (0 <= i < self._size):
		raise IndexError('bitvector index out of range')
	    (byte, offset) = divmod(i, 8)
	    b = self._blob[byte]
	    self._blob[byte] = (b & (1 << offset)-1) \
			     | ((b & (255 << offset+1)) >> 1)
	    for j in xrange(byte+1, len(self._blob)):
		if self._blob[j] & 1:
		    self._blob[j-1] |= 1 << 7
		self._blob[j] >>= 1
	    if self._size % 8 == 1:
		self._blob.pop()
	    self._size -= 1

    def __invert__(self):
	inverse = bitvector()
	inverse._blob = array('B', [~b & 255 for b in self._blob])
	if self._size % 8 != 0:
	    inverse._blob[-1] &= (1 << self._size % 8) - 1
	inverse._size = self._size
	return inverse

    def __int__(self):
	#return int(''.join('1' if b else '0' for b in reversed(self)), 2)
	#return int(''.join(bin(b).zfill(8) for b in reversed(self._blob)), 2)
	return reduce(lambda x,b: x*256 + b, reversed(self._blob), 0)
	#return reduce(operator.__or__, [b << i*8 for (i,b) in enumerate(self._blob)])
	#return sum(b << i*8 for (i,b) in enumerate(self._blob))

    def __long__(self): return long(int(self))

    def __copy__(self): return bitvector(self)

    copy = __copy__

    def extend(self, other): self += other #; return None

    def __len__(self): return self._size

    def __nonzero__(self): return any(b != 0 for b in self._blob)

    def __add__(self, other):
	new = bitvector(self)
	new.extend(other)
	return new

    def __radd__(self, other):
	new = bitvector(other)
	new.extend(self)
	return new

    def __iadd__(self, other):
	other = bitvector(other)
	offset = self._size % 8
	self._size += other._size
	if offset != 0:
	    other <<= offset
	    self._blob[-1] |= other._blob[0]
	    self._blob.extend(other.blob[1:])
	else:
	    self._blob.extend(other.blob)
	return self

    def __lshift__(self, n):
	new = bitvector(self)
	new <<= n
	return new

    def __ilshift__(self, n):
	if n < 0:
	    self >>= -n
	else:
	    (pads, offset) = divmod(n, 8)
	    if offset != 0:
		carry = 0
		for (i,b) in enumerate(self._blob):
		    (carry, self._blob[i]) = divmod((b << offset) | carry, 256)
		self._blob.append(carry)
	    if pads != 0:
		self._blob[0:0] = array('B', [0] * pads)
	    self._size += n
	return self

    def __rshift__(self, n):
	new = bitvector(self)
	new >>= n
	return new

    def __irshift__(self, n):
	if n < 0:
	    self <<= -n
	else:
	    del self[0:n]
	return self

    def __iter__(self):
	i=0
	for byte in self._blob:
	    for j in xrange(8):
		if i >= self._size:
		    break
		yield bool(byte & (1 << j))
		i += 1

    def __cmp__(self, other):
	return cmp(type(self), type(other)) or \
	       cmp((self._blob, self._size), (other._blob, other._size))

    def __repr__(self):
	return 'bitvector(%#x, width=%d)' % (int(self), len(self))
	#return 'bitvector(%s, width=%d)' % (bin(int(self)), len(self))

    def __and__(self, other):
	other = bitvector(other)
	other &= self
	return other

    __rand__ = __and__

    def __iand__(self, other):
	if not isinstance(other, bitvector):
	    other = bitvector(other)
	self.setWidth(min(self._size, other._size))
	for i in xrange(len(self._blob)):
	    self._blob[i] &= other._blob[i]
	return self

    def __or__(self, other):
	other = bitvector(other)
	other |= self
	return other

    __ror__  = __or__

    def __ior__(self, other):
	if not isinstance(other, bitvector):
	    other = bitvector(other)
	self.setWidth(max(self._size, other._size), False)
	for i in xrange(len(other._blob)):
	    self._blob[i] |= other._blob[i]
	return self

    def __xor__(self, other):
	other = bitvector(other)
	other ^= self
	return other

    __rxor__ = __xor__

    def __ixor__(self, other):
	if not isinstance(other, bitvector):
	    other = bitvector(other)
	self.setWidth(max(self._size, other._size), False)
	for i in xrange(len(other._blob)):
	    self._blob[i] ^= other._blob[i]
	return self

    def setWidth(self, width, fill=False):
	### TODO: Rename "resize"?
	if width < 0: width = 0
	if width < len(self):
	    self._blob = self._blob[0 : (width+7)//8]
	    if width % 8 != 0:
		self._blob[-1] &= (1 << (width % 8)) - 1
	else:
	    extra = width - len(self)
	    padBits = 0 if len(self) % 8 == 0 else (8 - len(self) % 8)
	    padBytes = (extra - padBits + 7) // 8
	    if fill and padBits > 0:
		self._blob[-1] |= ((1 << padBits) - 1) << (8 - padBits)
	    self._blob.extend([255 if fill else 0] * padBytes)
	self._size = width

    def toggle(self, i):
	if isinstance(i, slice):
	    (start, stop, step) = i.indices(len(self))
	    if start >= stop: return
	    if step == 1:
		(byte1, offset1) = divmod(start, 8)
		(byte2, offset2) = divmod(stop, 8)
		if byte1 == byte2:
		    self._blob[byte1] ^= ((1 << offset2-offset1) - 1) << offset1
		else:
		    self._blob[byte1] ^= 255 << offset1
		    for j in xrange(byte1+1, byte2):
			self._blob[j] ^= 255
		    self._blob[byte2] ^= (1 << offset2) - 1
	    else:
		for j in xrange(start, stop, step):
		    self.toggle(j)
	else:
	    if i < 0:
		i += self._size
	    if not (0 <= i < self._size):
		raise IndexError('bitvector.toggle index out of range')
	    (byte, offset) = divmod(i, 8)
	    self._blob[byte] ^= 1 << offset

    def append(self, x):
	if self._size % 8 == 0:
	    self._blob.append(1 if x else 0)
	elif x:
	    (byte, offset) = divmod(self._size, 8)
	    self._blob[byte] |= 1 << offset
	self._size += 1

    def toBytes(self, ascending=True):
	return map(None if ascending else revbyte, self._blob)

    def pop(self, i=-1):
	x = self[i]
	del self[i]
	return x

    def __reversed__(self):
	if len(self) == 0: return
	maxI = len(self) % 8 or 7
	for byte in reversed(self._blob):
	    for i in xrange(maxI, -1, -1):
		yield bool(byte & (1 << i))
	    maxI = 7

    def __mul__(self, n):
	if n <= 0: return bitvector()
	prod = bitvector(self)
	for _ in xrange(n-1):
	    prod.extend(self)
	return prod

    __rmul__ = __mul__

    def __imul__(self, n):
	if n <= 0:
	    self.clear()
	else:
	    tmp = self.copy()
	    for _ in xrange(n-1):
		self.extend(tmp)
	return self

    def __contains__(self, other):
	if other is True:
	    return any(self._blob)
	elif other is False:
	    (bytes, offset) = divmod(self._size, 8)
	    if any(b != 255 for b in self._blob[:bytes]):
		return True
	    return offset != 0 and ~self._blob[bytes] & ((1<<offset) - 1) != 0
	else: 
	    return self.find(other) != -1

    def count(self, sub, start=0, end=None):
	### TODO: Add an optimization for when `other` is a bool
	if not isinstance(sub, bitvector):
	    sub = bitvector(sub)
	if len(sub) == 0:
	    return len(self) + 1
	qty = 0
	while True:
	    try:
		start = self.index(sub, start, end)
	    except ValueError:
		return qty
	    else:
		qty += 1
		start += len(sub)

    def insert(self, i, x):
	### TODO: Should insertion of bitvectors also be supported, or just
	### bools?
	if i < 0: i += len(self)
	if i < 0: i = 0
	if i == len(self):
	    self.append(x)
	elif i < len(self):
	    (byte, offset) = divmod(i, 8)
	    b = self._blob[byte]
	    carry = 1 if b & 128 else 0
	    self._blob[byte] = (b & (1 << offset)-1) \
			     | ((b & (255 << offset)) << 1)
	    if x:
		self._blob[byte] |= 1 << offset
	    for j in xrange(byte+1, len(self._blob)):
		(carry, self._blob[j]) = divmod(self._blob[j] << 1 | carry, 256)
	    if len(self) % 8 == 0:
		self._blob.append(carry)
	    self._size += 1
	else:
	    raise IndexError('bitvector.insert index out of range')

    def remove(self, sub):
	if not isinstance(sub, bitvector):
	    sub = bitvector(sub)
	dex = self.index(sub)
	del self[dex : dex+len(sub)]

    def reverse(self):
	if len(self) % 8 != 0:
	    self <<= 8 - len(self) % 8
	self._blob.reverse()
	for i in xrange(len(self._blob)):
	    self._blob[i] = revbyte(self._blob[i])

    def find(self, sub, start=0, end=None):
	if not isinstance(sub, bitvector):
	    sub = bitvector(sub)
	if start < 0: start += len(self)
	if start < 0: start = 0
	if len(sub) == 0: return start
	if start > len(self): return -1
	if end is None: end = len(self)
	if end > len(self): end = len(self)
	if end < 0: end += len(self)
	if end < 0: end = 0
	shifted = [sub] + [None] * 7
	masks = [bitvector(True, width=len(sub))] + [None] * 7
	while start < end and end-start >= len(sub):
	    (byte, offset) = divmod(start, 8)
	    if shifted[offset] is None:
		shifted[offset] = sub << offset
		masks[offset] = bitvector([False] * offset,
					  width=(offset+len(sub)),
					  fill=True)
	    maskBytes = masks[offset]._blob
	    targetBytes = shifted[offset]._blob
	    if all((slf & mask) == target
		   for (slf, mask, target)
		   in zip(self._blob[byte:], maskBytes, targetBytes)):
		return start
	    start += 1
	return -1

    def index(self, sub, start=0, end=None):
	dex = self.find(sub, start, end)
	if dex == -1: raise ValueError('bitvector.index(x): x not in bitvector')
	else: return dex

    def clear(self):
	self._blob = array('B')
	self._size = 0

    def listSetBits(self):
	"""Returns an iterator of the indices of all set bits in the
	   `bitvector`"""
	i=0
	for byte in self._blob:
	    for j in xrange(8):
		if i >= self._size:
		    break
		if byte & (1 << j):
		    yield i
		i += 1

    def listUnsetBits(self):
	"""Returns an iterator of the indices of all unset bits in the
	   `bitvector`"""
	i=0
	for byte in self._blob:
	    for j in xrange(8):
		if i >= self._size:
		    break
		if not (byte & (1 << j)):
		    yield i
		i += 1

    @classmethod
    def fromSetBits(cls, bits, width=None):
	"""Constructs a `bitvector` from an iterable of indices of bits to set.
	   If a width is given, indices greater than or equal to `width` will
	   be discarded.  If a width is not given, the width of the resulting
	   `bitvector` will be the largest index plus 1.

	   If a negative index is encountered, `width` is added to it first; if
	   the value is still negative or if `width` is `None`, a `ValueError`
	   is raised."""
	bytes = defaultdict(int)
	maxB = 0
	for b in bits:
	    if b < 0 and width is not None:
		b += width
	    if b < 0:
		raise ValueError('negative index')
	    (byte, offset) = divmod(b,8)
	    bytes[byte] |= 1 << offset
	    maxB = max(b+1, maxB)
	obj = cls()
	if width is not None:
	    if width < 0: raise ValueError('negative width')
	    maxB = width
	if maxB > 0:
	    obj._blob = array('B', [bytes[i] for i in range((maxB+7) // 8)])
	    obj._size = maxB
	    if width is not None and width % 8 != 0:
		obj._blob[width//8] &= (1 << (width % 8)) - 1
	return obj

    def rstrip(self, val=False):
	"""Removes leading zero bits from the `bitvector`, that is, zero bits
	   starting at index 0.  The `bitvector` is modified in place.  If
	   `val` is `True`, one-bits are removed instead."""
	val = bool(val)
	for (i,b) in enumerate(self._blob):
	    if b != (0xFF if val else 0):
		for j in xrange(8):
		    if bool(b & (1 << j)) != val:
			break
		del self[:i*8+j]
		return
	self.clear()

    def lstrip(self, val=False):
	"""Removes trailing zero bits from the `bitvector`, that is, zero bits
	   starting at index `len(self)-1`.  The `bitvector` is modified in
	   place.  If `val` is `True`, one-bits are removed instead."""
	val = bool(val)
	for i in xrange(len(self._blob)-1, -1, -1):
	    b = self._blob[i]
	    if i == len(self._blob)-1 and self._size % 8 != 0 and val == True:
		b = (b | (0xFF << (self._size % 8))) & 0xFF
	    if b != (0xFF if val else 0):
		for j in xrange(7, -1, -1):
		    if bool(b & (1 << j)) != val:
			break
		self.setWidth(i*8+j+1)
		return
	self.clear()

    def strip(self, val=False):
	self.lstrip(val)
	self.rstrip(val)


def revbyte(b):  # internal helper function
    b2 = 0
    for i in xrange(8):
	b2 <<= 1
	if b & 1: b2 |= 1
	b >>= 1
    return b2
