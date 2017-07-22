from __future__ import division
import math as m
from sklearn.preprocessing import Normalizer
a = [301,448,305,309,522,650,490,300,274,575,530,363,409,479,445,410]
b = [8,18,5,3,11,3,9,10,7,20,22,14,10,15,13,9]
c = [4,8,9,6,10,13,5,7,12,15,9,6,8,7,11,5]

h = Normalizer(copy=False)
X = h.fit_transform(a)
Y = h.fit_transform(b)
Z = h.fit_transform(c)
print ('X is ',X)
print("")
print ('Y is ',Y)
print("")
print ('Z is',Z)
print(max(a),min(a),max(b),min(b),max(c),min(c))
# print(m.log(2/7,2))
def callog(x):
 a = -x*m.log(x,2)
 return a

result = (callog(2/7)+callog(3/7)+callog(2/7))*7
result1 = 8*(callog(4/8)+callog(1/8)+callog(3/8))
# print (result)
# print (result1)
# print(result+result1)
print ((result+result1)/15)
result2 = (8*(callog(4/8)+callog(1/8)+callog(3/8))+(callog(2/7)+callog(3/7)+callog(2/7))*7)/15
print(result2)
result3 =((callog(3/5)+callog(2/5))*5+(callog(1/3)+callog(1/3)+callog(1/3))*6+
(callog(1/4)+callog(1/2)+callog(1/4))*4)/15
print(result3)
result4 = (7*(callog(2/7)+callog(4/7)+callog(1/7))+(callog(4/8)+callog(4/8))*8)/15
print(result4)
result5 = (7*(callog(2/7)+callog(4/7)+callog(1/7))+(callog(5/8)+callog(3/8))*8)/15
print(result5)
result6 = callog(6/15)+callog(4/15)+callog(5/15)
print(result6)
result7 = (callog(2/4)+callog(2/4))*4/4
print(result7)
result8 = (2*(callog(1/2)+callog(1/2))+(callog(4/6)+callog(2/6))*6)/8
print(result8)
result9 = callog(4/7)+callog(2/7)+callog(1/7)
print(result9)
