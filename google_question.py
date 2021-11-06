import os
import sys
import time
import random
from itertools import dropwhile
from functools import reduce

def valid_parentheses(string):
    parentList = []

    for parent in string:
        if parent == '(' or parent == ')':
            parentList.append(parent)

    if len(parentList) == 0:
        return True

    if len(parentList) != 0 and len(parentList)%2 != 0:
        return False
    else:
        stack = []
        while len(parentList) != 0:
            startChar = parentList[0]
            parentList = parentList[1:]
            if startChar == '(':
                stack.append(startChar)
            elif startChar == ')':
                if len(stack) == 0 or stack[-1] != '(':
                    return False
                else:
                    stack.pop(-1)
        if len(stack) == 0:
            return True
        else:
            return False

def vlaid(st):
    cnt = 0
    for char in st:
        if char == '(': cnt += 1
        if char == ')': cnt -= 1
        if cnt < 0: return False
    return True if cnt == 0 else False

#has calculation
def zeross(n):
    znum = 0
    number = 1
    while n != 0:
        number = number*n
        n -= 1
    number = list(str(number))[::-1]
    for i in number:
        if i != '0':
            break
        else:
            znum += 1
    return znum

#no calculation
def zeros(n):
  x = n // 5
  return x+zeros(x) if x else 0

class Node:
    # a = (Node(Node(None, Node(None, None, 4), 2), Node(Node(None, None, 5), Node(None, None, 6), 3), 1))
    def __init__(self, L, R, n):
        self.left = L
        self.right = R
        self.value = n

def preorder(root):
    global res
    if root:
        res.append(root.value)
        preorder(root.left)
        preorder(root.right)

def ineorder(root):
    global res
    if root:
        preorder(root.left)
        res.append(root.value)
        preorder(root.right)

def tree_by_levels(node):
    ''' 別人的
    queue = [tree]
    values = []
    while queue:
        node = queue.pop(0)
        if node:
            queue += [node.left, node.right]
            values.append(node.value)
    return values
    '''
    que = [] ; dis = []
    if node == None:return que
    else:
        que.append(node)
        dis.append(node.value)
        while len(que) > 0:
            root = que.pop(0)
            if root.left != None:
                que.append(root.left)
                dis.append(root.left.value)
            if root.right != None:
                que.append(root.right)
                dis.append(root.right.value)
        return dis

class BigInt: # longlong operation
    def __init__(self, val):
        self.value = BigInt.parse(val) if isinstance(val, str) else val

    def __str__(self):
        v = BigInt.toComplement(self.value) \
            if BigInt.isNegative(self.value) else self.value
        builder = ['%04d' % v[i] for i in range(len(v) - 1, -1, -1)]
        clist = list(dropwhile(lambda c: c == '0', list(''.join(builder))))
        return '0' if len(clist) == 0 else ''.join(
            ((['-'] + clist) if BigInt.isNegative(self.value) else clist))

    def __add__(self, that):
        return (self - BigInt(BigInt.toComplement(that.value))) \
            if BigInt.isNegative(that.value) else self.add(that)

    def add(self, that):
        length = max(len(self.value), len(that.value))
        op1 = BigInt.copyOf(self.value, length)
        op2 = BigInt.copyOf(that.value, length)
        sum = BigInt.addForEach(op1, op2, 0)
        return BigInt(
            (((sum[0:-1] + [1]) if BigInt.isPositive(op1) else []) + [0] * 8)
            if sum[-1] == 1
            else (sum[0:-1] + [0 if BigInt.isPositive(op1) else 9999])
        )

    def __sub__(self, that):
        return (self + BigInt(BigInt.toComplement(that.value))) \
            if BigInt.isNegative(that.value) else self.sub(that)

    def sub(self, that):
        length = max(len(self.value), len(that.value))
        op1 = BigInt.copyOf(self.value, length)
        op2 = BigInt.copyOf(that.value, length)
        remain = BigInt.subForEach(op1, op2, 0)
        return BigInt(
            ((remain[0:-1] + [9998] if BigInt.isNegative(op1) else [])
             + [9999] * 8) if remain[-1] == 1
            else (remain[0:-1] +
                  [9999 if BigInt.isNegative(op1) else 0])
        )

    def multiply(self, val, shift):
        product = [0] * shift + \
                  BigInt.multiplyForEach(self.value, val, 0)
        return BigInt((product[0:-1] + product[-1:] + [0] * 8) \
                          if product[-1] != 0 else (product[0:-1] + [0]))

    def __mul__(self, that):
        op1 = BigInt(BigInt.toComplement(self.value)) \
            if BigInt.isNegative(self.value) else self
        op2 = BigInt.toComplement(that.value) \
            if BigInt.isNegative(that.value) else that.value
        result = reduce(BigInt.__add__,
                        [op1.multiply(op2[i], i)
                         for i in range(len(op2) - 1)], BigInt('0'))
        return BigInt(BigInt.toComplement(result.value)) \
            if self.value[-1] + that.value[-1] == 9999 else result

    def __ge__(self, that):
        return False if BigInt.isNegative((self - that).value) else True

    def isLessOrEqualsQuotient(self, op1, op2):
        return True if op1 >= (self * op2) else False

    def __floordiv__(self, that):
        op1 = BigInt(BigInt.toComplement(self.value)) \
            if BigInt.isNegative(self.value) else self
        op2 = BigInt(BigInt.toComplement(that.value)) \
            if BigInt.isNegative(that.value) else that
        one = BigInt('1')

        def quotient(left, right):
            if right >= left:
                x = (left + right).divide(2)
                l, r = ((x + one, right)
                        if x.isLessOrEqualsQuotient(op1, op2)
                        else (left, x - one))
                return quotient(l, r)
            else:
                return left - one

        result = quotient(BigInt('0'), op1)
        return BigInt(BigInt.toComplement(result.value)) \
            if self.value[-1] + that.value[-1] == 9999 else result

    @staticmethod
    def divideForEach(op, val, remain):
        if op == []:
            return []
        else:
            tmp = op[-1] + remain
            nextRemain = (tmp % val) * 10000
            return [tmp // val] + \
                   BigInt.divideForEach(op[0:-1], val, nextRemain)

    def divide(self, that):
        result = BigInt.divideForEach(self.value, that, 0)
        return BigInt(result[::-1] + [0] * (8 - (len(result) % 8)))

    @staticmethod
    def parse(val):
        v = val[1:] if val[0] == '-' else val
        digits = [int(v[i if i >= 0 else 0: i + 4])
                  for i in range(len(v) - 4, -4, -4)]
        zeros = [0] * ((len(digits) // 8 + 1) * 8 - len(digits))
        return BigInt.toComplement(digits + zeros) \
            if val[0] == '-' else (digits + zeros)

    @staticmethod
    def addForEach(op1, op2, carry):
        if op1 == []:
            return [carry]
        else:
            s = op1[0] + op2[0] + carry
            nextCarry, c = (0, s) if s < 10000 else (1, s - 10000)
            return [c] + BigInt.addForEach(op1[1:], op2[1:], nextCarry)

    @staticmethod
    def subForEach(op1, op2, borrow):
        if op1 == []:
            return [borrow]
        else:
            r = op1[0] - op2[0] - borrow
            nextBorrow, c = (0, r) if r > -1 else (1, r + 10000)
            return [c] + BigInt.subForEach(op1[1:], op2[1:], nextBorrow)

    @staticmethod
    def multiplyForEach(op, val, carry):
        if op == []:
            return [carry]
        else:
            tmp = op[0] * val + carry
            nextCarry = tmp // 10000
            return [tmp % 10000] + \
                   BigInt.multiplyForEach(op[1:], val, nextCarry)

    @staticmethod
    def toComplement(v):
        c = [9999 - i for i in v]
        return [c[0] + 1] + c[1:]

    @staticmethod
    def copyOf(original, newLength):
        return original + [0 if BigInt.isPositive(original) else 9999
                           for i in range(len(original), newLength)]

    @staticmethod
    def isNegative(list):
        return list[-1] == 9999

    @staticmethod
    def isPositive(list):
        return list[-1] == 0

def vampire_number(x,y):
    xy = str(x*y)
    print(xy)
    xyList = [i for i in str(x)] + [i for i in str(y)]
    print(xyList)
    if len(xy) != len(xyList):
        return False
    else:
        for i in xyList:
            if i not in xy:
                return False
            else:
                return True

def twoSum(nums, target):
    dic = dict()
    for index,value in enumerate(nums):
        sub = target - value
        if sub in dic:
            return [dic[sub],index]
        else:
            dic[value] = index

def singNum(nums):
    res = 0
    for i in range(32):
        cnt = 0
        mask = 1 << i
        for num in nums:
            if num & mask:
                cnt += 1
        if cnt % 3 == 1:
            res |= mask
    if res >= 2 ** 31:
        res -= 2 ** 32
    return res

if __name__ == '__main__':
    a = 'abc'
    print(list(a))
