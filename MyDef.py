import os
import re
import sys
import time
import enum
import ctypes
import serial
import random
import logging
import platform
import getpass
import subprocess
import urllib.request
import requests
# import unittest
import shutil
import json
import glob
import tkinter as tk
import docx
import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
import collections
import csv
import pandas as pd
import multiprocessing as mp
import datetime
import pytz
import arrow
import traceback
import itertools

# region Set Looger
'''
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
DisplayHandle = logging.StreamHandler()
DisplayHandle.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(name)s][%(levelname)s]%(message)s")
DisplayHandle.setFormatter(formatter)
logger.addHandler(DisplayHandle)
'''
# endregion

# region Base Function
def leto(times):
    leto = []
    letodic = {}
    for i in range(times):
        random.seed()
        s = set() #建立一個空的集合
        while len(s)<6:
            s.add(random.randint(1,49)) #隨機增加數字到s集合內，因為集合的特性有獨一性，數字若重覆將會取消新增，直到增加到6個不同的數字為止
        # print ("大樂透中獎號碼：",s)
        leto.append(sorted(s))
    for x in range(49):
        letodic[str(x+1)] = 0
    for le in leto:
        for l in le:
            if str(l) in letodic.keys():
                letodic[str(l)] = letodic[str(l)] + 1
    return leto, letodic

def leto2(times):
    let = []
    for i in range(times):
        random.seed()
        s = set()
        while len(s)<6:
            s.add(random.randint(1,49))
        s = sorted(s)
        if s in let:
            print ("大樂透中獎號碼：",s)
        else:let.append(s)
    print(len(let))
    print('\n\n')
    for x in let:
        print(x)

def list2split(_tosplit,spNum):
    # _tosplit is want list or str.. / spNum is how step to split
    sp = [_tosplit[i:i+spNum] for i in range(0,len(_tosplit),spNum)]
    # or as below
    ssp = []
    for i in range(0,len(_tosplit),spNum):
        ssp.append(_tosplit[i:i+spNum])

def byte2int(_byte):
    x = int.from_bytes(_byte,byteorder='big')
    # or as below
    y = int(bytes.hex(_byte),16)
    return x,y

def byte2bytehex(_byte):
    # to str
    s = bytes.decode(_byte)
    # to bytehex
    s = bytes.fromhex(s)
    # or as below
    return bytes.fromhex(bytes.decode(_byte))

def str2byte():
    b = b"example"
    s = "example"
    bytes(s, encoding = "utf8")  # str to bytes
    str(b, encoding = "utf-8")  # bytes to str
    str.encode(s)   # an alternative method str to bytes
    bytes.decode(b)   # bytes to str

def conver():
    hex(97)              # '0x61',Int to Hex
    chr(97)              # 'a'   ,Int to Char
    str(97)              # '97'  ,Int to String
    int('0x61', 16)      # 97    ,Hex to int
    chr(int('0x61', 16)) # 'a'   ,Hex to Char
    ord('a')             # 97    ,Char to Int
    hex(ord('a'))        # '0x61',Char to Hex
    int('97')            # 97    ,String to Int
    string = '61626364'
    ''.join(chr(int(string[i:i+2], 16)) for i in range(0, len(string), 2))  # 'abcd' ,Hex to String
    string = 'abcd'
    ''.join([hex(ord(x))[2:] for x in string])  # '61626364'  ,String to Hex

def Reverse(_torever):
    return _torever[::-1]

def average(_listNum):
    # using for loop
    n = 0
    total = 0
    for x in _listNum:
        n += 1
        total += x
    avg1 = total / n
    # using while loop
    n = len(_listNum)
    i = 0
    total = 0
    while i < 5:
        total += _listNum[i]
        i += 1
    avg2 = total / n
    return avg1  # or avf2

def basicLoop(_loopcount):
    while _loopcount > 0:
        # 迴圈工作區
        print(_loopcount)
        _loopcount -= 1 # 調整控制變數值

    #for loop
    for i in range(_loopcount, 0, -1): #range(第一個參數為起始值，第二個參數為結束值，第三個參數為遞增值)
        print(i)

def transferParameter():
    #intValue = int(sys.argv[1])#如果要將變數搞成數字的話可以使用 int()來轉
    print(sys.argv[0])          # *.py也算一個參數
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print("===============")
    print(len(sys.argv))#參數一共有幾個
    print("===============")
    for x in sys.argv:
        print(x)

def foundSyspath():
    return sys.path[0]

def removeDuplicates(nums) -> int:
        if len(nums) <= 1:
            return len(nums)

        s = 0

        for f in range(1, len(nums)):
            if nums[s] != nums[f]:
                s += 1
                nums[s] = nums[f]
        return s + 1

def removeElement(nums, val: int) -> int:
    count = 0
    for num in nums:
        if num != val:
            count+=1
    return count

def folder_compare(path_a,path_b,out_diff=None,out_a=None,out_b=None,):
    alist , blist = [],[]
    for root, dirs, files in os.walk(path_a):
        alist.extend(files)

    for root, dirs, files in os.walk(path_b):
        blist.extend(files)

    sa = set([os.path.splitext(i)[0] for i in alist])
    sb = set([os.path.splitext(i)[0] for i in blist])

    diff = []

    for i in sa.difference(sb):
        diff.append(i)


    for i in sb.difference(sa):
        diff.append(i)

    sd = set(diff)

    if out_diff:
        print('the difference as below:')
        for i in sd:
            print(i)

    if out_a:
        print('the {0} files as below'.format(path_a))
        for i in alist:
            print(i)

    if out_b:
        print('the {0} files as below'.format(path_b))
        for i in blist:
            print(i)

def fun():#fun()()()
    print("this is fun")
    def _fun():
        print("this is _fun")
        def __fun():
            print("this is __fun")
        return __fun
    return _fun

def decorateApple(f):
    def d_f(*args, **kargs):
        print("apple before call")
        result = f(*args, **kargs)
        print("apple after call")
        return result
    return d_f

@decorateApple # ==> hello = decorateApple(print_hello)
def hello():
    print("hello first time.")

def cleanup_function():
    print(sys._getframe().f_code.co_name)
    #program run as below
    '''
    print('registering')
    atexit.register(cleanup_function)
    print('registered')
    '''

def LogSample():
    logger = logging.getLogger("simple_example")
    logger.setLevel(logging.DEBUG)
    # 建立一個filehandler來把日誌記錄在文件裏，級別為debug以上
    fh = logging.FileHandler("spam.log")
    fh.setLevel(logging.DEBUG)
    # 建立一個streamhandler來把日誌打在CMD窗口上，級別為error以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 設置日誌格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    #將相應的handler添加在logger對象中
    logger.addHandler(ch)
    logger.addHandler(fh)
    # 開始打日誌
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")

def MakeList(num):
    return list(range(num))

def HciReset(comNumStr):
    com = serial.Serial(comNumStr, 115200, timeout=8, parity=serial.PARITY_NONE)
    com.write(b'\x01\x03\x0C\x00')
    rev = com.read(7)
    print(rev.hex().upper())
    com.close()

def isVampire(x,y):
    multiply = str(x*y)
    xylist = [i for i in str(x)] + [i for i in str(y)]
    if len(xylist) != len(multiply):
        return False
    else:
        for fang in xylist:
            if fang not in multiply:
                return  False
        return True

def factorial(num):
    ans = 1
    for i in range(1,(num+1)):
        ans*=i
    return ans

def Fibonacci_1(num):
    if num ==0 or num ==1:
        return num
    else:
        return Fibonacci_1(num-1) + Fibonacci_1(num-2)

def Fibonacci_2(num):
    if num == 0 or num == 1:
        return num
    else:
        x = 0 ; y = 1
        for i in range(num -1):
            tmp = y
            y = x + y
            x = tmp
        return y

def guessnumber():
    que = int(random.random() * 100) + 1
    min = 1
    max = 100
    while 1:
        ans = int(input('please input number {0} ~ {1}:'.format(min, max)))

        if ans >= min and ans <= max:
            if ans == que:
                print('yes')
                break

            if ans > que:
                max = ans
                print('too big')

            if ans < que:
                min = ans
                print('too small')

        else:
            print('out of range')

def matrix_print(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print('row            = matrix[{0}][{1}] : {2}'.format(i, j, matrix[i][j]))  # rows

            print('column         = matrix[{0}][{1}] : {2}'.format(j, i, matrix[j][i]))  # column

            if i == j:
                print('right diagonal = matrix[{0}][{1}] : {2}'.format(i, j, matrix[i][j]))  # right diagonal

        print('left diagonal  = matrix[{0}][{1}] : {2}'.format(i,len(matrix) - 1 - i,
                                                              matrix[i][len(matrix) - 1 - i]))  #left diagonal

def matrix_reverse(matrix):
    a = [list(i) for i in zip(*matrix)]
    # or
    b = [[row[col] for row in matrix]
          for col in range(len(matrix[0]))]
    '''
    b = []
    for c in range(len(matrix[0])):
        t = []
        for r in matrix:
            t.append(r[c])
        b.append(t)
    '''

    return a # or return b

def generateSquare(n):
        # 2-D array with all
        # slots set to 0
        magicSquare = [[0 for x in range(n)]
                       for y in range(n)]

        # initialize position of 1
        i = n / 2
        j = n - 1

        # Fill the magic square
        # by placing values
        num = 1
        while num <= (n * n):
            if i == -1 and j == n:  # 3rd condition
                j = n - 2
                i = 0
            else:

                # next number goes out of
                # right side of square
                if j == n:
                    j = 0

                # next number goes
                # out of upper side
                if i < 0:
                    i = n - 1

            if magicSquare[int(i)][int(j)]:  # 2nd condition
                j = j - 2
                i = i + 1
                continue
            else:
                magicSquare[int(i)][int(j)] = num
                num = num + 1

            j = j + 1
            i = i - 1  # 1st condition

        # Printing magic square
        print("Magic Squre for n =", n)
        print("Sum of each row or column",
              n * (n * n + 1) / 2, "\n")

        for i in range(0, n):
            for j in range(0, n):
                print('%2d ' % (magicSquare[i][j]),
                      end='')

                # To display output
                # in matrix form
                if j == n - 1:
                    print()

def creat_new_excel():
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.worksheets[0]
    ws.title = 'mySheet'
    wb.save('test.xlsx')

def cityTime(city_name):
    if type(city_name) != str:
        return 'input type not string'

    city = city_name.title()

    city_timezone = None

    for country in pytz.country_timezones:
        for tz in pytz.country_timezones(country):
            if city in tz:
                city_timezone = pytz.timezone(tz)

    if not city_timezone:
        return 'The {0} timezone not found ...'.format(city)

    return datetime.datetime.now(city_timezone)

def files_collections(pathA=None,pathB=None,diff=None):
    A_path , B_path = [] , []
    if pathA:
        for files in os.walk(pathA):
            A_path.extend(files[2])

    if pathB:
        for files in os.walk(pathB):
            B_path.extend(files[2])

    if diff:
        if len(A_path) >= len(B_path):
            print(pathA)
            for Afiles in A_path:
                if Afiles not in B_path:
                    print(Afiles)
        else:
            print(pathB)
            for Bfiles in B_path:
                if Bfiles not in A_path:
                    print(Bfiles)

def error_message():
    cl, exc, tb = sys.exc_info()
    for lastCallStack in traceback.extract_tb(tb):
        errMessage = ''.join(['\n######################## Error Message #############################\n'
                              '    Error class        : {}\n'.format(cl),
                              '    Error info         : {}\n'.format(exc),
                              '    Error fileName     : {}\n'.format(lastCallStack[0]),
                              '    Error fileLine     : {}\n'.format(lastCallStack[1]),
                              '    Error fileFunction : {}'.format(lastCallStack[2])])
        print(errMessage)
# endregion

# region Class group
class testClassOverride:
    def __init__(self,name,data):
        self._name = name
        self.data = data

    def kk(self,x,y):
        return x+y
    @property
    def name(self):
        return self._name

    # getattr(testClassOverride('aaa',123),'kk')(2,4) >>> 6
    # if hasattr(testClassOverride,'kk'): >>> True

class overridetest(testClassOverride):
    def __init__(self,name,data):
        testClassOverride.__init__(self,name,data)
        # or  -> super().__init__(name,data)

    def kk(self,x,y):
        return x*y

class ListNode:
    def __init__(self, data):
      self.val = data
      self.next = None

class SingleLinkedList:
    def __init__(self):
      self.head = None
      self.tail = None

    def add_list_item(self, item):
        if not isinstance(item, ListNode):
            item = ListNode(item)
        if self.head is None:
            self.head = item
        else:
            self.tail.next = item
        self.tail = item

class magic_square:
    def sub_square(self, grid):
        matrix = []
        '''
        for x in range(len(grid) - 2):
            for y in range(len(grid[0]) - 2):
                sub_matrix = []
                tempmatrix.append(grid[x][y:y + 3])
                tempmatrix.append(grid[x + 1][y: y + 3])
                tempmatrix.append(grid[x + 2][y: y + 3])
                matrix.append(sub_matrix)
        '''

        for row in range(len(grid) - 2): # -2 : mean is want n*n sub matrix range(len(grid) - (n-1))
            for col in range(len(grid[0]) - 2): # -2 : mean is want n*n sub matrix range(len(grid) - (n-1))
                # i and j mean is want n*n sub matrix range(n)
                sub_matrix = [[grid[row + i][col + j] for j in range(3)] for i in range(3)]
                matrix.append(sub_matrix)

        return matrix

    def is_magic_square(self, matrix):
        '''
        #----------检查条件1-9
        digit = 0 for i in range(0, 10)]

        for row in ma:
            for d in row:
                if d > 9:
                    return False
                digit[d] = 0

        for i in range(1, 10):
            if digit[i] != 0:
                return False

        #----检查每一行
        s = sum(ma[0])

        for i in range(1, 3):
            if sum(ma[i]) != s:
                return False

        #---检查对角线
        sdia = 0
        for i in range(0, 3):
            sdia +=ma[i][i]
        if sdia != s:
            return False

        sdia -= ma[2][0] + ma[1][1] + ma[0][2]
        if sdia != 0:
            return False

        #---检查每一列
        for j in range(0, 3):
            sc = 0
            for i in range(0,3):
                sc += ma[i][j]
            if sc != s:
                return False

        return True
        '''

        is_number_right = all(1 <= matrix[i][j] <= 9 for i in range(3) for j in range(3)) # i and j mean is want n*n sub matrix range(n)
        is_row_right = all(sum(row) == 15 for row in matrix) # == 15 mean is magic square sum() = n(n^2+1)/2
        is_col_right = all(sum(col) == 15 for col in [[matrix[i][j] for i in range(3)] for j in range(3)]) # i and j mean is want n*n sub matrix range(n)
        is_diagonal_right = matrix[1][1] == 5 and matrix[0][0] + matrix[-1][-1] == 10 and matrix[0][-1] + \
                            matrix[-1][0] == 10 # 5 is in matrix central and sum corner == 10
        is_repeat_right = len(set(matrix[i][j] for i in range(3) for j in range(3))) == 9  # i and j mean is want n*n sub matrix range(n)
        return is_number_right and is_row_right and is_col_right and is_diagonal_right and is_repeat_right

def reverseList(head):
    prev = None

    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev

# endregion

def insert(intervals, newInterval):
    '''
    :param intervals: List[List[int]]
    :param newInterval: List[int]
    :return: List[List[int]]
    '''
    import bisect
    if not intervals: return [newInterval]

    n = len(intervals)
    start_list = [x[0] for x in intervals]
    end_list = [x[1] for x in intervals]
    start = newInterval[0]
    end = newInterval[1]

    i = bisect.bisect_left(end_list,start)
    if i == n or end < intervals[i][0]:
        return intervals[:i] + [[start,end]] + intervals[i:]

    start = min(start,intervals[i][0])
    end = max(end,intervals[i][1])

    j = i + bisect.bisect_right(start_list[i:],end)
    end = max(end,intervals[j-1][1])

    return intervals[:i] + [[start,end]] + intervals[j:]

def isPossibleDivide(nums, k):
    '''
    :param nums: List[int]
    :param k: int
    :return: bool
    '''
    if len(nums) % k != 0:
        return False

    cnt = collections.Counter(nums)

    nums.sort()

    for n in nums:
        t = cnt[n]

        if t:
            for i in range(n, n + k):
                if cnt[i] < t:
                    return False
                cnt[i] -= t

    return True
# region Tample

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def divide(dividend, divisor):
    '''
    :param dividend: int
    :param divisor: int
    :return: int
    '''
    flag = (dividend < 0) ^ (divisor < 0)
    dividend, divisor = abs(dividend), abs(divisor)
    ans = 0
    while divisor <= dividend:
        temp = 1
        div = divisor
        while (div << 1) <= dividend:
            div <<= 1
            temp <<= 1
        dividend -= div
        ans += temp
        if ans >= 0x7fffffff:
            if flag and ans == 0x80000000:
                return -0x80000000
            return 0x7fffffff
    return ans if not flag else -ans

def matrixReshape(nums, r, c):
    '''
    :param nums: List[List[int]]
    :param r: int
    :param c: int
    :return: List[List[int]]
    '''
    a = sum(nums, [])
    if (len(a) != (r * c)):
        return nums
    res = [[0] * c for _ in range(r)]
    for i in range(0, len(a)):
        res[i // c][i % c] = a[i]
    return res

def coinChange(coins, amount):
    '''
    coins = [2,5,7] , amount = 27
    :param coins: list[int]
    :param amount: int
    :return:
    '''
    rs = [float('inf')] * (amount+1)
    rs[0] = 0
    for i in range(1, amount+1):
        for c in coins:
            if i >= c:
                rs[i] = min(rs[i], rs[i-c] + 1)

    if rs[amount] == float('inf'):
        return -1

    return rs[amount]

def longestArithSeqLength(A):
    '''
    :param A: List[int]
    :return:  int
    '''
    n = len(A)
    if n == 0:
        return 0

    res = 0

    tmp = [{} for i in range(n)]

    for i in range(n):
        for j in range(0, i):
            diff = A[i] - A[j]

            if diff in tmp[j]:
                tmp[i][diff] = tmp[j][diff] + 1

            else:
                tmp[i][diff] = 2

            res = max(res, tmp[i][diff])

    return res

class Solution:
    def __init__(self):
        pass

    def rangeSumBST(self, root, L, R):
        '''
        :param root: TreeNode
        :param L: int
        :param R: int
        :return: int
        '''
        self.total = 0

        def dfs(node, L, R):
            if not node:
                return 0

            if L <= node.val <= R:
                print(node.val)
                self.total += node.val

            dfs(node.left, L, R)

            dfs(node.right, L, R)

        dfs(root, L, R)

        return self.total

def buddyStrings(A, B):
    '''
    :param A: str
    :param B: str
    :return: bool
    '''
    if not A or len(A) != len(B):
        return False

    diff = []

    A = list(A)

    B = list(B)

    for i in range(len(A)):
        if A[i] != B[i]:
            diff.append(i)

    if len(diff) > 2:
        return False

    A[diff[0]], A[diff[1]] = A[diff[1]], A[diff[0]]

    print(A)
    print(B)

    if A != B:
        return False

    return True

# endregion

def findUnsortedSubarray(nums):
    '''
    :param nums: List[int]
    :return: int
    '''
    si = ei = 0
    ls = len(nums)

    # Find Start Index
    for i in range(ls - 1):
        if nums[i] > nums[i + 1]:
            si = i
            break

    # Find End Index
    for i in range(ls - 1, 0, -1):
        if nums[i] < nums[i - 1]:
            ei = i
            break

    '''Check if si is really si or not [1,3,4,{2,8,9,12,9}, 11] : 
       si points to 2 which is not right'''

    min_val = min(nums[si:ei + 1])

    for i in range(si):
        if nums[i] > min_val:
            si = i
            break

    '''If we sort {2<->9} still it's not answer is not correct as 12 is greater than 11 so we 
       need to find orginal End index'''
    max_val = max(nums[si:ei + 1])
    for i in range(ls - 1, si, -1):
        if nums[i] < max_val:
            ei = i
            break

    return (ei - si + 1) if ei - si > 0 else 0

def can_segment_string(s, dictionary):
    for i in range(len(s) - 1):
        print(s[0:i+1],s[i+1:])

        if s[0:i+1] in dictionary and s[i:] in dictionary:
            return True

    return False

if __name__ == '__main__':
    try:
        # t1 = TreeNode(1,TreeNode(3,left=TreeNode(5)),TreeNode(2))
        # t2 = TreeNode(2, TreeNode(1, right=TreeNode(5)), TreeNode(3, right=TreeNode(7)))
        '''
        flowerbed = [1, 0, 0, 0, 1]
        n = 1
        
        ans, start, left = 0, -1, -2
        flowerbed += [0, 1]

        for i in (flowerbed):
            start += 1
            if i == 1:
                ans += (start - left - 2) // 2
                left = start
                if ans >= n:
                    print( True)

        print(False)
        '''
        '''
        mat =[[1, 1, 0, 0, 0],
              [1, 1, 1, 1, 0],
              [1, 0, 0, 0, 0],
              [1, 1, 0, 0, 0],
              [1, 1, 1, 1, 1]]
        k = 3


        d = {i:sum(mat[i]) for i in range(len(mat))}

        print([i[0] for i in sorted(d.items(),key=lambda item:item[1])[:k]])
        '''
        '''
        matrix = [[4,3,8,4],
                  [9,5,1,9],
                  [2,7,6,2]]
        '''

        '''
        flowerbed = [1, 0, 0, 0, 1]
        n = 1

        l = [0] + flowerbed + [0]

        for i in range(1, len(l) - 1):
            print(l[i - 1:i + 2])
            if sum(l[i - 1:i + 2]) == 0:
                l[i] = 1
                n -= 1
        '''

        n = 257761

        m = list(str(n))  ## n = 257761
        l = len(m)  ## l = 6
        d = {}
        res = str(n)
        for i, c in enumerate(m[::-1]):
            if not d:
                d[c] = 1
            else:
                if all(c >= x for x in d):
                    d[c] = d.get(c, 0) + 1
                else:
                    d[c] = d.get(c, 0) + 1
                    res = ''.join(m[:l - 1 - i])
                    stock = sorted(list(d.keys()))
                    cplus = stock[stock.index(c) + 1]
                    res += cplus
                    d[cplus] -= 1
                    res += ''.join([x * d[x] for x in stock])

                    break

        print(int(res)) if n < int(res) < (2 ** 31 - 1) else -1

    except:
        error_message()
