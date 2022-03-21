import os
import sys
import random
import logging
import unittest
import traceback
import importlib

# region Non using module
'''
import re
import time
import enum
import ctypes
import tkinter.filedialog
import serial
import platform
import getpass
import subprocess
import urllib.request
import requests

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
import itertools
'''
# endregion

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

# region Class group
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def link_list(nums):
    '''
    :param nums: list[int]
    :return: link list
    '''
    head = ListNode(nums[0])

    link = head

    for i in nums[1:]:
        head.next = ListNode(i)
        head = head.next

    return link

def reverseList(head):
    prev = None

    while head:
        current = head
        head = head.next
        current.next = prev
        prev = current
    return prev

# endregion

# region Base Function
def gcd(a, b):
    '''
    :param a: int
    :param b: int
    :return: int
    this function is greatest common divisor
    equal
    GCD = lambda a, b: GCD(b, a % b) if a % b else b
    '''
    if a % b:
        return gcd(b, a % b)
    else:
        return b

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

def foundSyspath():
    return sys.path[0]

def folder_compare(path_a, path_b, out_diff=None, out_a=None, out_b=None, ):
    alist, blist = [], []
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
    new_matrix = []

    for col in range(len(matrix[0])):
        cell = []

        for row in matrix:
            cell.append(row[col])

        new_matrix.append(cell)

    return new_matrix

def matrix_rotate_180_degree(matrix):
    ''' rotate in place'''
    head = 0
    tail = len(matrix) - 1

    while head <= tail:
        if head == tail:
            matrix[head] = matrix[head][::-1]

        else:
            matrix[head] = matrix[head][::-1]
            matrix[tail] = matrix[tail][::-1]
            matrix[head], matrix[tail] = matrix[tail], matrix[head]

        head += 1
        tail -= 1

        return matrix

def matrix_rotate_left_90_degree(matrix,left = None):
    '''
    matrix = [[ 1, 2, 3, 4, 5],
              [ 6, 7, 8, 9,10],
              [11,12,13,14,15],
              [16,17,18,19,20],
              [21,22,23,24,25]]

    matrix = [[1,2,3],
              [4,5,6],
              [7,9,8]]
    '''
    l = len(matrix)

    if left :
        for i in range(len(matrix)):
            matrix[i] = matrix[i][::-1]

    for i in range(l):
        for j in range(i + 1, l):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        if not left:
            matrix[i] = matrix[i][::-1]

    return matrix

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

def dynamic_load_lib(libpath):
    if not os.path.exists(libpath):
        return f'{libpath} not exist , please make sure the lib path'

    path_spilt = os.path.splitext(libpath)[0].split('\\')

    os.chdir('\\'.join(path_spilt[:-1]))

    return importlib.import_module(path_spilt[-1])

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

def error_messages_display():
    cl, exc, tb = sys.exc_info()
    for lastCallStack in traceback.extract_tb(tb):
        errMessage = (f'\n######################## Error Message #############################\n'
                      f'    Error class        : {cl}\n'
                      f'    Error info         : {exc}\n'
                      f'    Error fileName     : {lastCallStack[0]}\n'
                      f'    Error fileLine     : {lastCallStack[1]}\n'
                      f'    Error fileFunction : {lastCallStack[2]}')
        print(errMessage)
# endregion

if __name__ == '__main__':
    '''
    root = TreeNode(val   = 0,
                    left  = TreeNode(val   = 1,
                                     left  = TreeNode(val   = 3,
                                                      left  = TreeNode(val = 6),
                                                      right = TreeNode(val = 7)
                                                      ),
                                     right = TreeNode(val = 4)
                                     ),
                    right = TreeNode(val   = 2,
                                     right = TreeNode(val   = 5,
                                                      left  = TreeNode(val = 8),
                                                      right = TreeNode(val = 9)
                                                      )
                                     )
                    )

    res = []
    q1 = [root]
    q2 = []

    while q1:
        node = q1.pop()
        q2.append(node)

        if node:
            if node.left:
                q1.append(node.left)

            if node.right:
                q1.append(node.right)

    while q2:
        node = q2.pop()
        res.append(node.val)


    print(res)


    d ={1:0}
    d.removed
    '''
