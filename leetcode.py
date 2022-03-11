import collections
import sys
import time
import traceback
import bisect

# link list
from typing import List, Any

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        # t1 = TreeNode(1,TreeNode(3,left=TreeNode(5)),TreeNode(2))
        # t2 = TreeNode(2, TreeNode(1, right=TreeNode(5)), TreeNode(3, right=TreeNode(7)))

    def tree_by_levels(self,node):
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
        que = []
        dis = []
        if node == None:
            return que

        que.append(node)

        while que:
            root = que.pop(0)
            dis.append(root.value)

            if root.left:
                que.append(root.left)

            if root.right:
                que.append(root.right)

        return dis

    def preorder(self,root,res):
        '''
        if root:
            res.append(root.value)
            self.preorder(root.left,res)
            self.preorder(root.right,res)
        '''
        q = [root]

        while q:
            node = q.pop()
            res.append(node.val)

            if node.right:
                q.append(node.right)

            if node.left:
                q.append(node.left)


    def inorder(self,root,res):
        '''
        if root:
            self.inorder(root.left,res)
            res.append(root.value)
            self.inorder(root.right,res)
        '''
        q = []

        while 1:
            while root:
                q.append(root)
                root = root.left

            if not q:
                break

            root = q.pop()
            res.append(root.val)
            root = root.right

    def postorder(self,root,res):
        '''
        if root:
            self.inorder(root.left,res)
            self.inorder(root.right,res)
            res.append(root.value)
        '''
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

    def mergeTrees(self, t1, t2):
        '''
        :param t1: TreeNode
        :param t2: TreeNode
        :return:   TreeNode
        '''

        if t1 is None:
            return t2

        if t2 is None:
            return t1

        t1.val += t2.val

        t1.left = self.mergeTrees(t1.left, t2.left)

        t1.right = self.mergeTrees(t1.right, t2.right)

        return t1

class LeetCode_Easy:
    def twoSum(self,nums, target):
        '''
        :param nums: List[int]
        :param target: int
        :return: List[int]
        1. Two Sum
        '''
        d = dict()

        for index,value in enumerate(nums):
            if target - value in d:
                return [d[target - value],index]

            d[value] = index

    def reverse(self,x):
        '''
        :param x:
        :return:
        leetcode easy: 7. Reverse Integer
        '''
        s = str(abs(x))

        rev = int(s[::-1])

        if rev > 2 ** 31:
            return 0

        return rev if x > 0 else (rev * -1)

    def isPalindrome(self, x):
        '''
        :param x: int
        :return: bool
        leetcode easy: 9. Palindrome Number
        '''
        if x == 0:
            return True

        if x < 0:
            return False

        '''
        using covert to string
        if str(x) == str(x)[::-1] or x == 0:
            return True
        '''

        div = 1

        while x / div >= 10:
            div *= 10

        while x:
            first = x // div
            last = x % 10

            if first != last:
                return False

            x = (x - (div * last) - last)

            if x < 10:
                return True

            x /= 10
            div /= 100

    def romanToInt(self,s):
        '''
        :param s: str
        :return: int
        leetcode easy: 13. Roman to Integer
        '''
        roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

        res = roman[s[-1]]

        for i in range(len(s) - 1):
            if roman[s[i]] < roman[s[i+1]]:
                res -= roman[s[i]]

            else:
                res += roman[s[i]]

        return res

    def longestCommonPrefix(self,strs):
        '''
        :param strs: str
        :return: str
        leetcode easy: 14. Longest Common Prefix

        Input: strs = ["flower","flow","flight"]
        Output: "fl"
        '''

        # sort for strs list let shorter in left side
        for i in range(1,len(strs)):
            if len(strs[i-1]) > len(strs[i]):
                strs[i-1] , strs[i] = strs[i],strs[i-1]

        common = strs.pop(0)

        while strs:
            if len(common) == 0:
                return ""

            node = strs.pop(0)

            while 1:
                if len(common) == 0:
                    return ""

                if common == node[:len(common)]:
                    break

                common = common[:-1]

        return common

    def isValid(self,s):
        '''
        :param s: str
        :return: bool
        leetcode easy: 20. Valid Parentheses
        '''
        stack = list()
        d     = {'(': ')', '{': '}', '[': ']'}

        for c in s:
            if c in d:
                stack.append(c)
            else:
                if len(stack) == 0 or d[stack.pop()] != c:
                    return False

        return len(stack) == 0

    def mergeTwoLists(self, l1, l2):
        '''
        :param l1: Optional[ListNode]
        :param l2: Optional[ListNode]
        :return: Optional[ListNode]
        leetcode easy: 21. Merge Two Sorted Lists
        '''
        if not l1 or not l2:
            return l1 or l2

        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1

        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    def removeDuplicates(self,nums):
        '''
        inpit:[0,0,1,1,1,2,2,3,3,4]
        :param nums: List[int]
        :return:int
        leetcode easy: 26. Remove Duplicates from Sorted Array
        '''
        i = 0

        for j in range(len(nums)):
            if nums[j] != nums[i]:
                i += 1
                nums[i] = nums[j]

        return i + 1

    def removeElement1(self,nums, val):
        '''
        :param nums: List[int]
        :param val: int
        :return: int
        leetcode easy: 27. Remove Element
        '''
        while val in nums:
            for i in range(len(nums)):
                if nums[i] == val:
                    nums.pop(i)
                    break
        return len(nums)

    def removeElement2(self,nums, val):
        '''
        :param nums: List[int]
        :param val: int
        :return: int
        leetcode easy: 27. Remove Element
        '''
        index = 0
        while val in nums:
            if nums[index] == val:
                nums.pop(index)
                if index > 0:
                    index -= 1
            index+=1
        return len(nums)

    def strStr(self,haystack, needle):
        '''
        :param haystack: str
        :param needle: str
        :return: int
        leetcode easy: 28. Implement strStr()
        '''
        if len(needle) == 0:
            return 0
        return haystack.find(needle)

    def searchInsert(self,nums, target):
        '''
        Input: [1,3,5,6], 5
        Output: 2
        :param nums: list[int]
        :param target: int
        :return: int
        leetcode easy: 35. Search Insert Position
        '''
        lo, up = 0, len(nums) - 1
        mi = (lo + up) // 2
        while lo <= up:
            if nums[mi] == target:
                return mi
            elif nums[mi] > target:
                up = mi - 1
            else:
                lo = mi + 1
            mi = (lo + up) // 2
        return lo

    def countAndSay(self,n):
        '''
        :param n: int
        :return: str
        leetcode easy: 38. Count and Say
        '''
        res = "1"

        for i in range(n - 1):
            prev  = res[0]
            count = 1
            ans   = ""

            for j in range(1, len(res)):
                cur = res[j]

                if prev != cur:
                    ans = ans + str(count) + str(prev)
                    prev = cur
                    count = 0

                count += 1

            res = ans + str(count) + str(prev)

        return res

    def maxsunarry(self,nums):
        '''
        :param nums: int
        :return: int
        leetcode easy: 53. Maximum Subarray
        Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
        Output: 6
        Explanation: [4,-1,2,1] has the largest sum = 6.
        '''
        # it's non dynamic program solution
        max_ending = max_current = nums[0]

        for num in nums[1:]:
            max_ending  = max(num, max_ending + num)
            max_current = max(max_current, max_ending)

        return max_current

    def lengthOfLastWord(self,s):
        '''
        :param s: str
        :return: int
        leetcode easy: 58. Length of Last Word
        '''
        return len(s.rstrip().split(' ')[-1])

    def plusOne(self, digits):
        '''
        :param digits: List[int]
        :return: List[int]
        leetcode easy: 66. Plus One

        Input: digits = [4,3,2,1]
        Output: [4,3,2,2]
        Input: digits = [9]
        Output: [1,0]
        '''
        # solution of python built-in of str method
        # return [int(i) for i in str(int(''.join([str(n) for n in digits]))+1)]

        carry_over = 0

        digits[-1] += 1

        for i in range(len(digits) - 1, -1, -1):
            digits[i] += carry_over
            carry_over = digits[i] // 10
            digits[i] %= 10

        if carry_over != 0:
            digits.insert(0, carry_over)

        return digits

    def addBinary(self, a, b):
        '''
        :param a: str
        :param b: str
        :return: str
        leetcode easy: 67. Add Binary

        Input: a = "11", b = "1"
        Output: "100"

        Input: a = "1010", b = "1011"
        Output: "10101"
        '''
        # return bin(int(a, 2) + int(b, 2))[2:]

        '''
        Funtion of half adder

        x = int(a,2)
        y = int(b,2)                # string transform to int

        while y:                    # y represent carry bit
            value = x ^ y           # x xor y to get sum  
            carry = (x & y) << 1    # x and y to get carry , remember left shift 1 bit  

            x = value
            y = carry               

        return bin(x)[2:]
        '''

        a = list(a)
        b = list(b)

        carry = 0
        res = ''

        while a or b or carry:
            if a:
                carry += int(a.pop())

            if b:
                carry += int(b.pop())

            res += str(carry % 2)
            carry //= 2

        return res[::-1]

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        leetcode easy: 88. Merge Sorted Array
        """
        p1 = m - 1
        p2 = n - 1

        for p in range(n + m - 1, -1, -1):
            if p2 < 0:
                break

            elif p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1

            else:
                nums1[p] = nums2[p2]
                p2 -= 1

        '''
        # using build-in sort method
        for i in range(n):
            nums1[m+i] = nums2[i]
        nums1.sort()
        '''

    def isSameTree(self,p, q):
        '''
        Definition for a binary tree node.
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        :param p: TreeNode
        :param q: TreeNode
        :return:  bool
        leetcode easy: 100. Same Tree
        '''
        p_queue = [p]
        p_value = []
        q_queue = [q]
        q_value = []

        while p_queue:
            node = p_queue.pop(0)
            if node:
                p_queue += [node.left, node.right]
                p_value.append(node.val)
            else:
                p_value.append(None)

        while q_queue:
            node = q_queue.pop(0)
            if node:
                q_queue += [node.left, node.right]
                q_value.append(node.val)
            else:
                q_value.append(None)

        if p_value == q_value:
            return True
        else:
            return False

    def isSymmetric(self, root):
        '''
        :param root: Optional[TreeNode]
        :return: bool
        leetcode easy: 101. Symmetric Tree
        '''
        '''
        nums = [1,2,2,3,4,4,3,5,6,7,8,8,7,6,5]
        
        root = TreeNode(val   = 1,                        
                        left  = TreeNode(val   = 2,
                                         left  = TreeNode(val   = 3,
                                                          left  = TreeNode(5),
                                                          right = TreeNode(6)
                                                          ),
                                         right = TreeNode(val   = 4,
                                                          left  = TreeNode(7),
                                                          right = TreeNode(8)
                                                          )
                                         ),               
                        right = TreeNode(val   = 2,
                                         left  = TreeNode(val   = 4,
                                                          left  = TreeNode(8),
                                                          right = TreeNode(7)
                                                          ),
                                         right = TreeNode(val   = 3,
                                                          left  = TreeNode(6),
                                                          right = TreeNode(5)
                                                          )
                                         )
                        )
        '''

        def isMirror(left, right):
            '''
            :param left: TreeNode
            :param right: TreeNode
            :return: bool
            '''
            if not left and not right:
                return True

            elif not left or not right:
                return False

            elif left.val != right.val:
                return False

            else:
                return isMirror(left.left, right.right) and isMirror(left.right, right.left)

        if not root:
            return True

        return isMirror(root.left, root.right)

    def levelOrderBottom(self, root):
        '''
        :param root: TreeNode
        :return: List[List[int]]
        leetcode easy: 107. Binary Tree Level Order Traversal II
        '''
        if not root:
            return []

        q     = [[root]]
        value = [[root.val]]

        while q:
            node = q.pop(0)
            tmp_value = []
            tmp_node = []

            for sub in node:
                if sub.left:
                    tmp_value.append(sub.left.val)
                    tmp_node.append(sub.left)

                if sub.right:
                    tmp_value.append(sub.right.val)
                    tmp_node.append(sub.right)

            if tmp_value:
                value.append(tmp_value)

            if tmp_node:
                q.append(tmp_node)

        return value[::-1]

    def generate(self, numRows):
        '''
        :param numRows: int
        :return: List[List[int]]
        leetcode easy: 118. Pascal's Triangle
        '''
        pascals = []

        for i in range(numRows):
            row = [1] * (i + 1)

            for j in range(1, i):
                row[j] = pascals[i - 1][j - 1] + pascals[i - 1][j]

            pascals.append(row)

        return pascals

    def maxProfit(self, prices):
        '''
        Input: [7,1,5,3,6,4]
        Output: 5
        :param prices: List[int]
        :return: int
        leetcode easy: 121. Best Time to Buy and Sell Stock
        '''
        minPrice = prices[0]

        profit = 0

        for price in prices[1:]:
            if price < minPrice:
                minPrice = price

            if price - minPrice > profit:
                profit = price - minPrice

        return profit

    def isPalindrome_125(self,s):
        '''
        :param s:str
        :return:bool
        leetcode easy: 125. Valid Palindrome
        '''
        slist =[]

        for i in s:
            if i.isalpha() or i.isdigit():
                slist.append(i.lower())

        if slist == slist[::-1]:
            return True

        else:
            return False

    def singleNumber(self,nums):
        """
        :type nums: List[int]
        :rtype: int
        leetcode easy: 136. Single Number
        """
        res = nums[0]

        for num in nums[1:]:
            res ^= num

        return res

    def convertToTitle(self, n):
        '''
        :param n: int
        :return: str
        leetcode easy: 168. Excel Sheet Column Title
        '''
        dic = {}
        for i in range(1, 27):
            dic[i] = chr(i + 64)

        ans = ''

        while n > 26:
            x = n % 26

            if x != 0:
                ans = dic[n % 26] + ans

            else:
                ans = dic[26] + ans
                n -= 1

            n //= 26

        return dic[n] + ans

    def majorityElement(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode easy: 169. Majority Element

        Input: nums = [2,2,1,1,1,2,2]
        Output: 2
        '''
        nums.sort()
        return nums[len(nums)//2]

    def titleToNumber(self, columnTitle):
        '''
        :param columnTitle: str
        :return: int
        leetcode easy: 171. Excel Sheet Column Number

        Input: columnTitle = "A"
        Output: 1

        Input: columnTitle = "AB"
        Output: 28

        Input: columnTitle = "ZY"
        Output: 701
        '''
        '''
        res = 0
        for c in columnTitle:
            res *= 26
            res += (ord(c) - ord('A') + 1)

        return res
        '''

        total = 0

        for c, v in enumerate(columnTitle[::-1]):
            total +=  (ord(v) - 64) * (26**c)

        return total

    def trailingZeroes1(self,n):
        '''
        :param n: int
        :return: int
        leetcode easy: 172. Factorial Trailing Zeroes
        '''
        i = 5
        res = 0

        while i <= n:
            res += n // i
            i *= 5

        return res

    def reverseBits(self, n):
        '''
        :param n: int
        :return: int
        leetcode easy: 190. Reverse Bits
        Input: n = 00000010100101000001111010011100
        Output:    964176192 (00111001011110000010100101000000)
        '''
        # using build-in library transform to string type then convert to int type.
        # return int(bin(n)[2:].zfill(32)[::-1], 2)

        res = 0

        for i in range(32):
            bit = (n >> i) & 1 # check has bit == 1
            res |= (bit << (31 - i)) # shift to reverse position then 'or'/'add' previous res

        return res

    def rob(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode medium: 198. House Robber

        Input: [2,7,9,3,1]
        Output: 12
        '''
        # it's non dp solution

        rob1, rob2 = 0, 0

        for num in nums:
            rob1 , rob2 = rob2 , max(num + rob1, rob2)

        return rob2

    def isHappy(self, n):
        '''
        :param n: int
        :return: bool
        leetcode easy: 202. Happy Number
        '''
        dic = {}
        while True:
            l = list(map(int, str(n)))

            n = sum(map(lambda x:x*x,l))

            if n in dic:
                return False

            if n == 1:
                return True

            dic[n] = n

    def countPrimes(self, n):
        '''
        :param n: int
        :return:  int
        leetcode easy: 204. Count Primes
        '''
        if n <= 2:
            return 0

        output = [1] * n
        output[0], output[1] = 0, 0

        for i in range(2, int(n ** 0.5) + 1):
            if output[i] == 1:
                # del has i factor number
                output[i * i:n:i] = [0] * len(output[i * i:n:i])

        return sum(output)

    def isPowerOfTwo(self, n):
        '''
        :param n: int
        :return: bool
        leetcode easy: 231. Power of Two

        Input: n = 16
        Output: true
        Explanation: 24 = 16

        Input: n = 3
        Output: false
        '''
        # return n > 0 and n & (n-1) == 0

        if n < 1:
            return False

        while n % 2 == 0:
            n /= 2

        return n == 1

    def missingNumber(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode easy: 268. Missing Number
        Input: nums = [9,6,4,2,3,5,7,0,1]
        Output: 8
        '''
        # sum(arithmetic sequence) = (n+1)n/2
        return (len(nums) + 1) * len(nums) // 2 - sum(nums)

    def moveZeroes(self, nums):
        '''
        :param nums: List[int]
        :return: None
        leetcode easy: 283. Move Zeroes
        Duobple pointer
        Input: nums = [0,1,0,3,12]
        Output: [1,3,12,0,0]
        '''
        """
        Do not return anything, modify nums in-place instead.
        """
        index = 0

        for i in range(len(nums)):
            if nums[i] != 0:
                nums[index] , nums[i] = nums[i] , nums[index]
                index += 1

    def isPowerOfThree(self, n: int) -> bool:
        '''
        :param n: int
        :return: boot
        leetcode easy: 326. Power of Three
        Input: n = 27
        Output: true

        This way can finish any number of power of number
        '''
        if n < 1:
            return False

        while n % 3 == 0:
            n /= 3

        return n == 1

    def reverseString(self, s):
        '''
        :param s: List[str]
        :return: None
        :Do not return anything, modify s in-place instead.
        leetcode easy: 344. Reverse String
        '''
        for i in range(len(s)//2):
            s[i],s[-i-1] = s[-i-1],s[i]

    def intersect(self, nums1, nums2):
        '''
        :param nums1: List[int]
        :param nums2: List[int]
        :return: List[int]
        leetcode easy: 350. Intersection of Two Arrays II
        '''
        d = collections.Counter(nums1)
        res = []

        for n in nums2:
            if d[n] > 0:
                res.append(n)
                d[n] -= 1

        return res

    def isPerfectSquare(self, num):
        '''
        :param num: int
        :return: bool
        leetcode easy: 367. Valid Perfect Square
        '''
        x = num
        while x * x > num:
            x = int((x + num / x) / 2)
        return (x * x == num)

    def canConstruct(self, ransomNote, magazine):
        '''
        :param ransomNote: str
        :param magazine: str
        :return: bool
        leetcode easy: 383. Ransom Note
        '''
        for key, value in collections.Counter(ransomNote).items():
            if key not in magazine:
                return False

            if collections.Counter(magazine)[key] < value:
                return False

        return True

    def isSubsequence(self, s: str, t: str) -> bool:
        '''
        :param s : str
        :param t : str
        :return: bool
        leetcode easy: 392. Is Subsequence

        Input: s = "abc", t = "ahbgdc"
        Output: true
        '''
        if not s:
            return True

        pos = 0

        for char in t:
            if pos <= len(s) - 1:
                if char == s[pos]:
                    pos += 1

        if pos == len(s):
            return True

        return False

    def sumOfLeftLeaves(self, root):
        '''
        :param root: TreeNode
        :return: int
        leetcode easy: 404. Sum of Left Leaves
        '''
        def isLeaf(node):
            if not node:
                return False

            if not node.left and not node.right:
                return True

            return False

        res = 0

        if root:
            if isLeaf(root.left):
                res += root.left.val

            else:
                res += self.sumOfLeftLeaves(root.left)

            res += self.sumOfLeftLeaves(root.right)

        return res

    def arrangeCoins(self, n):
        '''
        :param n: int
        :return: int
        leetcode easy: 441. Arranging Coins
        '''
        lo, hi = 1, n

        while lo <= hi:
            mid = (lo + hi) // 2

            total = mid * (mid + 1) // 2

            if total == n:
                return mid

            elif total > n:
                hi = mid - 1

            else:
                lo = mid + 1

        # return int(((1 + 8*n)**0.5 - 1)//2)
        '''
        1 + 2 + ... + k <= N
        先转换成等式
        k = (-1 + sqrt(1 + 8 * n)) / 2
        最后的答案是k的取整
        '''

        return hi

    def hammingDistance(self, x, y):
        '''
        :param x: int
        :param y: int
        :return: int
        leetcode easy : 461. Hamming Distance
        '''
        # solution : do x XOR y first then count how many bit == '1'

        distance = 0
        n = x ^ y

        while n > 0:
            if n & 1:
                distance += 1

            n = n >> 1 # n // 2

        return distance
        # return bin(x ^ y).count('1')

    def islandPerimeter(self, grid):
        '''
        :param grid: List[List[int]]
        :return: int
        leetcode easy : 463. Island Perimeter
        '''
        output = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    output += 4

                    if i > 0 and grid[i - 1][j] == 1:
                        output -= 2

                    if j > 0 and grid[i][j - 1] == 1:
                        output -= 2

        return output

    def findComplement(self, num):
        '''
        :param num: int
        :return: int
        leetcode easy : 476. Number Complement

        Input: num = 5
        Output: 2
        Explanation: 101 -> 010
        '''
        # this solution is num all bit XOR 1111...11 can getting num complement
        # Note: This question is the same as 1009

        # return 2 ** (len(bin(n)) - 2) - 1 - n

        x = 1

        while x <= num:
            x <<= 1 # ->> x *= 2 , this operation is count how many bit of num of binary representation.

        return (x - 1) ^ num

    def matrixReshape(self, nums, r, c):
        '''
        :param nums: List[List[int]]
        :param r: int
        :param c: int
        :return: List[List[int]]
        leetcode Easy: 566. Reshape the Matrix
        '''
        a = sum(nums,[])
        if (len(a) != (r*c)):
            return nums
        res = [[0]*c for i in range(r)]
        for i in range(0,len(a)):
            res[i//c][i%c] = a[i]
        return res

    def findLHS(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode Easy: 594. Longest Harmonious Subsequence
        input : [1,3,2,2,5,2,3,7]
        output: 5 , because = [3,2,2,2,3]
        '''
        D = collections.Counter(nums)

        ans = [0]

        for k in list(D):
            if k + 1 in D:
                ans.append(D[k] + D[k+1])

        return max(ans)

    def canPlaceFlowers(self, flowerbed, n):
        '''
        :param flowerbed: : List[int]
        :param n: int
        :return: bool
        leetcode Easy: 605. Can Place Flowers

        Input: flowerbed = [1,0,0,0,1], n = 1
        Output: true

        Input: flowerbed = [1,0,0,0,1], n = 2
        Output: false
        '''
        if n == 0:
            return True

        flowerbed = [0] + flowerbed + [0]

        for i in range(1, len(flowerbed) - 1):

            if not (flowerbed[i] or flowerbed[i - 1] or flowerbed[i + 1]):
                flowerbed[i] = 1
                n -= 1

                if not n:
                    break

        return not n

    def findErrorNums(self, nums):
        '''
        :param nums: List[int]
        :return: List[int]
        leetcode Easy: 645. Set Mismatch
        Input: nums = [1,2,2,4]
        Output: [2,3]
        Output first number is repetition of one number of output
        Output second number is loss of another number of output
        '''
        seen = [0] * (len(nums) + 1)
        seen[0] = 1

        duplicated = 0
        for n in nums:
            if seen[n]:
                duplicated = n
            seen[n] = 1

        missing = 0
        for idx, n in enumerate(seen):
            if n == 0:
                missing = idx

        return [duplicated, missing]

    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        leetcode Easy: 766. Toeplitz Matrix
        matrix = [[1, 2, 3, 4],
                  [5, 1, 2, 3],
                  [9, 5, 1, 2]]
        """
        return all(matrix[row+1][1:] == matrix[row][:-1] for row in range(len(matrix)-1))

    def rotatedDigits(self, N):
        '''
        :param N: int
        :return: int
        leetcode easy :788. Rotated Digits
        '''
        counter = 0
        for i in range(1, N + 1):
            str_num_list = list(str(i))

            is_good_number = False

            for digit in str_num_list:

                if digit in ['3', '4', '7']:
                    is_good_number = False
                    break

                elif digit in ['2', '5', '6', '9']:
                    is_good_number = True

            if is_good_number:
                counter += 1

        return counter

    def middleNode(self, head):
        '''
        :param head: Optional[ListNode]
        :return: Optional[ListNode]
        leetcode easy: 876. Middle of the Linked List

        Input: head = [1,2,3,4,5]
        Output: [3,4,5]
        Explanation: The middle node of the list is node 3
        '''
        '''
        # solution of count head index
        length = 0
        curr = head

        while curr:
            length += 1
            curr = curr.next


        index = 0
        curr = head

        while index < (length // 2):
            curr = curr.next
            index += 1

        return curr
        '''
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow

    def projectionArea(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        leetcode easy: 883. Projection Area of 3D Shapes
        """
        top, front, side = 0, 0, 0
        n = len(grid)

        for i in range(n):
            x, y = 0, 0
            for j in range(n):
                if grid[i][j] != 0:
                    top += 1

                x = max(x, grid[i][j])
                y = max(y, grid[j][i])

            front += x
            side += y
        return top + front + side

    def sortArrayByParity(self, A):
        '''
        :param A:List[int]
        :return: List[int]
        leetcode easy: 905. Sort Array By Parity
        '''
        lasteven = -1

        for index, num in enumerate(A):
            if num % 2 == 0:
                lasteven += 1
                A[lasteven], A[index] = A[index], A[lasteven]

        return A

    def smallestRangeII(self, A, K):
        '''
        :param A: List[int]
        :param K: int
        :return: int
        leetcode easy: 910. Smallest Range II
        '''
        A.sort()
        res = A[-1] - A[0]  # possible result

        for i in range(len(A) - 1):
            a, b = A[i], A[i + 1]
            hi = max(A[-1] - K, a + K)  # possible max num
            lo = min(A[0] + K, b - K)  # possible min num
            res = min(res, hi - lo)

        return res

    def sortArrayByParityII(self, nums):
        '''
        :param nums:List[int]
        :return: List[int]
        leetcode easy: 922. Sort Array By Parity II
        solve in-place
        '''
        even = 0
        odd = 1

        while even < len(nums) and odd < len(nums):

            if nums[even] % 2 == 0:
                even += 2

            else:
                if nums[odd] % 2 != 0:
                    odd += 2

                else:
                    nums[even], nums[odd] = nums[odd], nums[even]

                    even += 2
                    odd += 2

        return nums

    def rangeSumBST(self, root, L, R):
        '''
        :param root: TreeNode
        :param L: int
        :param R: int
        :return: int
        leetcode easy: 938. Range Sum of BST
        '''
        self.total = 0

        def dfs(node, L, R):
            if not node:
                return 0

            if L <= node.val <= R:
                self.total += node.val

            dfs(node.left, L, R)

            dfs(node.right, L, R)

        dfs(root, L, R)

        return self.total

    def validMountainArray(self, arr):
        '''
        :param arr: List[int]
        :return: bool
        leetcode easy: 941. Valid Mountain Array
        '''
        n = len(arr)

        if n < 3 or arr[0] > arr[1] or arr[-1] > arr[-2]:
            return False

        i = 1
        while arr[i - 1] < arr[i]:
            i += 1

        while i < n and arr[i - 1] > arr[i]:
            i += 1

        return i == n

    def allCellsDistOrder(self, R, C, r0, c0):
        '''
        :param R: int
        :param C: int
        :param r0: int
        :param c0: int
        :return: List[List[int]]
        leetcode easy: 1030. Matrix Cells in Distance Order
        '''
        dis = []
        for r in range(R):
            for c in range(C):
                dis.append((abs(r0 - r) + abs(c0 - c), [r, c]))
        dis.sort()
        return [x for d, x in dis]

    def gcdOfStrings(self, str1, str2):
        '''
        :param str1: str
        :param str2: str
        :return: str
        leetcode easy: 1071. Greatest Common Divisor of Strings
        '''
        if str1 == str2 and len(str1) != 0:
            return str1

        if str1.count(str2) != len(str1) // len(str2):
            return ''

        count = []

        for i in range(1, len(str1)):
            if len(str1) % i == 0 and len(str2) % i == 0:
                count.append(i)

        return [str1[i:i + count[-1]] for i in range(0, len(str1), count[-1])][0]

    def findOcurrences(self, text, first, second):
        '''
        :param text: str
        :param first: str
        :param second: str
        :return: list[str]
        leetcode easy: 1078. Occurrences After Bigram
        '''
        '''
        :return: List[str]
        Input: text = "alice is a good girl she is a good student"
               first = "a"
               second = "good"
        Output: ["girl","student"]
        '''
        ans = []

        text_spilt = text.split(' ')

        for x, y in enumerate(text_spilt):
            if y == first and x != len(text_spilt) - 1:
                if text_spilt[x + 1] == second and x + 1 != len(text_spilt) - 1:
                    ans.append(text_spilt[x + 2])

        return ans

    def countCharacters(self, words, chars):
        '''
        :param words: List[str]
        :param chars: str
        :return: int
        leetcode easy: 1160. Find Words That Can Be Formed by Characters
        Input: words = ["cat","bt","hat","tree"]
               chars = "atach"
        Output: 6
        '''
        ans = []

        for word in words:
            inside = True
            for w in word:
                if w not in chars:
                    inside = False
                    break

                if chars.count(w) < word.count(w):
                    inside = False
                    break

            if inside:
                ans.append(word)

        lenth = 0
        for i in ans:
            lenth += len(i)

        return lenth

    def distanceBetweenBusStops(self, distance, start, destination):
        '''
        :param distance: List[int]
        :param start: int
        :param destination: int
        :return: int
        leetcode easy: 1184. Distance Between Bus Stops
        '''
        if start == destination:
            return 0

        x = min(start, destination)
        y = max(start, destination)

        d1 = sum(distance[x:y])
        d2 = sum(distance[y:]) + sum(distance[:x])

        return min(d1, d2)

    def minimumAbsDifference(self, arr):
        '''
        :param arr: List[int]
        :return: List[List[int]]
        leetcode easy: 1200. Minimum Absolute Difference
        Input: arr = [4,2,1,3]
        Output: [[1,2],[2,3],[3,4]]
        Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.
        '''
        arr.sort()

        if len(arr) == 2:
            return [[arr[0], arr[1]]]

        res = [[arr[0], arr[1]]]
        diff = arr[1] - arr[0]

        for i in range(2, len(arr)):
            if arr[i] - arr[i - 1] < diff:
                diff = arr[i] - arr[i - 1]
                res = [[arr[i - 1], arr[i]]]

            elif arr[i] - arr[i - 1] == diff:
                res.append([arr[i - 1], arr[i]])

        return res

    def minCostToMoveChips(self, position):
        '''
        :param position: List[int]
        :return: int
        leetcode easy: 1217. Minimum Cost to Move Chips to The Same Position

        Input: position = [1,2,3]
        Output: 1
        explain solution: compare which are smaller between move all chips to 1th and move all chips to 2th
        '''
        odd_numbers  = 0
        even_numbers = 0

        for num in position:
            if num % 2 == 0:
                even_numbers += 1
            else:
                odd_numbers += 1

        return min(odd_numbers, even_numbers)

    def checkStraightLine(self, coordinates):
        '''
        input :[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
              :[[1,1],[2,2],[3,4],[4,5],[5,6],[7,7]]
        :param coordinates: List[List[int]]
        :return: bool
        leetcode easy: 1232. Check If It Is a Straight Line
        '''
        if len(coordinates) == 2:
            return True
        ans = []
        for i in range(1, len(coordinates)):
            if coordinates[i][0] - coordinates[i-1][0] == 0:
                k = float("inf")
            else:
                k = (coordinates[i][1] - coordinates[i-1][1])/(coordinates[i][0] - coordinates[i-1][0])
            ans.append(k)
        return len(set(ans)) == 1

    def kWeakestRows(self, mat, k):
        '''
        :param mat:List[List[int]]
        :param k: int
        :return: List[int]
        leetcode easy: 1337. The K Weakest Rows in a Matrix
        '''
        d = {i:sum(mat[i]) for i in range(len(mat))}
        return [i[0] for i in sorted(d.items(),key=lambda item:item[1])[:k]]

    def getDecimalValue(self, head):
        '''
        :param head: ListNode
        :return: int
        Input: head = [1,0,1]
        Output: 5
        Explanation: (101) in base 2 = (5) in base 10
        '''
        ans = 0

        while head:
            ans *= 2

            ans += head.val

            head = head.next

        return ans

    def checkIfExist(self, arr):
        '''
        :param arr: List[int]
        :return: bool
        leetcode easy: 1346. Check If N and Its Double Exist
        '''
        if len(arr) < 1:
            return False

        if arr.count(0) > 1:
            return True

        if 0 in arr:
            arr.pop(arr.index(0))

        for i in arr:
            if i * 2 in set(arr):
                return True

        return False

    def luckyNumbers(self, matrix):
        '''
        :param matrix: List[List[int]]
        :return: List[int]
        leetcode easy: 1380. Lucky Numbers in a Matrix
        '''
        '''
        min_matrix = []

        for i in matrix:
            min_matrix.append(min(i))

        reverse = []

        for column in range(len(matrix[0])):
            tmp = []

            for row in matrix:
                tmp.append(row[column])

            reverse.append(tmp)

        max_matrix = []

        for i in reverse:
            max_matrix.append(max(i))

        for i in min_matrix:
            if i in max_matrix:
                return [i]

        return []
        '''

        for i in [min(i) for i in matrix]:
            if i in [max(x) for x in [list(i) for i in zip(*matrix)]]:
                return [i]
        return []

    def countOdds(self, low, high):
        '''
        :param low:  int
        :param high: int
        :return: int

        leetcode easy: 1523. Count Odd Numbers in an Interval Range
        '''
        if low % 2 == 0:
            low += 1

        if high % 2 == 0:
            high -= 1

        return (1 + (high - low) // 2)

    def findKthPositive(self, arr, k):
        '''
        :param arr: List[int]
        :param k: int
        :return: int
        leetcode easy: 1539. Kth Missing Positive Number
        '''
        num = 0
        count = 0

        while count < k:
            num += 1
            if num not in arr:
                count += 1

        return num

    def diagonalSum(self, mat):
        '''
        :param mat: List[List[int]]
        :return: int

        leetcode easy: 1572. Matrix Diagonal Sum
        '''
        sum = 0
        for i, row in enumerate(mat):
            if i == len(row)-i-1:
                sum += (row[i])
            else:
                sum += (row[i] + row[len(row)-i-1])
        return sum

    def canFormArray(self, arr, pieces):
        '''
        :param arr: List[int]
        :param pieces: List[List[int]]
        :return: bool

        leetcode easy: 1640. Check Array Formation Through Concatenation

        Input: arr = [49,18,16], pieces = [[16,18,49]] ,Output: false
        Input: arr = [91,4,64,78], pieces = [[78],[4,64],[91]], Output: true
        '''
        mapping = {}
        for piece in pieces:
            mapping[piece[0]] = piece

        ans = []
        for num in arr:
            if num in mapping:
                ans += mapping[num]

        return ans == arr

    def findRotation(self, mat, target):
        '''
        :param mat: List[List[int]]
        :param target: List[List[int]]
        :return: bool

        leetcode easy: 1886. Determine Whether Matrix Can Be Obtained By Rotation

        Input: mat = [[0,0,0],[0,1,0],[1,1,1]], target = [[1,1,1],[0,1,0],[0,0,0]]
        Output: true
        Explanation: We can rotate mat 90 degrees clockwise two times to make mat equal target.
        '''

        '''
        for _ in range(4): 
            if mat == target: 
                return True
            mat = [list(x)[::-1] for x in zip(*mat)]
        return False 
        '''

        if mat == target:
            return True

        count = 3

        while count != 0:
            l = len(mat)

            for i in range(l):
                for j in range(i + 1, l):
                    if i != j:
                        mat[i][j], mat[j][i] = mat[j][i], mat[i][j]

                mat[i] = mat[i][::-1]

            if mat == target:
                return True

            count -= 1

        return False

    def validPath(self, n, edges, source, destination):
        '''
        :param n: int
        :param edges: List[List[int]]
        :param source: int
        :param destination: int
        :return: bool

        leetcode easy: 1971. Find if Path Exists in Graph

        Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
        Output: true
        Explanation: There are two paths from vertex 0 to vertex 2:
        - 0 → 1 → 2
        - 0 → 2
        '''
        if source == destination:
            return True

        graph = [[] for i in range(n)]

        for s, d in edges:
            graph[s].append(d)
            graph[d].append(s)

        visited = [False for _ in range(n)]

        stack = [source]

        while stack:
            key = stack.pop(0)
            visited[key] = True

            if destination in graph[key]:
                return True

            for vertex in graph[key]:
                if not visited[vertex]:
                    stack.append(vertex)

        return False

    def findGCD(self, nums):
        '''
        :param nums: List[int]
        :return: int

        leetcode easy: 1971. Find if Path Exists in Graph

        Input: nums = [2,5,6,9,10]
        Output: 2
        Explanation:
        The smallest number in nums is 2.
        The largest number in nums is 10.
        The greatest common divisor of 2 and 10 is 2.
        '''
        '''
        small = min(nums)
        large = max(nums)
        
        for i in range(small, 0, -1):
            if small % i == 0 and large % i == 0:
                return i
        '''
        min_num = min(nums)
        max_num = max(nums)

        while min_num:
            min_num, max_num = max_num % min_num, min_num

        return max_num

    def countEven(self, num):
        '''
        :param num: int
        :return: int
        leetcode easy: 2180. Count Integers With Even Digit Sum

        Input: num = 30
        Output: 14
        Explanation:
        The 14 integers less than or equal to 30 whose digit sums are even are
        2, 4, 6, 8, 11, 13, 15, 17, 19, 20, 22, 24, 26, and 28.
        '''
        n, dSum = num, 0

        # Calculate digit sum of numbers
        while n > 0:
            dSum += n % 10
            n //= 10

        if num % 2 == 0 and dSum % 2 == 1:
            return num // 2 - 1

        return num//2

class LeetCode_Medium:
    def addTwoNumbers(self, l1, l2):
        '''
        :param l1: Optional[ListNode]
        :param l2: Optional[ListNode]
        :return: Optional[ListNode]
        leetcode medium: 2. Add Two Numbers
        '''
        head = None
        ans = None

        tmp = 0

        while l1 or l2:
            v1, v2 = 0, 0

            if l1:
                v1 = l1.val

            if l2:
                v2 = l2.val

            v = v1 + v2 if tmp == 0 else v1 + v2 + tmp

            tmp = 0

            if v >= 10:
                tmp = v // 10

                if not head:
                    head = ListNode(v % 10)
                    ans = head

                else:
                    head.next = ListNode(v % 10)
                    head = head.next

            else:
                if not head:
                    head = ListNode(v)
                    ans = head
                else:
                    head.next = ListNode(v)
                    head = head.next

            if l1 and l1.next:
                l1 = l1.next
            else:
                l1 = None

            if l2 and l2.next:
                l2 = l2.next
            else:
                l2 = None

        if tmp != 0:
            head.next = ListNode(tmp)
            head = head.next

        return ans

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        leetcode medium: 3. Longest Substring Without Repeating Characters
        """
        d = {}

        start = 0
        maxlen = 0

        for i, v in enumerate(s):
            if v in d and start <= d[v]:
                start = d[v] + 1

            else:
                maxlen = max(maxlen, i - start + 1)

            d[v] = i

        return maxlen

    def myAtoi(self, s):
        '''
        :param s: str
        :return: int
        leetcode medium: 8. String to Integer (atoi)
        Input: s = "4193 with words"
        Output: 4193
        '''
        s = s.strip()
        if len(s) == 0 or (len(s) > 0 and not (s[0] in ['+', '-'] or s[0].isdigit())):
            return 0

        value = 0
        sign = -1 if s[0] == '-' else 1
        i = 1 if not s[0].isdigit() else 0

        while i < len(s) and s[i].isdigit():
            value = (value * 10) + (ord(s[i]) - ord('0'))
            i += 1

        if sign == -1:
            return -(2 ** 31) if value * sign < -(2 ** 31) else value * sign

        else:
            return (2 ** 31) - 1 if value > (2 ** 31) - 1 else value

    def maxArea(self, height):
        '''
        :param height: List[int]
        :return: int
        leetcode medium: 11. Container With Most Water
        '''
        maxArea = 0
        left    = 0
        right   = len(height)-1

        while left != right:
            width = right - left

            if height[left] <= height[right]:
                area = width * height[left]
                left += 1

            else:
                area = width * height[right]
                right -= 1

            if area > maxArea:
                maxArea = area

        return maxArea

    def threeSum(self,nums):
        '''
        :param nums: List[int]
        :return: List[List[int]]
        leetcode medium: 15. 3Sum
        time : O(n)^2 , space : O(1)
        '''
        if len(nums) < 3:
            return []

        nums.sort()
        ans = []

        for i in range(len(nums) - 2):
            # reduce run time if the number and next number is same
            if i != 0 and nums[i] == nums[i - 1]:
                continue

            left = i + 1
            right = len(nums) - 1

            while left < right:
                sum_ = nums[i] + nums[left] + nums[right]

                if sum_ < 0:
                    left += 1

                elif sum_ > 0:
                    right -= 1

                else:
                    ans.append([nums[i], nums[left], nums[right]])

                    left += 1
                    right -= 1

                    # reduce run time if the left number and next left number is same
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1

                    # reduce run time if the right number and right left number is same
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1

        return ans

    def threeSumClosest(self, nums, target):
        '''
        :param nums: List[int]
        :param target: int
        :return: int
        leetcode medium: 16. 3Sum Closest
        '''
        nums.sort()
        result = float('inf')

        for index in range(len(nums)):
            # reduce run time if the number and next number is same
            if index != 0 and nums[index] == nums[index - 1]:
                continue

            left  = index + 1
            right = len(nums) - 1

            while left < right:
                curSum = nums[left] + nums[right] + nums[index]

                if curSum == target:
                    return target

                if abs(result - target) > abs(curSum - target):
                    result = curSum

                if curSum > target:
                    right -= 1

                else:
                    left += 1

        return result

    def generateParenthesis(self, n):
        '''
        :param n: int
        :return: List[str]
        leetcode medium: 22. Generate Parentheses
        '''
        allOutput = set([])

        def dfs(output, numOfLeft, numOfRight, n):
            if len(output) == n * 2:
                allOutput.add(output)
                return

            if numOfLeft < n:
                dfs(output + '(', numOfLeft + 1, numOfRight, n)

            if numOfLeft > numOfRight and numOfRight < n:
                dfs(output + ')', numOfLeft, numOfRight + 1, n)

        dfs('(', 1, 0, n)
        return allOutput

    def swapPairs(self, head):
        '''
        :param head: ListNode
        :return: ListNode
        leetcode medium: 24. Swap Nodes in Pairs
        '''
        if not head:
            return head

        if not head.next:
            return head

        val = []

        while head:
            val.append(head.val)
            head = head.next

        L = len(val)

        if len(val) % 2 == 1:
            L -= 1

        for i in range(0, L, 2):
            val[i], val[i + 1] = val[i + 1], val[i]

        head = ListNode(val[0])

        ans = head

        for i in val[1:]:
            head.next = ListNode(i)
            head = head.next

        return ans

    def swapPairs_easy_understand(self, head):
        '''
        :param head: ListNode
        :return: ListNode
        leetcode medium: 24. Swap Nodes in Pairs
        '''
        if not head:
            return head

        ans = head

        while head and head.next:
            head.val , head.next.val = head.next.val , head.val
            head = head.next.next

        return ans

    def divide(self, dividend, divisor):
        '''
        :param dividend: int
        :param divisor: int
        :return: int
        leetcode medium: 29. Divide Two Integers
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

    def isValidSudoku(self, board):
        '''
        :param board: List[List[str]]
        :return: bool
        leetcode medium: 36. Valid Sudoku
        Input: board =  [["5","3",".",".","7",".",".",".","."]
                        ,["6",".",".","1","9","5",".",".","."]
                        ,[".","9","8",".",".",".",".","6","."]
                        ,["8",".",".",".","6",".",".",".","3"]
                        ,["4",".",".","8",".","3",".",".","1"]
                        ,["7",".",".",".","2",".",".",".","6"]
                        ,[".","6",".",".",".",".","2","8","."]
                        ,[".",".",".","4","1","9",".",".","5"]
                        ,[".",".",".",".","8",".",".","7","9"]]
        Output: true
        '''
        row = [[] for _ in range(9)]
        col = [[] for _ in range(9)]
        area = [[] for _ in range(9)]

        for i in range(9):
            for j in range(9):
                e = board[i][j]
                if e != '.':
                    area_id = i // 3 * 3 + j // 3
                    if e in row[i] or e in col[j] or e in area[area_id]:
                        return False
                    else:
                        row[i].append(e)
                        col[j].append(e)
                        area[area_id].append(e)

        return True

    def combinationSum(self, candidates, target):
        '''
        :param candidates: List[int]
        :param target:     int
        :return:           List[List[int]]
        leetcode medium: 39. Combination Sum
        '''
        candidates.sort()

        res, length = [], len(candidates)

        def dfs(target, start, vlist):
            if target == 0:
                return res.append(vlist)

            for i in range(start, length):
                if target < candidates[i]:
                    break

                else:
                    dfs(target - candidates[i], i, vlist + [candidates[i]])

        dfs(target, 0, [])

        return res

    def multiply(self, num1, num2):
        '''
        :param num1: str
        :param num2: str
        :return: str
        leetcode medium: 43. Multiply Strings
        '''
        res = 0

        imult = 1

        for i in reversed(num1):
            jmult = 1

            for j in reversed(num2):
                res += imult * (int(j) * int(i)) * jmult
                jmult *= 10

            imult *= 10
        return str(res)

    def permute(self, nums):
        '''
        :param nums: List[int]
        :return: list[List[int]]
        leetcode medium: 46. Permutations
        Input: nums = [1,2,3]
        Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
        '''
        def dfs(number, item, res):
            if len(item) == len(nums):
                res.append(item)
                return

            for i in range(len(number)):
                n_nums = number[:i] + number[i + 1:]
                dfs(n_nums, item + [number[i]], res)

        res = []

        dfs(nums, [], res)

        return res

    def permuteBFS(self, nums):
        '''
        :param nums: List[int]
        :return: list[List[int]]
        leetcode medium: 46. Permutations
        '''
        ans = []
        bfs = [([], nums)]

        while bfs:
            pre, nex = bfs.pop(0)
            if len(nex) == 1:
                ans.append(pre + nex)
            else:
                for i in range(len(nex)):
                    bfs.append((pre + [nex[i]], nex[:i] + nex[i + 1:]))
        return ans

    def rotate(self, matrix):
        '''
        :param matrix: List[List[int]]
        :return: None
        leetcode medium: 48. Rotate Image
        '''
        """
        Do not return anything, modify matrix in-place instead.
        """

        for i in range(len(matrix)):
            for j in range(i,len(matrix)):
                if i != j:
                    # transpose
                    matrix[i][j] , matrix[j][i] = matrix[j][i] ,matrix[i][j]
            # reverse by row
            matrix[i] = matrix[i][::-1]

    def groupAnagrams(self, strs):
        '''
        :param strs: List[str]
        :return: List[List[str]]
        leetcode medium: 49. Group Anagrams
        '''
        D = dict()

        for i in strs:
            isort = ''.join(sorted(i))

            if isort in D:
                D[isort].append(i)

            else:
                D[isort] = [i]

        return [i for i in D.values()]

    def myPow_recurcve(self, x, n):
        '''
        :param x: float
        :param n: int
        :return: float
        leetcode medium: 50. Pow(x, n)
        '''
        if n == 0:
            return 1

        if n < 0:
            x = 1 / x
            n = -n

        if n % 2:
            return x * self.myPow_recurcve(x * x, n // 2)
        else:
            return self.myPow_recurcve(x * x, n // 2)

    def myPow(self, x, n):
        '''
        :param x: float
        :param n: int
        :return: float
        leetcode medium: 50. Pow(x, n)
        '''
        result = 1

        if n < 0:
            x = 1 / x
            n = -n
        power = n

        while power:
            if power & 1:
                result = result * x
            x = x * x
            power = power >> 1

        return result

    def spiralOrder(self, matrix):
        '''
        :param matrix: List[List[int]]
        :return: List[int]

        leetcode medium: 54. Spiral Matrix

        Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
        Output: [1,2,3,6,9,8,7,4,5]
        '''
        '''
        spiral_path = []

        while matrix:
            # pop the top-most row
            spiral_path.extend(matrix.pop(0))

            # get the upside-down of matrix transpose
            matrix = [*zip(*matrix)][::-1]

        return spiral_path
        '''
        m = len(matrix)
        n = len(matrix[0])

        left, right = 0, n - 1
        top, bottom = 0, m - 1
        ans = []

        while left <= right and top <= bottom:

            for i in range(left, right + 1):
                ans.append(matrix[top][i])
            top += 1

            for i in range(top, bottom + 1):
                ans.append(matrix[i][right])
            right -= 1

            if left > right or top > bottom:
                break

            for i in range(right, left - 1, -1):
                ans.append(matrix[bottom][i])
            bottom -= 1

            for i in range(bottom, top - 1, -1):
                ans.append(matrix[i][left])
            left += 1

        return ans

    def canJump(self, nums):
        '''
        :param nums: List[int]
        :return: bool
        leetcode medium: 55. Jump Game
        '''
        N = len(nums)

        if N <= 1:
            return True

        j = N - 1

        for i in range(N - 2, -1, -1):
            if i + nums[i] >= j:
                j = i

        return j == 0

    def merge(self, intervals):
        '''
        :param intervals: List[List[int]]
        :return:  List[List[int]]
        leetcode medium: 56. Merge Intervals
        '''
        intervals.sort()
        res = [intervals[0]]

        for i in range(1, len(intervals)):
            last_x, last_y = res[-1]    # last_x = res[-1][0], last_y = res[-1][1]
            cur_x, cur_y = intervals[i] # cur_x = intervals[i][0], cur_y = intervals[i][1]

            if cur_x <= last_y:
                res[-1][1] = max(cur_y, last_y)

            else:
                res.append([cur_x, cur_y])

        return res

    def insert(self, intervals, newInterval):
        '''
        :param intervals: List[List[int]]
        :param newInterval: List[int]
        :return: List[List[int]]
        leetcode medium: 57. Insert Interval
        Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
        Output: [[1,5],[6,9]]
        '''
        if not intervals:
            return [newInterval]

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

    def generateMatrix(self, n):
        '''
        :param n: int
        :return: List[List[int]]
        leetcode medium: 59. Spiral Matrix II
        '''
        def spiral(start, i, j):
            if i == 1 and j == 1:
                return [[start]]

            next_spiral = spiral(start + i, j - 1, i)

            return [list(range(start, start + i))] + [list(a) for a in zip(*next_spiral[::-1])]

        return spiral(1, n, n)

    def rotateRight(self, head, k):
        '''
        :param head: ListNode
        :param k: int
        :return: ListNode
        leetcode medium: 61. Rotate List
        '''
        prev = None
        cur = head

        if not head or not head.next:
            return head

        length = 0
        while cur:
            length += 1
            cur = cur.next
        cur = head

        for i in range(k % length):
            while cur.next:
                prev = cur
                cur = cur.next

            cur.next = head
            head = cur
            prev.next = None

        return head

    def setZeroes(self, matrix):
        '''
        :param matrix: List[List[int]]
        :return: None
        leetcode medium: 73. Set Matrix Zeroes
        '''
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])

        y = []

        for i in range(0, m):
            indices = [index for index, x in enumerate(matrix[i]) if x == 0]
            if len(indices) > 0:
                y += indices
                matrix[i] = [0] * n

        y = set(y)

        for column in y:
            for i in range(0, m):
                matrix[i][column] = 0

    def combine(self, n, k):
        '''
        :param n: int
        :param k: int
        :return: List[List[int]]
        leetcode medium: 77. Combinations
        Input: n = 4, k = 2
        Output:
        [ [2,4],
          [3,4],
          [2,3],
          [1,2],
          [1,3],
          [1,4]]
        '''
        def dfs(number, item, res):
            if len(item) == k:
                res.append(item)
                return

            for i in range(len(number)):
                n_nums = number[i + 1:]
                dfs(n_nums, item + [number[i]], res)

        res = []

        dfs(list(range(1, n + 1)), [], res)

        return res

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        :input: [1,2,2]
        :output : [[],[1],[1,2],[1,2,2],[2],[2,2]]
        leetcode medium: 78. Subsets
        """
        if not nums:
            return []

        nums = sorted(nums)

        def dfs(index, item, res):
            res.append(item)

            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i - 1]:
                    continue

                dfs(i + 1, item + [nums[i]], res)

        res = []
        dfs(0, [], res)

        return res

    def exist(self, board , word):
        '''
        :param board: List[List[str]]
        :param word: str
        :return: bool
        leetcode medium: 79. Word Search
        '''
        def dfs(board, word, char, visited, i, j):
            if char >= len(word):
                return True

            m = len(board)
            n = len(board[0])

            if j + 1 < n and board[i][j + 1] == word[char] and not visited[i][j + 1]:
                visited[i][j + 1] = 1

                result = dfs(board, word, char + 1, visited, i, j + 1)

                if result:
                    return True

                visited[i][j + 1] = 0

            if i + 1 < m and board[i + 1][j] == word[char] and not visited[i + 1][j]:
                visited[i + 1][j] = 1

                result = dfs(board, word, char + 1, visited, i + 1, j)

                if result:
                    return True

                visited[i + 1][j] = 0

            if i - 1 >= 0 and board[i - 1][j] == word[char] and not visited[i - 1][j]:
                visited[i - 1][j] = 1

                result = dfs(board, word, char + 1, visited, i - 1, j)

                if result:
                    return True

                visited[i - 1][j] = 0

            if j - 1 >= 0 and board[i][j - 1] == word[char] and not visited[i][j - 1]:
                visited[i][j - 1] = 1

                result = dfs(board, word, char + 1, visited, i, j - 1)

                if result:
                    return True

                visited[i][j - 1] = 0

            return False

        m = len(board)
        n = len(board[0])

        visited = [[0] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    visited[i][j] = 1

                    result = dfs(board, word, 1, visited, i, j)

                    if result:
                        return True

                    visited[i][j] = 0
        return False

    def numTrees(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 96. Unique Binary Search Trees
        '''

        res = 1
        
        for i in range(n):
            res = res * 2 * (2 * i + 1) // (i + 2)
            
        return res

    def levelOrder(self, root):
        '''
        :param root: TreeNode
        :return: List[List[int]]
        leetcode medium: 102. Binary Tree Level Order Traversal
        '''
        stack = [] ; display = []
        if root == None:
            return display
        else:
            stack.append(root)
            while len(stack) > 0:
                display.append([node.val for node in stack])
                new = []
                for node in stack:
                    if node.left:
                        new.append(node.left)
                    if node.right:
                        new.append(node.right)
                stack = new
            return display

    def maxProfit(self, prices):
        '''
        :param prices: List[int]
        :return: int
        leetcode medium: 122. Best Time to Buy and Sell Stock II
        Input: prices = [7,1,5,3,6,4]
        Output: 7
        '''
        previous = prices[0]
        profit   = 0

        for price in prices[1:]:
            if price > previous:
                profit += price - previous

            previous = price

        return profit

    def solve(self, board):
        '''
        :param board: List[List[str]]
        :return: NONE
        leetcode medium: 130. Surrounded Regions
        Input: board = [["X","X","X","X"],
                        ["X","O","O","X"],
                        ["X","X","O","X"],
                        ["X","O","X","X"]]

        Output: [["X","X","X","X"],
                 ["X","X","X","X"],
                 ["X","X","X","X"],
                 ["X","O","X","X"]]
        '''
        """
        Do not return anything, modify board in-place instead.
        """
        rlen = len(board)
        clen = len(board[0])

        # find borders
        borders = set()

        for i in range(rlen):
            for j in [0, clen - 1]:
                borders.add((i, j))

        for i in [0, rlen - 1]:
            for j in range(clen):
                borders.add((i, j))

        # mark "escaped" cells with 'E'
        direct = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        for i, j in borders:
            if board[i][j] == 'O':
                to_visit = [(i, j)]

                while to_visit:
                    curr_i, curr_j = to_visit.pop(0)
                    board[curr_i][curr_j] = 'E'

                    for diff_r, diff_c in direct:
                        r, c = curr_i + diff_r, curr_j + diff_c

                        if 0 <= r < rlen and 0 <= c < clen and board[r][c] == 'O':
                            board[r][c] = 'E'
                            to_visit.append((r, c))

        for i in range(rlen):
            for j in range(clen):
                if board[i][j] == 'O':
                    board[i][j] = 'X'

                elif board[i][j] == 'E':
                    board[i][j] = 'O'

    def partition(self, s):
        '''
        :param s: str
        :return: List[List[str]]
        leetcode medium: 131. Palindrome Partitioning
        Input: s = "aab"
        Output: [["a","a","b"],["aa","b"]]
        '''
        def dfs(index,item,res):
            if index == len(s):
                res.append(item)
                return

            for i in range(index, len(s)):
                tmpStr = s[index:i + 1]
                if tmpStr == tmpStr[::-1]:
                    dfs(i + 1,item + [tmpStr], res)

        res = []

        dfs(0,[],res)

        return res

    def singleNumberII(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode medium: 137. Single Number II
        Input: nums = [0,1,0,1,0,1,99]
        Output: 99
        '''
        '''
        let each number to three and sum , then subtract sum(nums) then divide the result by 2 is answer
        return (3 * sum(set(nums)) - sum(nums)) // 2
        '''

        one = two = 0

        for n in nums:
            # 1 & ~1 = 0
            one = (one ^ n) & ~two # if n in one both n in two ,& ~two == (n & ~n)
            two = (two ^ n) & ~one

        return one

    def reorderList(self, head):
        '''
        :param head: Optional[ListNode]
        :return: None
        leetcode medium: 143. Reorder List

        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next

        Input: head = [1,2,3,4,5]
        Output: [1,5,2,4,3]
        '''
        """
        Do not return anything, modify head in-place instead.
        """
        slow = head
        fast = head

        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        head2 = slow.next
        slow.next = None

        prev = None
        curr = head2

        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp

        head2 = prev

        head1 = head

        while head1 and head2:
            temp1 = head1.next
            temp2 = head2.next
            head1.next = head2
            head2.next = temp1
            head2 = temp2
            head1 = temp1

    def maxProduct(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode medium: 152. Maximum Product Subarray
        Input: nums = [2,3,-2,4]
        Output: 6
        Explanation: [2,3] has the largest product 6.
        '''
        result    = nums[0]
        min_value = nums[0]
        max_value = nums[0]

        for num in nums[1:]:
            tmp       = max_value
            max_value = max(num, tmp * num, min_value * num)
            min_value = min(num, tmp * num, min_value * num)
            result    = max(result, min_value, max_value)

        return result

    def largestNumber(self, nums):
        '''
        :param nums: List[int]
        :return: str
        leetcode medium: 179. Largest Number
        '''
        def merge_sort(nums):
            if len(nums) <= 1:
                return nums

            length = len(nums) // 2

            l = merge_sort(nums[:length])
            r = merge_sort(nums[length:])
            return merge(l, r)

        def merge(l, r):
            result = []
            i = 0
            j = 0

            while i < len(l) and j < len(r):
                if int(str(l[i]) + str(r[j])) > int(str(r[j]) + str(l[i])):
                    result.append(l[i])
                    i += 1
                else:
                    result.append(r[j])
                    j += 1

            while i < len(l):
                result.append(l[i])
                i += 1

            while j < len(r):
                result.append(r[j])
                j += 1

            return result

        new_nums = merge_sort(nums)

        return str(int("".join(map(str, new_nums))))

    def rotate_189(self, nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: None
        leetcode medium: 189. Rotate Array
        '''
        """
        Do not return anything, modify nums in-place instead.
        """
        # This solution is use -k % len(nums) (if k > len(nums) to reorganize nums self
        # nums[:] = nums[-k % len(nums):] + nums[:-k % len(nums)]
        # -k or len(num) - k

        while k:
            nums.insert(0,nums.pop())
            k -= 1

    def numIslands(self, grid):
        '''
        :param grid: List[List[str]]
        :return: int
        leetcode medium: 200. Number of Islands
                grid = [['1','1','1','1','0'],
                ['1','1','0','1','0'],
                ['1','1','0','0','0'],
                ['0','0','0','0','0']]
        '''
        def dfs(grid, i, j):
            grid[i][j] = '0'

            checks = [[0, -1], [0, 1], [-1, 0], [1, 0]]

            for check in checks:
                nr = i + check[0]
                nc = j + check[1]

                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                    if grid[nr][nc] == '1':
                        dfs(grid, nr, nc)

        cnt = 0

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':
                    cnt += 1
                    dfs(grid, r, c)

        return cnt

    def rangeBitwiseAnd(self, m, n):
        '''
        :param m: int
        :param n: int
        :return: int
        leetcode medium: 201. Bitwise AND of Numbers Range
        '''
        if m == n:
            return n
        mask = 1
        dif = n - m
        bit = 1
        while dif >= (1 << bit):
            mask = mask | (mask << 1)
            bit += 1

        return m & (~mask) & n

    def minSubArrayLen(self, s, nums):
        '''
        :param s: int
        :param nums: List[int]
        :return: int
        leetcode medium: 209. Minimum Size Subarray Sum
        '''
        ind, res, l = 0, 0, float('inf')

        for i in range(len(nums)):
            res += nums[i]

            while res >= s:
                l = min(l, i - ind + 1)
                res -= nums[ind]
                ind += 1

        return l if l != float('inf') else 0

    def findOrder(self, numCourses, prerequisites):
        '''
        :param numCourses: int
        :param prerequisites: List[List[int]]
        :return: List[int]
        leetcode medium: 210. Course Schedule II

        Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
        Output: [0,2,1,3] or [0,1,2,3]

        Explanation: There are a total of 4 courses to take.
        To take course 3 you should have finished both courses 1 and 2.
        Both courses 1 and 2 should be taken after you finished course 0.
        So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3]
        '''
        indegree = [0] * numCourses
        edge = collections.defaultdict(list)

        for x, y in prerequisites:
            indegree[x] += 1
            edge[y].append(x)

        q = [i for i in range(numCourses) if indegree[i] == 0]
        res = []

        while q:
            cur = q.pop()
            res.append(cur)

            for n in edge[cur]:
                indegree[n] -= 1

                if indegree[n] == 0:
                    q.append(n)

        return res if len(res) == numCourses else []

    def maximalSquare(self, matrix):
        '''
        :param matrix: List[List[str]]
        :return: int
        leetcode medium: 221. Maximal Square

        Input: matrix = [["1","0","1","0","0"],
                         ["1","0","1","1","1"],
                         ["1","1","1","1","1"],
                         ["1","0","0","1","0"]]
        Output: 4

        this question has DP solution and more quickly
        '''
        def longestConsectutive1Bits(n):
            c = 0

            while n:
                n &= n << 1
                c += 1
            return c

        m = len(matrix)
        n = len(matrix[0])

        rols = []
        maxsqlen = 0
        # turn each row into a number
        # time O(mn)
        for r in range(m):
            v = 0
            for c in range(n - 1, -1, -1):
                v += int(matrix[r][c]) << (n - c - 1)
            if v > 0:  # there are at least one 1
                maxsqlen = 1
            rols.append(v)

        # iterate each value at ith againt the rest (i+1 to m-1)
        for (i, n) in enumerate(rols):
            # pruning when the rest operation cannot find a L which is greater than maxsqlen
            if maxsqlen >= m - i:
                break
            for j in range(i + 1, m):
                n &= rols[j]
                L = j - i + 1
                # if there is a squre in the rows between i and j,
                # which the length is graeter than L,
                # there must be at least L consectutive 1 bits.
                if longestConsectutive1Bits(n) < L:
                    break

                maxsqlen = max(maxsqlen, L)

        return maxsqlen * maxsqlen

    def singleNumberIII(self, nums):
        '''
        :param nums: List[int]
        :return: List[int]
        leetcode medium: 260. Single Number III
        Input: nums = [1,2,1,3,2,5]
        Output: [3,5]
        '''
        bitmask = nums[0]

        for num in nums[1:]: # if same number XOR = 0
            bitmask ^= num

        num1 = bitmask

        for num in nums:
            if num ^ bitmask > num: # This step is find bigger of the 2 numbers
                num1 ^= num

        num2 = num1 ^ bitmask

        return [num1, num2]

    def numSquares(self,n):
        '''
        :param n: int
        :return: int
        leetcode medium: 279. Perfect Squares
        '''
        dp    = [n] * (n + 1)
        dp[0] = 0
        dp[1] = 1

        for i in range(2, n + 1):
            j = 1

            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1

        return dp[-1]

    def gameOfLife(self, board):
        '''
        :param board: List[List[int]]
        :return: None
        leetcode medium: 289. Game of Life
        '''
        """
        Do not return anything, modify board in-place instead.
        """
        copy = []

        for i in range(len(board)):

            arr = []
            for j in range(len(board[0])):
                arr.append(board[i][j])

            copy.append(arr)

        m = len(board)
        n = len(board[0])

        for row in range(m):
            for col in range(n):
                cell = copy[row][col]

                sum = 0

                for r in range(row - 1, row + 2):
                    for c in range(col - 1, col + 2):
                        if (r >= 0 and r < m) and (c >= 0 and c < n):
                            sum += copy[r][c]

                sum -= cell

                if sum < 2 and cell == 1:
                    board[row][col] = 0

                elif (sum == 2 or sum == 3) and cell == 1:
                    board[row][col] = 1

                elif sum > 3 and cell == 1:
                    board[row][col] = 0

                elif sum == 3 and cell == 0:
                    board[row][col] = 1

    def getHint(self, secret, guess):
        '''
        :param secret: str
        :param guess:  str
        :return: str
        leetcode medium: 299. Bulls and Cows
        '''
        a = 0
        b = 0

        count = collections.Counter(secret)

        for i in range(len(secret)):
            if secret[i] == guess[i]:
                a += 1

            if guess[i] in secret and count[guess[i]] != 0:
                b += 1
                count[guess[i]] -= 1

        b -= a

        return ''.join([str(a), 'A', str(b), 'B'])

    def maxProfit_309(self, prices):
        '''
        :param prices: List[int]
        :return: int
        leetcode medium: 309. Best Time to Buy and Sell Stock with Cooldown
        Input: prices = [1,2,3,0,2]
        Output: 3
        '''
        dp = [[0 for _ in range(3)] for _ in range(len(prices))]

        dp[0][1] -= prices[0]

        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = dp[i - 1][1] + prices[i]

        return max(dp[-1])

    def oddEvenList(self, head):
        '''
        :param head: : Optional[ListNode]
        :return: : Optional[ListNode]
        leetcode medium: 328. Odd Even Linked List
        Input: head = [2,1,3,5,6,4,7]
        Output: [2,3,6,7,1,5,4]
        '''
        if not head or not head.next or not head.next.next:
            return head

        origin_head = head
        odds = head
        evens = head.next
        origin_evens = evens
        previor = odds

        while evens and evens.next:
            odds.next = evens.next
            evens.next = evens.next.next
            evens = evens.next
            previor = odds
            odds = odds.next

        if not odds:
            previor.next = origin_evens

        else:
            odds.next = origin_evens

        return origin_head

    def topKFrequent(self, nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: List[int]
        leetcode medium: 347. Top K Frequent Elements
        '''
        d = collections.Counter(nums)
        return [i[0] for i in sorted(d.items(),key=lambda item:item[1])][::-1][:k]

    def largestDivisibleSubset(self, nums):
        '''
        :param nums: List[int]
        :return: List[int]
        leetcode medium: 368. Largest Divisible Subset
        Input: nums = [1,2,3]
        Output: [1,2]
        Explanation: [1,3] is also accepted.

        Input: nums = [1,2,4,8]
        Output: [1,2,4,8]
        '''
        if not nums:
            return []

        nums.sort()
        dp = [0] * len(nums)

        # used to construct the subset after we find the largest size
        construct = [-1] * len(nums)
        best, best_i = -1, -1

        for i in range(len(nums)):
            for j in range(i):
                if nums[i] % nums[j] == 0 and dp[i] < dp[j]:
                    construct[i], dp[i] = j, dp[j]

            dp[i] += 1

            if dp[i] > best:
                best, best_i = dp[i], i
        # print(dp, construct)

        # construct the result
        result = [nums[best_i]]

        while (construct[best_i] > -1):
            best_i = construct[best_i]
            result.append(nums[best_i])

        return result[::-1]

    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        leetcode medium: 371. Sum of Two Integers
        """
        while b != 0:
            carry = a & b
            a = (a ^ b)
            b = (carry << 1)
        return a if a <= 0x7FFFFFFF else a|(~0x100000000+1)

    def kSmallestPairs(self, nums1, nums2, k):
        '''
        :param nums1: List[int]
        :param nums2: List[int]
        :param k: int
        :return: List[List[int]]
        leetcode medium: 373. Find K Pairs with Smallest Sums
        '''
        res = []
        for u in nums1:
            for v in nums2:
                res.append([u,v])

        res = sorted(res, key = lambda x:x[0]+x[1])

        return res[:k]

    def longestSubstring(self, s, k):
        '''
        :param s: str
        :param k: int
        :return: int
        leetcode medium: 395. Longest Substring with At Least K Repeating Characters
        '''
        if k > len(s):
            return 0

        f = self.longestSubstring

        char_occ_dict = collections.Counter(s)

        for character, occurrence in char_occ_dict.items():
            if occurrence < k:
                return max(f(sub_string, k) for sub_string in s.split(character))

        return len(s)

    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        s1 = "3[a]2[bc]"        # return "aaabcbc".
        s2 = "3[a2[c]]"         # return "accaccacc".
        s3 = "2[abc]3[cd]ef"    # return "abcabccdcdcdef".
        leetcode medium: 394. Decode String
        """
        curnum = 0
        curstring = ''
        stack = []
        for char in s:
            if char == '[':
                stack.append(curstring)
                stack.append(curnum)
                curstring = ''
                curnum = 0
            elif char == ']':
                prenum = stack.pop()
                prestring = stack.pop()
                curstring = prestring + prenum * curstring
            elif char.isdigit():
                curnum = curnum * 10 + int(char)
            else:
                curstring += char
        return curstring

    def integerReplacement(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 397. Integer Replacement
        '''
        cnt = 0
        while n != 1:
            if n % 2 == 0:
                n //= 2

            elif n % 4 == 1 or n == 3:
                n -= 1

            else:
                n += 1

            cnt += 1
        return cnt

    def removeKdigits(self, num, k):
        '''
        :param num: str
        :param k: int
        :return: str
        leetcode medium: 402. Remove K Digits

        Input: num = "1432219", k = 3
        Output: "1219"
        Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
        '''
        numStack = []

        for digit in num:
            while k and numStack and numStack[-1] > digit:
                numStack.pop()
                k -= 1

            numStack.append(digit)

        finalStack = numStack[:-k] if k else numStack

        return "".join(finalStack).lstrip('0') or "0"

    def canPartition(self, nums):
        '''
        :param nums: List[int]
        :return: bool
        leetcode medium: 416. Partition Equal Subset Sum

        Input: nums = [1,5,11,5]
        Output: true
        Explanation: The array can be partitioned as [1, 5, 5] and [11].
        '''

        dp = {0}

        for num in nums:
            dp = {num - i for i in dp} | {num + i for i in dp}

        return 0 in dp

    def findDuplicates(self, nums):
        '''
        :param nums: : List[int]
        :return: : List[int]
        leetcode medium: 442. Find All Duplicates in an Array
        '''
        ans = []

        for i in nums:
            if nums[abs(i) - 1] < 0:
                ans.append(abs(i))

            else:
                nums[abs(i) - 1] *= -1

        return ans
        # return [num for num,count in collections.Counter(nums).items() if count == 2]

    def deleteNode(self, root, key):
        '''
        :param root: Optional[TreeNode]
        :param key: int
        :return: Optional[TreeNode]
        leetcode medium: 450. Delete Node in a BST

        root = TreeNode(val   = 5,
                left  = TreeNode(val   = 3,
                                 left  = TreeNode(val = 2),
                                 right = TreeNode(val = 4)
                                 ),
                right = TreeNode(val   = 6,
                                 right = TreeNode(val = 7)
                                 )
                )

        Input: root = [5,3,6,2,4,null,7], key = 3
        Output: [5,4,6,2,null,null,7]
        '''
        def getMin(root):
            while root.left:
                root = root.left

            return root

        if not root:
            return None

        if root.val < key:
            root.right = self.deleteNode(root.right, key)

        if root.val > key:
            root.left = self.deleteNode(root.left, key)

        if root.val == key:
            if not root.left:
                return root.right

            if not root.right:
                return root.left

            minNode = getMin(root.right)

            root.val = minNode.val

            root.right = self.deleteNode(root.right, minNode.val)

        return root

    def findMinArrowShots(self, points):
        '''
        :param points: List[List[int]]
        :return: int
        leetcode medium: 452. Minimum Number of Arrows to Burst Balloons

        Input: points = [[10,16],[2,8],[1,6],[7,12]]
        Output: 2
        Explanation: The balloons can be burst by 2 arrows:
                      - Shoot an arrow at x = 6, bursting the balloons [2,8] and [1,6].
                      - Shoot an arrow at x = 11, bursting the balloons [10,16] and [7,12].
        '''

        '''
        sortedPoints = sorted(points, key= lambda x:x[1])
        overlapped = 0
        tail = sortedPoints[0][1]

        for i in sortedPoints[1:]:
            if i[0] <= tail:
                overlapped += 1
            else:
                tail = i[1]

        return len(sortedPoints) - overlapped
        '''

        points.sort()
        right = -float("inf")
        out = 0

        for point in points:
            if point[0] <= right:
                if point[1] < right:
                    right = point[1]
            else:
                right = point[1]
                out += 1

        return out

    def magicalString(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 481. Magical String
        '''
        if n == 0:
            return 0

        if n <= 3:
            return 1

        s, index = [1, 2, 2], 2

        while len(s) < n:
            s += [3 - s[-1]] * s[index]
            index += 1

        return s[:n].count(1)

    def findTargetSumWays(self,nums, S):
        '''
        :param nums: : List[int]
        :param S: int
        :return: int
        leetcode medium: 494. Target Sum
        '''
        '''
        D = collections.defaultdict(int)

        D[nums[0]] += 1
        D[-nums[0]] += 1

        for num in nums[1:]:
            D_tmp = collections.defaultdict(int)

            for k in D.keys():
                D_tmp[k + num] += D[k]
                D_tmp[k - num] += D[k]

            D = D_tmp

        return D[S]
        '''

        dp = collections.Counter()
        dp[0] = 1

        for n in nums:
            ndp = collections.Counter()

            for sgn in (1, -1):
                for k in dp.keys():
                    ndp[k + n * sgn] += dp[k]

            dp = ndp

        return dp[S]

    def findDiagonalOrder(self, matrix):
        '''
        :param matrix: List[List[int]]
        :return: List[int]
        leetcode medium: 498. Diagonal Traverse
        '''
        '''
        Input:
        [
         [ 1, 2, 3 ],
         [ 4, 5, 6 ],
         [ 7, 8, 9 ]
        ]
        
        Output:  [1,2,4,7,5,3,6,8,9]
        '''
        if not matrix:
            return []

        n = 0
        m = 0

        isUp = True
        res = []

        while n < len(matrix) and m < len(matrix[0]):
            res.append(matrix[n][m])

            if isUp:
                if n > 0 and m < len(matrix[0]) - 1:
                    n -= 1
                    m += 1
                else:
                    isUp = False
                    if m < len(matrix[0]) - 1:
                        m += 1
                    else:
                        n += 1
            else:
                if m > 0 and n < len(matrix) - 1:
                    n += 1
                    m -= 1
                else:
                    isUp = True

                    if n < len(matrix) - 1:
                        n += 1
                    else:
                        m += 1

        return res

    def findPairs(self, nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: int
        leetcode medium: 532. K-diff Pairs in an Array
        '''
        res = set()

        nums = sorted(nums)

        i = 0
        j = 1

        while i < j and j < len(nums):
            if nums[j] - nums[i] == k:
                res.add((nums[i], nums[j]))
                i += 1
                j = i + 1

            elif nums[j] - nums[i] > k:
                i += 1
                j = i + 1

            else:
                j += 1

        return len(res)

    def nextGreaterElement(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 556. Next Greater Element III
        '''
        m = list(str(n))  ## n = 257761
        l = len(m)        ## l = 6
        d = {}
        res = str(n)

        ## reading character backwards: 1->6->7->7->5 break
        for i, c in enumerate(m[::-1]):
            if not d:
                d[c] = 1  ## d = {'1':1}
            else:
                if all(c >= x for x in d):
                    d[c] = d.get(c, 0) + 1              ## d = {'1':1,'6':1,'7':2}

                else:
                    d[c]  = d.get(c, 0) + 1             ## d = {'1':1,'5':1,'6':1,'7':2}
                    res   = ''.join(m[:l - 1 - i])      ## res = '2'
                    stock = sorted(list(d.keys()))      ## stock = ['1','5','6','7']
                    cplus = stock[stock.index(c) + 1]   ## cplus = '6' just > '5'
                    res += cplus                        ## res = '26'
                    d[cplus] -= 1                       ## d = {'1':1,'5':1,'6':0,'7':2}
                    res += ''.join([x * d[x] for x in stock])
                    ## res = '26' + '1577'

                    break

        return int(res) if n < int(res) < (2 ** 31 - 1) else -1

    def arrayNesting(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode medium: 565. Array Nesting
        '''
        _set = set()
        maxd = 1

        for i in range(len(nums)):
            index = i
            curr = nums[index]

            if index not in _set:
                d = 0

                while index != curr and index not in _set:
                    _set.add(index)

                    index = curr
                    curr = nums[index]

                    d += 1

                maxd = max(maxd, d)

        return maxd

    def findUnsortedSubarray(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode medium: 581. Shortest Unsorted Continuous Subarray
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

    def judgeSquareSum(self, c):
        '''
        :param c: int
        :return: bool
        leetcode medium: 633. Sum of Square Numbers
        '''
        left = 0
        right = int(c ** 0.5)
        while left <= right:
            cur = left ** 2 + right ** 2
            if cur < c:
                left += 1
            elif cur > c:
                right -= 1
            else:
                return True
        return False

    def predictPartyVictory(self, senate):
        '''
        :param senate: str
        :return: str
        leetcode medium: 649. Dota2 Senate
        '''
        # Difficulty:middle
        l = len(senate)
        R = [i for i in range(l) if senate[i] == 'R']
        D = [i for i in range(l) if senate[i] == 'D']

        while len(D) and len(R):
            if D[0] > R[0]:
                R.append(R[0] + l)
            else:
                D.append(D[0] + l)

            D.pop(0)
            R.pop(0)

        return "Dire" if len(D) != 0 else "Radiant"

    def canPartitionKSubsets(self,nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: bool
        leetcode medium: 698. Partition to K Equal Sum Subsets
        '''
        def dfs(nums, k, index, target):
            if index == len(nums):
                return True

            num = nums[index]

            for i in range(k):
                if target[i] >= num:
                    target[i] -= num

                    if dfs(nums, k, index + 1, target):
                        return True

                    target[i] += num

            return False

        div = sum(nums) // k

        if not nums or len(nums) < k or sum(nums) % k or max(nums) > div:
            return False

        nums.sort(reverse=True)

        target = [div] * k

        return dfs(nums, k, 0, target)

    def accountsMerge(self, accounts):
        '''
        :param accounts: List[List[str]]
        :return: List[List[str]]
        leetcode medium: 721. Accounts Merge
        Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],
                           ["John","johnsmith@mail.com","john00@mail.com"],
                           ["Mary","mary@mail.com"],
                           ["John","johnnybravo@mail.com"]]

        Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],
                 ["Mary","mary@mail.com"],
                 ["John","johnnybravo@mail.com"]]
        '''
        graph = collections.defaultdict(set)

        for acct in accounts:
            center = acct[1]
            for email in acct[2:]:
                graph[center].add(email)
                graph[email].add(center)

        seen = set()

        def dfs(email, acct):
            stack = [email]
            seen.add(email)

            while stack:
                cur = stack.pop()
                acct.append(cur)
                for nei in graph[cur]:
                    if nei not in seen:
                        seen.add(nei)
                        stack.append(nei)

        ans = []

        for acct in accounts:
            name = acct[0]
            email = acct[1]

            if email not in seen:
                cur = []
                dfs(email, cur)
                ans.append([name] + sorted(cur))

        return ans

    def splitListToParts(self, head, k):
        '''
        :param head: : Optional[ListNode]
        :param k: int
        :return: List[Optional[ListNode]]
        leetcode medium: 725. Split Linked List in Parts
        '''
        nums = []

        while head:
            nums.append(head.val)
            head = head.next

        _list = []

        if len(nums) < k:

            _list = []

            while k:
                if nums:
                    _list.append(ListNode(nums.pop(0)))

                else:
                    _list.append(ListNode(''))

                k -= 1

            return _list

        point = len(nums) // k
        other = len(nums) % k

        _list = [[] for _ in range(k)]

        for i in _list:
            for j in range(point):
                i.append(nums.pop(0))

            if other:
                i.append(nums.pop(0))
                other -= 1

        ans = []

        for i in _list:
            head = ListNode(i[0])
            h = head

            for j in i[1:]:
                head.next = ListNode(j)
                head = head.next

            ans.append(h)

        return ans

    def dailyTemperatures(self, temperatures):
        '''
        :param temperatures: List[int]
        :return: List[int]
        leetcode medium: 739. Daily Temperatures

        Input: temperatures = [73,74,75,71,69,72,76,73]
        Output: [1,1,4,2,1,1,0,0]
        Output: [1,1,4,2,1,1,0,0]
        '''
        stack = []
        answer = [0] * len(temperatures)

        for i in range(len(temperatures) - 1, -1, -1):
            while stack and temperatures[i] >= temperatures[stack[-1]]:
                stack.pop()

            if stack:
                answer[i] = stack[-1] - i

            stack.append(i)

        return answer

    def deleteAndEarn(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode easy: 740. Delete and Earn

        Input: nums = [3,4,2]
        Output: 6
        Explanation: You can perform the following operations:
        - Delete 4 to earn 4 points. Consequently, 3 is also deleted. nums = [2].
        - Delete 2 to earn 2 points. nums = [].
        You earn a total of 6 points.
        '''
        c = collections.Counter(nums)

        keys = sorted(c.keys())
        prev = 0

        ans = cur = c[keys[0]] * keys[0]

        for i in range(1, len(keys)):
            if keys[i] == keys[i - 1] + 1:
                prev, cur = cur, max(cur, prev + keys[i] * c[keys[i]])

            else:
                prev, cur = cur, cur + keys[i] * c[keys[i]]

            ans = max(ans, cur)

        return ans

    def reachNumber(self, target):
        '''
        :param target: int
        :return: int
        leetcode medium: 754. Reach a Number
        '''
        t = abs(target)

        n = int(((1 + 8 * t) ** 0.5 - 1) / 2)

        LT = n * (n + 1) // 2

        if LT == t:
            return n

        if (LT + n + 1 - t) % 2 == 0:
            return n + 1

        return n + 3 - n % 2

    def letterCasePermutation(self, S):
        '''
        :param S: str
        :return: List[str]
        leetcode medium: 784. Letter Case Permutation
        Input: S = "a1b2"
        Output: ["a1b2","a1B2","A1b2","A1B2"]
        '''
        def dfs(index, item, ans):
            if index == len(S):
                ans.append(item)
                return

            if S[index].isalpha():
                dfs(index + 1, item + S[index].lower(), ans)

            dfs(index + 1, item + S[index].upper(), ans)

        ans = []

        dfs(0, "", ans)

        return ans

    def numTilings(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 790. Domino and Tromino Tiling
        '''
        if n <= 2:
            return n

        t1 = 2
        t2 = 1
        p1 = 1  # one tromino
        q  = 10 ** 9 + 7

        for _ in range(3, n + 1):
            T = t1 + t2 + 2 * p1
            P = p1 + t2

            t1, t2 = T % q, t1 % q
            p1 = P % q

        return t1

    def champagneTower(self, poured, query_row, query_glass):
        '''
        :param poured: int
        :param query_row: int
        :param query_glass: int
        :return: float
        leetcode medium: 799. Champagne Tower
        '''
        if poured == 0:
            return 0

        if query_row == 0:
            return 1

        curr = [poured]

        for row in range(1, query_row + 1):
            nxt = [0] * (row + 1)

            for i in range(len(curr)):
                remaining = curr[i] - 1

                if remaining < 0.0:
                    continue

                nxt[i] += remaining / 2
                nxt[i + 1] += remaining / 2

            curr = nxt

        return min(curr[query_glass], 1)

    def numMagicSquaresInside(self, grid):
        '''
        :param grid:List[List[int]]
        :return: int
        leetcode medium: 840. Magic Squares In Grid
        '''

        def is_magic(i, j):
            sb_grid = [grid[i]    [j:j + 3],
                       grid[i + 1][j:j + 3],
                       grid[i + 2][j:j + 3]]

            nums = [sb_grid[i][j] for i in range(3) for j in range(3)]

            for num in range(1, 10):
                if num not in nums:
                    return False

            ans = [0, 0]

            for sb_i in range(3):
                ans.append(sum(sb_grid[sb_i]))
                ans[0] += sb_grid[sb_i][sb_i]
                ans[1] += sb_grid[sb_i][3 - 1 - sb_i]

            for sb_i in range(3):
                for sb_j in range(sb_i, 3):
                    if sb_i != sb_j:
                        sb_grid[sb_i][sb_j], sb_grid[sb_j][sb_i] = sb_grid[sb_j][sb_i], sb_grid[sb_i][sb_j]

                ans.append(sum(sb_grid[sb_i]))

            return len(set(ans)) == 1

        if len(grid) < 3 or len(grid[0]) < 3:
            return 0

        count = 0

        for i in range(len(grid) - 2):
            for j in range(len(grid[0]) - 2):
                if is_magic(i, j):
                    count += 1

        return count

    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        leetcode medium: 841. Keys and Rooms
        Input: [[1],[2],[3],[]] ,Output: true
        Input: [[1,3],[3,0,1],[2],[0]], Output: false
        """
        q, path = [0], [0]

        while q:
            index = q.pop()
            for v in rooms[index]:
                if v not in path:
                    q.append(v)
                    path.append(v)

        return len(path) == len(rooms)

    def decodeAtIndex(self, S, K):
        '''
        :param S: str
        :param K: int
        :return: str
        leetcode medium: 880. Decoded String at Index
        '''
        lens, n = [0], len(S)

        for c in S:
            if c.isdigit():
                lens.append(lens[-1]*int(c))
            else:
                lens.append(lens[-1] + 1)

        for i in range(n, 0, -1):
            K %= lens[i]
            if K == 0 and S[i - 1].isalpha():
                return S[i - 1]

    def brokenCalc(self, X, Y):
        '''
        :param X: int
        :param Y: int
        :return: int
        leetcode medium: 991. Broken Calculator
        Input: X = 2, Y = 3
        Output: 2
        Explanation: Use double operation and then decrement operation {2 -> 4 -> 3}.
        '''
        '''
        recursive :
        def brokenCalc(self, X: int, Y: int) -> int:
            if X >= Y:
                return X - Y
            if Y % 2 == 0:
                return 1 + self.brokenCalc(X, Y // 2)
            if Y % 2 == 1:
                return 2 + self.brokenCalc(X, (Y + 1) // 2)            
        
        Bit manipulation, O(log log n)
        def brokenCalc(self, X: int, Y: int) -> int:
            count = 0
            while Y > X:
                if Y&1:
                    Y += 1
                    count += 1
                Y >>= 1
                count += 1
                
            return count + X-Y     
        '''

        count = 0
        while X < Y:
            count += 1

            if Y % 2:
                Y += 1
            else:
                Y //= 2

        return X - Y + count

    def orangesRotting(self, grid):
        '''
        :param grid: List[List[int]]
        :return: int
        leetcode medium: 994. Rotting Oranges
        Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
        Output: 4
        '''
        M      = len(grid)
        N      = len(grid[0])
        offset = [(1, 0), (-1, 0), (0, -1), (0, 1)]

        rotten = []

        for i in range(M):
            for j in range(N):
                if grid[i][j] == 2:
                    rotten.append((i, j))

        minute = 0

        while rotten:
            rotten2 = []

            for i, j in rotten:
                adj = [(i + x, j + y) for x, y in offset]
                for x, y in adj:
                    if 0 <= x < M and 0 <= y < N and grid[x][y] == 1:
                        rotten2.append((x, y))
                        grid[x][y] = 2

            minute += 1 if rotten2 else 0
            rotten = rotten2

        for i in range(M):
            for j in range(N):
                if grid[i][j] == 1:
                    return -1

        return minute

    def minDominoRotations(self, A, B):
        '''
        :param A: List[int]
        :param B: List[int]
        :return: int
        leetcode medium: 1007. Minimum Domino Rotations For Equal Row
        '''
        d = {i: set() for i in range(1, 7)}

        for i in range(len(A)):
            d[A[i]].add(i)
            d[B[i]].add(i)

        res = float('inf')
        flag = True

        for k in d:
            if len(d[k]) == len(A):
                flag = False
                top, bottom = 0, 0

                for i in range(len(A)):
                    if A[i] != k:
                        top += 1

                    if B[i] != k:
                        bottom += 1

                res = min([res, top, bottom])

        return -1 if flag else res

    def bstFromPreorder(self, preorder):
        '''
        :param preorder: List[int]
        :return: Optional[TreeNode]
        leetcode medium: 1008. Construct Binary Search Tree from Preorder Traversal
        '''
        if not preorder:
            return None

        root = TreeNode(preorder[0])

        for num in preorder[1:]:
            node = root

            while True:
                if node.val > num:
                    if node.left is None:
                        node.left = TreeNode(num)
                        break

                    else:
                        node = node.left

                else:
                    if node.right is None:
                        node.right = TreeNode(num)
                        break

                    else:
                        node = node.right

        return root

    def smallestRepunitDivByK(self, k):
        '''
        :param K: int
        :return: int
        leetcode medium: 1015. Smallest Integer Divisible by K
        '''
        if (k % 2 == 0) or (k % 5 == 0):
            return -1

        remainder = 0

        for n in range(1, k + 1):
            remainder = (remainder * 10 + 1) % k # >> 222 = (22*10) + 2 , 5555 = (555*10) + 5

            if remainder == 0:
                return n

    def carPooling(self, trips, capacity):
        '''
        :param trips: List[List[int]]
        :param capacity: int
        :return: bool
        leetcode medium: 1094. Car Pooling

        Input: trips = [[2,1,5],[3,3,7]], capacity = 4
        Output: false

        Input: trips = [[2,1,5],[3,3,7]], capacity = 5
        Output: true

        Input: trips = [[2,1,5],[3,5,7]], capacity = 3
        Output: true
        '''
        passengers = [0] * 1001 # make a list let it match the condition ,
                                # it each position represent how much passenger in the position at same time
        for trip in trips:
            trip_nump = trip[0]
            trip_from = trip[1]
            trip_to   = trip[2]

            for i in range(trip_from, trip_to):
                passengers[i] += trip_nump

                if passengers[i] > capacity: # judge the position have how much passenger
                    return False

        return True

    def longestCommonSubsequence(self, text1, text2):
        '''
        :param text1: str
        :param text2: str
        :return: int
        leetcode medium: 1143. Longest Common Subsequence
        '''
        m, n = len(text1), len(text2)

        dp = [[0 for _ in range(n)] for _ in range(m)]

        if text1[0] == text2[0]:
            dp[0][0] = 1

        for j in range(1, n):
            if text2[j] == text1[0]:
                for k in range(j, n):
                    dp[0][k] = 1

                break

            else:
                dp[0][j] = dp[0][0]

        for i in range(1, m):
            if text1[i] == text2[0]:
                for k in range(i, m):
                    dp[k][0] = 1

                break

            else:
                dp[i][0] = dp[0][0]

        for i in range(1, m):
            for j in range(1, n):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1

                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m - 1][n - 1]

    def smallestDivisor(self, nums, threshold):
        '''
        :param nums: List[int]
        :param threshold: int
        :return: int
        leetcode medium: 1283. Find the Smallest Divisor Given a Threshold
        '''
        start = 1
        end = max(nums)

        while start < end:
            mid = (start + end) // 2
            temp = 0

            for num in nums:
                temp += (num + mid - 1) // mid

            if temp > threshold:
                start = mid + 1

            else:
                end = mid

        return start

    def isPossibleDivide(self, nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: bool
        leetcode medium: 1296. Divide Array in Sets of K Consecutive Numbers
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

    def canReach(self, arr, start):
        '''
        :param arr: List[int]
        :param start: int
        :return: bool
        leetcode medium: 1306. Jump Game III
        Input: arr = [3,0,2,1,2], start = 2
        Output: false
        Explanation: There is no way to reach at index 1 with value 0.
        This solution is BFS
        '''
        jumps = [start]

        while jumps:
            current = jumps.pop(0)

            if arr[current] == -1:
                continue

            elif arr[current] == 0:
                return True

            next_steps = current + arr[current], current - arr[current]

            # Add next steps to the queue
            if next_steps[0] < len(arr):
                jumps.append(next_steps[0])

            if next_steps[1] >= 0:
                jumps.append(next_steps[1])

            # Mark the current index as visited
            arr[current] = -1

        return False

    def goodNodes(self, root):
        '''
        :param root: : TreeNode
        :return: int
        leetcode medium: 1448. Count Good Nodes in Binary Tree
        '''
        count = 0
        stack = [(root.val, root)]

        while stack:
            max_value, node = stack.pop()

            if node.val >= max_value:
                count += 1

            if node.right:
                stack.append((max(node.right.val, max_value), node.right))

            if node.left:
                stack.append((max(node.left.val, max_value), node.left))

        return count

    def countVowelStrings(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 1641. Count Sorted Vowel Strings
        '''
        a, e, i, o, u = 1, 1, 1, 1, 1
        for _ in range(n):
            a, e, i, o, u = a+e+i+o+u, e+i+o+u, i+o+u, o+u, u
        return a

    def findBall(self, grid):
        '''
        :param grid: List[List[int]]
        :return: List[int]
        leetcode medium: 1706. Where Will the Ball Fall
        '''
        m, n = len(grid), len(grid[0])
        answer = list(range(n))

        for r in range(m):
            for i in range(n):
                c = answer[i]

                if c == -1:
                    continue

                c_nxt = c + grid[r][c]

                if c_nxt < 0 or c_nxt >= n or grid[r][c_nxt] == -grid[r][c]:
                    answer[i] = -1
                    continue

                answer[i] += grid[r][c]

        return answer

class LeetCode_Hard:
    def findMedianSortedArrays(self, nums1, nums2):
        '''
        :param nums1: List[int]
        :param nums2: List[int]
        :return: float
        leetcode Hard: 4. Median of Two Sorted Arrays
        '''
        nums1.extend(nums2)

        nums1.sort()

        l = len(nums1)
        if l % 2 == 0:
            return (nums1[(l // 2) - 1] + nums1[l // 2]) / 2
        else:
            return nums1[(l + 1) // 2 - 1]

    def largestRectangleArea(self, heights):
        '''
        :param heights: List[int]
        :return: int
        leetcode Hard: 84. Largest Rectangle in Histogram
        '''
        heights.append(0)
        stack, area = [], 0

        for i in range(len(heights)):
            while stack and heights[stack[-1]] >= heights[i]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                area = max(area, h * w)

            stack.append(i)
        return area

    def maximalRectangle(self, matrix):
        '''
        :param matrix: List[List[str]]
        :return: int
        leetcode Hard: 85. Maximal Rectangle
        Input: matrix = [["1","0","1","0","0"],
                         ["1","0","1","1","1"],
                         ["1","1","1","1","1"],
                         ["1","0","0","1","0"]]
        Output: 6
        '''
        res, histRow = 0, ([0 for _ in matrix[0]]) if matrix else None

        for rowNums in matrix:
            stack = []

            for c, num in enumerate(rowNums):
                histRow[c] = (histRow[c] + 1) if num == '1' else 0

            for i, n in enumerate(histRow + [0]):
                while stack and histRow[stack[-1]] > n:
                    h = histRow[stack.pop()]
                    res = max(res, h * ((i - stack[-1] - 1) if stack else i))

                stack.append(i)

        return res

    def findKthNumber(self, m: int, n: int, k: int) -> int:
        '''
        :param m: int
        :param n: int
        :param k: int
        :return: int
        leetcode Hard: 668. Kth Smallest Number in Multiplication Table
        Input: m = 3, n = 3, k = 5
        Output: 3
        Explanation: The 5th smallest number is 3.
        '''
        if m > n:
            m, n = n, m

        def search(number):
            ans = 0

            for i in range(1, min(m, number) + 1):
                ans += min(n, number // i)

            return ans

        left = 1
        right = m * n

        while (left < right):
            mid = (left + right) // 2

            if search(mid) < k:
                left = mid + 1

            else:
                right = mid

        return left

    def orderlyQueue(self, s, k):
        '''
        :param s: str
        :param k: int
        :return: str
        leetcode Hard: 899. Orderly Queue
        '''
        if k == 1:
            '''
            _list = []
            
            for i in range(len(s)):
                _list.append(s[i:] + s[:i])
            
            return min(_list) # [cba,bac,acb] as following 
            '''
            return min(s[i:] + s[:i] for i in range(len(s)))

        return ''.join(sorted(s))

    def uniquePathsIII(self, grid):
        '''
        :param grid: List[List[int]
        :return: int
        leetcode Hard: 980. Unique Paths III
        '''
        m, n = len(grid), len(grid[0])
        start = None
        count = 0

        for i in range(m):
            for j in range(n):
                count += grid[i][j] == 0

                if not start and grid[i][j] == 1:
                    start = (i, j)

        def backtrack(i, j):
            '''
            :param i: int
            :param j: int
            :return: int
            '''
            nonlocal count
            result = 0
            for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):

                if 0 <= x < m and 0 <= y < n:
                    if grid[x][y] == 0:

                        grid[x][y] = -1
                        count -= 1
                        result += backtrack(x, y)

                        grid[x][y] = 0
                        count += 1

                    elif grid[x][y] == 2:

                        result += count == 0

            return result

        return backtrack(start[0], start[1])

    def findNumOfValidWords(self, words, puzzles):
        '''
        :param words: List[str]
        :param puzzles: List[str]
        :return: List[int]
        leetcode Hard: 1178. Number of Valid Words for Each Puzzle
        Input: words = ["aaaa","asas","able","ability","actt","actor","access"],
               puzzles = ["aboveyz","abrodyz","abslute","absoryz","actresz","gaswxyz"]
        Output: [1,1,3,2,4,0]
        '''
        def getBitMask(word):
            mask = 0

            for c in word:
                i = ord(c) - ord('a')
                mask |= 1 << i

            return mask

        letterFrequencies = {}

        for word in words:
            mask = getBitMask(word)
            letterFrequencies[mask] = letterFrequencies.get(mask, 0) + 1

        solution = [0] * len(puzzles)

        for i in range(len(puzzles)):
            puzzle  = puzzles[i]
            mask    = getBitMask(puzzle)
            subMask = mask
            total   = 0

            firstBitIndex = ord(puzzle[0]) - ord('a')

            while True:

                if subMask >> firstBitIndex & 1:
                    total += letterFrequencies.get(subMask, 0)

                if subMask == 0:
                    break

                subMask = (subMask - 1) & mask

            solution[i] = total

        return solution

    def minJumps(self, arr):
        '''
        :param arr: List[int]
        :return: int
        leetcode Hard: 1345. Jump Game IV
        '''
        n = len(arr)

        # Connect all indexes that have the same value.
        connections = collections.defaultdict(list)

        for i, num in enumerate(arr):
            connections[num].append(i)

        # Breadth First Search layer by layer using lists
        distance = 0
        layer = [0]
        visited = {0}

        while layer:
            next_layer = []

            for i in layer:
                # Return the distance if we have arrived at the last index.
                if i == n - 1:
                    return distance

                # Add the direct unvisited neighbors of i to the stack.
                if i > 0 and i - 1 not in visited:
                    next_layer.append(i - 1)
                    visited.add(i - 1)
                if i + 1 < n and i + 1 not in visited:
                    next_layer.append(i + 1)
                    visited.add(i + 1)

                # Add all unvisited connected indexes that share the
                # same value to the stack.
                if arr[i] in connections:
                    for j in connections[arr[i]]:
                        if j not in visited:
                            next_layer.append(j)
                            visited.add(j)

                    # The earliest possibility to visit the nodes in
                    # connections[arr[i]] has been found. Therefore
                    # we can now discard the list to prevent further
                    # visits.
                    del connections[arr[i]]

            distance += 1
            layer = next_layer

        return -1

    def countOrders(self, n: int) -> int:
        '''
        :param n: int
        :return: int
        leetcode hard: 1359. Count All Valid Pickup and Delivery Options

        Input: n = 2
        Output: 6
        Explanation: All possible orders:
        (P1,P2,D1,D2), (P1,P2,D2,D1), (P1,D1,P2,D2), (P2,P1,D1,D2), (P2,P1,D2,D1) and (P2,D2,P1,D1).
        This is an invalid order (P1,D2,P2,D1) because Pickup 2 is after of Delivery 2.

        Input: n = 3
        Output: 90

        solution : one pickup service have (2*1)-1 delivery service (1*3*5...2n-1)
                    equal = n! (1*3*5...2n-1)
        '''
        count = 1

        for i in range(2, n + 1):
            count = count * (2 * i - 1) * i

        return count % (10 ** 9 + 7)
        # return 1 if n == 1 else (self.countOrders(n-1) * (2*n-1) * n) % (10**9+7)

class DP:
    def searchInsert(self,nums ,target):
        '''
        :param nums: list[int]
        :param target: int
        :return: int
        leetcode easy: 35. Search Insert Position
        '''
        for index in range(len(nums)):
            if nums[index] == nums[-1] and nums[index] < target:
                return index+1
            if nums[index] == target or nums[index] > target:
                return index

    def combinationSum(self, candidates, target):
        '''
        :param candidates: List[int]
        :param target:     int
        :return:           List[List[int]]
        leetcode medium: 39. Combination Sum
        '''
        if target < min(candidates):
            return None

        dp = [[[]]] + [[] for _ in range(target)]

        for candidate in candidates:
            for i in range(candidate, target + 1):
                dp[i] += [sublist + [candidate] for sublist in dp[i - candidate]]

        return dp[target]

    def maxsubarray(self,nums):
        '''
        :param nums: int
        :return: int
        leetcode easy: 53. Maximum Subarray
        '''
        dp = [0] * len(nums)
        dp[0] = nums[0]

        for i in range(1, len(nums)):
            # explanation: if len(nums) > 1 , compare between nums[i] with nums[i] + dp[i-1]
            # because dp[i-1] is record previous step compare result

            dp[i] = max(nums[i], nums[i] + dp[i - 1])

        return max(dp)

    def permute(self, nums):
        '''
        :param nums: List[int]
        :return: list[List[int]]
        leetcode medium: 46. Permutations
        '''
        L = [[nums[0]]]
        for n in nums[1:]:
            new_L = []
            for perm in L:
                for i in range(len(perm) + 1):
                    new_L.append(perm[:i] + [n] + perm[i:])
                    L = new_L
        return L

    def canJump1(self,A):
        '''
        input : [2,3,1,1,4] :true
                [3,2,1,0,4] :False
        :param A: List
        :return: bool
        leetcode medium: 55. Jump Game
        '''
        if not A:
            return False
        n = len(A)
        dp = [False for _ in range(n)]
        dp[0] = True
        for i in range(n):
            for j in range(i):
                if dp[j] and j+A[j] >=i:
                    dp[i] = True
                    break
        return dp[n-1]

    def canJump2(self,A):
        '''
        input : [2,3,1,1,4]
        :param A: List
        :return: bool
        leetcode medium: 55. Jump Game
        '''
        if not A:
            return False
        n = len(A)
        dp = [float('inf')] * n
        dp[0] = 0
        for i in range(1,n):
            for j in range(i):
                if dp[j] != float('inf') and j + A[j] >= i:
                    dp[i] = min(dp[i],dp[j]+1)
        return dp[n-1]

    def uniquePaths(self,m,n):
        '''
        :param m: int
        :param n: int
        :return: int
        leetcode medium: 62. Unique Paths
        '''
        dp = [[1] * n for _ in range(m)]

        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]

        return dp[m - 1][n - 1]

    def uniquePathsWithObstacles(self, obstacleGrid):
        '''
        :param obstacleGrid:  List[List[int]]
        :return: int
        leetcode medium: 63. Unique Paths II
        '''
        dp = [[0] * (len(obstacleGrid[0]) + 1) for _ in range(len(obstacleGrid) + 1)]

        dp[0][1] = 1

        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if obstacleGrid[i - 1][j - 1] == 0:  # can go there
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[-1][-1]

    def minPathSum(self, grid):
        '''
        input :[[1,3,1],[1,5,1],[4,2,1]]
        output:7
        ans : 1 -> 3 -> 1 -> 1 -> 1
        :param grid: List[List[int]]
        :return: int
        leetcode medium: 64. Minimum Path Sum
        '''
        m = len(grid)
        n = len(grid[0])
        if m == 0 or n == 0:
            return 0
        dp = [[0] * n for i in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]

        for i in range(1, n):
            dp[0][i] = dp[0][i - 1] + grid[0][i]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        return dp[m - 1][n - 1]

    def climbStairs(self,n):
        '''
        :param n: int
        :return: int
        leetcode easy: 70. Climbing Stairs
        '''
        dp = [1] * (n + 1)

        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]

    def climbStairs_nonDP(self,n):
        '''
        :param n: int
        :return: int
        leetcode easy: 70. Climbing Stairs
        '''
        if n == 1:
            return 1
        if n == 2:
            return 2
        s1, s2 = 1, 2
        for _ in range(n - 2):
            s1, s2 = s2, s1 + s2
        return s2

    def mindistance(self,word1,word2):
        '''
        :param word1:str
        :param word2:str
        :return: int
        leetcode Hard: 72. Edit Distance
        '''
        l1 = len(word1) + 1
        l2 = len(word2) + 1

        dp = [[0] * l2 for i in range(l1)]

        for i in range(l1):
            dp[i][0] = i

        for j in range(l2):
            dp[0][j] = j

        for i in range(1, l1):
            for j in range(1, l2):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (word1[i - 1] != word2[j - 1]))

        return dp[-1][-1]

    def maximalRectangle(self, matrix):
        '''
        :param matrix: List[List[str]]
        :return: int
        leetcode Hard: 85. Maximal Rectangle
        Input: matrix = [["1","0","1","0","0"],
                         ["1","0","1","1","1"],
                         ["1","1","1","1","1"],
                         ["1","0","0","1","0"]]
        Output: 6

        DP: left[i]   = max(left[i],currentLeft)
            right[i]  = min(right[i],currentRight)
            height[i] = height[i]+1 if row[i] == '1' else 0
        '''
        if not matrix:
            return 0

        n = len(matrix[0])
        left, right, height = [0] * n, [n] * n, [0] * n
        res = 0

        for row in matrix:
            # calculate right
            currentRight = n

            for i in range(n - 1, -1, -1):
                if row[i] == '1':
                    right[i] = min(right[i], currentRight)

                else:
                    right[i] = n
                    currentRight = i

            currentLeft = 0

            for i in range(n):
                # calculate height
                height[i] = height[i] + 1 if row[i] == '1' else 0

                # calculate left
                if row[i] == '1':
                    left[i] = max(left[i], currentLeft)

                else:
                    left[i] = 0
                    currentLeft = i + 1

                # calculate Rectangle
                res = max(res, height[i] * (right[i] - left[i]))

        return res

    def numDecodings(self, s):
        '''
        :param s: str
        :return: int
        leetcode medium: 91. Decode Ways
        '''
        dp = [0] * (len(s) + 1)
        dp[0] = 1

        for i in range(1, len(dp)):
            if s[i-1] != '0':
                dp[i] = dp[i-1]

            if i != 1 and '09' < s[i-2:i] < '27':
                dp[i] += dp[i-2]

        return dp[-1]

    def numTrees(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 96. Unique Binary Search Trees
        '''

        if n == 0:
            return 0

        dp = [0] * (1 + n)
        dp[0] = 1

        for i in range(1, n + 1, 1):
            for j in range(0, i):
                dp[i] += dp[j] * dp[i - 1 - j]

        return dp[n]

    def minimumTotal(self,triangle):
        '''
        input : [[2],[3,4],[6,5,7],[4,1,8,3]]
        :param triangle: list[list[int]]
        :return: int
        leetcode medium: 120. Triangle
        '''
        if not triangle:
            return 0

        for i in range(len(triangle)):
            for j in range(len(triangle[i])):
                if not i == 0:
                    if j == 0:
                        triangle[i][j] += triangle[i - 1][j]

                    elif j == i: # j == last index == i[-1]
                        triangle[i][j] += triangle[i - 1][j - 1]

                    else:
                        triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j])

        return min(triangle[-1])

    def wordBreak(self, s, wordDict):
        '''
        :param s: str
        :param wordDict: List[str]
        :return: List[str]
        leetcode medium: 139. Word Break
        '''
        if not s:
            return 1

        if not wordDict:
            return 0

        n = len(s)

        word_l = set([word.lower() for word in wordDict])

        dp = [0] * (n+1)
        dp[0] = 1

        for i in range(1,n + 1):
            for j in range(i):
                if s[j:i].lower() in word_l:
                    dp[i] += dp[j]

        return dp[n]

    def rob(self, nums):
        '''
        Input: [2,7,9,3,1]
        Output: 12
        :param nums: List[int]
        :return: int
        leetcode medium: 198. House Robber
        '''
        if len(nums) == 1:
            return nums[0]

        rob = [0] * len(nums)

        rob[0] = nums[0]
        rob[1] = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            rob[i] = max(rob[i - 1] , nums[i] + rob[i - 2])

        return rob[-1]

    def maximalSquare(self, matrix):
        '''
        :param matrix: List[List[str]]
        :return: int
        leetcode medium: 221. Maximal Square

        Input: matrix = [["1","0","1","0","0"],
                         ["1","0","1","1","1"],
                         ["1","1","1","1","1"],
                         ["1","0","0","1","0"]]
        Output: 4
        '''
        m = len(matrix)
        n = len(matrix[0])
        res = 0
        dp = [[0] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                    res = max(res, dp[i][j])

        return res ** 2

    def numSquares(self,n):
        '''
        :param n: int
        :return: int
        leetcode medium: 279. Perfect Squares
        '''
        dp    = [n] * (n + 1)
        dp[0] = 0
        dp[1] = 1

        for i in range(2, n + 1):
            j = 1

            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1

        return dp[-1]

    def lengthOfLIS(self,nums):
        '''
        Input: [10,9,2,5,3,7,101,18]
        Output: 4
        :param nums: List[int]
        :return:int
        leetcode medium: 300. Longest Increasing Subsequence
        '''
        if not nums:
            return 0
        dp = [1] * len(nums)

        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i],dp[j]+1)

        return max(dp)

    def maxProfit(self, prices):
        '''
        :param prices: List[int]
        :return: int
        leetcode medium: 309. Best Time to Buy and Sell Stock with Cooldown
        Input: prices = [1,2,3,0,2]
        Output: 3
        '''
        dp = [[0 for _ in range(3)] for _ in range(len(prices))]

        dp[0][1] -= prices[0]

        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            dp[i][2] = dp[i - 1][1] + prices[i]

        return max(dp[-1])

    def coinChange(self,coins, amount):
        '''
        coins = [2,5,7] , amount = 27
        :param coins: list[int]
        :param amount: int
        :return: int
        leetcode medium: 322. Coin Change
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

    def largestDivisibleSubset(self, nums):
        '''
        :param nums: List[int]
        :return: List[int]
        leetcode medium: 368. Largest Divisible Subset
        Input: nums = [1,2,3]
        Output: [1,2]
        Explanation: [1,3] is also accepted.

        Input: nums = [1,2,4,8]
        Output: [1,2,4,8]
        '''
        if not nums:
            return []

        nums.sort()

        dp        = [0]  * len(nums)
        construct = [-1] * len(nums)
        best      = -1
        best_i    = -1

        for i in range(len(nums)):
            for j in range(i):
                if nums[i] % nums[j] == 0 and dp[i] < dp[j]:
                    construct[i], dp[i] = j, dp[j]

            dp[i] += 1

            if dp[i] > best:
                best, best_i = dp[i], i

        result = [nums[best_i]]

        while (construct[best_i] > -1):
            best_i = construct[best_i]
            result.append(nums[best_i])

        return result[::-1]

    def longestPalindrome(self, s):
        '''
        :param s: str
        :return: str
        leetcode easy: 409. Longest Palindrome
        '''
        if len(set(s)) == 1: return s
        n = len(s)

        start, end, maxL = 0, 0, 0

        dp = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i):
                dp[j][i] = (s[j] == s[i]) & ((i - j < 2) | dp[j + 1][i - 1])

                if dp[j][i] and maxL < i - j + 1:
                    maxL = i - j + 1
                    start = j
                    end = i

            dp[i][i] = 1

        return s[start: end + 1]

    def numberOfArithmeticSlices(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode medium: 413. Arithmetic Slices

        input: nums = [1, 2, 3, 4]
        output: return: 3,
        for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself.
        '''
        if len(nums) < 3:
            return 0

        dp = [0] * len(nums)

        for i in range(2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1

        return sum(dp)

    def canPartition(self, nums):
        '''
        :param nums: List[int]
        :return: bool
        leetcode medium: 416. Partition Equal Subset Sum

        Input: nums = [1,5,11,5]
        Output: true
        Explanation: The array can be partitioned as [1, 5, 5] and [11].
        '''
        s = sum(nums)
        target = s // 2                                 # sum of one subset will be half of total sum
        if s & 1:                                       # if the sum is odd then subsets cannot be formed.
            return False

        dp = [False] * (target + 1)
        dp[0] = True

        for n in nums:
            for i in range(target, -1, -1):             # check if i element can be formed using array
                if i >= n:                              # either the i if already present in nums or
                    dp[i] = dp[i] or dp[i - n]          # i-n can be formed from array
                                                        # return the status of target index
        return dp[target]                               # if it can be formed then it will be set to True else False

    def findMaxForm(self, strs, m, n):
        '''
        :param strs: List[str]
        :param m: int
        :param n: int
        :return: int
        leetcode medium: 474. Ones and Zeroes
        '''
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zeros, ones = s.count("0"), s.count("1")
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(1 + dp[i - zeros][j - ones], dp[i][j])
        return dp[-1][-1]

    def findTargetSumWays(self,nums, S):
        '''
        :param nums: : List[int]
        :param S: int
        :return: int
        leetcode medium: 494. Target Sum
        '''
        '''
        D = collections.defaultdict(int)

        D[nums[0]] += 1
        D[-nums[0]] += 1

        for num in nums[1:]:
            D_tmp = collections.defaultdict(int)

            for k in D.keys():
                D_tmp[k + num] += D[k]
                D_tmp[k - num] += D[k]

            D = D_tmp

        return D[S]
        '''

        dp = collections.Counter()
        dp[0] = 1

        for n in nums:
            ndp = collections.Counter()

            for sgn in (1, -1):
                for k in dp.keys():
                    ndp[k + n * sgn] += dp[k]

            dp = ndp

        return dp[S]

    def fib(self,N):
        '''
        :param N: int
        :return: int
        leetcode easy: 509. Fibonacci Number
        '''
        dp = [0] * (N + 1)
        dp[1] = 1

        for i in range(2, N + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[N]

    def deleteAndEarn(self, nums):
        '''
        :param nums: List[int]
        :return: int
        leetcode easy: 740. Delete and Earn

        Input: nums = [3,4,2]
        Output: 6
        Explanation: You can perform the following operations:
        - Delete 4 to earn 4 points. Consequently, 3 is also deleted. nums = [2].
        - Delete 2 to earn 2 points. nums = [].
        You earn a total of 6 points.
        '''
        d = collections.Counter(nums)
        m = max(nums)
        dp = [0] * (m + 1)

        for i in range(1, m + 1):
            if i in d:
                if i == 1:
                    dp[i] = d[i] * i

                else:
                    dp[i] = max(dp[i - 1], dp[i - 2] + i * d[i])

            else:
                dp[i] = dp[i - 1]

        return dp[-1]

    def numTilings(self, n):
        '''
        :param n: int
        :return: int
        leetcode medium: 790. Domino and Tromino Tiling
        '''
        if n == 1:
            return 1

        mod = 10 ** 9 + 7
        dp_full = [0 for _ in range(n)]
        dp_incomp = [0 for _ in range(n)]

        dp_full[0] = 1
        dp_full[1] = 2
        dp_incomp[1] = 2

        for i in range(2, n):
            dp_full[i] = dp_full[i - 2] + dp_full[i - 1] + dp_incomp[i - 1]
            dp_incomp[i] = dp_full[i - 2] * 2 + dp_incomp[i - 1]

        return dp_full[-1] % mod

    def champagneTower(self, poured, query_row, query_glass):
        '''
        :param poured: int
        :param query_row: int
        :param query_glass: int
        :return: float
        leetcode medium: 799. Champagne Tower
        '''
        dp = [[0 for _ in range(x)] for x in range(1, query_row + 2)]

        dp[0][0] = poured

        for i in range(query_row):
            for j in range(len(dp[i])):
                temp = (dp[i][j] - 1) / 2.0

                if temp > 0:
                    dp[i + 1][j] += temp
                    dp[i + 1][j + 1] += temp

        return dp[query_row][query_glass] if dp[query_row][query_glass] <= 1 else 1

    def longestArithSeqLength(self, A):
        '''
        :param A: List[int]
        :return:  int
        leetcode medium: 1027. Longest Arithmetic Subsequence
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

    def longestCommonSubsequence(self, text1, text2):
        '''
        :param text1: str
        :param text2: str
        :return: int
        leetcode medium: 1143. Longest Common Subsequence
        '''
        m, n = len(text1), len(text2)

        dp = [[0 for _ in range(n)] for _ in range(m)]

        if text1[0] == text2[0]:
            dp[0][0] = 1

        for j in range(1, n):
            if text2[j] == text1[0]:
                for k in range(j, n):
                    dp[0][k] = 1

                break

            else:
                dp[0][j] = dp[0][0]

        for i in range(1, m):
            if text1[i] == text2[0]:
                for k in range(i, m):
                    dp[k][0] = 1

                break

            else:
                dp[i][0] = dp[0][0]

        for i in range(1, m):
            for j in range(1, n):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1

                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m - 1][n - 1]

    def countOrders(self, n):
        '''
        :param n: int
        :return: int
        leetcode hard: 1359. Count All Valid Pickup and Delivery Options

        Input: n = 2
        Output: 6
        Explanation: All possible orders:
        (P1,P2,D1,D2), (P1,P2,D2,D1), (P1,D1,P2,D2), (P2,P1,D1,D2), (P2,P1,D2,D1) and (P2,D2,P1,D1).
        This is an invalid order (P1,D2,P2,D1) because Pickup 2 is after of Delivery 2.

        Input: n = 3
        Output: 90
        '''
        dp = [0 for i in range(n + 1)]
        dp[1] = 1

        for i in range(2, len(dp)):
            dp[i] = dp[i - 1] * i * (2 * i - 1) % (10 ** 9 + 7)

        return dp[n]

    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        leetcode easy: 1464. Maximum Product of Two Elements in an Array
        """
        if not nums: return 0
        N = len(nums)
        f = [0] * N
        g = [0] * N
        f[0] = g[0] = res = nums[0]
        for i in range(1, N):
            f[i] = max(f[i - 1] * nums[i], nums[i], g[i - 1] * nums[i])
            g[i] = min(f[i - 1] * nums[i], nums[i], g[i - 1] * nums[i])
            res = max(res, f[i])
        return res

class DFS:
    def generateParenthesis(self, n):
        '''
        :param n: int
        :return: List[str]
        leetcode medium: 22. Generate Parentheses
        '''
        allOutput = set([])

        def dfs(output, numOfLeft, numOfRight, n):
            if len(output) == n * 2:
                allOutput.add(output)
                return

            if numOfLeft < n:
                dfs(output + '(', numOfLeft + 1, numOfRight, n)

            if numOfLeft > numOfRight and numOfRight < n:
                dfs(output + ')', numOfLeft, numOfRight + 1, n)

        dfs('(', 1, 0, n)
        return allOutput

    def combinationSum(self, candidates, target):
        '''
        :param candidates: List[int]
        :param target:     int
        :return:           List[List[int]]
        leetcode medium: 39. Combination Sum
        '''
        candidates.sort()

        res, length = [], len(candidates)

        def dfs(target, start, vlist):
            if target == 0:
                return res.append(vlist)

            for i in range(start, length):
                if target < candidates[i]:
                    break

                else:
                    dfs(target - candidates[i], i, vlist + [candidates[i]])

        dfs(target, 0, [])

        return res

    def permute(self, nums):
        '''
        :param nums: List[int]
        :return: list[List[int]]
        leetcode medium: 46. Permutations
        Input: nums = [1,2,3]
        Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
        '''
        def dfs(number, item, res):
            if len(item) == len(nums):
                res.append(item)
                return

            for i in range(len(number)):
                n_nums = number[:i] + number[i + 1:]
                dfs(n_nums, item + [number[i]], res)

        res = []

        dfs(nums, [], res)

        return res

    def combine(self, n, k):
        '''
        :param n: int
        :param k: int
        :return: List[List[int]]
        leetcode medium: 77. Combinations
        Input: n = 4, k = 2
        Output:
        [ [2,4],
          [3,4],
          [2,3],
          [1,2],
          [1,3],
          [1,4]]
        '''
        def dfs(number, item, res):
            if len(item) == k:
                res.append(item)
                return

            for i in range(len(number)):
                n_nums = number[i + 1:]
                dfs(n_nums, item + [number[i]], res)

        res = []

        dfs(list(range(1, n + 1)), [], res)

        return res

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        :input: [1,2,2]
        :output : [[],[1],[1,2],[1,2,2],[2],[2,2]]
        leetcode medium: 78. Subsets
        """
        if not nums:
            return []

        nums = sorted(nums)

        def dfs(index, item, res):
            res.append(item)

            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i - 1]:
                    continue

                dfs(i + 1, item + [nums[i]], res)

        res = []
        dfs(0, [], res)

        return res

    def exist(self, board , word):
        '''
        :param board: List[List[str]]
        :param word: str
        :return: bool

        Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
        word = "SEE"
        Output: true

        leetcode medium: 79. Word Search
        '''
        '''
        # The two dfs function is same , but now using dfs function more quick
        def dfs(board, word, char, i, j):
            if char >= len(word):
                return True

            checks = [[0, -1], [0, 1], [-1, 0], [1, 0]]

            for check in checks:
                ni = i + check[0]
                nj = j + check[1]

                if 0 <= ni < len(board) and 0 <= nj < len(board[0]) and board[ni][nj] == word[char]:
                    visited       = word[char]
                    board[ni][nj] = '0'

                    if dfs(board, word, char+1, ni, nj):
                        return True

                    board[ni][nj] = visited

            return False
        '''

        def dfs(board, word, char, i, j):
            if char >= len(word):
                return True

            if i + 1 < len(board) and board[i + 1][j] == word[char]:
                visited         = word[char]
                board[i + 1][j] = '0'

                if dfs(board, word, char + 1, i + 1, j):
                    return True

                board[i + 1][j] = visited

            if j + 1 < len(board[0]) and board[i][j + 1] == word[char]:
                visited         = word[char]
                board[i][j + 1] = '0'

                if dfs(board, word, char + 1, i, j + 1):
                    return True

                board[i][j + 1] = visited

            if i - 1 >= 0 and board[i - 1][j] == word[char]:
                visited         = word[char]
                board[i - 1][j] = '0'

                if dfs(board, word, char + 1, i - 1, j):
                    return True

                board[i - 1][j] = visited

            if j - 1 >= 0 and board[i][j - 1] == word[char]:
                visited         = word[char]
                board[i][j - 1] = '0'

                if dfs(board, word, char + 1, i, j - 1):
                    return True

                board[i][j - 1] = visited

            return False

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    visited     = word[0]
                    board[i][j] = '0'

                    if dfs(board, word, 1, i, j):
                        return True

                    board[i][j] = visited

        return False

    def solve(self, board):
        '''
        :param board: List[List[str]]
        :return: NONE
        leetcode medium: 130. Surrounded Regions
        Input: board = [["X","X","X","X"],
                        ["X","O","O","X"],
                        ["X","X","O","X"],
                        ["X","O","X","X"]]

        Output: [["X","X","X","X"],
                 ["X","X","X","X"],
                 ["X","X","X","X"],
                 ["X","O","X","X"]]
        '''
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])

        if m < 3 or n < 3:
            return

        def dfs(board, row, col):
            board[row][col] = 'E'
            if row > 0 and board[row - 1][col] == 'O':
                dfs(board, row - 1, col)

            if row < m - 1 and board[row + 1][col] == 'O':
                dfs(board, row + 1, col)

            if col > 0 and board[row][col - 1] == 'O':
                dfs(board, row, col - 1)

            if col < n - 1 and board[row][col + 1] == 'O':
                dfs(board, row, col + 1)

        for row in range(m):
            if board[row][0] == 'O':
                dfs(board, row, 0)

            if board[row][n - 1] == 'O':
                dfs(board, row, n - 1)

        for col in range(1, n - 1):
            if board[0][col] == 'O':
                dfs(board, 0, col)

            if board[m - 1][col] == 'O':
                dfs(board, m - 1, col)

        for row in range(m):
            for col in range(n):
                if board[row][col] == 'O':
                    board[row][col] = 'X'

                elif board[row][col] == 'E':
                    board[row][col] = 'O'

    def partition(self, s):
        '''
        :param s: str
        :return: List[List[str]]
        leetcode medium: 131. Palindrome Partitioning
        Input: s = "aab"
        Output: [["a","a","b"],["aa","b"]]
        '''
        def dfs(index,item,res):
            if index == len(s):
                res.append(item)
                return

            for i in range(index, len(s)):
                tmpStr = s[index:i + 1]
                if tmpStr == tmpStr[::-1]:
                    dfs(i + 1,item + [tmpStr], res)

        res = []

        dfs(0,[],res)

        return res

    def numIslands(self, grid):
        '''
        :param grid: List[List[str]]
        :return: int
        leetcode medium: 200. Number of Islands
                grid = [['1','1','1','1','0'],
                        ['1','1','0','1','0'],
                        ['1','1','0','0','0'],
                        ['0','0','0','0','0']]
        '''
        def dfs(grid, i, j):
            grid[i][j] = '0'

            checks = [[0, -1], [0, 1], [-1, 0], [1, 0]]

            for check in checks:
                nr = i + check[0]
                nc = j + check[1]

                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                    if grid[nr][nc] == '1':
                        dfs(grid, nr, nc)

        cnt = 0

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':
                    cnt += 1
                    dfs(grid, r, c)

        return cnt

    def canPartitionKSubsets(self,nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: bool
        leetcode medium: 698. Partition to K Equal Sum Subsets
        '''
        _sum = sum(nums)
        groups = [_sum/ k] * k
        nums.sort(reverse=True)

        def dfs(i):
            if i == len(nums):
                return True

            for group in range(k):
                if groups[group] >= nums[i]:
                    groups[group] -= nums[i]

                    if dfs(i + 1):
                        return True

                    groups[group] += nums[i]

            return False

        return dfs(0)

    def canPartitionKSubsets_698(self,nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: bool
        leetcode medium: 698. Partition to K Equal Sum Subsets
        '''
        def dfs(nums, k, index, target):
            if index == len(nums):
                return True

            num = nums[index]

            for i in range(k):
                if target[i] >= num:
                    target[i] -= num

                    if dfs(nums, k, index + 1, target):
                        return True

                    target[i] += num

            return False

        div = sum(nums) // k

        if not nums or len(nums) < k or sum(nums) % k or max(nums) > div:
            return False

        nums.sort(reverse=True)

        target = [div] * k

        return dfs(nums, k, 0, target)

    def accountsMerge(self, accounts):
        '''
        :param accounts: List[List[str]]
        :return: List[List[str]]
        leetcode medium: 721. Accounts Merge
        Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],
                           ["John","johnsmith@mail.com","john00@mail.com"],
                           ["Mary","mary@mail.com"],
                           ["John","johnnybravo@mail.com"]]

        Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],
                 ["Mary","mary@mail.com"],
                 ["John","johnnybravo@mail.com"]]
        '''
        graph = collections.defaultdict(set)

        for acct in accounts:
            center = acct[1]
            for email in acct[2:]:
                graph[center].add(email)
                graph[email].add(center)

        seen = set()

        def dfs(email, acct):
            stack = [email]
            seen.add(email)

            while stack:
                cur = stack.pop()
                acct.append(cur)
                for nei in graph[cur]:
                    if nei not in seen:
                        seen.add(nei)
                        stack.append(nei)

        ans = []

        for acct in accounts:
            name = acct[0]
            email = acct[1]

            if email not in seen:
                cur = []
                dfs(email, cur)
                ans.append([name] + sorted(cur))

        return ans

    def letterCasePermutation(self, S):
        '''
        :param S: str
        :return: List[str]
        leetcode medium: 784. Letter Case Permutation
        Input: S = "a1b2"
        Output: ["a1b2","a1B2","A1b2","A1B2"]
        '''
        def dfs(index, item, ans):
            if index == len(S):
                ans.append(item)
                return

            if S[index].isalpha():
                dfs(index + 1, item + S[index].lower(), ans)

            dfs(index + 1, item + S[index].upper(), ans)

        ans = []

        dfs(0, "", ans)

        return ans

    def rangeSumBST(self, root, L, R):
        '''
        :param root: TreeNode
        :param L: int
        :param R: int
        :return: int
        leetcode easy: 938. Range Sum of BST
        '''
        self.total = 0

        def dfs(node, L, R):
            if not node:
                return 0

            if L <= node.val <= R:
                self.total += node.val

            dfs(node.left, L, R)

            dfs(node.right, L, R)

        dfs(root, L, R)

        return self.total

    def canReach(self, arr, start):
        '''
        :param arr: List[int]
        :param start: int
        :return: bool
        leetcode medium: 1306. Jump Game III
        Input: arr = [4,2,3,0,3,1,2], start = 5
        Output: true
        Explanation:
        All possible ways to reach at index 3 with value 0 are:
        index 5 -> index 4 -> index 1 -> index 3
        index 5 -> index 6 -> index 4 -> index 1 -> index 3
        '''
        if start < 0 or start >= len(arr):
            return False

        if arr[start] == -1:
            return False

        if arr[start] == 0:
            return True

        temp = arr[start]

        # Mark the current index as visited
        arr[start] = -1

        if self.canReach(arr, start + temp):
            return True

        if self.canReach(arr, start - temp):
            return True

        return False

    def goodNodes(self, root):
        '''
        :param root: : TreeNode
        :return: int
        leetcode medium: 1448. Count Good Nodes in Binary Tree
        '''
        self.res = 0

        def goodguys(root, val):
            if not root:
                return
            if root.val >= val:
                self.res += 1
                val = root.val
            goodguys(root.left, val)
            goodguys(root.right, val)

        goodguys(root, float("-inf"))
        return self.res

class algorithm:
    def sieve_algorithm(self, n):
        if n <= 2:
            # Corner case handle
            return 0

        is_prime = [True for _ in range(n)]

        # Base case initialization
        is_prime[0] = False
        is_prime[1] = False

        upper_bound = int(n ** 0.5)
        for i in range(2, upper_bound + 1):

            if not is_prime[i]:
                # only run on prime number
                continue

            for j in range(i * i, n, i):
                # mark all multiples of i as "not prime"
                is_prime[j] = False

        return sum(is_prime)

    def countPrimes(self, n):
        '''
        :param n: int
        :return: int
        '''
        return self.sieve_algorithm(n)

    def bfs(self,graph,start):
        queue,visited = [start],[start]
        while queue:
            vertex=queue.pop()
            for i in graph[vertex]:
                if i not in visited:
                    visited.append(i)
                    queue.append(i)
        return visited

    def BFS(self,graph, s):
        queue = []  # 初始化一个空队列
        queue.append(s) # 将所有节点入队列
        seen = set()
        seen.add(s)
        parent = {s : None}
        while(len(queue) > 0):
            vertex = queue.pop(0)
            nodes = graph[vertex]
            for w in nodes:
                if w not in seen:
                    queue.append(w)
                    seen.add(w)
                    parent[w] = vertex
            print(vertex)
        return parent

    def dfs(self,graph,vertex,queue):
        '''
        :param graph:
        :param vertex:
        :param queue: list()
        :return: list()
        '''
        queue.append(vertex)

        for i in graph[vertex]:
            if i not in queue:
                queue=self.dfs(graph,i,queue)

        return queue

    def DFS(self,graph, s):
        stack = []
        stack.append(s)
        seen = set()
        seen.add(s)
        while(len(stack) > 0):
            vertex = stack.pop()
            nodes = graph[vertex]
            for w in nodes:
                if w not in seen:
                    stack.append(w)
                    seen.add(w)
            print(vertex)

    def how_manny_permutation(self,n,k):
        '''
        :param n int
        :param k: int
        :return: int

        given n numbers then pick kth , how many permutation

        ans = n! / (n-k)!
        '''
        total  = 1
        diff   = 1

        for i in range(1,n + 1):
            total  *= i

            if i <= n - k:
                diff  *= i

        return total // diff

    def combination(self,n,k):
        '''
        :param nums: int
        :param target: int
        :return: list[list[int]]
        ans = n! / (n-k)! (k!)
        '''
        total  = 1
        diff   = 1
        repeat = 1

        for i in range(1,n + 1):
            total  *= i

            if i <= n - k:
                diff  *= i

            if i <= k:
                repeat *= i

        return total // (diff * repeat)

class Amazon:
    def fresh_promotion(self,code_list, shopping_cart):
        '''
        :param code_list: List[List[str]]
        :param shopping_cart: List[str]
        :return: int

        Amazon is running a promotion in which customers receive prizes for purchasing a secret combination of fruits.
        The combination will change each day, and the team running the promotion wants to use a code list to make it easy to change the combination.
        The code list contains groups of fruits. Both the order of the groups within the code list and the order of the fruits within the groups matter.
        However, between the groups of fruits, any number, and type of fruit is allowable. The term "anything" is used to allow for any type of fruit to appear in that location within the group.

        Consider the following secret code list: [[apple, apple], [banana, anything, banana]]
        Based on the above secret code list, a customer who made either of the following purchases would win the prize:
        orange, apple, apple, banana, orange, banana
        apple, apple, orange, orange, banana, apple, banana, banana

        Write an algorithm to output 1 if the customer is a winner else output 0.

        Input
        The input to the function/method consists of two arguments:
        codeList, a list of lists of strings representing the order and grouping of specific fruits that must be purchased in order to win the prize for the day.
        shoppingCart, a list of strings representing the order in which a customer purchases fruit.

        Output
        Return an integer 1 if the customer is a winner else return 0.

        Note
        'anything' in the codeList represents that any fruit can be ordered in place of 'anything' in the group. 'anything' has to be something, it cannot be "nothing."
        'anything' must represent one and only one fruit.
        If secret code list is empty then it is assumed that the customer is a winner.

        Example 1:

        Input: codeList = [[apple, apple], [banana, anything, banana]] shoppingCart = [orange, apple, apple, banana, orange, banana]
        Output: 1
        Explanation:
        codeList contains two groups - [apple, apple] and [banana, anything, banana].
        The second group contains 'anything' so any fruit can be ordered in place of 'anything' in the shoppingCart.
        The customer is a winner as the customer has added fruits in the order of fruits in the groups and the order of groups in the codeList is also maintained in the shoppingCart.

        Example 2:

        Input: codeList = [[apple, apple], [banana, anything, banana]]
        shoppingCart = [banana, orange, banana, apple, apple]
        Output: 0
        Explanation:
        The customer is not a winner as the customer has added the fruits in order of groups but group [banana, orange, banana] is not following the group [apple, apple] in the codeList.

        Example 3:

        Input: codeList = [[apple, apple], [banana, anything, banana]] shoppingCart = [apple, banana, apple, banana, orange, banana]
        Output: 0
        Explanation:
        The customer is not a winner as the customer has added the fruits in an order which is not following the order of fruit names in the first group.

        Example 4:

        Input: codeList = [[apple, apple], [apple, apple, banana]] shoppingCart = [apple, apple, apple, banana]
        Output: 0
        Explanation:
        The customer is not a winner as the first 2 fruits form group 1, all three fruits would form group 2, but can't because it would contain all fruits of group 1.
        '''

        cart_idx = 0
        code_list_idx = 0

        while code_list_idx in range(len(code_list)) and cart_idx < len(shopping_cart):
            code = code_list[code_list_idx]
            code_idx = 0

            while code_idx < len(code) and cart_idx < len(shopping_cart):
                if code[code_idx] == shopping_cart[cart_idx] or code[code_idx] == 'anything':
                    code_idx += 1

                else:
                    code_idx = 0

                cart_idx += 1

            if code_idx == len(code):
                code_list_idx += 1

        if code_list_idx == len(code_list):
            return 1

        return 0

    def aircraft(self,maxTravelDist, forwardRouteList, returnRouteList):
        '''
        :param maxTravelDist: int
        :param forwardRouteList: List[List[int]]
        :param returnRouteList: List[List[int]]
        :return: List[List[int]]
        test 1 :
            maxTravelDist    = 10000
            forwardRouteList = [[1, 3000], [2, 5000], [3, 7000], [4, 10000]]
            returnRouteList  = [[1, 2000], [2, 3000], [3, 4000], [4, 5000]]

            output : [[2,4],[3,2]]

        test 2:
            maxTravelDist    = 7000
            forwardRouteList = [[1, 2000], [2, 4000], [3, 6000]]
            returnRouteList  = [[1, 2000]]

            output : [[2,1]]
        '''
        for i in range(len(forwardRouteList)):
            for j in range(i + 1, len(forwardRouteList)):
                if forwardRouteList[i][1] > forwardRouteList[j][1]:
                    forwardRouteList[i], forwardRouteList[j] = forwardRouteList[j], forwardRouteList[i]

        for i in range(len(returnRouteList)):
            for j in range(i + 1, len(returnRouteList)):
                if returnRouteList[i][1] > returnRouteList[j][1]:
                    returnRouteList[i], returnRouteList[j] = returnRouteList[j], returnRouteList[i]

        d = {}

        for f in forwardRouteList:
            for r in returnRouteList:
                if f[1] + r[1] > maxTravelDist:
                    continue

                if f[1] + r[1] <= maxTravelDist:
                    if f[1] + r[1] in d:
                        d[f[1] + r[1]].append([f[0], r[0]])

                    else:
                        d[f[1] + r[1]] = [[f[0], r[0]]]

        return d[max(d)]

class Google:
    def valid_parentheses(self,st):
        '''
        :param st: str
        :return: bool
        codewar
        '''
        cnt = 0
        for char in st:
            if char == '(': cnt += 1
            if char == ')': cnt -= 1
            if cnt < 0: return False
        return True if cnt == 0 else False

    def sumlist(self,nums,target):
        '''
        :param nums: List[int]
        :param target: int
        :return: List[int]
        google question
        '''
        ans = [] ; tmp = []
        for num in nums:
            sub = target - num
            if sub in nums:
                tmp.append(sub)
                tmp.append(num)
            if len(tmp) != 0 and len(tmp) == 2:
                tmp = sorted(tmp)
                if tmp not in ans:
                    ans.append(tmp)
                    tmp = []
        return ans

    def vampire_number(self,x,y):
        '''
        :param x: int
        :param y: int
        :return: bool
        '''
        xy = x*y
        if len(str(xy)) != len(str(x))+len(str(y)):
            return False

        xy_list = list(str(x))+(list(str(y)))

        for i in xy_list:
            if i not in str(xy):
                return False
            else:
                if list(str(xy)).count(i) != xy_list.count(i):
                    return False

        return True

if __name__ == '__main__':
    pass