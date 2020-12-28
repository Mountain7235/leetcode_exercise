import collections
import sys
import time
import bisect

# link list
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class keysandrooms:
    # True  = [[1],[2],[3],[]]
    # False = [[1,3],[3,0,1],[2],[0]]
    def canVisitAllRoomsDFS1(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        q, path = [0], [0]
        while q:
            index = q.pop()
            for v in rooms[index]:
                if v not in path:
                    q.append(v)
                    path.append(v)
        return len(path) == len(rooms)

    def canVisitAllRoomsDFS2(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        return len(self.dfs2(rooms, [])) == len(rooms)

    def dfs2(self, rooms, path, source=0):
        path += [source]
        for i in rooms[source]:
            if i not in path:
                self.dfs2(rooms, path, i)
        return path

    def canVisitAllRoomsBFS(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        q, path = [0], [0]
        while q:
            index = q.pop(0)
            for v in rooms[index]:
                if v not in path:
                    q.append(v)
                    path.append(v)
        return len(path) == len(rooms)

class Island:
    def numIslands1(self,grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        res = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == "1":
                    self.dfs1(grid, r, c)
                    res += 1
        return res

    def dfs1(self,grid, i, j):
        dirs = [[-1, 0], [0, 1], [0, -1], [1, 0]]
        grid[i][j] = "0"
        for dir in dirs:
            nr, nc = i + dir[0], j + dir[1]
            if nr >= 0 and nc >= 0 and nr < len(grid) and nc < len(grid[0]):
                if grid[nr][nc] == "1":
                    self.dfs1(grid, nr, nc)

    def numIslands2(self, grid):
        if not grid or len(grid) == 0:
            return 0

        row, columns = len(grid), len(grid[0])
        count = 0
        for i in range(row):
            for j in range(columns):
                if grid[i][j] == '1':
                    self.dfs2(grid, i, j, row, columns)
                    count += 1
        return count

    def dfs2(self, grid, i, j, row, columns):
        if i >= row or i < 0 or j >= columns or j < 0 or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        self.dfs2(grid, i - 1, j, row, columns)
        self.dfs2(grid, i, j - 1, row, columns)
        self.dfs2(grid, i + 1, j, row, columns)
        self.dfs2(grid, i, j + 1, row, columns)

    def numIslands3(self, grid):
        '''
        grid = [['1','1','1','1','0'],
                ['1','1','0','1','0'],
                ['1','1','0','0','0'],
                ['0','0','0','0','0']]
        :param grid: List[List[str]
        :return: int
        '''
        if not grid or not grid[0]:
            return 0
        visited = [[False] * len(grid[0]) for _ in range(len(grid))]
        res = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1' and not visited[i][j]:
                    res += 1
                    self.dfs3(grid, visited, i, j)
        return res

    def dfs3(self, grid, visited, i, j):
        '''
        :param grid: List[List[str]]
        :param visited: List[List[bool]]
        :param i: int
        :param j: int
        :return: None
        '''
        if i >= 0 and j >= 0 and i < len(grid) and j < len(grid[0]) and not visited[i][j] and grid[i][j] == '1':
            visited[i][j] = True
            self.dfs3(grid, visited, i + 1, j)
            self.dfs3(grid, visited, i - 1, j)
            self.dfs3(grid, visited, i, j + 1)
            self.dfs3(grid, visited, i, j - 1)

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

            if root.right != None:
                que.append(root.right)

        return dis

    def preorder(self,root,res):
        if root:
            res.append(root.value)
            self.preorder(root.left,res)
            self.preorder(root.right,res)

    def inorder(self,root,res):
        if root:
            self.inorder(root.left,res)
            res.append(root.value)
            self.inorder(root.right,res)

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

class LeetCode:
    # region easy level
    def twoSum(self,nums, target):
        '''
        :param nums: List[int]
        :param target: int
        :return: List[int]
        1. Two Sum
        '''
        dic = dict()
        for index,value in enumerate(nums):
            sub = target - value
            if sub in dic:
                return [dic[sub],index]
            else:
                dic[value] = index

    def reverse(self,x):
        '''
        :param x:
        :return:
        leetcode easy: 7. Reverse Integer
        '''
        if x > 0:
            return int(str(x)[::-1])
        else:
            return (0 - int(str(-x)[::-1]))

    def romanToInt(self,s):
        '''
        :param s: str
        :return: int
        leetcode easy: 13. Roman to Integer
        '''
        roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        res = roman[s[-1]]
        N = len(s)
        for i in range(N - 2, -1, -1):
            if roman[s[i]] < roman[s[i + 1]]:
                res -= roman[s[i]]
            else:
                res += roman[s[i]]
        return res

    def isValid(self,s):
        '''
        :param s: str
        :return: bool
        leetcode easy: 20. Valid Parentheses
        '''
        if len(s) % 2 != 0:
            return False

        stack = list()

        dic = {'(': ')', '{': '}', '[': ']'}

        for i, c in enumerate(s):
            if s[i] in dic:
                stack.append(s[i])
            else:
                if len(stack) != 0 and dic[stack[-1]] == s[i]:
                    stack.pop()
                else:
                    return False
        if stack:
            return False
        else:
            return True

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

    def addTwoNumbers(self, l1, l2):
        '''
        :param l1: ListNode
        :param l2: ListNode
        :return: ListNode
        leetcode easy: 21. Merge Two Sorted Lists
        '''
        num1 = ''
        num2 = ''

        while l1:
            num1 += str(l1.val)
            l1 = l1.next

        while l2:
            num2 += str(l2.val)
            l2 = l2.next

        total = str(int(num1[::-1]) + int(num2[::-1]))[::-1]

        head = ListNode(int(total[0]))
        ans = head

        for i in total[1:]:
            head.next = ListNode(int(i))
            head = head.next

        return ans

    def removeDuplicates(self,nums):
        '''
        inpit:[0,0,1,1,1,2,2,3,3,4]
        :param nums: List[int]
        :return:int
        leetcode easy: 26. Remove Duplicates from Sorted Array
        '''
        if not nums:
            return 0
        i, j = 0, 1
        while j < len(nums):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]
            j += 1
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
        # '1'的翻譯是'11'，等於多翻譯了一次，所以把n-1
        n -= 1
        s = '1'
        # 因爲每個由上一個計算而來，所以要循環n次
        while (n > 0):
            res, i = '', 0
            # 計算int翻譯爲str
            while (i < len(s)):
                # 用來統計相鄰相同數字的個數
                count = 1
                while (i + 1 < len(s) and s[i] == s[i + 1]):
                    count += 1
                    i += 1
                # 把數字個數與該數字轉爲str再相加
                res += str(count) + str(s[i])
                i += 1
            s = res
            n -= 1
        return s

    def lengthOfLastWord(self,s):
        '''
        :param s: str
        :return: int
        leetcode easy: 58. Length of Last Word
        '''
        return len(s.rstrip().split(' ')[-1])

    def addBinary(self,a, b):
        '''
        :param a: str
        :param b: str
        :return: str
        leetcode easy: 67. Add Binary
        '''
        count = 0
        suma = 0
        for i in a[::-1]:
            suma = suma+(int(i)*(2**count))
            count+=1

        count = 0
        sumb = 0
        for i in b[::-1]:
            sumb = sumb+(int(i)*(2**count))
            count+=1

        total = suma + sumb
        ans = ''
        while total != 0:
            ans = ans+str(total%2)
            total//=2

        # str(bin(suma+sumb))[2:]
        return ans

    def merge1(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        leetcode easy: 88. Merge Sorted Array
        """
        i, j = m - 1, n - 1
        k = m + n - 1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1

    def merge2(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        leetcode easy: 88. Merge Sorted Array
        """
        nums1 = nums1[m:] + nums1[:m]
        nums2.sort()
        for num in nums2:
            for index, value in enumerate(nums1):
                if num < value:
                    nums1.insert(index, num)
                    nums1.pop(0)
                    break
                if index == len(nums1) - 1 and value < num:
                    nums1.append(num)
                    nums1.pop(0)
                    break

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
        ans = []
        if not numRows:
            return ans

        for i in range(numRows):
            ans.append([])
            ans[i].append(1)

            for j in range(1, i):
                ans[i].append(ans[i - 1][j - 1] + ans[i - 1][j])

            if numRows != 0 and len(ans) !=1:
                ans[i].append(1)

        return ans

    def maxProfit(self, prices):
        '''
        Input: [7,1,5,3,6,4]
        Output: 5
        :param prices: List[int]
        :return: int
        leetcode easy: 121. Best Time to Buy and Sell Stock
        '''
        if not prices:
            return 0
        minPrice = float('inf')
        profit = 0

        for price in prices:
            minPrice = min(minPrice, price)
            profit = max(profit, price - minPrice)

        return profit

    def isPalindrome(self,x):
        '''
        :param x:str
        :return:bool
        leetcode easy: 125. Valid Palindrome
        '''
        if x < 0:
            return False
        if x == 0:
            return True
        div = 1
        while x / div >= 10:
            div *= 10

        while x:
            first = x // div
            last = x%10

            if first != last:
                return False
            else:
                x = (x - (div * last) - last)
                if x < 10 :
                    return True
                x = x/10
                div = div/100

    def sig1(self,nums):
        """
        :type nums: List[int]
        :rtype: int
        leetcode easy: 136. Single Number
        """
        num = [nums.count(i) for i in nums]
        return nums[num.index(1)]

    def singleNumber2(self,A):
        one = 0; two = 0; three = 0
        for i in range(len(A)):
            two |= A[i] & one
            one = A[i] ^ one
            three = ~(one & two)
            one &= three
            two &= three
        return one

    def singNum3(self,nums):
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

    def singleNumber4(self,nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = nums[0]
        for i in range(1, len(nums)):
            res ^= nums[i]
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

    def moveZeroes(self, nums):
        '''
        :param nums: List[int]
        :return: None
        leetcode easy: 283. Move Zeroes
        Duobple pointer
        '''
        """
        Do not return anything, modify nums in-place instead.
        """
        index = 0

        for i in range(len(nums)):
            if nums[i] != 0:
                nums[index] = nums[i]
                index += 1

        if index:
            for i in range(index, len(nums)):
                nums[i] = 0

    def reverseString(self, s):
        '''
        :param s: List[str]
        :return: None
        :Do not return anything, modify s in-place instead.
        leetcode easy: 344. Reverse String
        '''
        for i in range(len(s)//2):
            s[i],s[-i-1] = s[-i-1],s[i]

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

    def hammingDistance(self, x, y):
        '''
        :param x: int
        :param y: int
        :return: int
        leetcode easy : 461. Hamming Distance
        '''
        distance = 0
        n = x ^ y

        while n > 0:
            if n & 1 != 0:
                distance += 1

            n = n >> 1
        return distance

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

    def canPlaceFlowers(self, flowerbed, n):
        '''
        :param flowerbed: : List[int]
        :param n: int
        :return: bool
        leetcode Easy: 605. Can Place Flowers
        '''
        l = [0] + flowerbed + [0]

        for i in range(1, len(l) - 1):
            if sum(l[i - 1:i + 2]) == 0:
                l[i] = 1
                n -= 1

        return n <= 0

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

    def minCostToMoveChips(self, position):
        '''
        :param position: List[int]
        :return: int
        leetcode easy: 1217. Minimum Cost to Move Chips to The Same Position
        '''
        odd_numbers = 0
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

    def kWeakestRows(self, mat, k):
        '''
        :param mat:List[List[int]]
        :param k: int
        :return: List[int]
        1337. The K Weakest Rows in a Matrix
        '''
        d = {i:sum(mat[i]) for i in range(len(mat))}
        return [i[0] for i in sorted(d.items(),key=lambda item:item[1])[:k]]

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
    # endregion

    # region medium level
    def vampire_number(self,x,y):
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

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        leetcode medium: 3. Longest Substring Without Repeating Characters
        """
        left, right = 0, 0
        res = 0
        chars = dict()
        for right in range(len(s)):
            if s[right] in chars:
                left = max(left, chars[s[right]] + 1)
            chars[s[right]] = right
            res = max(res, right - left + 1)
        return res

    def maxArea_1(self, height):
        '''
        :param height: List[int]
        :return: int
        leetcode medium: 11. Container With Most Water
        '''
        ans = 0
        l = 0
        r = len(height) - 1

        while l < r:
            ans = max(ans, min(height[l], height[r]) * (r - l))

            if height[l] < height[r]:
                l += 1
            else:
                r -= 1

        return ans

    def maxArea_2(self, height):
        '''
        :param height: List[int]
        :return: int
        leetcode medium: 11. Container With Most Water
        '''
        maxArea = 0
        left = 0
        right = len(height)-1

        while left != right:
            width = right - left
            if height[left] <= height[right]:
                area = width*height[left]
                left += 1
            else:
                area = width*height[right]
                right -= 1

            if area > maxArea:
                maxArea = area

        return maxArea

    def maxArea_3(self, height):
        '''
        :param height: List[int]
        :return: int
        leetcode medium: 11. Container With Most Water
        '''
        maxArea = 0
        left = 0
        right = len(height)-1
        while left!= right:
            width=right-left
            if height[left] <= height[right]:
                area = width*height[left]
                left += 1
            else:
                area = width*height[right]
                right -= 1
            if area>maxArea:
                maxArea = area
        return maxArea

    def threeSum(self,nums):
        '''
        :param nums: List[int]
        :return: List[List[int]]
        leetcode medium: 15. 3Sum
        '''
        if len(nums) < 3:
            return []
        nums.sort()
        res = set()
        for i, v in enumerate(nums[:-2]):
            if i >= 1 and v == nums[i - 1]:
                continue
            d = {}
            for x in nums[i + 1:]:
                if x not in d:
                    d[-v - x] = None
                else:
                    res.add((v, -v - x, x))
        return list(res)

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
        '''
        L = [[nums[0]]]
        for n in nums[1:]:
            new_L = []
            for perm in L:
                for i in range(len(perm) + 1):
                    new_L.append(perm[:i] + [n] + perm[i:])
                    L = new_L
                return L

    def permuteDFS(self, nums):
        '''
        :param nums: List[int]
        :return: list[List[int]]
        leetcode medium: 46. Permutations
        '''
        def dfs(nums, item, res):
            if not nums:
                res.append(item)

            for i in range(len(nums)):
                num_remain = nums[:i] + nums[i + 1:]
                dfs(num_remain, item + [nums[i]], res)

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

    def groupAnagrams(self, strs):
        '''
        :param strs: List[str]
        :return: List[List[str]]
        leetcode medium: 49. Group Anagrams
        '''
        D = dict()

        for i in strs:
            isort = ''.join(sorted(list(i)))

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
        '''
        spiral_path = []

        while matrix:
            # pop the top-most row
            spiral_path.extend(matrix.pop(0))

            # get the upside-down of matrix transpose
            matrix = [*zip(*matrix)][::-1]

        return spiral_path

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

    def combine1(self, n, k):
        '''
        :param n: int
        :param k: int
        :return: List[List[int]]
        leetcode medium: 77. Combinations
        '''
        res = []  # 1

        def dfs(nums, k, index, path, res):  # 4
            if k == 0:  # 7
                res.append(path)  # 8
                return  # backtracking  #9

            for i in range(index, len(nums)):  # 10
                dfs(nums, k - 1, i + 1, path + [nums[i]], res)  # 11

        dfs(range(1, n + 1), k, 0, [], res)  # 2

        return res  # 3

    def combine2(self, n, k):
        '''
        :param n: int
        :param k: int
        :return: List[List[int]]
        leetcode medium: 77. Combinations
        '''
        res = []
        def dfs(A):
            if len(A)==k:
                return res.append(list(A))

            x = A[-1] +1 if A else 1
            for i in range(x,n+1):
                A.append(i)
                dfs(A)
                A.pop()
        dfs([])
        return res

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        :input: [1,2,2]
        :output : [[],[1],[1,2],[1,2,2],[2],[2,2]]
        leetcode medium: 78. Subsets
        """
        res = []
        if not nums:
            return res

        nums = sorted(nums)

        def dfs_help(index, item, res):
            res.append(item)
            for i in range(index, len(nums)):
                # skip the same neibor
                if i > index and nums[i] == nums[i - 1]:
                    continue

                dfs_help(i + 1, item + [nums[i]], res)

        dfs_help(0, [], res)

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

    def partition(self, s):
        '''
        :param s: str
        :return: List[List[str]]
        leetcode medium: 131. Palindrome Partitioning
        '''
        ans = []

        def dfs(currList, k):
            if k == len(s):
                ans.append(currList)
                return

            for i in range(k, len(s)):
                tmpStr = s[k:i + 1]
                if tmpStr == tmpStr[::-1]:
                    dfs(currList + [tmpStr], i + 1)

        dfs([], 0)

        return ans

    def rotate(self, nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: None
        leetcode medium: 189. Rotate Array
        '''
        """
        Do not return anything, modify nums in-place instead.
        """
        while k:
            a = nums.pop(-1)
            nums.insert(0,a)
            k-=1

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

    def topKFrequent(self, nums, k):
        '''
        :param nums: List[int]
        :param k: int
        :return: List[int]
        leetcode medium: 347. Top K Frequent Elements
        '''
        d = collections.Counter(nums)
        return [i[0] for i in sorted(d.items(),key=lambda item:item[1])][::-1][:k]

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

    def numMagicSquaresInside(self, grid):
        '''
        :param grid:List[List[int]]
        :return: int
        leetcode medium: 840. Magic Squares In Grid
        '''
        def is_magic(matrix):
            num = [matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0]))]
            for i in range(1, 10):
                if i not in num:
                    return 0

            result = set()

            for r in matrix:
                result.add(sum(r))

            for c in zip(*matrix):
                result.add(sum(c))

            rdig = 0
            ldig = 0
            for i in range(len(matrix)):
                rdig += matrix[i][i]
                ldig += matrix[i][len(matrix[0]) - 1 - i]

            result.add(rdig)
            result.add(ldig)
            return 1 if len(result) == 1 else 0

        if len(grid) < 3 or len(grid[0]) < 3:
            return 0

        count = 0
        for i in range(len(grid) - 2):
            for j in range(len(grid[0]) - 2):
                candi_matrix = [grid[i][j:j + 3], grid[i + 1][j:j + 3], grid[i + 2][j:j + 3]]

                if is_magic(candi_matrix):
                    count += 1
        return count

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
                lens.append(lens[-1] * int(c))
            else:
                lens.append(lens[-1] + 1)

        for i in range(n, 0, -1):
            K %= lens[i]
            if K == 0 and S[i - 1].isalpha():
                return S[i - 1]

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

    def smallestRepunitDivByK(self, K):
        '''
        :param K: int
        :return: int
        leetcode medium: 1015. Smallest Integer Divisible by K
        '''
        if (K % 2 == 0) or (K % 5 == 0):
            return -1

        remainder = 0

        for N in range(1, K+1):
            remainder = (remainder * 10 + 1) % K
            if remainder == 0:
                return N

    def goodNodes1(self, root):
        '''
        :param root: : TreeNode
        :return: int
        1448. Count Good Nodes in Binary Tree
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

    def goodNodes2(self, root):
        '''
        :param root: : TreeNode
        :return: int
        1448. Count Good Nodes in Binary Tree
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
        '''
        if start < 0 or start >= len(arr):
            return False

        if arr[start] == -1:
            return False

        if arr[start] == 0:
            return True

        temp = arr[start]

        arr[start] = -1

        if self.canReach(arr, start + temp):
            return True

        if self.canReach(arr, start - temp):
            return True

        return False

    def canReach_bfs(self, arr, start):
        '''
        :param arr: List[int]
        :param start: int
        :return: bool
        leetcode medium: 1306. Jump Game III
        '''
        _len_= len(arr)
        seen = set()
        queue = collections.deque()
        queue.append(start)

        while(queue):
            idx = queue.popleft()

            if arr[idx] == 0:
                return True

            seen.add(idx)

            for var in (idx - arr[idx], idx + arr[idx]):
                if (var not in seen) and (-1 < var < _len_):
                    queue.append(var)

        return False

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
    # endregion

    # region Hard level
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

    # endregion

class DP:
    def longestCommonPrefix(self,strs):
        '''
        :param strs: str
        :return: str
        leetcode easy: 14. Longest Common Prefix
        '''
        base = strs[0]
        command = [False]*len(strs)

        for s in strs:
            if len(s) < len(base):
                base = s

        while False in command:
            strs = [i[:len(base)] for i in strs]

            for s in range(len(strs)):
                if base == strs[s]:
                    command[s] = True

            if False not in command:
                break
            else:
                base = base[:-1]

        return base

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

    def combinationSum_DP1(self, candidates, target):
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

    def combinationSum_DP2(self, candidates, target):
        '''
        :param candidates: List[int]
        :param target:     int
        :return:           List[List[int]]
        leetcode medium: 39. Combination Sum
        '''
        if target < min(candidates):
            return None

        answer = []

        universe = [([], 0)]

        for n in candidates:
            for (ls, v) in universe:
                if v + n == target:
                    answer.append(ls + [n])
                elif v + n < target:
                    universe.append((ls + [n], v + n))

        return answer

    def combinationSum_DFS(self, candidates, target):
        '''
        :param candidates: List[int]
        :param target:     int
        :return:           List[List[int]]
        leetcode medium: 39. Combination Sum
        '''

        #下面有一個目標值小於某一元素就break，所以先排序
        candidates.sort()
        #儲存返回的二維列表
        res, length = [], len(candidates)

        #遞歸，目標值，起始位置，當前組合
        def dfs(target, start, vlist):
            #目標值爲0，表明當前遞歸完成，把當前遞歸結果加入res並返回
            if target == 0:
                return res.append(vlist)
            #從開始下標循環
            for i in range(start, length):
                #candidates有序，只要當前大於目標後面都大於，直接break
                if target < candidates[i]:
                    break
                #否則目標值減當前值，i爲新的起始位置，把當前值加入當前組合
                else:
                    dfs(target - candidates[i], i, vlist + [candidates[i]])

        dfs(target, 0, [])

        return res

    def combinationSum_dfs(self, candidates, target):
        '''
        :param candidates: List[int]
        :param target:     int
        :return:           List[List[int]]
        leetcode medium: 39. Combination Sum
        '''
        def dfs(nums, target, index, path, res):
            if target < 0:
                return  # backtracking
            if target == 0:
                res.append(path)
                return
            for i in range(index, len(nums)):
                dfs(nums, target - nums[i], i, path + [nums[i]], res)

        res = []
        candidates.sort()
        dfs(candidates, target, 0, [], res)
        return res

    def maxsunarry1(self,nums):
        '''
        :param nums: int
        :return: int
        leetcode easy: 53. Maximum Subarray
        '''
        l = g = -100000000000
        for num in nums:
            l = max(num,l+num)
            g = max(l,g)
        return g

    def maxsubarray2(self,nums):
        '''
        :param nums: int
        :return: int
        leetcode easy: 53. Maximum Subarray
        '''
        l = len(nums)
        if l == 0: return 0
        res = now = 0
        for i in range(l):
            if now > 0:
                now += nums[i]
            else:
                now = nums[i]
            if now > res:
                res = now
        return res

    def maxsubarray3(self,nums):
        '''
        :param nums: int
        :return: int
        leetcode easy: 53. Maximum Subarray
        '''
        sum = 0
        ma = nums[0]
        for i in range(len(nums)):
            if (sum < 0):
                sum = nums[i]
            else:
                sum += nums[i]

            ma = max(ma, sum)
        return ma

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

    def uniquePaths1(self,m,n):
        '''
        :param m: int
        :param n: int
        :return: int
        leetcode medium: 62. Unique Paths
        '''
        dp = [[1] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    continue

                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]

        return dp[m - 1][n - 1]

    def uniquePaths2(self,m,n):
        '''
        :param m: int
        :param n: int
        :return: int
        leetcode medium: 62. Unique Paths
        '''
        c = [0 for _ in range(m)]

        for i in range(n):
            for j in range(m):
                if j == 0:
                    c[j] = 1
                else:
                    c[j] = c[j] + c[j - 1]
        return c[m - 1]

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
        if n == 1:
            return 1
        dp = [0] * (n+1)
        dp[0] = 1
        dp[1] = 1

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

    def minimumTotal(self,triangle):
        '''
        input : [[2],[3,4],[6,5,7],[4,1,8,3]]
        :param triangle: list[list[int]]
        :return: int
        leetcode medium: 120. Triangle
        '''
        if not triangle:
            return 0

        m = len(triangle)
        dp = [[0] * (n+1) for n in range(m)]
        dp[0][0] = triangle[0][0]
        for i in range(1,m):
            dp[i][0] = dp[i-1][0] + triangle[i][0]
            dp[i][i] = dp[i-1][i-1]+triangle[i][i]

        for i in range(1,m):
            for j in range(1,i):
                dp[i][j] = min(dp[i-1][j-1],dp[i-1][j]) + triangle[i][j]

        return min(dp[-1])

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

    def rob1(self, nums):
        '''
        Input: [2,7,9,3,1]
        Output: 12
        :param nums: List[int]
        :return: int
        leetcode medium: 198. House Robber
        '''
        if len(nums) == 0:
            return 0

        if len(nums) == 1:
            return nums[0]

        rob = [0] * len(nums)

        rob[0] = nums[0]
        rob[1] = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            rob[i] = max(nums[i] + rob[i - 2], rob[i - 1])

        return rob[-1]

    def rob2(self, nums):
        '''
        Input: [2,7,9,3,1]
        Output: 12
        :param nums: List[int]
        :return: int
        leetcode medium: 198. House Robber
        '''
        rob1, rob2 = 0, 0

        for n in nums:
            temp = max(n + rob1, rob2)
            rob1 = rob2
            rob2 = temp

        return rob2

    def numSquares(self,n):
        '''
        :param n: int
        :return: int
        leetcode medium: 279. Perfect Squares
        '''
        dp = [n] * (n + 1)
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

    def countPrimes(self, n: int) -> int:
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

if __name__ == '__main__':
    coins = [2,5,7]
    amount = 27


    print(DP().coinChange(coins,amount))