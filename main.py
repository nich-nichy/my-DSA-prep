# 1). Python program for prime factor

# 36 = 3 * 3 * 2 * 2

# def productPrimeFactor(n):
#   product = 1
#   for i in range(2, n+1):
#      if (n % i == 0): #n % i
#            isPrime = 1
#            for j in range(2, int(i/2 + 1)): #i / 2 + 1
#                 if(i % j == 0): # i % j
#                     isPrime = 0
#                 if(isPrime):
#                     product = product * i
#     return product
# n = 18  # n + 1 = 19
# print(productPrimeFactor(n))


# 2). Convert time from 12 hour to 24 hour

# def convert(str1):
#     if str1[-2:] == 'AM' and str1[:2] == '12':
#         return str[2:-2]
#     elif str1[-2:] == 'AM':
#         return str1[:-2]
#     elif str1[-2:] == "PM" and str1[:2] == "12":
#         return str[:-2]
#     else:
#         return str(int(str1[:2]) + 12) + str[2:8]
#
# print(convert("04:00:00 PM"))

# 3). Multiway selection in python

# def add(numb1, numb2):
#     return numb1 + numb2
# def substract(numb1, numb2):
#     return numb1 - numb2
# def multiply(numb1, numb2):
#     return numb1 * numb2
# def divide(numb1, numb2):
#     numb1 / numb2
# print("Select an operation vroo: - \n"
#       "1. Addition \n"
#       "2. Subraction \n"
#       "3. Multiplication \n"
#       "4. Division \n")
# select = int(input("Select operations on 1, 2, 3, 4: "))
#
# numb1 = int(input("Enter first num: "))
# numb2 = int(input("Enter second num: "))
#
# if select == 1:
#     print(numb1, "+", numb2, "=", add(numb1, numb2))
# elif select == 2:
#     print(numb1, "-", numb2, "=", substract(numb1, numb2))
# elif select == 3:
#     print(numb1, "*", numb2, "=", multiply(numb1, numb2))
# elif select == 4:
#     print(numb1, "/", numb2, "=", divide(numb1, numb2))


# 4). Calculate the string A and calculate the string B total

# word_count = 0
# char_count = 0
# usr_input = input("Enter the string: ")
# split_string = usr_input.split()
# word_count = len(split_string)
# for i in split_string:
#     char_count = len(i)
#     print("Total words: {} ".format(word_count))
#     print("Total chars is : {}".format(char_count))

# 5). Tower of Hanoi

# def towerOfHanoi(n, source, dest, aux):
#     if n == 1:
#         print("Move disk 1 from source", source, "to destination", dest)
#         return
#     towerOfHanoi(n-1, source, aux, dest)
#     print("Move disk 1 from source", source, "to destination", dest)
#     towerOfHanoi(n-1, aux, dest, source)
#
#
#
# n = 4
# towerOfHanoi(n, 'A', 'B', 'C')

# Unit 2
# 1). Tuple functions

# tuple1, tuple2 = ('apple', 'banana', 'strawberry'), ('berry', 'mango')
# list1 = ['hai', 'hello', 'we', 'are', 'best', 'friends']
# print("elements on tuple 1: ", max(tuple1))
# print("elements on tuple 1: ", min(tuple1))
# print("elements on tuple 2: ", max(tuple2))
# print("elements on tuple 2: ", max(tuple2))
# print("elements on tuple 1: ", len(tuple1))
# print("elements on tuple 2: ", len(tuple2))
# tuple3 = tuple(list1)
# print(tuple3)

# ii). min, max, sorted

# list = [(2,3), (4, 7), (8, 11), (3, 6)]
# print("List size is " + str(list))
# res1 = min(list)[0], max(list)[0]
# res2 = min(list)[1], max(list)[1]
# print(str(res1))
# print(str(res2))


# import calendar
# yy = int(input("Year"))
# mm = int(input("Month"))
# print(calendar.month(yy, mm))

# 2). Binary search

# def binarySearch(x, high, low, arr):
#     if high >= low:
#         mid = (high + low - 1) // 2
#         if arr[mid] == x:
#             return mid
#         elif arr[mid] > x:
#             return binarySearch(arr, low, mid - 1, x)
#         else:
#             return binarySearch(arr, high, mid + 1, x)
#     else:
#         return -1
#
# arr = [2, 3, 4, 10, 40]
# x = 10
# result = binarySearch(x, 0, len(arr) - 1, arr)
# if result != 1:
#     print(str(result))
# else:
#     print("Element is not present in array")

# 4). Count and sum the even and odd number

# maximum= int(input("Max"))
# even_total = 0
# odd_total = 0
#
# for i in range(1, maximum+1):
#     if (i % 2 == 0):
#         even_total = even_total + i
#     else:
#         odd_total = odd_total + i
#
# print("{0} = {1}". format(i, even_total))
# print("{0} = {1}". format(i, odd_total))

# 4). ii). linear search

# def linear_search(list1, key, n):
#     for i in range(0, n):
#         if (list1[i] == key):
#             return i
#         return -1
# list1 = [2, 4, 6, 8, 9]
# key = 7
# n = len(list1)
# result = linear_search(list1, key, n)
# if result == -1:
#     print("No")
# else:
#     print(result)

# Unit 3
# 1). Duplicate chars in string

# def removeDuplicate(str, n):
#     index = 0
#     for i in range(0, n):
#         for j in range(0, i+1):
#             if str[i] == str[j]:
#                 break
#         if j == i:
#             str[index] = str[i]
#             index += 1
#     return " ".join(str[:index])
# str="nichfornich"
# n = len(str)
# print(removeDuplicate(list(str), n))


#  2). Palindrome

# def isPalindrome(s):
#     return s == s[::-1]
# s = 'malayalam'
# ans = isPalindrome(s)
# if ans:
#     print("palindrome")
# else:
#     print("not a palindrome")

# from collections import Counter
# def winner(input):
#     votes = Counter(input)
#     dict = {}
#     for value in votes.values():
#         dict[value] = []
#     for (key,value) in votes.items():
#         dict[value].append(key)
# maxVote = sorted(dict.keys(),reverse=True)[0]
# if len(dict[maxVote])>1:
#     print (sorted(dict[maxVote])[0])
# else:
#     print (dict[maxVote][0])
# # Driver program
# if __name__ == "__main__":
#  input =['john','johnny','jackie','johnny',
#  'john','jackie','jamie','jamie',
#  'john','johnny','jamie','johnny',
#  'john']
#  winner(input)

#################################################################################
#################################################################################


# Data structures and algorithms practice
# Day 1
# Reversed linked list


# def reverseLinkedList(head):
#     prev, curr = None, head;
#
#     while curr:
#         nxt = curr.next
#         curr.next = prev
#         prev = curr
#         curr = nxt
#     return prev

# Pivot index
# def pivotIndex(nums):
#     total = sum(nums)
#     leftSum = 0
#     for i in range(len(nums)):
#         rightSum = total - nums[i] - leftSum
#         if leftSum == rightSum:
#             return i
#         leftSum += nums[i]
#     return -1
# nums = [1,7,3,6,5,6]
# print(pivotIndex(nums))

# Isomorphic string:
# def isomorphicStrings(s, t):
#     mapST, mapTS = {}, {}
#     for i in range(len(s)):
#         c1, c2 = s[i], t[i]
#         if (c1 in mapST and mapST[c1] != c2) or (c2 in mapTS and mapTS[c2] != c1):
#             return False
#         mapST[c1] = c2
#         mapTS[2] = c1
#     return True
#
# s = "foo"
# t = "bar"
# print(isomorphicStrings(s, t))


# Reverse linked
# def reverseLinked(head):
#     prev, curr = None, head
#     # prev -> curr -> next
#     while curr:
#         nxt = curr.next
#         curr.next = prev
#         prev = curr
#         curr = nxt
#     return prev


#  Subsequence
# def isSubsequence(s, t):
#     i, j = 0, 0
#     while i < len(s) and j < len(t):
#         if s[i] == t[j]:
#             i += 1
#         j += 1
#     return True if i == len(s) else False
#
# s = "abc"
# t = "abdc"
# print(isSubsequence(s, t))

# Sum of 1d array
# def runningSum(nums):
#     for i in range(1, len(nums)):
#         nums[i] = nums[i - 1]
#     return nums
# nums = [1,2,3,4,5]
# print(runningSum(nums))

# Pivot index
# def pivotIndex(num):
#     total = sum(num)
#
#     leftSum = 0
#
#     for i in range(len(num)):
#         rightSum = total - num[i] - leftSum
#         if leftSum == rightSum:
#             return i
#         leftSum += num[i]
#     return -1
#
# num = [1,7,3,6,5,6]
# print(pivotIndex(num))

# Merge two sorted list

# class ListNode:
#     def __init__(self, val = 0, next = None):
#         self.val = val
#         self.next = next
#
# class Solution:
#     def mergeList(self, l1 : ListNode, l2: ListNode) -> ListNode:
#         dummy = ListNode()
#         tail = dummy
#
#         while l1 and l2:
#             if l1.val < l2.val:
#                 tail.next = l1
#                 l1 = l1.next
#             else:
#                 tail.next = l2
#                 l2 = l2.next
#             tail = tail.next
#             if l1:
#                 tail.next = l1
#             elif l2:
#                 tail.next = l2
#             return dummy.next
# l1 = [1, 2, 3]
# l2 = [4, 5, 6]
# Solution.mergeList(l1, l2)


# Profit and loss in a stock
#     import collections
#     from typing import List

# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         l, r = 0, 1
#         maxP = 0
#         while r < len(prices):
#             if prices[l] < prices[r]:
#                 profit = prices[r] - prices[l]
#                 maxP = max(maxP, profit)
#             else:
#                 l = r
#             r += 1
#         return maxP
#
# # prices = [7,1,5,3,6,4]
# # print(profitStock(prices))
# print(Solution.maxProfit(prices=[7,1,5,3,6,4]))


# Longest Palindrome
# class Solution:
#     def longestPalindrome(s) -> int:
#         res = ""
#         resLen = 0
#         for i in range(len(s)):
#             #odd one
#             l, r = i, i
#             while l >= 0 and r < len(s) and s[l] == s[r]:
#                 if (r - l + 1) > resLen:
#                     res = s[l : r + 1]
#                     resLen = r - l + 1
#                 l -= 1
#                 r += 1
#             #even one
#             l, r = i, i + 1
#             while l >= 0 and r < len(s) and s[l] == s[r]:
#                 if (r - l + 1) > resLen:
#                     res = s[l : r + 1]
#                     resLen = r - l + 1
#                 l -= 1
#                 r += 1
#         return len(res)
# s = "abaddd"
# print(Solution.longestPalindrome(s))


# Tree preorder traversal

# def preOrderTraversal(s, root):
#     output = []
#     def dfs(node):
#         if not node:
#             return
#         output.append(node.val)
#         for i in node.children:
#             dfs(i)
#     dfs(root)
#     return output

# Binary tree level order traversal

# def binaryTreeTraversal(self, root: TreeNode) -> List[List[int]]:
#     res = []
#     q = collections.deque()
#     q.append(root)
#
#     while q:
#         qLen = len(q)
#         level= []
#         for i in range(qLen):
#             node = q.popleft()
#             if node:
#                 level.append(node.val)
#                 q.append(node.left)
#                 q.append(node.right)
#         if level:
#             res.append(level)
#     return res

# N ary tree using breath for search algorothm
# class Node:
#     def __init__(self, val=None, children=None):
#         self.val = val
#         self.children = children
# class Solution:
#     def preOrder(self, root: 'Node') -> List[int]:
#         output = []
#         def naryTree(node):
#             if not node:
#                 return
#             output.append(node.val)
#
#             for i in node.children:
#                 naryTree(i)
#             naryTree(root)
#             return output

# Binary tree order traversal
# import collections
# def binarySearch(root):
#     res = []
#     q = collections.deque
#     q.append(root)
#
#     while q:
#         qLen = len(q)
#         level = []
#         for i in range(qLen):
#             node = q.popleft()
#             if node:
#                 level.append(node.val)
#                 q.append(node.left)
#                 q.append(node.right)
#         if level:
#             res.appent(level)
#     return res
#

# Binary search

# -1, 0, 3, 5, 9, 12
#  L               R
# def binarySearch(nums, target):
#     l, r = 0, len(nums) - 1
#
#     while l <= r:
#         m = (l + r) // 2
#         if nums[m] > target:
#             r = m - 1
#         elif nums[m] < target:
#             r = m + 1
#         else:
#             return m
#     return -1
#
#
# nums = [-1, 0, 1, 2, 3]
# target = 2
# print(binarySearch(nums, target))

# Binary search
# def binarySearch(nums, target):
#     l, r = 0, len(nums) - 1
#
#     while l <= r:
#         m = (l + r) // 2
#         if nums[m] > target:
#             r = m - 1
#         elif nums[m] < target:
#             l = m + 1
#         else:
#             return m
#     return -1
# nums = [1, 2, 3, 4, 5 ]
# target = 2
# print(binarySearch(nums, target))

# Bad version

# def firstBadVersion(n, target):
#     low = 1
#     high = n
#     mid = 0
#     result = n
#     while (low <= high):
#         mid = (low + high) // 2
#         if mid:
#             result = mid
#             high = mid - 1
#         else:
#             low = mid - 1
#     return result
# n = 5
# target = 4
# print(firstBadVersion(n, target))

# def loop(x):
#     print(x*3)
#
# def map_simple(crazy, list):
#     for i in list:
#         crazy(i)
# list = ['biriyani', True, 3, '4', 5, 6 ]
# map_simple(crazy= [int], list = [1, 2, 3, 4, 5, 6 ])
# fruits = ['apple', 'orange', 'kiwi', 'pineapple']
# newFruit = [i for i in fruits if 'z' in i ]
# print(newFruit)
# print(list[1:-4])

# Binary Search Tree

# def binarySearchTree(node, left, right):
#     if not node:
#         return True
#      if not (node.val < right and node.val > left):
#         return False
#       return (binarySearchTree(node.left, left, node.val) and binarySearchTree(node.right, right, node.val))
#  return (binarySearchTree(root, float(-inf) and binarySearchTree(float(inf))

# Lowest common ancestor of Binary Search Tree

# class TreeNode:
#      def __init__(self, x):
#          self.val = x
#          self.left = None
#          self.right = None
# class Solution:
#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         curr = root
#
#         if (p.val > curr.val and q.val > curr.val):
#             curr = curr.right
#         elif (p.val < curr.val and q.val > curr.val):
#             curr = curr.left
#         else:
#             return curr

# class Human:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
# h1 = Human('Nich', 21)
# print(h1.age)

# def square(x):
#     return x * x
#
# numbers = [1, 2, 3, 4, 5]
#
# listSquares = map(square, numbers)
# print(list(listSquares))


# def complainManagemant(complain):
#     if complain:
#         print("You have an complaint and it has been stored successfully!... :)")
#         print("We may take immediate action")
#         complain = save.append(complain)
#         print(complain)
#     else:
#         return -1
# complain = input("Enter your complaints here: ")
# save = []
# print(complainManagemant(complain))

# def generateIp(s):
#
#     if s == " ":
#         print(index(s) + ".")
#     else:
#         return -1
#
# # Main
# s = print(int(input()))
# print(generateIp(s))

# def characterReplacement(s, k):
#     count = {}
#     res = 0
#
#     l = 0
#     maxf = 0
#
#     for r in range(len(s)):
#         count[s[r]] = count.get(s[r] , 0)
#         maxf = max(maxf, count[s[r]])
#         while (r - l + 1) - maxf > k:
#             count[s[l]] -= 1
#             l += 1
#         res = max(res, r - 1 + 1)
#     return res

# def findAnagrams(s, p):
#     if len(p) > len(s): return []
#     pCount, sCount = {}, {}
#     for i in range(len(p)):
#         pCount[p[i]] = 1 + pCount.get(p[i], 0)
#         sCount[s[i]] = 1 + sCount.get(s[i], 0)
#     res = [0] if sCount == pCount else []
#     l = 0
#     for r in range(len(p), len(s)):
#         sCount[s[r]] = 1 + sCount.get(s[r], 0)
#         sCount[s[l]] -= 1
#
#         if sCount[s[l]] == 0:
#             sCount.pop(s[l])
#         l += 1
#         if sCount == pCount:
#             res.append(l)
#     return res

# class Solution:
#     def twoSum(nums, target):
#         prevMap = {}
#         for i, n in enumerate(nums):
#             diff = target - n
#             if diff in prevMap:
#                 return [prevMap[diff], i]
#             prevMap[n] = i
#         return





# def findTriplets(arr, n):
#     for i in arr:
#         if arr[i] == True:
#             res = sorted(arr)
#             # print(res)
#             for j in res:
#                 if (0 < res[j]) != (0 > res[j]):
#                     print("j:", j)
#
#         # m = (l + r) // 2
#
# n = 5
# arr = [0, -1, 2, -3, 1]
#
# print(findTriplets(arr, n))




#
# def findTriplets(arr, n):
#     found = False
#     for i in range(0, n - 2):
#         for j in range(i + 1, n - 1):
#             for k in range(j + 1, n):
#                 if (arr[i] + arr[j] + arr[k] == 0):
#                     print(1)
#                     found = True
#     if (found == False):
#         print(0)


# Driver code
# arr = [0, -1, 2, -3, 1]
# n = len(arr)
# findTriplets(arr, n)


# def learningArr(v):
#     for i in range(0, v - 2):
#         for j in range(i + 1, v - 1):
#             for k in range(j + 1, n):
#                 print(i, j, k)
# v = [0, -1, 2, -3, 1]
# print(learningArr(v))


# def backspaceCompare(s):
#     arr = []
#     for i in range(len(s)):
#         res = arr.append(s)
#         if "#" in res:
#             res = res.pop(i)
#             print(res)
# s = "ab#cd"
# print(backspaceCompare(s))

# def decodestring(s):
#     stack = []
#
#     for i in range(len(s)):
#         if s[i] != "]":
#             stack.append(s[i])
#         else:
#             substr = ""
#             while stack[-1] != "[":
#                 substr = stack.pop() + substr
#             stack.pop()
#             k = ""
#             while stack and stack[-1].isdigit():
#                 k = stack.pop() + k
#                 stack.append(int(k) * substr)
#     return "".join(stack)

# Is the number happy

# def isHappy(self, n):
#     #  Hash set
#     visit = set()
#
#     while n not in visit:
#         visit.add(n)
#         n = self.sumofSquares(n)
#
#         if n == 1:
#             return True
#     return False
#
# def sumofSquares(n):
#     output = 0
#
#     while n:
#         digit = n % 10
#         digit = digit ** 2
#         output += digit
#         n = n // 10
#     return output


# def reverseArray(arr):
#     arr1, arr2 = [], []
#
#     for i in range(len(arr)):
#         l, r = min(range(arr[i])), max(range(arr[i]))
#         m = (l + r) // 2
#         if m < arr[i]:
#             arr1.append(m)
#             print("i" ,arr1)
#         elif m > arr[i]:
#             arr2.append(m)
#             print("j", arr2)
#
# arr = [1, 2, 3, 4, 5]
# print(reverseArray(arr))


# def reverseInGroups(arr, N, K):
#     i = 0
#     while(i<N):
#         if (i+K<N):
#             arr[i:i+K]=reversed(arr[i:i+K])
#             i += K
#     else:
#         arr[i:] = reversed(arr[i:])
#         i+=K
# arr = [1, 2, 3, 4, 5]
# N = 5
# K = 3
# print("this", reverseInGroups(arr, N, K))


# Program for spiral order
# class Solution:
#     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
#         res = []
#         left, right = 0, len(matrix[0])
#         top, bottom = 0, len(matrix)
#
#         while left < right and top < bottom:
#             for i in range(left, right):
#                 res.append(matrix[top][i])
#             top += 1
#
#             for i in range(top, bottom):
#                 res.append(matrix[i][right - 1])
#             right -= 1
#
#             if not (left < right and top < bottom):
#                 break
#
#             for i in range(right - 1, left - 1, -1):
#                 res.append(matrix[bottom - 1][i])
#             bottom -= 1
#
#             for i in range(bottom - 1, top - 1, -1):
#                 res.append(matrix[i][left])
#             left += 1
#         return res


# def findDups(nums):
#     for i in range(0, len(nums)):
#         if nums[i] == nums[i]:
#             arr2 = []
#             arr2.append(nums[i])
#             print(arr2)
#             return True
#         else:
#             return False
# nums = [1, 2, 3, 1]
# print(findDups(nums))

# def sortAndFindRepitations(arr):
#     s1, s2 = 0, len(arr)
#     for i in arr:
#         res = arr.sort()
#         for j in range(arr):
#             j -= 1
#             if arr[i] == arr[j]:
#                 print(arr[j])
# arr = [1, 1, 3, 2]
# print(sortAndFindRepitations(arr))

# def containDuplicates(nums):
#     size = len(nums)
#     repeated = []
#     for i in range(size):
#         k = i + 1
#         for j in range(k, size):
#             if nums[i] == nums[j] and nums[i] not in repeated:
#                 repeated.append(nums[i])
#                 return True
#     return False
# nums = [1, 2, 3, 4]
# print(containDuplicates(nums))

# def dups(nums):
#     return not len(nums) == len(set(nums))
# nums = [1, 2, 3, 1]
# print(dups(nums))

def sort012(a, arr_size):
    lo = 0
    hi = arr_size - 1
    mid = 0
    # Iterate till all the elements
    # are sorted
    while mid <= hi:
        # If the element is 0
        if a[mid] == 0:
            a[lo], a[mid] = a[mid], a[lo]
            lo = lo + 1
            mid = mid + 1
        # If the element is 1
        elif a[mid] == 1:
            mid = mid + 1
        # If the element is 2
        else:
            a[mid], a[hi] = a[hi], a[mid]
            hi = hi - 1
    return a
# Function to print array
def printArray(a):
    for k in a:
        print(k, end=' ')
# Driver Program
arr = [0, 2, 1, 2, 0]
arr_size = len(arr)
arr = sort012(arr, arr_size)
printArray(arr)



