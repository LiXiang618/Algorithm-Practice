#two sum
a = [11,52,33,42,42]
b = [1,2,3,4,5]
ls = sorted(zip(a,b))
[x for (x,y) in ls]
#solution1: use for loop
#solution2: sort first then two pointers; however, be careful when the indexes are out of order

#contains duplicate
nums = [2,3,2,4,1,2,3,4,5]
d = {}
d[nums[0]] = 1
d.get(1,0)
for i in range(1,len(nums)):
    if(d.get(nums[i],0)==0):
        d[nums[i]]=1
    else:
        print("True")
        break
hash()
#solution1: sort then find
#solution2: hashtable dictionary

#contains duplicate II
nums = [2,3,2,4,1,2,3,4,5]
k=2
#solution1: use dictionary
d = {}
d[nums[0]] = 0
for i in range(1,len(nums)):
    if(d.get(nums[i],-1)==-1):
        d[nums[i]]=i
    else:
        if(i-d[nums[i]]<=k):
            print(True)
        else:
            d[nums[i]]=i
print(False)
#Warn: update the dictionary!!!
#solution2: make some improvement
d = {}
for i,v in enumerate(nums):
    if(v in d and i-d[v]<=k):   #if v not in d, it will neglect the rest
        print(True)
    d[v]=i
print(False)

#contains duplicate III
#solution1: for and while, time limit exceeded
#solution2: use mode to reduce the value
def containsNearbyAlmostDuplicate(self, nums, k, t):
    d = {}
    if(t!=0):
        for i,v1 in enumerate(nums):
            if(v1//t in d):
                for ls in d[v1//t]:
                    if(i-ls[0]<=k and abs(v1-ls[1])<=t):
                        return(True)
            if((v1//t)+1 in d):
                for ls in d[(v1//t)+1]:
                    if(i-ls[0]<=k and abs(v1-ls[1])<=t):
                        return(True)
            if(d.get(v1//t,-1)==-1):
                d[v1//t] = [] + [[i,v1]]
            else:
                d[v1//t] = d[v1//t] + [[i,v1]]
            if(d.get((v1//t)+1,-1)==-1):
                d[(v1//t)+1] = [] + [[i,v1]]
            else:
                d[(v1//t)+1] = d[(v1//t)+1]+[[i,v1]]
    else:
        for i,v1 in enumerate(nums):
            if(v1 in d and i-d[v1]<=k):
                return(True)
            d[v1] = i
    print(d)
    return(False)
#solution2:
def containsNearbyAlmostDuplicate(self, nums, k, t):
    """
    :type nums: List[int]
    :type k: int
    :type t: int
    :rtype: bool
    """
    d = {}
    if (t != 0):
        for i, v1 in enumerate(nums):
            bucknum = v1 // t
            for x in range(bucknum - 1, bucknum + 2):
                if (x in d and abs(v1 - d[x]) <= t):
                    return (True)
            d[bucknum] = v1                 #key is the mod, value is the number
            if (i >= k):
                del d[nums[i - k] // t]     #don't need to worry deleting right elements, the "right" would return true if it was right
    else:
        for i, v1 in enumerate(nums):
            if (v1 in d and i - d[v1] <= k):
                return (True)
            d[v1] = i
    return (False)
#solution3: make some improvement
def containsNearbyAlmostDuplicate(self, nums, k, t):
    """
    :type nums: List[int]
    :type k: int
    :type t: int
    :rtype: bool
    """
    if (t < 0):             #This is essential, without which, line "x" will go wrong
        return (False)
    d = {}
    if (t != 0):
        for i, v1 in enumerate(nums):
            bucknum = v1 // t
            if (bucknum in d):      #line "x"
                return (True)
            elif (bucknum + 1 in d and abs(v1 - d[bucknum + 1]) <= t):
                return (True)
            elif (bucknum - 1 in d and abs(v1 - d[bucknum - 1]) <= t):
                return (True)
            d[bucknum] = v1
            if (i >= k):
                del d[nums[i - k] // t]
    else:
        for i, v1 in enumerate(nums):
            if (v1 in d and i - d[v1] <= k):
                return (True)
            d[v1] = i

    return (False)

#Rectangle Area
def computeArea(self, A, B, C, D, E, F, G, H):
    """
    :type A: int
    :type B: int
    :type C: int
    :type D: int
    :type E: int
    :type F: int
    :type G: int
    :type H: int
    :rtype: int
    """
    S1 = (C - A) * (D - B)
    S2 = (G - E) * (H - F)
    if (S1 == 0 or S2 == 0):
        return (S1 + S2)
    X = 0
    Y = 0
    if ((C - G) * (E - A) > 0):
        X = min(C - A, G - E)
    else:
        X = (abs(abs(A - G) - abs(C - E)) != (C - A) + (G - E)) * (
        max(abs(A - G), abs(C - E)) - abs(abs(A - G) - abs(C - E)))
    if ((D - H) * (F - B) > 0):
        Y = min(D - B, H - F)
    else:
        Y = (abs(abs(D - F) - abs(B - H)) != (D - B) + (H - F)) * (
        max(abs(D - F), abs(B - H)) - abs(abs(D - F) - abs(B - H)))
    S3 = X * Y
    return (S1 + S2 - S3)
#solution1:
#case1: area = 0
#case2: separated
#case3: contained
#case4: overlapped
#solution2:
def computeArea(self, A, B, C, D, E, F, G, H):
    """
    :type A: int
    :type B: int
    :type C: int
    :type D: int
    :type E: int
    :type F: int
    :type G: int
    :type H: int
    :rtype: int
    """
    overlap = max(min(C, G) - max(A, E), 0) * max(min(D, H) - max(B, F), 0)
    return (A - C) * (B - D) + (E - G) * (F - H) - overlap

#Add Digits
def addDigits(self, num):
    """
    :type num: int
    :rtype: int
    """
    if num == 0:
        return (0)
    a = num % 9
    if a == 0:
        return (9)
    else:
        return (a)

#Majority Element
#solution1: disctionary
def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    d = {}
    for i, v in enumerate(nums):
        if (v in d):
            d[v] += 1
        else:
            d[v] = 1
    for key, value in d.items():
        if (value > int(len(nums) / 2)):
            return (key)
#solution2: smart method
def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    major = nums[0]
    count = 1
    for i, v in enumerate(nums[1:]):
        if (count == 0):
            major = v
        if (v == major):
            count += 1
        else:
            count -= 1
    return (major)

#Majority Element II
#solution1: efficiency is low
def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    if (len(nums) == 0):
        return ([])
    num12 = [nums[0], None]
    count12 = [1, 0]
    for i, v in enumerate(nums[1:]):
        if (v in num12):
            if (num12[0] == v):
                count12[0] += 1
            else:
                count12[1] += 1
        else:
            if (None in num12):
                if (num12[0] == None):
                    num12[0] = v
                    count12[0] = 1
                else:
                    num12[1] = v
                    count12[1] = 1
            else:
                count12[0] -= 1
                count12[1] -= 1
                if (count12[0] == 0):
                    num12[0] = None
                if (count12[1] == 0):
                    num12[1] = None
    for i, v in enumerate(num12):
        count12[i] = 0
        for j, w in enumerate(nums):
            if (w == v):
                count12[i] += 1
    ls = []
    crit = int(len(nums) / 3)
    if (count12[0] > crit):
        ls.append(num12[0])
    if (count12[1] > crit):
        ls.append(num12[1])
    return (ls)
#solution2:
def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    if not nums:
        return ([])
    num1, num2 = nums[0], None
    count1, count2 = 1, 0
    for v in nums[1:]:
        if (num1 == v):
            count1 += 1
        elif (num2 == v):
            count2 += 1
        elif (count1 == 0):
            num1, count1 = v, 1
        elif (count2 == 0):
            num2, count2 = v, 1
        else:
            count1, count2 = count1 - 1, count2 - 1
    return ([x for x in (num1, num2) if nums.count(x) > len(nums) // 3])
#Warn: a,b = 1,2 is much more efficient than a = 1 b = 2

#Add Two Numbers
#solution1
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
def addTwoNumbers(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    if ((not l1) and (not l2)):
        return ([])
    elif (not l1):
        return (l2)
    elif (not l2):
        return (l1)
    ll1 = l1
    ll2 = l2
    ths = (ll1.val + ll2.val) % 10
    nex = (ll1.val + ll2.val) // 10
    ls = ListNode(ths)
    p = ls
    while (ll1.next or ll2.next):
        if (ll1.next and ll2.next):
            ll1 = ll1.next
            ll2 = ll2.next
            ths = ll1.val + ll2.val + nex
            nex = ths // 10
            ths = ths % 10
            p.next = ListNode(ths)
            p = p.next
        elif (ll1.next):
            ll1 = ll1.next
            ths = ll1.val + nex
            nex = ths // 10
            ths = ths % 10
            p.next = ListNode(ths)
            p = p.next
        elif (ll2.next):
            ll2 = ll2.next
            ths = ll2.val + nex
            nex = ths // 10
            ths = ths % 10
            p.next = ListNode(ths)
            p = p.next
    if (nex > 0):
        p.next = ListNode(nex)
    return (ls)
#solution2: make some change in the while loop
def addTwoNumbers(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    if ((not l1) and (not l2)):
        return ([])
    elif (not l1):
        return (l2)
    elif (not l2):
        return (l1)
    ths = (l1.val + l2.val) % 10
    nex = (l1.val + l2.val) // 10
    ls = ListNode(ths)
    p = ls
    while (l1.next or l2.next):
        if (l1.next and l2.next):
            l1 = l1.next
            l2 = l2.next
            ths = l1.val + l2.val + nex
            nex = ths // 10
            ths = ths % 10
            p.next = ListNode(ths)
            p = p.next
        elif (l1.next):
            while (l1.next):                #Here is the difference
                l1 = l1.next
                ths = l1.val + nex
                nex = ths // 10
                ths = ths % 10
                p.next = ListNode(ths)
                p = p.next
            if (nex > 0):
                p.next = ListNode(nex)
                return (ls)
        elif (l2.next):
            while (l2.next):
                l2 = l2.next
                ths = l2.val + nex
                nex = ths // 10
                ths = ths % 10
                p.next = ListNode(ths)
                p = p.next
            if (nex > 0):
                p.next = ListNode(nex)
                return (ls)
    if (nex > 0):
        p.next = ListNode(nex)
    return (ls)
#solution3: concise version but the efficiency is lower than solution2
def addTwoNumbers(self, l1, l2):
    carry = 0
    root = n = ListNode(0)
    while l1 or l2 or carry:
        v1 = v2 = 0
        if l1:
            v1 = l1.val
            l1 = l1.next
        if l2:
            v2 = l2.val
            l2 = l2.next
        carry, val = divmod(v1 + v2 + carry, 10)
        n.next = ListNode(val)
        n = n.next
    return root.next

#gcd and lcm

#Longest Substring Without Repeating Characters
#solution1:
def lengthOfLongestSubstring(self, s):
    """
    :type s: str
    :rtype: int
    """
    max_str = ""
    max_c = 0
    tmp_str = ""
    tmp_c = 0
    for a in s:
        if (a in tmp_str):
            index = tmp_str.find(a)
            tmp_str = tmp_str[index + 1:] + a
            if (index == 0):
                max_str = tmp_str
            else:
                tmp_c = len(tmp_str)
        else:
            tmp_str = tmp_str + a
            tmp_c += 1
            if (tmp_c > max_c):
                max_str = tmp_str
                max_c += 1
    return (max_c)
#solution2: dictionary
def lengthOfLongestSubstring(self, s):
    """
    :type s: str
    :rtype: int
    """
    d = {}
    start = max_len = 0
    for i in range(len(s)):
        if (s[i] in d and d[s[i]] >= start):
            start = d[s[i]] + 1
        else:
            max_len = max(max_len, i - start + 1)
        d[s[i]] = i
    return (max_len)

#Median of Two Sorted Arrays
#Solution1: classification is essential!
def findMedianSortedArrays(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    aa = nums1
    bb = nums2
    if (len(aa) == 0 or len(bb) == 0):
        if (len(aa) == 0):
            cc = aa
            aa = bb
            bb = cc
        if (len(aa) % 2 == 1):
            mid = len(aa) // 2
            return (float(aa[mid]))
        else:
            right = len(aa) // 2
            left = right - 1
            return (float(aa[left] + aa[right]) / 2)
    while (True):
        if (len(aa) == 1 and len(bb) == 1):  # 1 and 1
            return (float(aa[0] + bb[0]) / 2)
        elif ((len(aa) % 2 == 1 and len(bb) == 1) or (len(bb) % 2 == 1 and len(aa) == 1)):  # 1 and odd
            if (len(bb) % 2 == 1 and len(aa) == 1):
                cc = aa
                aa = bb
                bb = cc
            mid = len(aa) // 2
            if (bb[0] == aa[mid]):
                return (float(bb[0]))
            elif (bb[0] > aa[mid + 1]):
                return (float(aa[mid] + aa[mid + 1]) / 2)
            elif (bb[0] < aa[mid - 1]):
                return (float(aa[mid] + aa[mid - 1]) / 2)
            else:
                return (float(bb[0] + aa[mid]) / 2)
        elif (len(bb) == 1 or len(aa) == 1):  # 1 and even
            if (len(aa) == 1):
                cc = aa
                aa = bb
                bb = cc
            right = len(aa) // 2
            left = right - 1
            if (bb[0] < aa[left]):
                return (float(aa[left]))
            elif (bb[0] > aa[right]):
                return (float(aa[right]))
            else:
                return (float(bb[0]))
        elif (len(aa) % 2 == 0 and len(bb) % 2 == 0):  # even and even
            right_a = len(aa) // 2
            left_a = right_a - 1
            a = float(aa[right_a] + aa[left_a]) / 2
            right_b = len(bb) // 2
            left_b = right_b - 1
            b = float(bb[right_b] + bb[left_b]) / 2
            # if(a==b):
            #    return(float(a))
            if (aa[left_a] <= bb[left_b] and bb[right_b] <= aa[right_a]):
                return (float(bb[left_b] + bb[right_b]) / 2)
            elif (bb[left_b] <= aa[left_a] and aa[right_a] <= bb[right_b]):
                return (float(aa[left_a] + aa[right_a]) / 2)
            elif (a < b):
                length = min(right_a, right_b)
                aa = aa[length:]
                bb = bb[0:-length]
            else:
                length = min(right_a, right_b)
                aa = aa[0:-length]
                bb = bb[length:]
        elif (len(aa) % 2 == 1 and len(bb) % 2 == 1):  # odd and odd
            mid_a = len(aa) // 2
            mid_b = len(bb) // 2
            a = aa[mid_a]
            b = bb[mid_b]
            if (a == b):
                return (float(a))
            elif (a < b):
                length = min(mid_a, mid_b)
                aa = aa[length:]
                bb = bb[0:-length]
            else:
                length = min(mid_a, mid_b)
                aa = aa[0:-length]
                bb = bb[length:]
        else:  # odd and even
            if (len(bb) % 2 == 1):
                cc = aa
                aa = bb
                bb = cc
            mid = len(aa) // 2
            right = len(bb) // 2
            left = right - 1
            a = aa[mid]
            b = float(bb[right] + bb[left]) / 2
            if (bb[left] <= a and a <= bb[right]):
                return (float(a))
            elif (a < b):
                length = min(mid, right)
                aa = aa[length:]
                bb = bb[0:-length]
            elif (a > b):
                length = min(mid, right)
                aa = aa[0:-length]
                bb = bb[length:]

#Longest Palindromic Substring
#solution1:
def longestPalindrome(self, s):
    """
    :type s: str
    :rtype: str
    """
    def isPa(s):
        i = 0
        j = len(s) - 1
        while (i <= j):             # i!=j is not correct, because it may happen that i>j
            if (s[i] != s[j]):
                return (False)
            i += 1
            j -= 1
        return (True)
    max_len = 0
    max_str = ""
    for i in range(len(s)):
        j = len(s) - 1
        while (j >= i):
            if (s[j] == s[i] and isPa(s[i:j + 1]) and j - i + 1 > max_len):
                max_len = j - i + 1
                max_str = s[i:j + 1]
                break
            j -= 1
    return (max_str)
#solution2:
def longestPalindrome(self, s):
    """
    :type s: str
    :rtype: str
    """
    def isPa(s):
        i = 0
        j = len(s) - 1
        while (i <= j):
            if (s[i] != s[j]):
                return (False)
            i += 1
            j -= 1
        return (True)
    max_len = 0
    max_str = ""
    for i in range(len(s)):
        j = len(s) - 1
        while (j >= i and len(s) - i > max_len):    #make some change here
            if (s[j] == s[i] and isPa(s[i:j + 1]) and j - i + 1 > max_len):
                max_len = j - i + 1
                max_str = s[i:j + 1]
                break
            j -= 1
    return (max_str)
#solution3: Basic thought is simple.
    # When you increase s by 1 character,
    # you could only increase maxPalindromeLen by 1 or 2,
    # and that new maxPalindrome includes this new character.
def longestPalindrome(self, s):
    if len(s) == 0:
        return 0
    maxLen = 1
    start = 0
    for i in range(len(s)):
        if i - maxLen >= 1 and s[i - maxLen - 1:i + 1] == s[i - maxLen - 1:i + 1][::-1]:
            start = i - maxLen - 1
            maxLen += 2
            continue
        if i - maxLen >= 0 and s[i - maxLen:i + 1] == s[i - maxLen:i + 1][::-1]:
            start = i - maxLen
            maxLen += 1
    return s[start:start + maxLen]

#ZigZag Conversion
#solution1:
def convert(self, s, numRows):
    """
    :type s: str
    :type numRows: int
    :rtype: str
    """
    if (numRows <= 1):
        return (s)
    ls = [[] for i in range(numRows)]
    index = 0
    down = True
    for i in s:
        ls[index].append(i)
        if (down):
            if (index == numRows - 1):
                down = False
                index -= 1
            else:
                index += 1
        else:
            if (index == 0):
                down = True
                index += 1
            else:
                index -= 1
    return (''.join([''.join(x) for x in ls]))

#Palindrome Number
#solution1:
def isPalindrome(self, x):
    """
    :type x: int
    :rtype: bool
    """
    if (x < 0):
        return (False)
    if (x < 10):
        return (True)
    n = x
    i = j = 0
    while (n >= 10):
        n = n // 10
        i += 1
    while (i >= j):
        if ((x % (10 ** (i + 1))) // (10 ** i) != (x % (10 ** (j + 1))) // (10 ** j)):
            return (False)
        i -= 1
        j += 1
    return (True)

#Regular Expression Matching
#Solution1: time limit exceeded
def isMatch(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """
    def myFun(s, p):
        lenp = len(p)
        lens = len(s)
        if (lenp == 0):
            if (lens == 0):
                return (True)
            else:
                return (False)
        i = 0
        j = 0
        while (i < lenp):
            if (p[i] == '.'):
                if (i + 1 < lenp):
                    if (p[i + 1] == '*'):
                        return (any([myFun(s[j:], p[i + 2:]), myFun(s[j:], p[i] + p[i:])]))
                    else:
                        if (j < lens):
                            j += 1
                            i += 1
                        else:
                            return (False)
                else:
                    if (j == lens - 1):
                        return (True)
                    else:
                        return (False)
            else:
                if (i + 1 < lenp):
                    if (p[i + 1] == '*'):
                        return (any([myFun(s[j:], p[i + 2:]), myFun(s[j:], p[i] + p[i:])]))
                    else:
                        if (j < lens and p[i] == s[j]):
                            j += 1
                            i += 1
                        else:
                            return (False)
                else:
                    if (j == lens - 1 and p[i] == s[j]):
                        return (True)
                    else:
                        return (False)
    return (myFun(s, p))
#solution2: Instead of using any(), using "if else" could improve efficiency.
def isMatch(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """

    def myFun(s, p):
        lenp = len(p)
        lens = len(s)
        if (lenp == 0):
            if (lens == 0):
                return (True)
            else:
                return (False)
        i = 0
        j = 0
        while (i < lenp):
            if (p[i] == '.'):
                if (i + 1 < lenp):
                    if (p[i + 1] == '*'):
                        if (j < lens):
                            if (myFun(s[j + 1:], p[i:])):
                                return (True)
                            elif (myFun(s[j:], p[i + 2:])):
                                return (True)
                            else:
                                return (False)
                                # return(any([myFun(s[j+1:],p[i:]),myFun(s[j:],p[i+2:])]))
                        else:
                            return (myFun(s[j:], p[i + 2:]))
                    else:
                        if (j < lens):
                            j += 1
                            i += 1
                        else:
                            return (False)
                else:
                    if (j == lens - 1):
                        return (True)
                    else:
                        return (False)
            else:
                if (i + 1 < lenp):
                    if (p[i + 1] == '*'):
                        if (j < lens):
                            if (s[j] == p[i] and myFun(s[j + 1:], p[i:])):
                                return (True)
                            elif (myFun(s[j:], p[i + 2:])):
                                return (True)
                            else:
                                return (False)
                                # return(any([s[j]==p[i] and myFun(s[j+1:],p[i:]),myFun(s[j:],p[i+2:])]))
                        else:
                            return (myFun(s[j:], p[i + 2:]))
                    else:
                        if (j < lens and p[i] == s[j]):
                            j += 1
                            i += 1
                        else:
                            return (False)
                else:
                    if (j == lens - 1 and p[i] == s[j]):
                        return (True)
                    else:
                        return (False)
    return (myFun(s, p))
#solution3: DP
def isMatch(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """
    m = len(s)  # i,str
    n = len(p)  # j,pattern
    # ls = [[False]*(n+1)]*(m+1)
    ls = [[False for a in range(n + 1)] for b in range(m + 1)]
    ls[0][0] = True
    for i in range(m + 1):
        for j in range(1, n + 1):
            if (p[j - 1] != '*'):
                ls[i][j] = i > 0 and ls[i - 1][j - 1] and (p[j - 1] == s[i - 1] or p[j - 1] == '.')
            else:
                ls[i][j] = ls[i][j - 2] or (i > 0 and (s[i - 1] == p[j - 2] or p[j - 2] == '.') and ls[i - 1][j])
    return (ls[m][n])

#Container With Most Water
#solution1: DP; time limit exceeded; In fact, we don't need DP.
def maxArea(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    n = len(height)
    ls = [[0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(j - 1, -1, -1):
            ls[i][j] = max(ls[i + 1][j], ls[i][j - 1], (j - i) * min(height[i], height[j]))
    return (ls[0][n - 1])
#solution2: two pointers
def maxArea(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    n = len(height)
    left = []
    right = []
    tmp = maxl = 0
    while (tmp < n):
        if (height[tmp] > maxl):
            maxl = height[tmp]
            left.append((tmp, maxl))
        tmp += 1
    tmp -= 1
    maxArea = ii = jj = 0
    len_left = len(left)
    len_right = len(right)
    while (ii < len_left and jj < len_right):
        (i, vi) = left[ii]
        (j, vj) = right[jj]
        area = min(vi, vj) * (j - i)
        maxArea = area if area > maxArea else maxArea
        if (vi >= vj):
            jj += 1
        else:
            ii += 1
    return (maxArea)

#Roman to Integer
#Solution:
def romanToInt(self, s):
    """
    :type s: str
    :rtype: int
    """
    d = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    sum = 0
    for i in range(len(s) - 1):
        this = d[s[i]]
        next = d[s[i + 1]]
        if (this < next):
            sum -= this
        else:
            sum += this
    sum += d[s[-1]]
    return (sum)

#Integer to Roman
#solution1:
def intToRoman(self, num):
    """
    :type num: int
    :rtype: str
    """
    s = "IVXLCDMZ"
    ls = [1, 5, 10, 50, 100, 500, 1000, 5000]
    i = 0
    ret = ""
    while (num > 0):
        if (num > ls[i]):
            i += 1
        else:
            if (s[i] in "VLDZ"):
                if (num == ls[i]):
                    ret = ret + s[i]
                    num = num - ls[i]
                elif (num >= 0.8 * ls[i]):
                    ret = ret + s[i - 1] + s[i]
                    num = num + ls[i - 1] - ls[i]
                else:
                    ret = ret + s[i - 1]
                    num = num - ls[i - 1]
            else:
                if (ls[i] == num):
                    ret = ret + s[i]
                    num = num - ls[i]
                elif (num >= 0.9 * ls[i]):
                    ret = ret + s[i - 2] + s[i]
                    num = num + ls[i - 2] - ls[i]
                else:
                    ret = ret + s[i - 1]
                    num = num - ls[i - 1]
            i = 0
    return (ret)
#solution2:
def intToRoman(self, num):
    """
    :type num: int
    :rtype: str
    """
    M = ["", "M", "MM", "MMM"]
    C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    return (M[num // 1000] + C[(num % 1000) // 100] + X[(num % 100) // 10] + I[num % 10]);

#Longest Common Prefix
#solution:
def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
        return ("")
    length = min([len(s) for s in strs])
    common = ""
    n = len(strs)
    for i in range(length):
        s = strs[0][i]
        if (all([strs[j][i] == s for j in range(1, n)])):
            common += s
        else:
            break
    return (common)

#3Sum
#solution:
def threeSum(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    # if(len(nums)<3 or min(nums)>0 or max(nums)<0):
    #    return([])
    nums = sorted(nums)
    ls = []
    for k in range(len(nums)):
        if (nums[k] > 0):
            break
        i = k + 1
        j = len(nums) - 1
        while (i < j):
            sum2 = nums[i] + nums[j]
            if (sum2 + nums[k] > 0):
                j -= 1
            elif (sum2 + nums[k] < 0):
                i += 1
            else:
                ls.append((nums[k], nums[i], nums[j]))
                j -= 1
                i += 1
    ls = set(ls)
    return ([[i, j, k] for (i, j, k) in ls])

#3Sum Cloest
#solution:
def threeSumClosest(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    m = float('Inf')
    ret = 0
    nums.sort()
    for k in range(len(nums)):
        i = k + 1
        j = len(nums) - 1
        while (i < j):
            sum3 = nums[k] + nums[i] + nums[j]
            if (abs(sum3 - target) < m):
                m = abs(sum3 - target)
                ret = sum3
            if (sum3 > target):
                j -= 1
            elif (sum3 < target):
                i += 1
            else:
                return (sum3)
    return (ret)

#Letter Combinations of a Phone Number
def letterCombinations(self, digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    if not digits:
        return ([])
    nums = [" ", "*", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
    ls = [s for s in nums[int(digits[0])]]
    i = 1
    while (i < len(digits)):
        ls = [a + b for a in ls for b in nums[int(digits[i])]]
        i += 1
    return (ls)

#4Sum
#solution:
def fourSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    if (len(nums) < 4):
        return ([])
    ls = []
    nums.sort()
    for i in range(len(nums) - 3):
        for j in range(i + 1, len(nums) - 2):
            k = j + 1
            l = len(nums) - 1
            while (k < l):
                sum4 = nums[i] + nums[j] + nums[k] + nums[l]
                if (sum4 > target):
                    l -= 1
                elif (sum4 < target):
                    k += 1
                else:
                    if ([nums[i], nums[j], nums[k], nums[l]] not in ls):
                        ls.append([nums[i], nums[j], nums[k], nums[l]])
                    l -= 1
                    k += 1
    return (ls)

#Remove Nth Node From End of List
#solution1: use stack to store the node
def removeNthFromEnd(self, head, n):
    """
    :type head: ListNode
    :type n: int
    :rtype: ListNode
    """
    if not head.next:
        return ([])
    this = head
    ls = []
    ls.append(this)
    while (this.next):
        this = this.next
        ls.append(this)
    while (n > 0):
        this = ls.pop()
        n -= 1
    if (this == head):
        return (this.next)
    else:
        father = ls.pop()
        father.next = this.next
        return (head)
#solution2: use two pointer, one fast, one slow
def removeNthFromEnd(self, head, n):
    fast = slow = head
    for _ in range(n):
        fast = fast.next
    if not fast:
        return head.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head

#Valid Parenthesis
#solution1: using stack
def isValid(self, s):
    """
    :type s: str
    :rtype: bool
    """
    ls = []
    for a in s:
        if (a in ['[', '(', '{']):
            ls.append(a)
        else:
            if not ls:
                return (False)
            ele = ls.pop()
            if ((a == ']' and ele != '[') or (a == ')' and ele != '(') or (a == '}' and ele != '{')):
                return (False)
    if (ls):
        return (False)
    return (True)
#solution2: stack with dictionary
def isValid(self, s):
    """
    :type s: str
    :rtype: bool
    """
    ls = []
    d = {'[': ']', '(': ')', '{': '}'}
    for a in s:
        if (a in d.keys()):
            ls.append(a)
        else:
            if ((not ls) or d[ls.pop()] != a):
                return (False)
    return (ls == [])   #very smart!

#Merge Two Sorted Lists
#solution:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def mergeTwoLists(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    p1 = l1
    p2 = l2
    ls = ListNode(0)
    p = ls
    while (p1 and p2):  # worth thinking
        if (p1.val <= p2.val):
            p.next = p1
            p = p.next
            p1 = p1.next
        else:
            p.next = p2
            p = p.next
            p2 = p2.next
    if not p1:
        p.next = p2
    else:
        p.next = p1
    return (ls.next)

#Generate Parenthesis
#solution1.1: DP
def generateParenthesis(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    array = [[""]]
    for k in range(1, n + 1):
        ls = []
        i = 1
        j = k - 1
        while (i <= j):
            ls = ls + [s1 + s2 for s1 in array[i] for s2 in array[j] if s1 + s2 not in ls]
            ls = ls + [s2 + s1 for s1 in array[i] for s2 in array[j] if s2 + s1 not in ls]
            i += 1
            j -= 1
        ls = ls + ['(' + s + ')' for s in array[k - 1] if s not in ls]
        array.append(ls)
    return (array[n])
#solution1.2: use set to improve
def generateParenthesis(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    array = [[""]]
    for k in range(1, n + 1):
        ls = []
        i = 1
        j = k - 1
        while (i <= j):
            ls = ls + [s1 + s2 for s1 in array[i] for s2 in array[j]]
            ls = ls + [s2 + s1 for s1 in array[i] for s2 in array[j]]
            i += 1
            j -= 1
        ls = ls + ['(' + s + ')' for s in array[k - 1]]
        array.append(list(set(ls)))
    return (array[n])

#Merge k Sorted Lists
#solution1: refer to Merge 2 Sorted Lists, long time
def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    ls = ListNode(0)
    p = ls
    ps = [x for x in lists if x]
    while (any(ps)):
        ps_val = [x.val for x in ps]
        index = ps_val.index(min(ps_val))
        p.next = ps[index]
        p = p.next
        ps[index] = ps[index].next
        if not ps[index]:
            del ps[index]
    return (ls.next)
#solution2: heap
def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    from heapq import heappush, heappop, heapreplace, heapify
    dummy = node = ListNode(0)
    heap = [(x.val, x) for x in lists if x]
    heapify(heap)
    while (heap):
        v, x = heap[0]
        if not x.next:
            heappop(heap)   #only change heap size when necessary
        else:
            heapreplace(heap, (x.next.val, x.next)) #heapreplace() is more efficient than heappop() followed by heappush()
        node.next = x
        node = node.next
    return (dummy.next)

#Swap Nodes in Pairs
#solution1:
def swapPairs(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    dummy = p = ListNode(0)
    p.next = head
    while (p.next and p.next.next):
        tmp = p.next
        p.next = p.next.next
        p = p.next
        tmp.next = p.next
        p.next = tmp
        p = p.next
    return (dummy.next)
#solution2: fast and slow
def swapPairs(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    dummy = fast = slow = ListNode(0)
    fast.next = head
    while (fast.next and fast.next.next):
        slow = fast.next
        fast.next = slow.next
        fast = slow.next
        slow.next = fast.next
        fast.next = slow
        fast = slow
    return (dummy.next)

#Reverse Nodes in k-Group
#solution:
def reverseKGroup(self, head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    dummy = fast = slow = ListNode(0)
    fast.next = head
    kk = k
    while (fast.next):
        fast = fast.next
        kk -= 1
        if (kk == 0):
            kk = k
            begin = slow
            end = fast.next
            slow = begin.next
            begin.next = fast
            p = slow
            tmp = ListNode(0)
            tmp.next = p.next
            x = end
            while (tmp.next != end):
                this = p
                tmp.next = p.next.next
                p = p.next
                this.next = x
                x = this
            p.next = x
            fast = slow
    return (dummy.next)

#Remove Duplicates from Sorted Array
#solution:
def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return (len([]))
    i = 1
    last = nums[0]
    while (i < len(nums)):
        if (nums[i] == last):
            del nums[i]
        else:
            last = nums[i]
            i += 1
    return (len(nums))

#Remove Element
#solution1:
def removeElement(self, nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    if not nums:
        return (0)
    i = 0
    while (i < len(nums)):
        if (nums[i] == val):
            del nums[i]
        else:
            i += 1
    return (len(nums))
#solution2: two pointers
def removeElement(self, nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    if not nums:
        return (0)
    i = 0
    j = len(nums) - 1
    while (j >= 0 and nums[j] == val):
        j -= 1
    if (j < 0):
        return (0)
    while (i < j):
        if (nums[i] == val and nums[j] != val):
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
        elif (nums[i] == val):
            j -= 1
        else:
            i += 1
    if (i == j and nums[j] == val):
        return (j)
    return (j + 1)

#Implement strStr()
#solution:
def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    len_needle = len(needle)
    for i in range(len(haystack) - len_needle + 1):
        if (haystack[i:i + len_needle] == needle):
            return (i)
    return (-1)

#Divide Two Integers
#solution:
def divide(self, dividend, divisor):
    """
    :type dividend: int
    :type divisor: int
    :rtype: int
    """
    if not divisor:
        return (-1)
    if not dividend:
        return (0)
#
#    if (dividend == -2147483648 and divisor == -1):
#        return (2147483647)
#
    sign = True
    if ((dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0)):
        sign = False
    aa = a = abs(dividend)
    dd = d = abs(divisor)
    c = 1
    total = 0
    while (True):
        aa -= dd
        if (aa < 0 and c == 1):
            total += 1
            break
        elif (aa < 0):  # c!=1
            c = 1
            aa += dd
            dd = d
        else:  # aa>=0
            total += c
            c += c
            dd += dd
    result = total - 1 if sign else 1 - total
    return (result)

#Substring with Concatenation of All Words
#solution: rob Peter to pay Paul
def findSubstring(self, s, words):
    """
    :type s: str
    :type words: List[str]
    :rtype: List[int]
    """
    if words[0] == "":
        return (list(range(len(s) + 1)))
    step = len(words[0])
    length = len(words) * step
    ls = []
    for start in range(0, step):
        i = j = start
        track = []
        w = list(words)
        while (i <= len(s) - length and j < len(s)):
            this = s[j:j + step]
            if (this in w):
                w.remove(this)
                track.append(this)
                j += step
                if (j - i == length):
                    ls.append(i)
                    i += step
                    w.append(track.pop(0))
            else:
                if (i == j):
                    i += step
                    j += step
                else:
                    i += step
                    w.append(track.pop(0))
    return (ls)

#Next Permutation
#solution:
def nextPermutation(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    j = len(nums) - 1
    i = j - 1
    while (i >= 0):
        if (nums[i] >= nums[i + 1]):
            i -= 1
        else:
            while (nums[i] >= nums[j]):
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
            left, right = i + 1, len(nums) - 1
            while (left < right):
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            return
    nums.sort()
#Warn: need to practice Permutation

#Longest Valid Parentheses
#solution1: stack without DP, O(n)
def longestValidParentheses(self, s):
    """
    :type s: str
    :rtype: int
    """
    if not s:
        return (0)
    stack = []
    for i, v in enumerate(s):
        if (stack and v == ")" and stack[-1][1] == "("):
            stack.pop()
        else:
            stack.append((i, v))
    if not stack:
        return (len(s))
    ma = stack[0][0] + 1
    for i in range(len(stack) - 1):
        diff = stack[i + 1][0] - stack[i][0]
        if (diff > ma):
            ma = diff
    ma = max(ma, len(s) - stack[-1][0])
    return (ma - 1)
#solution2: DP
def longestValidParentheses(self, s):
    """
    :type s: str
    :rtype: int
    """
    if not s:
        return (0)
    stack = []  #stores the longest length of valid parentheses which is end at i.
    for i, v in enumerate(s):
        if (v == '('):
            stack.append(0)
        else:
            if (i == 0):
                stack.append(0)
            elif (s[i - 1] == "("):
                if (i == 1):
                    stack.append(2)
                else:
                    stack.append(stack[i - 2] + 2)
            else:
                length = stack[i - 1]
                index = i - length - 1
                if index < 0:
                    stack.append(0)
                elif (s[index] == "("):
                    if (index - 1 >= 0):
                        stack.append(stack[index - 1] + 2 + length)
                    else:
                        stack.append(length + 2)
                else:
                    stack.append(0)
    return (max(stack))

#Search in Rotated Sorted Array
#solution1:
def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if not nums:
        return (-1)
    if len(nums) == 1:
        return (0 if target == nums[0] else -1)
    i = 0
    j = len(nums) - 1
    while (True):
        k = (i + j) // 2
        if (k == i):
            break
        if (nums[k] == target):
            return (k)
        elif (nums[k] < target):
            if (nums[k] < nums[j]):
                if (target > nums[j]):
                    j = k
                else:
                    i = k
            else:
                i = k
        else:
            if (nums[k] < nums[j]):
                j = k
            else:
                if (target > nums[j]):
                    j = k
                else:
                    i = k
    if (nums[i] == target):
        return (i)
    elif (nums[j] == target):
        return (j)
    return (-1)
#solution2: clarify when to end the loop
def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if not nums:
        return (-1)
    if len(nums) == 1:
        return (0 if target == nums[0] else -1)
    i = 0
    j = len(nums) - 1
    while (i <= j):
        k = (i + j) // 2
        if (nums[k] == target):
            return (k)
        elif (nums[k] < target):
            if (nums[k] < nums[j] < target):
                j = k - 1                           #use this way
            else:
                i = k + 1
        else:
            if (nums[k] >= nums[j] >= target):
                i = k + 1
            else:
                j = k - 1
    return (-1)

#Search for a Range
#solution1:
def searchRange(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    if not nums:
        return ([-1, -1])
    i = 0
    j = len(nums) - 1
    left = right = -1
    while (i < j):
        mid = (i + j) // 2
        if (nums[mid] == target):
            j = mid
        elif (nums[mid] < target):
            i = mid + 1
        else:
            j = mid - 1
    if (nums[i] == target):
        left = i
    i = 0
    j = len(nums) - 1
    while (i < j):
        mid = (i + j + 1) // 2
        if (nums[mid] == target):
            i = mid
        elif (nums[mid] < target):
            i = mid + 1
        else:
            j = mid - 1
    if (nums[j] == target):  # i may be go beyond, j is correct here
        right = j
    return ([left, right])
#solution2: make some slight modification
def searchRange(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    if not nums:
        return ([-1, -1])
    i = 0
    j = len(nums) - 1
    left = right = -1
    while (i < j):
        mid = (i + j) // 2
        if (nums[mid] < target):
            i = mid + 1
        else:
            j = mid
    if (nums[i] == target):
        left = i
    i = 0
    j = len(nums) - 1
    while (i < j):
        mid = (i + j + 1) // 2
        if (nums[mid] <= target):
            i = mid
        else:
            j = mid - 1
    if (nums[j] == target):  # i cannot go beyond this time
        right = j
    return ([left, right])

#Search Insert Position
#solution:
def searchInsert(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    i = 0
    j = len(nums) - 1
    while (i <= j):
        k = (i + j) // 2
        if (nums[k] == target):
            return (k)
        elif (nums[k] < target):
            i = k + 1
        else:
            j = k - 1
    return (i)

#Valid Sudoku
#solution:
def isValidSudoku(self, board):
    """
    :type board: List[List[str]]
    :rtype: bool
    """
    for i in board:
        d = {}
        for s in i:
            if (s != '.'):
                if (d and s in d):
                    return (False)
                d[s] = 1
    for j in range(9):
        d = {}
        ls = [s[j] for s in board]
        for a in ls:
            if (a != "."):
                if (d and a in d):
                    return (False)
                d[a] = 1
    for i in [1, 4, 7]:
        for j in [1, 4, 7]:
            ls = [board[i - 1][j - 1], board[i][j - 1], board[i + 1][j - 1], board[i - 1][j], board[i][j],
                  board[i + 1][j], board[i - 1][j + 1], board[i][j + 1], board[i + 1][j + 1]]
            d = {}
            for a in ls:
                if (a != "."):
                    if (d and a in d):
                        return (False)
                    d[a] = 1
    return (True)

#Sudoku Solver
#solution:
def solveSudoku(self, board):
    """
    :type board: List[List[str]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    def fun(i, j, b):
        if (i == 8 and j == 8):
            p = b[i][j] if b[i][j] != "." else "123456789"
            for n in p:
                if (isok(i, j, n, b)):
                    b[i][j] = n
                    return (b)
            tmp = "." if p == "123456789" else p
            b[i][j] = tmp
            return ([])
        else:
            p = b[i][j] if b[i][j] != "." else "123456789"
            for n in p:
                if (isok(i, j, n, b)):
                    b[i][j] = n
                    if (j < 8):
                        res = fun(i, j + 1, b)
                        if (res):
                            return (res)
                    else:
                        res = fun(i + 1, 0, b)
                        if (res):
                            return (res)
            tmp = "." if p == "123456789" else p
            b[i][j] = tmp
            return ([])
    def isok(i, j, n, b):
        # row
        d = {}
        this = b[i]
        for k in range(0, j):
            s = this[k]
            if (s != '.' and d and s in d):
                return (False)
            d[this[k]] = 1
        if (d and n in d):
            return (False)
        d[n] = 1
        for k in range(j + 1, 9):
            s = this[k]
            if (s != '.' and d and s in d):
                return (False)
            d[this[k]] = 1
        # column
        d = {}
        this = [s[j] for s in b]
        for k in range(0, i):
            s = this[k]
            if (s != '.' and d and s in d):
                return (False)
            d[this[k]] = 1
        if (d and n in d):
            return (False)
        d[n] = 1
        for k in range(i + 1, 9):
            s = this[k]
            if (s != '.' and d and s in d):
                return (False)
            d[this[k]] = 1
        # box
        d = {}
        ci = (i // 3) * 3 + 1
        cj = (j // 3) * 3 + 1
        this = [b[ci - 1][cj - 1], b[ci - 1][cj], b[ci - 1][cj + 1],
                b[ci][cj - 1], b[ci][cj], b[ci][cj + 1],
                b[ci + 1][cj - 1], b[ci + 1][cj], b[ci + 1][cj + 1]]
        bias = (i % 3) * 3 + (j % 3)
        this[bias] = n
        for k in range(9):
            s = this[k]
            if (s != '.' and d and s in d):
                return (False)
            d[this[k]] = 1
        return (True)
    board = fun(0, 0, board)

#Count and Say
#solution:
def countAndSay(self, n):
    """
    :type n: int
    :rtype: str
    """
    s = "1"
    for _ in range(n - 1):
        new = ""
        count = 1
        tmp = s[0]
        for i in range(1, len(s)):
            if (s[i] == tmp):
                count += 1
            else:
                new = new + str(count) + tmp
                tmp = s[i]
                count = 1
        s = new + str(count) + tmp
    return (s)

#Combination Sum
#solution1.1: dp
def combinationSum(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    dp = [[]]
    for i in range(1, target + 1):
        dp.append([])
        for j in candidates:
            if (i - j == 0):
                dp[i].append([j])
            elif (i - j > 0):
                dp[i] += [sorted(x + [j]) for x in dp[i - j] if sorted(x + [j]) not in dp[i]]
    return (dp[target])
#solution1.2:
def combinationSum(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    candidates.sort()
    dp = [[]]
    for i in range(1, target + 1):
        dp.append([])
        for j in candidates:
            if (i - j == 0):
                dp[i].append([j])
            elif (i - j > 0):
                dp[i] += [x + [j] for x in dp[i - j] if j >= x[-1]] # make some modification here, sort first
    return (dp[target])
#solution2: DFS

#Combination Sum II
#solution: Backtracking
def combinationSum2(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    candidates.sort()
    def fun(can, tar, p, res):
        if (tar == 0):
            if (list(p) not in res):
                res.append(list(p))
            return
        else:
            for i in range(len(can)):
                if (tar - can[i] >= 0):
                    p.append(can[i])
                    fun(can[(i + 1):], tar - can[i], p, res)
                    p.pop()
                else:
                    break
    ls = []
    fun(candidates, target, [], ls)
    return (ls)

#First Missing Positive
#solution:
def firstMissingPositive(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i = 0
    while (i < len(nums)):
        if (nums[i] <= 0 or nums[i] > len(nums) or nums[i] == i + 1):
            i += 1
        else:
            tmp = nums[i] - 1
            if (nums[i] == nums[tmp]):
                i += 1
            else:
                nums[i], nums[tmp] = nums[tmp], nums[i]
    for i in range(len(nums)):
        if (nums[i] != i + 1):
            return (i + 1)
    return (len(nums) + 1)

#Trapping Rain Water
#solution1: stack
def trap(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    i = total = 0
    stack = []
    while (i < len(height)):
        if (not stack) and height[i] > 0:
            stack.append((height[i], i))
            i += 1
        elif not stack:
            i += 1
        else:
            tmp_bef = 0
            while (True):
                if stack and height[i] >= stack[-1][0]:
                    tmp = stack.pop()
                    total += (i - tmp[1] - 1) * (tmp[0] - tmp_bef)
                    tmp_bef = tmp[0]
                elif not stack:
                    stack.append((height[i], i))
                    i += 1
                    break
                else:
                    total += (i - stack[-1][1] - 1) * (height[i] - tmp_bef)
                    stack.append((height[i], i))
                    i += 1
                    break
    return (total)
#solution2: two pointers
def trap(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    n = len(height)
    l, r, water, minHeight = 0, n - 1, 0, 0
    while l < r:
        while l < r and height[l] <= minHeight:
            water += minHeight - height[l]
            l += 1
        while r > l and height[r] <= minHeight:
            water += minHeight - height[r]
            r -= 1
        minHeight = min(height[l], height[r])
    return water

#Multiply Strings
#solution1:
def multiply(self, num1, num2):
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    def add(s1, s2):
        if (len(s1) > len(s2)):
            s2 = "0" * (len(s1) - len(s2)) + s2
        else:
            s1 = "0" * (len(s2) - len(s1)) + s1
        res = ""
        last = 0
        i = len(s1) - 1
        while (i >= 0):
            tmp = int(s1[i]) + int(s2[i]) + last
            last, tmp = divmod(tmp, 10)
            res = str(tmp) + res
            i -= 1
        if (last > 0):
            res = str(last) + res
        return (res)
    def multi(s1, s2):  # s2 is a single
        i = len(s1) - 1
        res = ""
        last = 0
        while (i >= 0):
            tmp = int(s1[i]) * int(s2) + last
            last, tmp = divmod(tmp, 10)
            res = str(tmp) + res
            i -= 1
        if (last > 0):
            res = str(last) + res
        return (res)
    ls = []
    i = 0
    length = len(num2)
    while (i < length):
        ls.append(multi(num1, num2[length - 1 - i]) + "0" * i)
        i += 1
    res = ls[0]
    for i in range(1, len(ls)):
        res = add(res, ls[i])
    while (len(res) > 1):
        if (res[0] == '0'):
            res = res[1:]
    return (res)
#solution2:
def multiply(self, num1, num2):
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    ls = [0] * (len(num1) + len(num2))
    for i in range(len(num1) - 1, -1, -1):
        for j in range(len(num2) - 1, -1, -1):
            a, ls[i + j + 1] = divmod(int(num1[i]) * int(num2[j]) + ls[i + j + 1], 10)
            ls[i + j] += a
    i = 0
    while (i < len(ls) - 1):
        if (ls[i] == 0):
            i += 1
        else:
            break
    return ("".join([str(x) for x in ls[i:]]))

#Wildcard Matching
#solution1: DP
def isMatch(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """
    m = len(s)  # i
    n = len(p)  # j
    dp = [[False for x in range(n + 1)] for y in range(m + 1)]
    dp[0][0] = True
    for i in range(1, n + 1):
        if (p[i - 1] == '*'):
            dp[0][i] = True
        else:
            break
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if (p[j - 1] != "*"):
                dp[i][j] = dp[i - 1][j - 1] and (p[j - 1] == '?' or s[i - 1] == p[j - 1])
            else:
                dp[i][j] = dp[i - 1][j - 1] or dp[i - 1][j] or dp[i][j - 1]
    return (dp[m][n])
#solution2: Backtracking
def isMatch(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """
    i = 0
    j = 0
    star = -1
    s_star = 0
    s_len = len(s)
    p_len = len(p)
    while i < s_len:
        if i < s_len and j < p_len and (s[i] == p[j] or p[j] == '?'):
            i += 1
            j += 1
        elif j < p_len and p[j] == '*':
            star = j
            s_star = i
            j += 1
        elif star != -1:
            j = star + 1
            s_star += 1
            i = s_star
        else:
            return False
    while j < p_len and p[j] == '*':
        j += 1
    return j == p_len

#Jump Game II
#solution1: DP
def jump(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    ls = [0] * len(nums)
    tmp = 0
    for i in range(len(nums) - 1):
        m = min(len(nums), i + 1 + nums[i])
        if (m - 1 > tmp):
            for j in range(tmp + 1, m):
                ls[j] = ls[i] + 1
            tmp = m - 1
    return (ls[-1])
#solution2: Greedy
def jump(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n, start, end, step = len(nums), 0, 0, 0
    while end < n - 1:
        step += 1
        maxend = end + 1
        for i in range(start, end + 1):
            if i + nums[i] >= n - 1:
                return step
            maxend = max(maxend, i + nums[i])
        start, end = end + 1, maxend
    return step

#Permutations
#solution:
def permute(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    def fun(ls, path, nums):
        if (len(nums) == 1):
            ls.append(path + nums)
            return
        else:
            for i in range(len(nums)):
                nums[0], nums[i] = nums[i], nums[0]
                fun(ls, path + [nums[0]], nums[1:])
                nums[0], nums[i] = nums[i], nums[0]
    ls = []
    path = []
    fun(ls, path, nums)
    return (ls)

#Permutation II
#solution:
def permuteUnique(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    ls = []
    path = []

    def fun(ls, path, nums):
        if (len(nums) == 1):
            ls.append(path + nums)
            return
        else:
            fun(ls, path + [nums[0]], nums[1:])
            tmp = [nums[0]]
            for i in range(1, len(nums)):
                if (nums[i] not in tmp):
                    nums[i], nums[0] = nums[0], nums[i]
                    tmp.append(nums[0])
                    fun(ls, path + [nums[0]], nums[1:])
                    nums[i], nums[0] = nums[0], nums[i]

    fun(ls, path, nums)
    return (ls)

#Rotate Image
#solution1: common method
def rotate(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    length = len(matrix)
    for n in range(length // 2):
        for i in range(length - 2 * n - 1):
            matrix[n][n + i], matrix[n + i][length - 1 - n], matrix[length - 1 - n][length - 1 - n - i], \
            matrix[length - 1 - n - i][n] = matrix[length - 1 - n - i][n], matrix[n][n + i], matrix[n + i][
                length - 1 - n], matrix[length - 1 - n][length - 1 - n - i]
#solution2: one line solution
def rotate(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    matrix[::] = zip(*matrix[::-1])

#Group Anagrams
#solution1:
def groupAnagrams(self, strs):
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    """
    d = {}
    for s in strs:
        tmp = "".join(sorted(s))
        if (tmp in d):
            d[tmp].append(s)
        else:
            d[tmp] = [s]
    ls = []
    for v in d.values():
        ls.append(v)
    return (ls)
#solution2: with detail improvement
def groupAnagrams(self, strs):
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    """
    d = {}
    for s in strs:
        tmp = tuple(sorted(s))
        if (tmp in d):
            d[tmp].append(s)
        else:
            d[tmp] = [s]
    # return(list(d.values()))
    return ([v for k, v in d.items()])

#Pow(x,n)
#solution1.1: recursion
def myPow(self, x, n):
    """
    :type x: float
    :type n: int
    :rtype: float
    """
    if (n == 0):
        return (1)
    if (n == 1):
        return (x)
    if (n < 0):
        x = 1 / x
        n = -n
    tmp = 1
    total = x
    while (True):
        if (tmp + tmp <= n):
            tmp += tmp
            total = total * total
        else:
            return (total * self.myPow(x, n - tmp))
#solution1.2:
def myPow(self, x, n):
    """
    :type x: float
    :type n: int
    :rtype: float
    """
    if (n == 0):
        return (1)
    if (n < 0):
        return (1 / self.myPow(x, -n))
    if (n % 2):
        return (x * self.myPow(x, n - 1))
    else:
        return (self.myPow(x * x, n / 2))
#solution2: iterative
def myPow(self, x, n):
    """
    :type x: float
    :type n: int
    :rtype: float
    """
    if n < 0:
        x = 1 / x
        n = -n
    pow = 1
    while n:
        if n & 1:
            pow *= x
        x *= x
        n >>= 1
    return pow

#N-Queens
#solution1: stack
def solveNQueens(self, n):
    """
    :type n: int
    :rtype: List[List[str]]
    """
    def isok(i, j, stack, n):
        if (j in [x[1] for x in stack]):
            return (False)
        if (i + j in [x[2] for x in stack]):
            return (False)
        if (i - j in [x[3] for x in stack]):
            return (False)
        return (True)
    ls = []
    stack = []
    i = j = 0
    flag = True
    while (flag):
        if (isok(i, j, stack, n)):
            stack.append((i, j, i + j, i - j))
            i += 1
            j = 0
            if (i == n):
                ls.append(list(stack))
                i, j = stack.pop()[0:2]
                j += 1
                while (j == n):
                    if (stack):
                        i, j = stack.pop()[0:2]
                        j += 1
                    else:
                        flag = False
                        break
        else:
            j += 1
            while (j == n):
                if (stack):
                    i, j = stack.pop()[0:2]
                    j += 1
                else:
                    flag = False
                    break
    return ([["." * i[1] + "Q" + "." * (n - i[1] - 1) for i in sol] for sol in ls])
#solution2: DFS
def solveNQueens(self, n):
    """
    :type n: int
    :rtype: List[List[str]]
    """
    def DFS(queens, xy_dif, xy_sum):
        p = len(queens)
        if p == n:
            result.append(queens)
            return None
        for q in range(n):
            if q not in queens and p - q not in xy_dif and p + q not in xy_sum:
                DFS(queens + [q], xy_dif + [p - q], xy_sum + [p + q])
    result = []
    DFS([], [], [])
    return [["." * i + "Q" + "." * (n - i - 1) for i in sol] for sol in result]

#N-Queens II
#solution:
def totalNQueens(self, n):
    """
    :type n: int
    :rtype: int
    """
    def DFS(queens, xy_dif, xy_sum):
        p = len(queens)
        if p == n:
            result.append(queens)
            return None
        for q in range(n):
            if q not in queens and p - q not in xy_dif and p + q not in xy_sum:
                DFS(queens + [q], xy_dif + [p - q], xy_sum + [p + q])
    result = []
    DFS([], [], [])
    return len(result)

#Maximum Subarray
#solution:
def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i = tmp = 0
    ma = -float("Inf")
    for i in nums:
        tmp += i
        ma = max(ma, tmp)
        tmp = max(tmp, 0)
    return (ma)

#Spiral Matrix
#solution1.1:
def spiralOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    i = 0
    ls = []
    while (matrix and matrix[0]):
        if (i == 0):
            ls += matrix.pop(0)
        elif (i == 1):
            ls += [x.pop() for x in matrix]
        elif (i == 2):
            ls += list(reversed(matrix.pop()))
        else:
            ls += list(reversed([x.pop(0) for x in matrix]))
        i += 1
        if (i == 4):
            i = 0
    return (ls)
#solution1.2:
def spiralOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    if not matrix:
        return ([])
    i = j = 0
    m = len(matrix)
    n = len(matrix[0])
    ls = []
    while (True):
        if (n):
            ls += matrix[i][j:j + n]
            j = j + n - 1
            i += 1
            m = m - 1
        else:
            break
        if (m):
            for i in range(i, i + m):
                ls.append(matrix[i][j])
            j -= 1
            n = n - 1
        else:
            break
        if (n):
            for j in range(j, j - n, -1):
                ls.append(matrix[i][j])
            i -= 1
            m = m - 1
        else:
            break
        if (m):
            for i in range(i, i - m, -1):
                ls.append(matrix[i][j])
            j += 1
            n = n - 1
        else:
            break
    return (ls)

#solution2: one line solution
#Notes: The and operator evaluates whether both of its arguments are tru-ish,
    # but in a slightly surprising way: First it examines its left argument.
    # If it is truish, then it returns its right argument.
    # If the left argument is falsish, then it returns the left argument.
def spiralOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    return matrix and list(matrix.pop(0)) + self.spiralOrder(zip(*matrix)[::-1])

#Jump Game
#solution:
def canJump(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    far = 0
    for i, v in enumerate(nums):
        if (i > far):
            return (False)
        far = max(far, i + v)
    return (True)

#Merge Intervals
#solution1:
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e
def merge(self, intervals):
    """
    :type intervals: List[Interval]
    :rtype: List[Interval]
    """
    if not intervals:
        return ([])
    intervals.sort(key=lambda x: x.start)
    ls = [intervals[0]]
    for v in intervals[1:]:
        if (v.start <= ls[-1].end):
            ls[-1].end = max(ls[-1].end, v.end)
        else:
            ls.append(v)
    return (ls)
#solution2: this is much faster
def merge(self, intervals):
    """
    :type intervals: List[Interval]
    :rtype: List[Interval]
    """
    ls = []
    for v in sorted(intervals, key=lambda v: v.start):
        if (ls and v.start <= ls[-1].end):
            ls[-1].end = max(ls[-1].end, v.end)
        else:
            ls.append(v)
    return (ls)

#Insert Interval
#solution:  binary search
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e
def insert(self, intervals, newInterval):
    """
    :type intervals: List[Interval]
    :type newInterval: Interval
    :rtype: List[Interval]
    """
    if not intervals:
        return ([newInterval])
    left = right = 0
    i = 0
    j = len(intervals) - 1

    if (newInterval.end < intervals[0].start):
        intervals = [newInterval] + intervals
        return (intervals)
    if (intervals[-1].end < newInterval.start):
        intervals.append(newInterval)
        return (intervals)

    while (i <= j):
        mid = (i + j) // 2
        if (intervals[mid].start <= newInterval.start and newInterval.start <= intervals[mid].end):
            left = mid
            break
        elif (intervals[mid].end < newInterval.start and newInterval.start < intervals[mid + 1].start):
            left = mid + 1
            break
        elif (newInterval.start < intervals[mid].start):
            j = mid - 1
        else:
            i = mid + 1
    if (i > j):
        left = i

    i = 0
    j = len(intervals) - 1
    if (newInterval.end >= intervals[-1].start):
        right = j
    else:
        while (i <= j):
            mid = (i + j) // 2
            if (intervals[mid].start <= newInterval.end and newInterval.end < intervals[mid + 1].start):
                right = mid
                break
            elif (newInterval.end < intervals[mid].start):
                j = mid - 1
            else:
                i = mid + 1
        if (i > j):
            right = i

    newIt = Interval(min(newInterval.start, intervals[left].start), max(newInterval.end, intervals[right].end))
    intervals[left:right + 1] = [newIt]
    return (intervals)

#Length of Last Word
#solution:
def lengthOfLastWord(self, s):
    """
    :type s: str
    :rtype: int
    """
    j = len(s) - 1
    while (j >= 0):
        if (s[j] != " "):
            break
        else:
            j -= 1
    length = 0
    while (j >= 0):
        if (s[j] != " "):
            j -= 1
            length += 1
        else:
            break
    return (length)
#solution2: one line solution
def lengthOfLastWord(self, s):
    return len(s.rstrip(' ').split(' ')[-1])

#Spiral Matrix II
#solution1:
def generateMatrix(self, n):
    """
    :type n: int
    :rtype: List[List[int]]
    """
    ls = [[0] * n for _ in range(n)]
    i = j = 0
    a = b = n
    c = 1
    while (True):
        if (a):
            for j in range(j, j + a):
                ls[i][j] = c
                c += 1
            b -= 1
            i += 1
        else:
            break
        if (b):
            for i in range(i, i + b):
                ls[i][j] = c
                c += 1
            a -= 1
            j -= 1
        else:
            break
        if (a):
            for j in range(j, j - a, -1):
                ls[i][j] = c
                c += 1
            b -= 1
            i -= 1
        else:
            break
        if (b):
            for i in range(i, i - b, -1):
                ls[i][j] = c
                c += 1
            a -= 1
            j += 1
        else:
            break
    return (ls)
#solution2: spiral
def generateMatrix(self, n):
    """
    :type n: int
    :rtype: List[List[int]]
    """
    A, lo = [], n * n + 1
    while lo > 1:
        lo, hi = lo - len(A), lo
        A = [range(lo, hi)] + zip(*A[::-1])
    return A

#Permutation Sequence
#solution1: time limit exceeded
def getPermutation(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: str
    """

    def fun(nums, path, k, tmp, res):
        if (len(nums) == 1):
            tmp[0] += 1
            if (tmp[0] == k):
                res.append(path + nums)
            return
        else:
            for i in range(len(nums)):
                nums[0], nums[i] = nums[i], nums[0]
                fun(nums[1:], path + [nums[0]], k, tmp, res)
                # nums[0],nums[i] = nums[i],nums[0]

    nums = list(range(1, 1 + n))
    res = []
    tmp = [0]
    fun(nums, [], k, tmp, res)
    return ("".join([str(x) for x in res[0]]))
#solution2:
def getPermutation(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: str
    """
    import math
    ls = list(range(1, 1 + n))
    res = []
    for n in range(n, 1, -1):
        fac = math.factorial(n - 1)
        index = (k - 1) // fac
        k = (k - 1) % fac + 1
        res.append(ls[index])
        del ls[index]
    res += ls
    return ("".join([str(x) for x in res]))

#Rotate List
#solution1:
def rotateRight(self, head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    if not head:
        return ([])
    length = 1
    p = head
    while (p.next):
        p = p.next
        length += 1
    q = head
    new = head
    k = k % length
    if (length != k and k != 0):
        for _ in range(length - k - 1):
            q = q.next
        new = q.next
        q.next = None
        p.next = head
    return (new)
#solution2: two pointer
def rotateRight(self, head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    if not head:
        return None
    if head.next == None:
        return head
    pointer = head
    length = 1
    while pointer.next:
        pointer = pointer.next
        length += 1
    rotateTimes = k % length
    if k == 0 or rotateTimes == 0:
        return head
    fastPointer = head
    slowPointer = head
    for a in range(rotateTimes):
        fastPointer = fastPointer.next
    while fastPointer.next:
        slowPointer = slowPointer.next
        fastPointer = fastPointer.next
    temp = slowPointer.next
    slowPointer.next = None
    fastPointer.next = head
    head = temp
    return (head)

#Unique Paths
#solution1: DP
def uniquePaths(self, m, n):
    """
    :type m: int
    :type n: int
    :rtype: int
    """
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return (dp[-1][-1])
#solution2: mathematics method
def uniquePaths(self, m, n):
    return math.factorial(m+n-2)/math.factorial(m-1)/math.factorial(n-1)

#Unique Path II
#solution:
def uniquePathsWithObstacles(self, obstacleGrid):
    """
    :type obstacleGrid: List[List[int]]
    :rtype: int
    """
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0] * n for _ in range(m)]
    if obstacleGrid[-1][-1] == 0:
        dp[-1][-1] = 1
    else:
        return (0)
    for i in range(m - 2, -1, -1):
        if (obstacleGrid[i][-1] == 0):
            dp[i][-1] = 1
        else:
            break
    for j in range(n - 2, -1, -1):
        if (obstacleGrid[-1][j] == 0):
            dp[-1][j] = 1
        else:
            break
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            dp[i][j] = 0 if obstacleGrid[i][j] == 1 else dp[i + 1][j] + dp[i][j + 1]
    return (dp[0][0])

#Minimum Path Sum
#solution:
def minPathSum(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    m, n = len(grid), len(grid[0])
    for j in range(n - 2, -1, -1):
        grid[-1][j] += grid[-1][j + 1]
    for i in range(m - 2, -1, -1):
        grid[i][-1] += grid[i + 1][-1]
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            grid[i][j] += min(grid[i + 1][j], grid[i][j + 1])
    return (grid[0][0])

#Valid Number
#solution1:
def isNumber(self, s):
    """
    :type s: str
    :rtype: bool
    """
    s = s.strip()
    if not s:
        return (False)
    dot = e = True
    e_ok = False
    index_e = 0
    for i in range(len(s)):
        if (s[i] in "0123456789"):
            e_ok = True
        elif (s[i] in "+-"):
            if (e):
                if (i != 0):
                    return (False)
            else:
                if (i != index_e + 1):
                    return (False)
            if (i + 1 >= len(s)):
                return (False)
        elif (s[i] == "."):
            if not e:
                return (False)
            else:
                if (dot):
                    tmp1 = i - 1 >= 0 and s[i - 1] in "0123456789"
                    tmp2 = i + 1 < len(s) and s[i + 1] in "0123456789"
                    if not (tmp1 or tmp2):
                        return (False)
                    dot = False
                else:
                    return (False)
        elif (s[i] == "e"):
            if not e_ok:
                return (False)
            if e:
                if (i + 1 >= len(s)):
                    return (False)
                elif (s[i + 1] in "+-"):
                    if (i + 2 >= len(s) or s[i + 2] not in "0123456789"):
                        return (False)
                e = False
                index_e = i
            else:
                return (False)
        else:
            return (False)
    return (True)
#solution2:
def isNumber(self, s):
    """
    :type s: str
    :rtype: bool
    """
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

#Plus One
#solution:
def plusOne(self, digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    i = len(digits) - 1
    while (i >= 0):
        digits[i] += 1
        if (digits[i] == 10):
            digits[i] = 0
            i -= 1
        else:
            return (digits)
    return ([1] + digits)

#Add Binary
#solution:
def addBinary(self, a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    diff_len = len(a) - len(b)
    i = len(a) - 1
    if (diff_len > 0):
        b = '0' * diff_len + b
    else:
        a = '0' * abs(diff_len) + a
        i -= diff_len
    tmp = 0
    res = ""
    while (i >= 0):
        tmp, this = divmod(int(a[i]) + int(b[i]) + tmp, 2)
        res = str(this) + res
        i -= 1
    if (tmp):
        res = '1' + res
    return (res)

#Text Justification
#solution:
def fullJustify(self, words, maxWidth):
    """
    :type words: List[str]
    :type maxWidth: int
    :rtype: List[str]
    """
    ls = []
    i = 0
    while (i < len(words)):
        tmp = [words[i]]
        width = len(words[i])
        while (True):
            if (i + 1 < len(words)):
                if (width + 1 + len(words[i + 1]) <= maxWidth):
                    tmp.append(" " + words[i + 1])
                    width += 1 + len(words[i + 1])
                    i += 1
                else:
                    c = len(tmp)
                    if (c == 1):
                        tmp.append(" " * (maxWidth - width))
                        break
                    a, b = divmod((maxWidth - width), c - 1)
                    for index in range(1, 1 + b):
                        tmp[index] = " " * (a + 1) + tmp[index]
                    for index in range(1 + b, c):
                        tmp[index] = " " * a + tmp[index]
                    break
            else:
                tmp.append(" " * (maxWidth - width))
                break
        ls.append("".join(tmp))
        i += 1
    return (ls)

#Sqrt(x)
#solution:
def mySqrt(self, x):
    """
    :type x: int
    :rtype: int
    """
    if (x <= 1):
        return (x)
    i = 2
    j = x
    while (i <= j):
        mid = (i + j) // 2
        if (mid * mid == x):
            return (mid)
        elif (mid * mid > x):
            j = mid - 1
        else:
            i = mid + 1
    return (j)

#Climbing Stairs
#solution1: Recursion, time limit exceeded
def climbStairs(self, n):
    """
    :type n: int
    :rtype: int
    """
    total = [0]
    def fun(n, total):
        if (n == 0):
            total[0] += 1
            return
        if (n == 1):
            total[0] += 1
            return
        else:
            fun(n - 1, total)
            fun(n - 2, total)
    fun(n, total)
    return (total[0])
#solution2: Dynamic Programming
def climbStairs(self, n):
    """
    :type n: int
    :rtype: int
    """
    tmp1 = tmp2 = 1
    for i in range(n):
        tmp1, tmp2 = tmp2, tmp1 + tmp2
    return (tmp1)

#Simplify Path
#solution:
def simplifyPath(self, path):
    """
    :type path: str
    :rtype: str
    """
    stack = []
    ls = path.split("/")
    for ele in ls:
        if (ele == ".." and stack):
            stack.pop()
        elif (ele not in ["..", ".", ""]):
            stack.append(ele)
    return ("/" + "/".join(stack))

#Edit Distance
#solution:
def minDistance(self, word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    len1 = len(word1)  # j
    len2 = len(word2)  # i
    dp = [[0] * (len1 + 1) for _ in range(len2 + 1)]
    dp[0] = list(range(len1 + 1))
    for i in range(len2 + 1):
        dp[i][0] = i
    for i in range(1, len2 + 1):
        for j in range(1, len1 + 1):
            if (word1[j - 1] == word2[i - 1]):
                dp[i][j] = dp[i - 1][j - 1]  # no need to change
                continue
            tmp1 = dp[i - 1][j - 1] + 1  # replace
            tmp2 = dp[i][j - 1] + 1  # delete new char
            tmp3 = dp[i - 1][j] + 1  # add new char
            dp[i][j] = min(tmp1, tmp2, tmp3)
    return (dp[-1][-1])

#Set Matrix Zeros
#solution1:
def setZeroes(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    m = len(matrix)
    n = len(matrix[0])
    for i in range(m):
        for j in range(n):
            if (matrix[i][j] == 0):
                for k in range(j):
                    if (matrix[i][k] != 0):
                        matrix[i][k] = None
                for k in range(j + 1, n):
                    if (matrix[i][k] != 0):
                        matrix[i][k] = None
                for k in range(i):
                    if (matrix[k][j] != 0):
                        matrix[k][j] = None
                for k in range(i + 1, m):
                    if (matrix[k][j] != 0):
                        matrix[k][j] = None
    for i in range(m):
        for j in range(n):
            if (matrix[i][j] == None):
                matrix[i][j] = 0
#solution2:
def setZeroes(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    m = len(matrix)
    n = len(matrix[0])
    this = col = row = False
    if (matrix[0][0] == 0):
        this = True
    for i in range(m):
        if (matrix[i][0] == 0):
            row = True
            matrix[i][0] = None
    for j in range(n):
        if (matrix[0][j] == 0):
            col = True
            matrix[0][j] = None
    for i in range(1, m):
        for j in range(1, n):
            if (matrix[i][j] == 0):
                matrix[i][0] = None
                matrix[0][j] = None
    for i in range(1, m):
        if (matrix[i][0] == None):
            matrix[i] = [0] * n
    for j in range(1, n):
        if (matrix[0][j] == None):
            for k in range(m):
                matrix[k][j] = 0
    if (this):
        for i in range(m):
            matrix[i][0] = 0
        for j in range(n):
            matrix[0][j] = 0
    if (row):
        for i in range(m):
            matrix[i][0] = 0
    if (col):
        for j in range(n):
            matrix[0][j] = 0

#Search a 2D Matrix
#solution1:
def searchMatrix(self, matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    if not matrix:
        return (False)
    if not matrix[0]:
        return (False)
    i1, i2 = 0, len(matrix) - 1
    while (i1 <= i2):
        mid = (i1 + i2) // 2
        if (matrix[mid][0] <= target and matrix[mid][-1] >= target):
            j1, j2 = 0, len(matrix[mid]) - 1
            while (j1 <= j2):
                m = (j1 + j2) // 2
                if (matrix[mid][m] == target):
                    return (True)
                elif (matrix[mid][m] > target):
                    j2 = m - 1
                else:
                    j1 = m + 1
            return (False)
        elif (matrix[mid][0] > target):
            i2 = mid - 1
        else:
            i1 = mid + 1
    return (False)
#solution2:
def searchMatrix(self, matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    if not matrix or target is None:
        return False
    rows, cols = len(matrix), len(matrix[0])
    low, high = 0, rows * cols - 1
    while low <= high:
        mid = (low + high) / 2
        num = matrix[mid / cols][mid % cols]
        if num == target:
            return True
        elif num < target:
            low = mid + 1
        else:
            high = mid - 1
    return False

#Sort Colors
#solution1:
def sortColors(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    a = b = 0
    for i in nums:
        if (i == 0):
            a += 1
        elif (i == 1):
            b += 1
    for i in range(a):
        nums[i] = 0
    for i in range(a, a + b):
        nums[i] = 1
    for i in range(a + b, len(nums)):
        nums[i] = 2
#solution2:
def sortColors(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    i = j = 0
    for k in xrange(len(nums)):
        v = nums[k]
        nums[k] = 2
        if v < 2:
            nums[j] = 1
            j += 1
        if v == 0:
            nums[i] = 0
            i += 1

#Minimum Window Substring
#solution1.1: time limit exceeded
def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    if (len(t) == 1):
        return t if t in s else ""
    d = {}
    m = float("Inf")
    n = (0, 0)
    for i in t:
        if (i in d):
            d[i].append(-1)
        else:
            d[i] = [-1]
    for i in range(len(s)):
        if (s[i] in d):
            ls = d[s[i]]
            ls[ls.index(min(ls))] = i
        this = sum(d.values(), [])
        if (min(this) >= 0):
            tmp = max(this) - min(this)
            if (tmp < m):
                m = tmp
                n = min(this), max(this)
    this = sum(d.values(), [])
    if (min(this) >= 0):
        return (s[n[0]:n[1] + 1])
    return ("")
#solution1.2:
def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    d = {}
    miss = list(t)
    m = float("Inf")
    n = (0, 0)
    for i in t:
        d[i] = []
    for i in range(len(s)):
        if (s[i] in d):
            if s[i] not in miss and d[s[i]] != []:
                d[s[i]].pop(0)
            elif s[i] in miss:
                miss.remove(s[i])
            d[s[i]].append(i)
        if (miss == []):
            ma = max([x[-1] for x in d.values()])
            mi = min([x[0] for x in d.values()])
            if (ma - mi < m):
                m = ma - mi
                n = mi, ma
    if (miss == []):
        return (s[n[0]:n[1] + 1])
    return ("")
#solution2: more difficult to understand, need to think again
def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    need, missing = collections.Counter(t), len(t)
    i = I = J = 0
    for j, c in enumerate(s, 1):
        missing -= need[c] > 0
        need[c] -= 1
        if not missing:
            while i < j and need[s[i]] < 0:
                need[s[i]] += 1
                i += 1
            if not J or j - i <= J - I:
                I, J = i, j
    return s[I:J]

#Combinations
#solution1.1:
def combine(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    def fun(ls, this, path, k):
        if (k == 0):
            ls.append(path)
        else:
            for i in range(len(this) - k + 1):
                fun(ls, this[i + 1:], path + [this[i]], k - 1)

    ls = []
    this = range(1, n + 1)
    path = []
    fun(ls, this, path, k)
    return (ls)
#solution1.2:   some modification
def combine(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    def fun(ls, this, path, k):
        if (k == 1):
            for i in this:
                ls.append(path + [i])
        else:
            for i in range(len(this) - k + 1):
                fun(ls, this[i + 1:], path + [this[i]], k - 1)
    ls = []
    this = range(1, n + 1)
    path = []
    fun(ls, this, path, k)
    return (ls)

#Subsets:
#solution1:
def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    def fun(ls, this, path, k):
        if (k == 0):
            ls.append(path)
        else:
            for i in range(len(this) - k + 1):
                fun(ls, this[i + 1:], path + [this[i]], k - 1)
    ls = []
    for i in range(len(nums) + 1):
        path = []
        fun(ls, nums, path, i)
    return (ls)
#solution2: so smart!
def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    result = [[]]
    for num in nums:
        result += [i + [num] for i in result]
    return result

#Word Search
#solution:
def exist(self, board, word):
    """
    :type board: List[List[str]]
    :type word: str
    :rtype: bool
    """
    def fun(i, j, board, word):
        if (board[i][j] != word[0]):
            return (False)
        else:
            if (len(word) == 1):
                return (True)
            tmp = board[i][j]
            board[i][j] = ""
            if (i - 1 >= 0):
                if (fun(i - 1, j, board, word[1:])):
                    board[i][j] = tmp
                    return (True)
            if (i + 1 < len(board)):
                if (fun(i + 1, j, board, word[1:])):
                    board[i][j] = tmp
                    return (True)
            if (j - 1 >= 0):
                if (fun(i, j - 1, board, word[1:])):
                    board[i][j] = tmp
                    return (True)
            if (j + 1 < len(board[0])):
                if (fun(i, j + 1, board, word[1:])):
                    board[i][j] = tmp
                    return (True)
            board[i][j] = tmp
            return (False)
    for i in range(len(board)):
        for j in range(len(board[0])):
            if (fun(i, j, board, list(word))):
                return (True)
    return (False)

#Remove Duplicates from Sorted Array II
#solution1:
def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return (0)
    i = 1
    count = 1
    last = nums[0]
    while (i < len(nums)):
        if (nums[i] == last):
            if (count == 2):
                del nums[i]
            else:
                i += 1
                count += 1
        else:
            last = nums[i]  #I forgot it!
            count = 1
            i += 1
    return (len(nums))
#solution2: So smart!
def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i = 0
    for n in nums:
        if i < 2 or n > nums[i - 2]:
            nums[i] = n
            i += 1
    return i

#Search in Rotated Sorted Array II
#solution1:
def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: bool
    """
    i, j = 0, len(nums) - 1
    while (i <= j):
        mid = (i + j) // 2
        if (nums[mid] == target):
            return (True)
        if (nums[i] == nums[j]):
            if (nums[i] == target):
                return (True)
            else:
                i += 1
                j -= 1
        elif (nums[i] < nums[j]):               #
            if (nums[mid] > target):            #could
                j = mid - 1                     #be
            else:                               #deleted
                i = mid + 1                     #
        else:
            if (nums[mid] >= nums[i]):
                if (nums[i] <= target and target < nums[mid]):
                    j = mid - 1
                else:
                    i = mid + 1
            else:
                if (target < nums[mid] or target >= nums[i]):
                    j = mid - 1
                else:
                    i = mid + 1
    return (False)

#Remove Duplicates from Sorted List II
#solution:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def deleteDuplicates(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return ([])
    begin = ListNode(0)
    begin.next = head
    this = begin
    i = head
    while (i and i.next):
        if (i.next.val == i.val):
            j = i.next
            while (j and j.val == i.val):
                j = j.next
            this.next = j
            i = this.next
        else:
            this = i
            i = i.next
    return (begin.next)

#Remove Duplicates from Sorted List
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#solution:
def deleteDuplicates(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return ([])
    i = head
    while (i.next):
        if (i.next.val == i.val):
            i.next = i.next.next
        else:
            i = i.next
    return (head)

#Largest Rectangle in Histogram
#solution1: time limit exceeded
def largestRectangleArea(self, heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    if not heights:
        return (0)
    length = len(heights)
    ma = max(heights)
    for i in range(length):
        mi = heights[i]
        for j in range(i + 1, length):
            mi = min(mi, heights[j])
            ma = max(ma, mi * (j - i + 1))
    return (ma)
#solution2:
def largestRectangleArea(self, heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    heights.append(0)
    ma = 0
    stack = [-1]  # smart trick
    for i in range(len(heights)):
        while (heights[i] < heights[stack[-1]]):
            h = heights[stack.pop()]
            w = i - 1 - stack[-1]
            ma = max(ma, h * w)
        stack.append(i)
    return (ma)

#Maximal Rectangle
#solution:  reduced from last problem, very smart! I need to review the last problem te prepare for interview.
def maximalRectangle(self, matrix):
    """
    :type matrix: List[List[str]]
    :rtype: int
    """
    if not matrix or not matrix[0]:
        return 0
    n = len(matrix[0])
    height = [0] * (n + 1)
    ans = 0
    for row in matrix:
        for i in xrange(n):
            height[i] = height[i] + 1 if row[i] == '1' else 0
        stack = [-1]
        for i in xrange(n + 1):
            while height[i] < height[stack[-1]]:
                h = height[stack.pop()]
                w = i - 1 - stack[-1]
                ans = max(ans, h * w)
            stack.append(i)
    return ans

#Partition List
#solution:
def partition(self, head, x):
    """
    :type head: ListNode
    :type x: int
    :rtype: ListNode
    """
    start = j = ListNode(0)  # trick
    j.next = head
    i = head
    while (i):
        if (i.val >= x):
            while (i.next):
                if (i.next.val < x):
                    tmp = j.next
                    j.next = i.next
                    i.next = i.next.next
                    j = j.next
                    j.next = tmp
                else:
                    i = i.next
            return (start.next)
        else:
            i = i.next
            j = j.next
    return (start.next)

#Scramble String
#solution1.1: time limit exceeded
def isScramble(self, s1, s2):
    """
    :type s1: str
    :type s2: str
    :rtype: bool
    """
    def fun(s, t):
        if (s == t):
            return (True)
        for i in range(1, len(s)):
            left = fun(s[0:i], t[0:i])
            right = fun(s[i:], t[i:])
            if (left and right):
                return (True)
            left = fun(s[0:i], t[len(t) - i:len(t)])
            right = fun(s[i:], t[0:len(t) - i])
            if (left and right):
                return (True)
        return (False)
    return (fun(s1, s2))
#solution1.2:
def isScramble(self, s1, s2):
    """
    :type s1: str
    :type s2: str
    :rtype: bool
    """
    def fun(s, t):
        n, m = len(s), len(t)
        if n != m or sorted(s) != sorted(t):    #only slight modification
            return False
        if n < 4 or s == t:
            return True
        for i in range(1, n):
            if (fun(s[0:i], t[0:i]) and fun(s[i:], t[i:])):
                return (True)
            if (fun(s[0:i], t[len(t) - i:len(t)]) and fun(s[i:], t[0:len(t) - i])):
                return (True)
        return (False)
    return (fun(s1, s2))

#Merge Sorted Array
#solution:
def merge(self, nums1, m, nums2, n):
    """
    :type nums1: List[int]
    :type m: int
    :type nums2: List[int]
    :type n: int
    :rtype: void Do not return anything, modify nums1 in-place instead.
    """
    while (m > 0 and n > 0):
        if (nums1[m - 1] > nums2[n - 1]):
            nums1[n + m - 1] = nums1[m - 1]
            m -= 1
        else:
            nums1[n + m - 1] = nums2[n - 1]
            n -= 1
    if (n > 0):
        nums1[:n] = nums2[:n]
    return

#Gray Code
#solution1: not in correct order
def grayCode(self, n):
    """
    :type n: int
    :rtype: List[int]
    """
    def fun(s, i, ls):
        ls.append(int(s, 2))
        if (i < 0):
            return
        for j in range(i, -1, -1):
            fun(s[0:j] + "1" + s[j + 1:], j - 1, ls)
    if (n == 0):
        return ([0])
    ls = []
    fun("0" * n, n - 1, ls)
    return (ls)
#solution2: one-line solution
def grayCode(self, n):
    """
    :type n: int
    :rtype: List[int]
    """
    return [(i >> 1) ^ i for i in xrange(2 ** n)]

#Subsets II
#solution:
def subsetsWithDup(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    nums.sort()
    result = [[]]
    for num in nums:
        result += [i + [num] for i in result if i + [num] not in result]
    return result

#Decode Ways
#solution:
def numDecodings(self, s):
    """
    :type s: str
    :rtype: int
    """
    if not s:
        return (0)
    length = len(s)
    dp = [0] * length
    if (s[0] != "0"):
        dp[0] = 1
    else:
        return (0)
    if (length == 1):
        return (dp[0])
    if (s[0:2] in ["10", "20"]):
        dp[1] = 1
    elif (s[1] == "0"):
        return (0)
    elif (int(s[0:2]) > 26):
        dp[1] = 1
    else:
        dp[1] = 2
    for i in range(2, length):
        if (s[i] == "0"):
            if (s[i - 1] not in "12"):
                return (0)
            else:
                dp[i] = dp[i - 1] = dp[i - 2]
        else:
            if (int(s[i - 1:i + 1]) < 10 or int(s[i - 1:i + 1]) > 26):
                dp[i] = dp[i - 1]
            else:
                dp[i] = dp[i - 1] + dp[i - 2]
    return (dp[-1])
#solution2: neat code!
def numDecodings(self, s):
    """
    :type s: str
    :rtype: int
    """
    if not s:
        return 0
    dp = [1] + [0] * len(s) #smart trick!
    for i in range(1, len(s) + 1):
        if s[i - 1] != "0":
            dp[i] += dp[i - 1]
        if i != 1 and s[i - 2:i] < "27" and s[i - 2:i] > "09":  # "01"ways = 0
            dp[i] += dp[i - 2]
    return dp[-1]

#Reverse Linked List II
#solution:
def reverseBetween(self, head, m, n):
    """
    :type head: ListNode
    :type m: int
    :type n: int
    :rtype: ListNode
    """
    i = start = ListNode(0)
    start.next = head       #count = n-m
    m -= 1
    n -= m
    while (m > 0):
        i = i.next
        m -= 1
    j = i
    while (n > 0):
        j = j.next
        n -= 1
    while (i.next != j):    #we can use count to control the loop
        tmp = i.next
        i.next = tmp.next
        tmp.next = j.next
        j.next = tmp
    return (start.next)

#Restore IP Address
#solution:
def restoreIpAddresses(self, s):
    """
    :type s: str
    :rtype: List[str]
    """
    def fun(slots, path, s, ls):
        if (len(s) < slots or len(s) > 3 * slots):
            return
        if (slots == 1):
            if (s[0] == "0" and len(s) > 1):
                return
            if (len(s) < 3 or s <= "255"):
                ls.append(path[1:] + "." + s)
        else:
            if (s[0] == "0"):
                fun(slots - 1, path + "." + s[0:1], s[1:], ls)
            else:
                for i in range(min(3, len(s))):
                    if (i < 2 or s[0:i + 1] <= "255"):
                        fun(slots - 1, path + "." + s[0:i + 1], s[i + 1:], ls)
    ls = []
    fun(4, "", s, ls)
    return (ls)

#Binary Tree Inorder Traversal
#solution1:  recursion
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def inorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    def fun(root, ls):
        if not root:
            return
        fun(root.left, ls)
        ls.append(root.val)
        fun(root.right, ls)
    ls = []
    fun(root, ls)
    return (ls)
#solution2: iteration
def inorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    res, stack = [], []
    while True:
        while root:
            stack.append(root)
            root = root.left
        if not stack:
            return res
        node = stack.pop()
        res.append(node.val)
        root = node.right

#Unique Binary Search Trees II
#solution: recursion
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def generateTrees(self, n):
    """
    :type n: int
    :rtype: List[TreeNode]
    """
    def fun(ls):
        trees = []
        for i in range(len(ls)):
            for leftSub in fun(ls[:i]):
                for rightSub in fun(ls[i + 1:]):
                    root = TreeNode(ls[i])
                    root.left = leftSub
                    root.right = rightSub
                    trees.append(root)
        return (trees or [None])
    if not n:
        return ([])
    return (fun(list(range(1, n + 1))))

#Unique Binary Search Trees
#solution1: recursion, time limit exceeded
def numTrees(self, n):
    """
    :type n: int
    :rtype: int
    """
    def fun(n):
        if (n <= 1):
            return (1)
        result = 0
        for i in range(n):
            l = fun(i)
            r = fun(n - i - 1)
            result += l * r
        return (result)
    return (fun(n))
#solution2: DP
def numTrees(self, n):
    """
    :type n: int
    :rtype: int
    """
    dp = [1] * (n + 1)
    for i in range(2, n + 1):
        result = 0
        for j in range(i):
            result += dp[j] * dp[i - j - 1]
        dp[i] = result
    return (dp[-1])
#solution3: Catalan Number
def numTrees(self, n):
    """
    :type n: int
    :rtype: int
    """
    return math.factorial(2 * n) / (math.factorial(n) * math.factorial(n + 1))

#Interleaving String
#solution: DP
def isInterleave(self, s1, s2, s3):
    """
    :type s1: str
    :type s2: str
    :type s3: str
    :rtype: bool
    """
    if (len(s1) + len(s2) != len(s3)):
        return (False)
    len1, len2 = len(s1) + 1, len(s2) + 1
    dp = [[False] * (len1) for _ in range(len2)]
    dp[0][0] = True
    for i in range(len1 - 1):
        if (s1[i] == s3[i]):
            dp[0][i + 1] = True
        else:
            break
    for j in range(len2 - 1):
        if (s2[j] == s3[j]):
            dp[j + 1][0] = True
        else:
            break
    for i in range(1, len2):
        for j in range(1, len1):
            if (dp[i - 1][j] and s2[i - 1] == s3[i + j - 1]):
                dp[i][j] = True
                continue
            if (dp[i][j - 1] and s1[j - 1] == s3[i + j - 1]):
                dp[i][j] = True
                continue
            dp[i][j] = False
    return (dp[-1][-1])

#Validate Binary Search Tree
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def isValidBST(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """

    def fun(root, l, u):
        this = root.val > l and root.val < u
        if not this:
            return (False)
        if (root.left):
            if not (fun(root.left, l, root.val) and root.left.val < root.val):
                return (False)
        if (root.right):
            if not (fun(root.right, root.val, u) and root.right.val > root.val):
                return (False)
        return (True)

    if not root:
        return (True)
    return (fun(root, -float("Inf"), float("Inf")))

#Same Tree
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def isSameTree(self, p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    def fun(p, q):
        if (not p and not q):
            return (True)
        if (not p or not q):
            return (False)
        ret = p.val == q.val
        ret = ret and fun(p.left, q.left)
        ret = ret and fun(p.right, q.right)
        return (ret)
    return (fun(p, q))

#Symmetric Tree
#solution1.1:   naive recursion
def isSymmetric(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    def fun1(root, ls):
        if (not root):
            ls.append(None)
            return
        ls.append(root.val)
        fun1(root.left, ls)
        fun1(root.right, ls)
    def fun2(root, ls):
        if (not root):
            ls.append(None)
            return
        ls.append(root.val)
        fun2(root.right, ls)
        fun2(root.left, ls)
    if not root:
        return (True)
    ls1 = []
    ls2 = []
    fun1(root.left, ls1)
    fun2(root.right, ls2)
    return (ls1 == ls2)
#solution1.2:
def isSymmetric(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    def fun(r1, r2):
        if (not r1 and not r2):
            return (True)
        if (not r1 or not r2):
            return (False)
        if (r1.val == r2.val):
            if not fun(r1.left, r2.right):
                return (False)
            if not fun(r1.right, r2.left):
                return (False)
            return True
        else:
            return False
    if not root:
        return (True)
    return (fun(root.left, root.right))
#solution2: iterative solution, with stack
def isSymmetric(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    if not root:
        return (True)
    ls = [[root.left, root.right]]
    while ls:
        left, right = ls.pop(0)
        if (not left and not right):
            continue
        if (not left or not right):
            return (False)
        if (left.val == right.val):
            ls.insert(0, [left.left, right.right])
            ls.insert(0, [left.right, right.left])
        else:
            return (False)
    return (True)

#Binary Tree Level Order Traversal
#solution:
def levelOrder(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    if not root:
        return ([])
    ls = [[root.val]]
    q1 = [root]
    q2 = []
    while (True):
        while (q1):
            tmp = q1.pop(0)
            q2.append(tmp.left) if tmp.left else 0
            q2.append(tmp.right) if tmp.right else 0
        if not q2:
            break
        ls.append([x.val for x in q2])
        q1 = q2
        q2 = []
    return (ls)

#Binary Tree Zigzag Level Order Traversal
#solution:
def zigzagLevelOrder(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    if not root:
        return ([])
    ls = [[root.val]]
    q1 = [root]
    q2 = []
    counter = 0
    while (True):
        while (q1):
            tmp = q1.pop(0)
            q2.append(tmp.left) if tmp.left else 0
            q2.append(tmp.right) if tmp.right else 0
        if not q2:
            break
        if (counter == 1):
            ls.append([x.val for x in q2])
            counter -= 1
        else:
            ls.append(list(reversed([x.val for x in q2])))
            counter += 1
        q1 = q2
        q2 = []
    return (ls)

#Maximum Depth of Binary Tree
#solution:
def maxDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root, depth):
        if not root:
            return (depth)
        depth += 1
        return (max(fun(root.left, depth), fun(root.right, depth)))
    return (fun(root, 0))

#Binary Tree Level Order Traversal II
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def levelOrderBottom(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    if not root:
        return ([])
    ls = [[root.val]]
    queue = [root]
    q = []
    while (True):
        while (queue):
            tmp = queue.pop(0)
            if tmp.left:
                q.append(tmp.left)
            if tmp.right:
                q.append(tmp.right)
        if not q:
            return (ls)
        queue = q
        ls = [[x.val for x in q]] + ls
        q = []
    return (ls)

#Convert Sorted Array to Binary Search Tree
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#solution1.1:
def sortedArrayToBST(self, nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    def fun(i, j, root):
        mid = (i + j) // 2
        root.val = nums[mid]
        if (i < mid):
            root.left = TreeNode(0)
            fun(i, mid - 1, root.left)
        if (mid < j):
            root.right = TreeNode(0)
            fun(mid + 1, j, root.right)
    if not nums:
        return ([])
    root = TreeNode(0)
    fun(0, len(nums) - 1, root)
    return (root)
#solution1.2:   Let the function return a TreeNode
def sortedArrayToBST(self, nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    def fun(i, j):
        mid = (i + j) // 2
        root = TreeNode(nums[mid])
        if (i < mid):
            root.left = fun(i, mid - 1)
        if (mid < j):
            root.right = fun(mid + 1, j)
        return (root)
    if not nums:
        return ([])
    return (fun(0, len(nums) - 1))

#Balanced Binary Tree
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def isBalanced(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    def fun(root):
        if not root:
            return (True, 0)
        tmp1 = fun(root.left)
        if not tmp1[0]:
            return ((False, 0))
        tmp2 = fun(root.right)
        if not tmp2[0]:
            return ((False, 0))
        return (abs(tmp1[1] - tmp2[1]) <= 1, max(tmp1[1], tmp2[1]) + 1)
    return (fun(root)[0])

#Minimum Depth of Binary Tree
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#solution1.1:
def minDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root):
        if not root:
            return (float("Inf"))
        if root.left or root.right:
            return (min(fun(root.left), fun(root.right)) + 1)
        else:
            return (1)
    if not root:
        return (0)
    return (fun(root))
#solution1.2:
def minDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root):
        if not root:
            return (0)
        if root.left and root.right:
            return (min(fun(root.left), fun(root.right)) + 1)
        else:
            return (fun(root.left) + fun(root.right) + 1)
    if not root:
        return (0)
    return (fun(root))

#Path Sum
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def hasPathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    def fun(root, target):
        if not root.left and not root.right:
            return (root.val == target)
        if (root.left):
            if (fun(root.left, target - root.val)):
                return (True)
        if (root.right):
            if (fun(root.right, target - root.val)):
                return (True)
        return (False)
    if not root:
        return (False)
    return (fun(root, sum))

#Path Sum II
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def pathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """
    def fun(root, path, ls, target):
        if not root.left and not root.right:
            if (root.val == target):
                ls.append(path + [target])
        if root.left:
            fun(root.left, path + [root.val], ls, target - root.val)
        if root.right:
            fun(root.right, path + [root.val], ls, target - root.val)
    if not root:
        return ([])
    ls = []
    fun(root, [], ls, sum)
    return (ls)

#Convert Sorted List to Binary Search Tree
#solution:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def sortedListToBST(self, head):
    """
    :type head: ListNode
    :rtype: TreeNode
    """
    def fun(i, j):
        if (i > j):
            return None
        else:
            mid = (i + j) // 2
            root = TreeNode(nums[mid])
            root.left = fun(i, mid - 1)
            root.right = fun(mid + 1, j)
            return (root)
    nums = []
    while (head):
        nums.append(head.val)
        head = head.next
    return (fun(0, len(nums) - 1))

#Construct Binary Tree from Preorder and Inorder Traversal
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def buildTree(self, preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    def fun(preorder, inorder):
        if not inorder:
            return
        tmp = preorder.pop(0)
        root = TreeNode(tmp)
        index = inorder.index(tmp)
        root.left = fun(preorder, inorder[:index])
        root.right = fun(preorder, inorder[index + 1:])
        return (root)
    return (fun(preorder, inorder))

#Construct Binary Tree from Inorder and Postorder Traversal
#solution:
def buildTree(self, inorder, postorder):
    """
    :type inorder: List[int]
    :type postorder: List[int]
    :rtype: TreeNode
    """
    def fun(inorder, postorder):
        if not inorder:
            return
        tmp = postorder.pop()
        root = TreeNode(tmp)
        index = inorder.index(tmp)
        root.right = fun(inorder[index + 1:], postorder)
        root.left = fun(inorder[:index], postorder)
        return (root)
    return (fun(inorder, postorder))

#Flatten Binary Tree to Linked List
#solution:
def flatten(self, root):
    """
    :type root: TreeNode
    :rtype: void Do not return anything, modify root in-place instead.
    """
    stack = []
    p = root
    while (p):
        if not p.left and not p.right:
            if stack:
                p.right = stack.pop()
                p = p.right
                continue
            else:
                return
        if p.left:
            if p.right:
                stack.append(p.right)
            p.right = p.left
            p.left = None
            p = p.right
        else:
            p = p.right

#Distinct Subsequences
#solution:
def numDistinct(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: int
    """
    lent = len(t)
    if not lent:
        return (1)
    lens = len(s)
    if lens < lent:
        return (0)
    dp = [[0] * lens for _ in range(lent)]
    for i in range(lent):
        if s[i] == t[i]:
            dp[i][i] = 1
        else:
            break
    tmp = t[0]
    for j in range(1, lens):
        dp[0][j] = dp[0][j - 1] + 1 if s[j] == tmp else dp[0][j - 1]
    for i in range(1, lent):
        for j in range(i + 1, lens):
            dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] if t[i] == s[j] else dp[i][j - 1]
    return (dp[-1][-1])

#Populating Next Right Pointers in Each Node
#solution:
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
def connect(self, root):
    if not root:
        return
    ls1 = [root]
    while (True):
        if (ls1 and ls1[0]):
            ls2 = []
            tmp = ls1.pop(0)
            ls2.append(tmp.left)
            ls2.append(tmp.right)
            while (ls1):
                tmp.next = ls1.pop(0)
                tmp = tmp.next
                ls2.append(tmp.left)
                ls2.append(tmp.right)
            tmp.next = None
            ls1 = ls2
        else:
            return

#Populating Next Right Pointers in Each Node II
#solution1: space is not constant
def connect(self, root):
    if not root:
        return
    ls1 = [root]
    while (True):
        if (ls1):
            ls2 = []
            tmp = ls1.pop(0)
            ls2.append(tmp.left)
            ls2.append(tmp.right)
            while (ls1):
                tmp.next = ls1.pop(0)
                tmp = tmp.next
                ls2.append(tmp.left)
                ls2.append(tmp.right)
            tmp.next = None
            ls1 = [x for x in ls2 if x]
        else:
            return
#solution2: constant space
def connect(self, root):
    tail = dummy = TreeLinkNode(0)
    while root:
        tail.next = root.left
        if tail.next:
            tail = tail.next
        tail.next = root.right
        if tail.next:
            tail = tail.next
        root = root.next
        if not root:
            tail = dummy
            root = dummy.next

#Pascal's Triangle
#solution1:
def generate(self, numRows):
    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    if not numRows:
        return ([])
    if numRows == 1:
        return ([[1]])
    ls = [[1]]
    for i in range(1, numRows):
        tmp = [1]
        for j in range(len(ls[i - 1]) - 1):
            tmp.append(ls[i - 1][j] + ls[i - 1][j + 1])
        tmp.append(1)
        ls.append(tmp)
    return (ls)
#solution2:
def generate(self, numRows):
    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    if not numRows:
        return ([])
    ls = [[1]]
    for i in range(1, numRows):
        tmp = map(lambda x, y: x + y, [0] + ls[-1], ls[-1] + [0])
        ls.append(tmp)
    return (ls)

#Pascal's Triangle II
#solution1:  map
def getRow(self, rowIndex):
    """
    :type rowIndex: int
    :rtype: List[int]
    """
    ls = [1]
    for i in range(rowIndex):
        ls = map(lambda x, y: x + y, [0] + ls, ls + [0])
    return (ls)
#solution2: zip
def getRow(self, rowIndex):
    """
    :type rowIndex: int
    :rtype: List[int]
    """
    ls = [1]
    for i in range(rowIndex):
        ls = [x + y for x, y in zip([0] + ls, ls + [0])]
    return (ls)

#Triangle
#solution1: time limit exceeded
def minimumTotal(self, triangle):
    """
    :type triangle: List[List[int]]
    :rtype: int
    """
    def fun(i, j, s, triangle, ls):
        if (i == len(triangle) - 1):
            ls.append(s + min(triangle[i][j], triangle[i][j + 1]))
            return
        fun(i + 1, j, s + triangle[i][j], triangle, ls)
        fun(i + 1, j + 1, s + triangle[i][j + 1], triangle, ls)
    if not triangle:
        return 0
    if len(triangle) == 1:
        return triangle[0][0]
    ls = []
    fun(1, 0, triangle[0][0], triangle, ls)
    return (min(ls))
#solution2:
def minimumTotal(self, triangle):
    """
    :type triangle: List[List[int]]
    :rtype: int
    """
    if not triangle:
        return 0
    if len(triangle) == 1:
        return triangle[0][0]
    ls = triangle[0]
    ma = float("Inf")
    for i in range(1, len(triangle)):
        tmp1 = map(lambda x, y: x + y, [ma] + ls, triangle[i])
        tmp2 = map(lambda x, y: x + y, ls + [ma], triangle[i])
        ls = map(lambda x, y: min(x, y), tmp1, tmp2)
    return (min(ls))

#Best Time to Buy and Sell Stock
#solution1: time limited exceeded
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if len(prices) <= 1:
        return (0)
    ma = 0
    for i in range(1, len(prices)):
        ma = max(max(map(lambda x, y: x - y, prices[i:], prices[:-i])), ma)
    return (ma)
#solution2.1: DP
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
        return (0)
    ma = 0
    mi = prices[0]
    for i in prices:
        ma = max(ma, i - mi)
        mi = min(mi, i)
    return (ma)
#solution2.2:   more quick
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
        return (0)
    ma = 0
    mi = prices[0]
    for i in prices:
        if i > mi:
            if ma < i - mi:
                ma = i - mi
        else:
            mi = i
    return (ma)

#Best Time to Buy and Sell Stock II
#solution:
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    return (sum(z for z in map(lambda x, y: x - y, prices[1:], prices[:-1]) if z > 0))

#Best Time to Buy and Sell Stock III
#solution1: time limit exceeded
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
        return (0)
    sum1 = []
    for index in range(len(prices)):
        first = prices[:index]
        if not first:
            sum1.append(0)
            continue
        ma = 0
        mi = first[0]
        for i in first:
            if i > mi:
                if ma < i - mi:
                    ma = i - mi
            else:
                mi = i
        sum1.append(ma)
    sum2 = []
    for index in range(len(prices)):
        second = prices[index:]
        if not second:
            sum2.append(0)
            continue
        ma = 0
        mi = second[0]
        for i in second:
            if i > mi:
                if ma < i - mi:
                    ma = i - mi
            else:
                mi = i
        sum2.append(ma)
    return (max([x + y for x, y in zip(sum1, sum2)]))
#solution2: two ways
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices:
        return (0)
    ma = 0
    mi = prices[0]
    ls1 = []
    for i in prices:
        if i > mi:
            if ma < i - mi:
                ma = i - mi
        else:
            mi = i
        ls1.append(ma)
    ma = 0
    mi = prices[-1]
    ls2 = []
    for i in prices[::-1]:
        if i < mi:
            if ma < mi - i:
                ma = mi - i
        else:
            mi = i
        ls2.append(ma)
    return (max([x + y for x, y in zip(ls1, ls2[::-1])]))

#Binary Tree Maximum Path Sum
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def maxPathSum(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root, ma):
        if not root.left and not root.right:
            ma[0] = max(ma[0], root.val)
            return (root.val)
        left = right = 0
        if root.left:
            left = max(0, fun(root.left, ma))
        if root.right:
            right = max(0, fun(root.right, ma))
        ma[0] = max(ma[0], root.val + left + right)
        return (root.val + max(left, right))
    if not root:
        return (0)
    ma = [root.val]
    fun(root, ma)
    return (ma[0])

#Valid Palindrome
#solution:
def isPalindrome(self, s):
    """
    :type s: str
    :rtype: bool
    """
    i, j = 0, len(s) - 1
    while (i <= j):
        if not s[i].isalnum():
            i += 1
            continue
        if not s[j].isalnum():
            j -= 1
            continue
        if (s[i].lower() != s[j].lower()):
            return (False)
        i += 1
        j -= 1
    return (True)

#Recover Binary Search Tree
#solution1: O(n) space
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def recoverTree(self, root):
    """
    :type root: TreeNode
    :rtype: void Do not return anything, modify root in-place instead.
    """
    def fun1(root, ls):
        if not root:
            return
        fun1(root.left, ls)
        ls.append(root.val)
        fun1(root.right, ls)
    def fun2(root, ls):
        if not root:
            return
        fun2(root.left, ls)
        root.val = ls.pop(0)
        fun2(root.right, ls)
    ls = []
    fun1(root, ls)
    ls.sort()
    fun2(root, ls)

#Longest Consecutive Sequence
#solution:
def longestConsecutive(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums = set(nums)        #retrieve element from set --- O(1)
    ma = 0
    while (nums):
        n = nums.pop()
        low = up = 0
        i = n + 1
        while (i in nums):
            nums.remove(i)  #necessary!
            i += 1
            up += 1
        i = n - 1
        while (i in nums):
            nums.remove(i)  #necessary!
            i -= 1
            low += 1
        ma = max(ma, low + 1 + up)
    return (ma)

#Sum Root to Leaf Numbers
#solution1.1:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def sumNumbers(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root, num):
        if not root.left and not root.right:
            return (num * 10 + root.val)
        s = 0
        if root.left:
            s += fun(root.left, num * 10 + root.val)
        if root.right:
            s += fun(root.right, num * 10 + root.val)
        return (s)
    if not root:
        return (0)
    return (fun(root, 0))
#solution1.2:   a little faster
def sumNumbers(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root, num):
        if not root:
            return (0)
        if not root.left and not root.right:
            return (num * 10 + root.val)
        return (fun(root.left, num * 10 + root.val) + fun(root.right, num * 10 + root.val))
    return (fun(root, 0))

#Surrounded Regions
#solution:  unknown error
def solve(self, board):
    """
    :type board: List[List[str]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    def fun(i, j, board):
        if (board[i][j] == "O"):
            if (i == 0 or j == 0 or i == len(board) - 1 or j == len(board[0]) - 1):
                return (False)
            board[i][j] = "X"
            tmp = fun(i + 1, j, board) and fun(i - 1, j, board) and fun(i, j + 1, board) and fun(i, j - 1, board)
            if not tmp:
                board[i][j] = "O"
            return (tmp)
        else:
            return (True)
    nrow = len(board)
    if not nrow:
        return
    ncol = len(board[0])
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (board[i][j] == "O"):
                fun(i, j, board)
    return

#Palindrome Partitioning
#solution:
def partition(self, s):
    """
    :type s: str
    :rtype: List[List[str]]
    """
    def isPalin(s):
        i, j = 0, len(s) - 1
        while (i <= j):
            if (s[i] != s[j]):
                return (False)
            i += 1
            j -= 1
        return (True)
    def fun(path, s, ls):
        if not s:
            ls.append(path)
            return
        for i in range(1, len(s) + 1):
            if (isPalin(s[:i])):
                fun(path + [s[:i]], s[i:], ls)
    ls = []
    fun([], s, ls)
    return (ls)

#Palindrome Partitioning II
#solution1: time limit exceeded
def minCut(self, s):
    """
    :type s: str
    :rtype: int
    """
    def isPalin(s):
        i, j = 0, len(s) - 1
        while (i <= j):
            if (s[i] != s[j]):
                return (False)
            i += 1
            j -= 1
        return (True)
    def fun(path, s, ls):
        if not s:
            ls.append(path)
            return
        for i in range(1, len(s) + 1):
            if (isPalin(s[:i])):
                fun(path + [s[:i]], s[i:], ls)
    ls = []
    fun([], s, ls)
    return (min([len(s) - 1 for s in ls]))
#solution2: time limit exceeded
def minCut(self, s):
    """
    :type s: str
    :rtype: int
    """
    def isPalin(s):
        i, j = 0, len(s) - 1
        while (i <= j):
            if (s[i] != s[j]):
                return (False)
            i += 1
            j -= 1
        return (True)
    dp = [0]
    for i in range(1, len(s)):
        if (isPalin(s[:i + 1])):
            dp.append(0)
            continue
        mi = []
        for j in range(1, i):
            if (isPalin(s[j:i + 1])):
                mi.append(dp[j - 1] + 1)
        mi.append(dp[-1] + 1)
        dp.append(min(mi))
    return (dp[-1])
#solution3.1:
def minCut(self, s):
    """
    :type s: str
    :rtype: int
    """
    cut = [x for x in range(-1, len(s))]
    for i in range(0, len(s)):
        for j in range(i, len(s)):
            if s[i:j] == s[j:i:-1]:
                cut[j + 1] = min(cut[j + 1], cut[i] + 1)
    return cut[-1]
#solution3.2:   not my code, study needed
def minCut(self, s):
    """
    :type s: str
    :rtype: int
    """
    # acceleration
    if s == s[::-1]: return 0
    for i in range(1, len(s)):
        if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
            return 1
    # algorithm
    cut = [x for x in range(-1, len(s))]  # cut numbers in worst case (no palindrome)
    for i in range(len(s)):
        r1, r2 = 0, 0
        # use i as origin, and gradually enlarge radius if a palindrome exists
        # odd palindrome
        while i - r1 >= 0 and i + r1 < len(s) and s[i - r1] == s[i + r1]:
            cut[i + r1 + 1] = min(cut[i + r1 + 1], cut[i - r1] + 1)
            r1 += 1
        # even palindrome
        while i - r2 >= 0 and i + r2 + 1 < len(s) and s[i - r2] == s[i + r2 + 1]:
            cut[i + r2 + 2] = min(cut[i + r2 + 2], cut[i - r2] + 1)
            r2 += 1
    return cut[-1]

#Gas Station
#solution1: time limit exceeded
def canCompleteCircuit(self, gas, cost):
    """
    :type gas: List[int]
    :type cost: List[int]
    :rtype: int
    """
    N = len(gas)
    for i in range(N):
        tank = 0
        count = 0
        j = i
        while (count < N):
            tank += gas[j]
            tank -= cost[j]
            if tank < 0:
                break
            j += 1
            if (j == N):
                j = 0
            count += 1
        if (count == N):
            return (i)
    return (-1)
#solution2:
def canCompleteCircuit(self, gas, cost):
    """
    :type gas: List[int]
    :type cost: List[int]
    :rtype: int
    """
    ls = map(lambda x, y: x - y, gas, cost)
    N = len(ls)
    start = 0
    count = 0
    tank = 0
    i = 0
    circle = False
    while (True):
        tank += ls[i]
        if (tank < 0):
            start = i
            while (ls[start] < 0):
                start += 1
                if (start == N or circle):
                    return (-1)
                count = 0
                tank = 0
                i = start
            continue
        i += 1
        if (i == N):
            i = 0
            circle = True
        count += 1
        if (count == N):
            return (start)
#solution3: when the sum is positive, there is definitely a solution
def canCompleteCircuit(self, gas, cost):
    """
    :type gas: List[int]
    :type cost: List[int]
    :rtype: int
    """
    if len(gas) == 0 or len(cost) == 0 or sum(gas) < sum(cost):
        return -1
    position = 0
    balance = 0  # current tank balance
    for i in range(len(gas)):
        balance += gas[i] - cost[i]  # update balance
        if balance < 0:  # balance drops to negative, reset the start position
            balance = 0
            position = i + 1
    return position

#Single Number
#solution1:
def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    d = {}
    for i in nums:
        if i in d:
            del d[i]
        else:
            d[i] = 1
    return (d.keys()[0])
#solution2:
def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    return reduce(lambda x, y: x ^ y, nums)

#Single Number II
#solution:
def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    return ((3 * sum(set(nums)) - sum(nums)) / 2)

#Copy List with Random Pointer
#solution1.1: too much time
# class RandomListNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
def copyRandomList(self, head):
    """
    :type head: RandomListNode
    :rtype: RandomListNode
    """
    def fun(head, root, ls):
        if (head.next):
            root.next = RandomListNode(head.next.label)
            if head.next not in ls:
                ls.append(head.next)
                fun(head.next, root.next, ls)
        if (head.random):
            root.random = RandomListNode(head.random.label)
            if head.random not in ls:
                ls.append(head.random)
                fun(head.random, root.random, ls)
    if not head:
        return None
    root = RandomListNode(head.label)
    ls = [head]
    fun(head, root, ls)
    return (root)
#solution1.2: dictionary, still too much time
def copyRandomList(self, head):
    """
    :type head: RandomListNode
    :rtype: RandomListNode
    """
    def fun(head, root, ls):
        if (head.next):
            root.next = RandomListNode(head.next.label)
            if head.next not in ls:
                ls.append(head.next)
                fun(head.next, root.next, ls)
        if (head.random):
            root.random = RandomListNode(head.random.label)
            if head.random not in ls:
                ls.append(head.random)
                fun(head.random, root.random, ls)
    if not head:
        return None
    root = RandomListNode(head.label)
    ls = [head]
    fun(head, root, ls)
    return (root)
#solution2.1:
def copyRandomList(self, head):
    dic = dict()
    m = n = head
    while m:
        dic[m] = RandomListNode(m.label)
        m = m.next
    while n:
        dic[n].next = dic.get(n.next)
        dic[n].random = dic.get(n.random)
        n = n.next
    return dic.get(head)
#solution2.2:   worth studying again
def copyRandomList(self, head):
    """
    :type head: RandomListNode
    :rtype: RandomListNode
    """
    dic = collections.defaultdict(lambda: RandomListNode(0))
    dic[None] = None
    n = head
    while n:
        dic[n].label = n.label
        dic[n].next = dic[n.next]
        dic[n].random = dic[n.random]
        n = n.next
    return dic[head]

#Word Break
#solution1: time limit exceeded
def wordBreak(self, s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    def fun(s, d):
        if (s in d):
            return (True)
        for i in range(1, len(s)):
            tmp = fun(s[:i], d) and fun(s[i:], d)
            if (tmp):
                return (True)
        return (False)
    return (fun(s, wordDict))
#solution2.1: DP
def wordBreak(self, s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    dp = [[False] * len(s) for _ in range(len(wordDict))]
    if not dp:
        return (False)
    for i in range(len(wordDict)):
        for j in range(len(s)):
            tmp = s[:j + 1] in wordDict[:i + 1]
            if (i > 1):
                tmp = tmp or dp[i - 1][j]
            if (tmp):
                dp[i][j] = tmp
                continue
            for w in wordDict[:i + 1]:
                lw = len(w)
                if (j - lw >= 0):
                    tmp = (s[j - lw + 1:j + 1] == w and dp[i][j - lw])
                    if (tmp):
                        dp[i][j] = tmp
                        continue
    return (dp[-1][-1])
#solution2.2:
def wordBreak(self, s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    d = [False] * len(s)
    for i in range(len(s)):
        for w in wordDict:
            if w == s[i - len(w) + 1:i + 1] and (d[i - len(w)] or i - len(w) == -1):    #the first index is tricky
                d[i] = True
    return d[-1]

#Linked List Cycle
#solution1.1:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def hasCycle(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if not head:
        return (False)
    p1 = p2 = head
    while (True):
        if (p1.next):
            p1 = p1.next
        else:
            return (False)
        if (p2.next):
            p2 = p2.next
            if (p2.next):
                p2 = p2.next
            else:
                return (False)
        else:
            return (False)
        if (p1 == p2):
            return (True)
#solution1.2:   neat code
def hasCycle(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            return True
    return False
#solution2: try and except
def hasCycle(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    try:
        slow = head
        fast = head.next
        while slow is not fast:
            slow = slow.next
            fast = fast.next.next
        return True
    except:
        return False

#Linded List Cycle II
#solution1.1:
def detectCycle(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return (None)
    i = j = head
    ifFound = False
    while (j.next and j.next.next):
        i = i.next
        j = j.next.next
        if (i == j):
            ifFound = True
            break
    if (ifFound):
        i = head
        while (True):
            if (i == j):
                return (i)
            i = i.next
            j = j.next
    return (None)
#solution1.2:   a little change
def detectCycle(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return (None)
    i = j = head
    while (j.next and j.next.next):
        i = i.next
        j = j.next.next
        if (i == j):
            i = head
            while (True):
                if (i == j):
                    return (i)
                i = i.next
                j = j.next
    return (None)

#Reorder List
#solution1:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def reorderList(self, head):
    """
    :type head: ListNode
    :rtype: void Do not return anything, modify head in-place instead.
    """
    if not head:
        return
    i = head
    stack = []
    while (i):
        stack.append(i)
        i = i.next
    i = head
    j = stack.pop()
    while (i != j):
        if (i.next == j):
            j.next = None
            return
        j.next = i.next
        i.next = j
        i = j.next
        j = stack.pop()
    i.next = None
    return
#solution2: faster
def reorderList(self, head):
    """
    :type head: ListNode
    :rtype: void Do not return anything, modify head in-place instead.
    """
    if not head:
        return
    # ensure the first part has the same or one more node
    fast, slow = head.next, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    # reverse the second half
    p = slow.next
    slow.next = None
    node = None
    while p:
        nxt = p.next
        p.next = node
        node = p
        p = nxt
    # combine head part and node part
    p = head
    while node:
        tmp = node.next
        node.next = p.next
        p.next = node
        p = p.next.next  # p = node.next
        node = tmp

#Binary Tree Preorder Traversal
#solution1: recursion
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def preorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    def fun(root, ls):
        if not root:
            return
        ls.append(root.val)
        fun(root.left, ls)
        fun(root.right, ls)
    ls = []
    fun(root, ls)
    return (ls)

#Binary Tree Postorder Traversal
#solution1: recursion
def postorderTraversal(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    def fun(root, ls):
        if not root:
            return
        fun(root.left, ls)
        fun(root.right, ls)
        ls.append(root.val)
    ls = []
    fun(root, ls)
    return (ls)

#Insertion Sort List
#solution:
def insertionSortList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    begin = ListNode(0)
    begin.next = head
    j = begin
    while (j.next):
        i = begin
        change = False
        while (i.next != j.next):
            if (j.next.val < i.next.val):
                tmp = j.next.next
                j.next.next = i.next
                i.next = j.next
                j.next = tmp
                change = True
                break
            else:
                i = i.next
        if not change:
            j = j.next
    return (begin.next)

#Reverse Words in s String
#solution1.1:
def reverseWords(self, s):
    """
    :type s: str
    :rtype: str
    """
    ls = s.split()
    ls.reverse()
    return (" ".join(ls))
#solution1.2:
def reverseWords(self, s):
    """
    :type s: str
    :rtype: str
    """
    return " ".join(s.strip().split()[::-1])

#Maximum Product Subarray
#solution:
def maxProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return (0)
    ma = nums[0]
    i = 0
    tmp = 1
    for i in nums:
        if (i):
            tmp = tmp * i
            ma = max(tmp, ma)
        else:
            ma = max(0, ma)
            tmp = 1
    tmp = 1
    for i in nums[::-1]:
        if (i):
            tmp = tmp * i
            ma = max(tmp, ma)
        else:
            ma = max(0, ma)
            tmp = 1
    return (ma)

#Find Minimum in Rotated Sorted Array
#solution1:
def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return (0)
    mi = nums[-1]
    i, j = 0, len(nums) - 1
    while (i < j):
        mid = (i + j + 1) // 2
        if (nums[i] > nums[j] and i + 1 < j):
            if (nums[mid] > nums[i]):
                i = mid
            else:
                j = mid
        elif (nums[i] < nums[j]):
            return (nums[i])
        else:
            return (nums[j])
    return (mi)
#solution2: much better
def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i = 0
    j = len(nums) - 1
    while i < j:
        m = i + (j - i) / 2     #good thoughts but not necessary here
        if nums[m] > nums[j]:
            i = m + 1
        else:
            j = m               #very important!
    return nums[i]

#Min Stack
#solution:
class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.mi = float("Inf")

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        self.mi = min(self.mi, x)

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()
        if not self.stack:
            self.mi = float("Inf")
        else:
            self.mi = min(self.stack)

    def top(self):
        """
        :rtype: int
        """
        return (self.stack[-1])

    def getMin(self):
        """
        :rtype: int
        """
        return (self.mi)

#Intersection of Two Linked Lists
#solution1:
def getIntersectionNode(self, headA, headB):
    """
    :type head1, head1: ListNode
    :rtype: ListNode
    """
    if not headA or not headB:
        return (None)
    i = headA
    j = headB
    count1 = count2 = 0
    while (i.next):
        count1 += 1
        i = i.next
    while (j.next):
        count2 += 1
        j = j.next
    if i != j:
        return (None)
    i = headA
    j = headB
    if (count1 > count2):
        for _ in range(count1 - count2):
            i = i.next
    elif (count1 < count2):
        for _ in range(count2 - count1):
            j = j.next
    while (i != j):
        i = i.next
        j = j.next
    return (i)
#solution:  very smart solution
def getIntersectionNode(self, headA, headB):
    """
    :type head1, head1: ListNode
    :rtype: ListNode
    """
    if headA is None or headB is None:
        return None
    pa = headA  # 2 pointers
    pb = headB
    while pa is not pb:
        pa = headB if pa is None else pa.next
        pb = headA if pb is None else pb.next
    return pa

#Two Sum II - Input array is sorted
#solution:
def twoSum(self, numbers, target):
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    i, j = 0, len(numbers) - 1
    while (True):
        s = numbers[i] + numbers[j]
        if (s == target):
            return ([i + 1, j + 1])
        elif (s > target):
            j -= 1
        else:
            i += 1

#Excel Sheet Column Number
#solution:
def titleToNumber(self, s):
    """
    :type s: str
    :rtype: int
    """
    sum = 0
    for i in range(len(s)):
        sum = sum * 26
        n = ord(s[i]) - ord("A") + 1
        sum += n
    return (sum)

#Find Peak Element
#solution:
def findPeakElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i, j = 1, len(nums)
    ls = [-float("Inf")] + nums + [float("Inf")]
    if (ls[i] > ls[i + 1]):
        return (i - 1)
    if (ls[j] > ls[j - 1]):
        return (j - 1)
    while (i < j):
        m = (i + j) // 2
        if (ls[m - 1] < ls[m] and ls[m] > ls[m + 1]):
            return (m - 1)
        if (ls[m - 1] < ls[m]):
            i = m + 1
        else:
            j = m
    return (i - 1)

#Excel Sheet Column Title
#solution:
def convertToTitle(self, n):
    """
    :type n: int
    :rtype: str
    """
    s = ""
    while (n > 0):
        n, tmp = divmod(n - 1, 26)
        s = chr(65 + tmp) + s
    return (s)

#Factorial Trailing Zeros
#solution1: time limit exceeded
def trailingZeroes(self, n):
    """
    :type n: int
    :rtype: int
    """
    leftover = n % 5
    n = n - leftover
    s = 0
    while (n > 0):
        tmp = n
        while (tmp and tmp % 5 == 0):
            tmp = tmp // 5
            s += 1
        n -= 5
    return (s)
#solution2:
def trailingZeroes(self, n):
    """
    :type n: int
    :rtype: int
    """
    res = 0
    while (n):
        n = n // 5
        res += n
    return (res)

#Binary Search Tree Iterator
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#solution1:
class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.ls = []

        def fun(root, ls):
            if not root:
                return
            fun(root.left, ls)
            ls.append(root.val)
            fun(root.right, ls)

        fun(root, self.ls)

    def hasNext(self):
        """
        :rtype: bool
        """
        return (bool(self.ls))

    def next(self):
        """
        :rtype: int
        """
        return (self.ls.pop(0))
#solution2:
    # @param root, a binary search tree's root node
    def __init__(self, root):
        self.stack = list()
        self.pushAll(root)

    # @return a boolean, whether we have a next smallest number
    def hasNext(self):
        return self.stack

    # @return an integer, the next smallest number
    def next(self):
        tmpNode = self.stack.pop()
        self.pushAll(tmpNode.right)
        return tmpNode.val

    def pushAll(self, node):
        while node is not None:
            self.stack.append(node)
            node = node.left

#Rotate Array
#solution1:
def rotate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    k = k % len(nums)
    last = nums[-k:]
    nums[k:] = nums[:-k]
    nums[:k] = last

#House Robber
#solution1:
def rob(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    if len(nums) <= 2:
        return (max(nums))
    dp = [nums[0], max(nums[0], nums[1])]
    for i in range(2, len(nums)):
        dp.append(max(dp[i - 1], dp[i - 2] + nums[i]))
    return (dp[-1])
#solution2:
def rob(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    tmp1 = tmp2 = 0
    for i in nums:
        tmp1, tmp2 = tmp2, max(tmp1 + i, tmp2)
    return (tmp2)

#Number of 1 Bits
#solution:
def hammingWeight(self, n):
    """
    :type n: int
    :rtype: int
    """
    count = 0
    while (n):
        n, tmp = divmod(n, 2)
        count += tmp
    return (count)

#Reverse Linked List
#solution:
def reverseList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    ls = []
    while (head):
        ls.append(head)
        head = head.next
    new = ListNode(0)
    i = new
    while (ls):
        i.next = ls.pop()
        i = i.next
    i.next = None
    return (new.next)

#Binary Tree Right Side View
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def rightSideView(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return ([])
    ls = [root]
    res = []
    while (True):
        res.append(ls[-1].val)
        tmp = []
        while (ls):
            this = ls.pop(0)
            if this.left:
                tmp.append(this.left)
            if this.right:
                tmp.append(this.right)
        if not tmp:
            return res
        ls = tmp

#Combination Sum III:
#solution:
def combinationSum3(self, k, n):
    """
    :type k: int
    :type n: int
    :rtype: List[List[int]]
    """
    def fun(nums, path, ls, k, n):
        if k == 0:
            if n == 0:
                ls.append(path)
            return
        for i in range(0, len(nums) - k + 1):
            fun(nums[i + 1:], path + [nums[i]], ls, k - 1, n - nums[i])
    ls = []
    fun(list(range(1, 10)), [], ls, k, n)
    return (ls)

#Remove Linked List Elements
#solution:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def removeElements(self, head, val):
    """
    :type head: ListNode
    :type val: int
    :rtype: ListNode
    """
    dummy = ListNode(0)
    dummy.next = head
    this = dummy
    while (this.next):
        if this.next.val == val:
            this.next = this.next.next
        else:
            this = this.next
    return (dummy.next)

#Maximum Gap
#solution1: heap
def maximumGap(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if (len(nums) < 2):
        return 0
    heapq.heapify(nums)
    tmp1 = heapq.heappop(nums)
    tmp2 = heapq.heappop(nums)
    ma = abs(tmp2 - tmp1)
    while (nums):
        tmp1, tmp2 = tmp2, heapq.heappop(nums)
        ma = max(ma, tmp2 - tmp1)
    return (ma)
#solution2: radix sort
def maximumGap(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    def radixSort(A):
        for k in xrange(10):
            s = [[] for i in xrange(10)]
            for i in A:
                s[i / (10 ** k) % 10].append(i)
            A = [a for b in s for a in b]
        return A
    A = radixSort(nums)
    ans = 0
    if len(A) == 0: return 0
    prev = A[0]
    for i in A:
        if i - prev > ans: ans = i - prev
        prev = i
    return ans

#Clone Graph
#solution:
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []
def cloneGraph(self, node):
    def fun(node, new, d):
        new.neighbors = []
        for i in node.neighbors:
            if i.label not in d:
                new2 = UndirectedGraphNode(i.label)
                d[new2.label] = new2
                new.neighbors.append(new2)
                fun(i, new2, d)
            else:
                new.neighbors.append(d[i.label])

    if not node:
        return (None)
    new = UndirectedGraphNode(node.label)
    d = {}
    d[new.label] = new
    fun(node, new, d)
    return (new)

#Candy
#solution1:
def candy(self, ratings):
    """
    :type ratings: List[int]
    :rtype: int
    """
    if not ratings:
        return (0)
    if len(ratings) == 1:
        return (1)
    nums = [0] * len(ratings)
    ls = [-1] * len(ratings)
    if ratings[0] > ratings[1]:
        ls[0] = 1
    else:
        ls[0] = 0
        nums[0] = 1
    if ratings[-2] >= ratings[-1]:
        ls[-1] = 0
        nums[-1] = 1
    else:
        ls[-1] = 1
    for i in range(1, len(ratings) - 1):
        if (ratings[i - 1] == ratings[i] and ratings[i] == ratings[i + 1]):
            ls[i] = 0
            nums[i] = 1
        elif (ratings[i - 1] <= ratings[i] and ratings[i] >= ratings[i + 1]):
            ls[i] = 1
        elif (ratings[i - 1] >= ratings[i] and ratings[i] <= ratings[i + 1]):
            ls[i] = 0
            nums[i] = 1
    bottom = top = 0
    d = 0
    i = 0
    while (i < len(ls)):
        if (ls[i] == 1):
            top = i
            d = top - bottom
            bottom = i
        elif (ls[i] == 0):
            bottom = i
            if nums[top] == 0:
                d = max(d, bottom - top)
                nums[top] = 1 + d
                d = 0
        i += 1
    for i in range(1, len(ratings)):
        if (nums[i - 1] and nums[i] == 0 and ratings[i - 1] < ratings[i]):
            nums[i] = nums[i - 1] + 1
        elif (nums[i - 1] and nums[i] == 0 and ratings[i - 1] == ratings[i]):
            nums[i] = nums[i]
    for i in range(len(ratings) - 2, -1, -1):
        if (nums[i + 1] and nums[i] == 0 and ratings[i] > ratings[i + 1]):
            nums[i] = nums[i + 1] + 1
        elif (nums[i + 1] and nums[i] == 0 and ratings[i] == ratings[i + 1]):
            nums[i] = nums[i + 1]
    return (sum(nums))
#solution2: so smart!
def candy(self, ratings):
    """
    :type ratings: List[int]
    :rtype: int
    """
    # use two pass scan from left to right and vice versa to keep the candy level up to now
    # similar to like the Trapping Rain Water question
    res = [1] * len(ratings)  # also compatable with [] input
    lbase = rbase = 1
    # left scan
    for i in xrange(1, len(ratings)):
        lbase = lbase + 1 if ratings[i] > ratings[i - 1] else 1
        res[i] = lbase
    # right scan
    for i in xrange(len(ratings) - 2, -1, -1):
        rbase = rbase + 1 if ratings[i] > ratings[i + 1] else 1
        res[i] = max(rbase, res[i])
    return sum(res)

#Compare Version Numbers
#solution:
def compareVersion(self, version1, version2):
    """
    :type version1: str
    :type version2: str
    :rtype: int
    """
    nums1 = [int(x) for x in version1.split(".")]
    nums2 = [int(x) for x in version2.split(".")]
    while (nums1 and nums2):
        tmp1 = nums1.pop(0)
        tmp2 = nums2.pop(0)
        if (tmp1 > tmp2):
            return (1)
        elif (tmp1 < tmp2):
            return (-1)
    if (nums1):
        if any([x > 0 for x in nums1]):
            return (1)
    elif (nums2):
        if any([x > 0 for x in nums2]):
            return (-1)
    return (0)

#Fraction to Recurring Decimal
#solution:
def fractionToDecimal(self, numerator, denominator):
    """
    :type numerator: int
    :type denominator: int
    :rtype: str
    """
    sign1 = sign2 = False
    if numerator < 0:
        sign1 = True
        numerator = -numerator
    if denominator < 0:
        sign2 = True
        denominator = -denominator
    res = ""
    tmp1, tmp2 = divmod(numerator, denominator)
    if (tmp2 == 0):
        if tmp1 == 0:
            return ("0")
        res = str(tmp1)
        if ((sign1 and sign2) or ((not sign1) and (not sign2))):
            return (res)
        else:
            return ("-" + res)
    d = {}
    res = str(tmp1) + "."
    while (tmp2):
        if tmp2 in d:
            res = res[:d[tmp2]] + "(" + res[d[tmp2]:] + ")"
            if ((sign1 and sign2) or ((not sign1) and (not sign2))):
                return (res)
            else:
                return ("-" + res)
        else:
            d[tmp2] = len(res)
            tmp1, tmp2 = divmod(tmp2 * 10, denominator)
            res += str(tmp1)
    if ((sign1 and sign2) or ((not sign1) and (not sign2))):
        return (res)
    else:
        return ("-" + res)

#Find Minimum in Rotated Sorted Array II
#solution:
def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i = 0
    j = len(nums) - 1
    while i < j:
        m = (j + i) / 2
        if nums[m] > nums[j]:
            i = m + 1
        else:
            j = m if nums[m] < nums[j] else j - 1   #tricky point
    return nums[i]

#Largest Number
#solution1:
def largestNumber(self, nums):
    strs = [str(x) for x in nums]
    strs0 = [x[-1] + x for x in strs]
    maxlen = max([len(x) for x in strs])
    strs2 = [x + x[0] * (maxlen - len(x)) for x in strs]
    combine = [(y, z, x) for x, y, z in zip(strs, strs2, strs0)]
    combine.sort(key=lambda x: x[1], reverse=True)
    combine.sort(key=lambda x: x[0], reverse=True)
    res = "".join([x[2] for x in combine])
    return (str(int(res)))
#solution2.1: pretty neat code
def largestNumber(self, nums):
    comp = lambda a, b: 1 if a + b > b + a else -1 if a + b < b + a else 0
    nums = map(str, nums)
    nums.sort(cmp=comp, reverse=True)
    return str(int("".join(nums)))
#solution2.2:
def largestNumber(self, nums):
    r = ''.join(sorted(map(str, nums), lambda x, y: [1, -1][x + y > y + x]))
    return r.lstrip('0') or '0'

#Word Ladder
#solution1:  time limit exceeded
def ladderLength(self, beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: int
    """
    def oneDiff(s1, s2):
        ls = [x == y for x, y in zip(s1, s2)]
        count = ls.count(False)
        if count == 1:
            return (True)
        return (False)
    def fun(this, endWord, wordList, count, lens):
        if (this == endWord):
            lens.append(count)
        else:
            for i in range(len(wordList)):
                if oneDiff(this, wordList[i]):
                    fun(wordList[i], endWord, wordList[:i] + wordList[i + 1:], count + 1, lens)
    lens = []
    fun(beginWord, endWord, wordList, 1, lens)
    if not lens:
        return (0)
    return (min(lens))
#solution2: time limit exceeded
def ladderLength(self, beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: int
    """
    queue = collections.deque([[beginWord, 1]])
    while queue:
        word, length = queue.popleft()
        if word == endWord:
            return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i + 1:]
                if next_word in wordList:
                    wordList.remove(next_word)
                    queue.append([next_word, length + 1])
    return 0
#solution3: preprocess
def ladderLength(self, beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: int
    """
    from collections import deque
    def construct_dict(word_list):
        d = {}
        for word in word_list:
            for i in range(len(word)):
                s = word[:i] + "_" + word[i + 1:]
                d[s] = d.get(s, []) + [word]
        return d
    def bfs_words(begin, end, dict_words):
        queue, visited = deque([(begin, 1)]), set()
        while queue:
            word, steps = queue.popleft()
            if word not in visited:
                visited.add(word)
                if word == end:
                    return steps
                for i in range(len(word)):
                    s = word[:i] + "_" + word[i + 1:]
                    neigh_words = dict_words.get(s, [])
                    for neigh in neigh_words:
                        if neigh not in visited:
                            queue.append((neigh, steps + 1))
        return 0
    d = construct_dict(wordList)
    return bfs_words(beginWord, endWord, d)

#Word Ladder II
#solution1: time limit exceeded
def findLadders(self, beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: List[List[str]]
    """
    from collections import deque
    def construct_dict(word_list):
        d = {}
        for word in word_list:
            for i in range(len(word)):
                s = word[:i] + "_" + word[i + 1:]
                d[s] = d.get(s, []) + [word]
        return d
    def bfs_words(begin, end, dict_words, ls, stop):
        queue = deque([(begin, [begin])])
        while queue:
            word, tmp = queue.popleft()
            if len(tmp) > stop:
                return
            if word == end:
                ls.append(tmp)
                length = len(tmp)
                while queue:
                    word, tmp = queue.popleft()
                    if len(tmp) == length and word == end:
                        ls.append(tmp)
                return
            for i in range(len(word)):
                s = word[:i] + "_" + word[i + 1:]
                neigh_words = dict_words.get(s, [])
                for neigh in neigh_words:
                    queue.append((neigh, tmp + [neigh]))
    d = construct_dict(wordList)
    ls = []
    bfs_words(beginWord, endWord, d, ls, len(wordList) + 1)
    return ls
#solution2: not my code, further study needed
def findLadders(self, begin, end, words_list):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: List[List[str]]
    """
    def construct_paths(source, dest, tree):
        if source == dest:
            return [[source]]
        return [[source] + path for succ in tree[source]
                for path in construct_paths(succ, dest, tree)]
    def add_path(tree, word, neigh, is_forw):
        if is_forw:
            tree[word] += neigh,
        else:
            tree[neigh] += word,
    def bfs_level(this_lev, oth_lev, tree, is_forw, words_set):
        if not this_lev: return False
        if len(this_lev) > len(oth_lev):
            return bfs_level(oth_lev, this_lev, tree, not is_forw, words_set)
        for word in (this_lev | oth_lev):
            words_set.discard(word)
        next_lev, done = set(), False
        while this_lev:
            word = this_lev.pop()
            for c in string.ascii_lowercase:
                for index in range(len(word)):
                    neigh = word[:index] + c + word[index + 1:]
                    if neigh in oth_lev:
                        done = True
                        add_path(tree, word, neigh, is_forw)
                    if not done and neigh in words_set:
                        next_lev.add(neigh)
                        add_path(tree, word, neigh, is_forw)
        return done or bfs_level(next_lev, oth_lev, tree, is_forw, words_set)
    if end not in words_list:
        return ([])
    tree, path, paths = collections.defaultdict(list), [begin], []
    is_found = bfs_level(set([begin]), set([end]), tree, True, set(words_list))
    return construct_paths(begin, end, tree)

#Surrounded Regions
#solution:
def solve(self, board):
    """
    :type board: List[List[str]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    if not board or not board[0]:
        return
    for i in range(len(board)):
        for j in range(len(board[0])):
            if (board[i][j] == "O"):
                ls = [(i, j)]
                visited = []
                tmp = False
                while (ls):
                    ii, jj = ls.pop(0)
                    if board[ii][jj] != "O":
                        continue
                    if (tmp or ii == 0 or ii == len(board) - 1 or jj == 0 or jj == len(board[0]) - 1):
                        tmp = True
                    else:
                        visited.append((ii, jj))
                    board[ii][jj] = "A"
                    if (ii - 1 >= 0 and board[ii - 1][jj] == "O"):
                        ls.append((ii - 1, jj))
                    if (ii + 1 <= len(board) - 1 and board[ii + 1][jj] == "O"):
                        ls.append((ii + 1, jj))
                    if (jj - 1 >= 0 and board[ii][jj - 1] == "O"):
                        ls.append((ii, jj - 1))
                    if (jj + 1 <= len(board[0]) - 1 and board[ii][jj + 1] == "O"):
                        ls.append((ii, jj + 1))
                if not tmp:
                    for iii, jjj in visited:
                        board[iii][jjj] = "X"
    for i in range(len(board)):
        for j in range(len(board[0])):
            if (board[i][j] == "A"):
                board[i][j] = "O"

#Repeated DNA Sequences
#solution1.1:
def findRepeatedDnaSequences(self, s):
    """
    :type s: str
    :rtype: List[str]
    """
    if len(s) <= 10:
        return ([])
    i = j = 0
    tmp = 0
    ls = []
    for k in range(10):
        if (s[j] == "A"):
            tmp += 10 ** k
        elif (s[j] == "C"):
            tmp += 2 * 10 ** k
        elif (s[j] == "G"):
            tmp += 3 * 10 ** k
        elif (s[j] == "T"):
            tmp += 4 * 10 ** k
        j += 1
    d = {}
    d[tmp] = s[i:j]
    i += 1
    while (j < len(s)):
        if (s[j] == "A"):
            tmp = tmp // 10 + 10 ** 9
        elif (s[j] == "C"):
            tmp = tmp // 10 + 2 * 10 ** 9
        elif (s[j] == "G"):
            tmp = tmp // 10 + 3 * 10 ** 9
        elif (s[j] == "T"):
            tmp = tmp // 10 + 4 * 10 ** 9
        j += 1
        if tmp in d:
            if d[tmp] not in ls:
                ls.append(d[tmp])
        else:
            d[tmp] = s[i:j]
        i += 1
    return (ls)
#solution1.2:
def findRepeatedDnaSequences(self, s):
    """
    :type s: str
    :rtype: List[str]
    """
    if len(s) <= 10:
        return ([])
    i = 0
    d = {}
    d[s[0:10]] = 1
    ls = []
    for i in range(1, len(s) - 9):
        tmp = s[i:i + 10]
        if (tmp in d):
            d[tmp] += 1
        else:
            d[tmp] = 1
        if (d[tmp] == 2):
            ls.append(tmp)
    return (ls)

#Happy Number
#solution1:
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    d = {}
    while (True):
        n = sum([x ** 2 for x in [int(x) for x in str(n)]])
        if n == 1:
            return (True)
        else:
            if n in d:
                return (False)
            else:
                d[n] = 1
#solution2:
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    mem = set()
    while n != 1:
        n = sum([int(i) ** 2 for i in str(n)])
        if n not in mem:
            mem.add(n)
        else:
            return False
    return True

#Reverse Bits
#solution:
def reverseBits(self, n):
    b = str(bin(n))[2:][::-1]
    b = b + "0" * (32 - len(b))
    return (int(b, 2))

#Number of Islands
#solution:
def numIslands(self, grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    if not grid or not grid[0]:
        return (0)
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if (grid[i][j] == "1"):
                count += 1
                ls = [(i, j)]
                while (ls):
                    ii, jj = ls.pop(0)
                    if (grid[ii][jj] == "1"):
                        grid[ii][jj] = "0"
                        if (ii - 1 >= 0 and grid[ii - 1][jj] == "1"):
                            ls.append((ii - 1, jj))
                        if (ii + 1 < len(grid) and grid[ii + 1][jj] == "1"):
                            ls.append((ii + 1, jj))
                        if (jj - 1 >= 0 and grid[ii][jj - 1] == "1"):
                            ls.append((ii, jj - 1))
                        if (jj + 1 < len(grid[0]) and grid[ii][jj + 1] == "1"):
                            ls.append((ii, jj + 1))
    return (count)

#Bitwise AND of Numbers Range
#solution:
def rangeBitwiseAnd(self, m, n):
    """
    :type m: int
    :type n: int
    :rtype: int
    """
    if m == n:
        return (m)
    res = 0
    while (m and n):
        mm = "{0:b}".format(m)
        nn = "{0:b}".format(n)
        if len(mm) == len(nn):
            res |= int("1" + "0" * (len(mm) - 1), 2)
            m = int(mm[1:], 2)
            n = int(nn[1:], 2)
        else:
            return (res)
    return (res)

#Isomorphic Strings
#solution1:
def isIsomorphic(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    d = {}
    for i, j in zip(s, t):
        if i in d:
            if j != d[i]:
                return (False)
        else:
            d[i] = j
    d = {}
    for i, j in zip(t, s):
        if i in d:
            if j != d[i]:
                return (False)
        else:
            d[i] = j
    return (True)
#solution2: smart but slow
def isIsomorphic(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    return [s.find(i) for i in s] == [t.find(j) for j in t]
#solution3:
def isIsomorphic(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))
#solution4:
def isIsomorphic(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    return map(s.find, s) == map(t.find, t)

#Count Primes
#solution:
def countPrimes(self, n):
    """
    :type n: int
    :rtype: int
    """
    ls = [True] * n
    count = 0
    for i in xrange(2, n):  #xrange is needed
        if not ls[i - 1]:
            continue
        else:
            count += 1
        for j in xrange(i * i, n, i):
            ls[j - 1] = False
    return (count)

#Course Schedule
#solution1:
def canFinish(self, numCourses, prerequisites):
    """
    :type numCourses: int
    :type prerequisites: List[List[int]]
    :rtype: bool
    """
    ls = [[] for _ in range(numCourses)]
    for i, j in prerequisites:
        ls[i].append(j)
    count = newCount = len(prerequisites)
    while (True):
        count = newCount
        for i in range(numCourses):
            for j in ls[i]:
                if not ls[j]:
                    ls[i].remove(j)
                    newCount -= 1
        if not newCount:
            return (True)
        if newCount == count:
            return (False)
    return (True)
#solution2: Topological Sort
def canFinish(self, numCourses, prerequisites):
    """
    :type numCourses: int
    :type prerequisites: List[List[int]]
    :rtype: bool
    """
    graph = collections.defaultdict(set)
    neighbors = collections.defaultdict(set)
    for course, pre in prerequisites:
        graph[course].add(pre)
        neighbors[pre].add(course)
    stack = [n for n in range(numCourses) if not graph[n]]
    count = 0
    while stack:
        node = stack.pop()
        count += 1
        for n in neighbors[node]:
            graph[n].remove(node)
            if not graph[n]:
                stack.append(n)
    return count == numCourses

#Maximal Square
#solution1: BFS, time limit exceeded
def maximalSquare(self, matrix):
    """
    :type matrix: List[List[str]]
    :rtype: int
    """
    if not matrix or not matrix[0]:
        return (0)
    ma = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]:
                queue = [(i, j, 1)]
                count = 0
                while (True):
                    ii, jj, c = queue.pop(0)
                    if (ii == len(matrix) or jj == len(matrix[0]) or matrix[ii][jj] == '0'):
                        count = c - 1
                        break
                    if (ii + 1, jj, c + 1) not in queue and (ii + 1, jj, c) not in queue:
                        queue.append((ii + 1, jj, c + 1))
                    if (ii + 1, jj + 1, c + 1) not in queue and (ii + 1, jj + 1, c) not in queue:
                        queue.append((ii + 1, jj + 1, c + 1))
                    if (ii, jj + 1, c + 1) not in queue and (ii, jj + 1, c) not in queue:
                        queue.append((ii, jj + 1, c + 1))
                ma = max(ma, count)
    return (ma ** 2)
#solution2: DP
def maximalSquare(self, matrix):
    """
    :type matrix: List[List[str]]
    :rtype: int
    """
    if not matrix or not matrix[0]:
        return (0)
    dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    count = 0
    for i in range(0, len(matrix)):
        dp[i][0] = int(matrix[i][0])
        count = max(count, dp[i][0])
    for j in range(0, len(matrix[0])):
        dp[0][j] = int(matrix[0][j])
        count = max(count, dp[0][j])
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            dp[i][j] = 1 + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]) if matrix[i][j] == "1" else 0
            count = max(count, dp[i][j])
    return (count ** 2)

#Invert Binary Tree
#solution:
def invertTree(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    def fun(root):
        if root:
            root.left, root.right = fun(root.right), fun(root.left)
        return (root)
    return (fun(root))

#Minimum Size Subarray Sum
#solution:
def minSubArrayLen(self, s, nums):
    """
    :type s: int
    :type nums: List[int]
    :rtype: int
    """
    if not nums or sum(nums) < s:
        return (0)
    i = j = 0
    length = len(nums) + 1
    sums = 0
    while (j < len(nums)):
        sums += nums[j]
        if (sums >= s):
            length = min(length, j - i + 1)
            while (i < j):
                sums -= nums[i]
                i += 1
                if (sums >= s):
                    length = min(length, j - i + 1)
                else:
                    break
        j += 1
    return (length)

#Kth Largest Element in an Array
#solution1: Heap
def findKthLargest(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    heapq.heapify(nums)
    for _ in range(len(nums) - k):
        heapq.heappop(nums)
    return (heapq.heappop(nums))
#solution2: Sort
def findKthLargest(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    nums.sort()
    return (nums[-k])

#Power of Two
#solution1:
def isPowerOfTwo(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n <= 0:
        return (False)
    if n == 1:
        return (True)
    while (n > 1):
        if (n % 2 == 0):
            n = n / 2
        else:
            return (False)
    return (True)
#solution2.1:
def isPowerOfTwo(self, n):
    """
    :type n: int
    :rtype: bool
    """
    return n > 0 and not (n & n - 1)
#solution2.2:   much faster
def isPowerOfTwo(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n <= 0:
        return (False)
    return not (n & (n - 1))

#Palindrome Linked List
#solution1.1:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def isPalindrome(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    i = head
    ls = []
    while (i):
        ls.append(i.val)
        i = i.next
    i, j = 0, len(ls) - 1
    while (i < j):
        if (ls[i] != ls[j]):
            return (False)
        i += 1
        j -= 1
    return (True)
#solution1.2:
def isPalindrome(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    vals = []
    while head:
        vals += head.val,
        head = head.next
    return vals == vals[::-1]
#solution2: O(n) time and O(1) space
def isPalindrome(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    if not head:
        return (True)
    mid = end = head
    count = 0
    while (end.next):
        count += 1
        end = end.next
    div, mod = divmod(count, 2)
    for _ in range(div):
        mid = mid.next
    if not mod:
        tmp = ListNode(mid.val)
        tmp.next = mid.next
        mid.next = tmp
    midnext = mid.next
    dummy = ListNode(0)
    i, j = dummy, head
    while (j != midnext):
        tmp = j.next
        j.next = i
        i, j = j, tmp
    while (midnext):
        if mid.val != midnext.val:
            return (False)
        mid = mid.next
        midnext = midnext.next
    return (True)

#Count Complete Tree Nodes
#solution1: time limit exceeded
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def countNodes(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return (0)
    count = 0
    stack = [root]
    while (stack):
        count += 1
        tmp = stack.pop(0)
        if tmp.left:
            stack.append(tmp.left)
        if tmp.right:
            stack.append(tmp.right)
    return (count)
#solution2:
def countNodes(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root):
        count = 0
        while (root):
            count += 1
            root = root.left
        return (count)
    if not root:
        return (0)
    count = 0
    height = fun(root)
    while (root):
        if fun(root.right) == height - 1:
            count += 2 ** (height - 1)
            root = root.right
        else:
            count += 2 ** (height - 2)
            root = root.left
        height -= 1
    return (count)

#Summary Ranges
#solution:
def summaryRanges(self, nums):
    """
    :type nums: List[int]
    :rtype: List[str]
    """
    if not nums:
        return ([])
    length = len(nums)
    ls = []
    begin = end = i = 0
    while (i + 1 < length):
        if nums[i] + 1 == nums[i + 1]:
            end = i + 1
            i += 1
        else:
            end = i
            if begin == end:
                ls.append(str(nums[begin]))
            else:
                ls.append(str(nums[begin]) + "->" + str(nums[end]))
            i += 1
            begin = end = i
    if begin == end:
        ls.append(str(nums[begin]))
    else:
        ls.append(str(nums[begin]) + "->" + str(nums[end]))
    return (ls)

#Implement Stack using Queues
#solution:
class MyStack(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = []
        self.t = None   #Be careful! The name of variable should not be the same as the function
    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.queue.append(x)
        self.t = x
    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        res = self.t
        length = len(self.queue)
        q = []
        for _ in range(length - 1):
            self.t = self.queue.pop(0)
            q.append(self.t)
        self.queue = q
        return (res)
    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return (self.t)
    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return (len(self.queue) == 0)
# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

#Implement Queue using Stacks
#solution1: one stack
class MyQueue(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []
        self.bottom = None
    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        if not self.stack:
            self.bottom = x
        self.stack.append(x)
    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        length = len(self.stack)
        s = []
        for _ in range(length - 1):
            self.bottom = self.stack.pop()
            s.append(self.bottom)
        res = self.stack.pop()
        while (s):
            self.stack.append(s.pop())
        return (res)
    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return (self.bottom)
    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return (len(self.stack) == 0)
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
#solution2: two stacks
class MyQueue(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []
        self.s = []
    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if self.s:
            return (self.s.pop())
        length = len(self.stack)
        for _ in range(length - 1):
            self.s.append(self.stack.pop())
        return (self.stack.pop())
    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if self.s:
            return (self.s[-1])
        while (self.stack):
            self.s.append(self.stack.pop())
        return (self.s[-1])
    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return (not self.stack and not self.s)

#Kth Smallest Element in a BST
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def kthSmallest(self, root, k):
    """
    :type root: TreeNode
    :type k: int
    :rtype: int
    """
    def fun(root, kk):
        if not root:
            return None
        tmp1 = fun(root.left, kk)
        if tmp1 is not None:    #tmp1 may be 0, so here we have to use "None"
            return (tmp1)
        if kk[0] == 1:
            return (root.val)
        else:
            kk[0] -= 1
        return (fun(root.right, kk)
    kk = [k]
    return (fun(root, kk))

#Basic Calculator
#solution:
def calculate(self, s):
    """
    :type s: str
    :rtype: int
    """

    def fun(s):
        if s[0] == "-":
            s = "0" + s
        ls = []
        for i, v in enumerate(s):
            if v == "+" or v == "-":
                ls.append(i)
        ls.append(len(s))
        res = int(s[:ls[0]])
        tmp = ls[0]
        for i in ls[1:]:
            if s[tmp] == "+":
                res += int(s[tmp + 1:i])
            else:
                res -= int(s[tmp + 1:i])
            tmp = i
        return (res)

    stack = []
    i = 0
    while (i < len(s)):
        if s[i] == "(":
            stack.append(i)
            i += 1
        elif s[i] == ")":
            tmp = stack.pop()
            res = str(fun(s[tmp + 1:i]))
            if (tmp - 1 >= 0 and s[tmp - 1] == res[0] == "-"):
                s = s[:tmp - 1] + "+" + res[1:] + s[i + 1:]
                i = tmp + len(res) - 1
            elif (tmp - 1 >= 0 and s[tmp - 1] == "-" and res[0] != "-"):
                s = s[:tmp] + res + s[i + 1:]
                i = tmp + len(res)
            else:
                if (res[0] == "-"):
                    s = s[:tmp] + "0" + res + s[i + 1:]
                    i = tmp + len(res) + 1
                else:
                    s = s[:tmp] + "0+" + res + s[i + 1:]
                    i = tmp + len(res) + 2
        else:
            i += 1
    if s[0] == "(":
        return (fun(s[1:-1]))
    return (fun(s))

#Basic Calculator II
#solution1: time limit exceeded
def calculate(self, s):
    """
    :type s: str
    :rtype: int
    """
    ls = []
    i = 0
    tmp = 0
    if s[0] == "-":
        s = "0" + s
    while (i < len(s)):
        if (s[i] in "+-*/"):
            ls.append(int(s[tmp:i]))
            ls.append(s[i])
            tmp = i + 1
        i += 1
    ls.append(int(s[tmp:i]))
    i = 0
    while (i < len(ls)):
        if (ls[i] == "*"):
            ls[i - 1] = ls[i - 1] * ls[i + 1]
            del ls[i]
            del ls[i]
        elif (ls[i] == "/"):
            ls[i - 1] = int(ls[i - 1] / ls[i + 1])
            del ls[i]
            del ls[i]
        else:
            i += 1
    i = 0
    while (i < len(ls)):
        if (ls[i] == "+"):
            ls[i - 1] = ls[i - 1] + ls[i + 1]
            del ls[i]
            del ls[i]
        elif (ls[i] == "-"):
            ls[i - 1] = ls[i - 1] - ls[i + 1]
            del ls[i]
            del ls[i]
        else:
            i += 1
    return (ls[0])
#solution2:
def calculate(self, s):
    """
    :type s: str
    :rtype: int
    """
    ls = []
    i = 0
    tmp = 0
    if s[0] == "-":
        s = "0" + s
    while (i < len(s)):
        if (s[i] in "+-*/"):
            ls.append(int(s[tmp:i]))
            ls.append(s[i])
            tmp = i + 1
        i += 1
    ls.append(int(s[tmp:i]))
    res = [ls[0]]   #The following parts are very smart!
    i = 1
    while (i < len(ls)):
        if (ls[i] == "*"):
            res[-1] *= ls[i + 1]
        elif (ls[i] == "/"):
            res[-1] = res[-1] // ls[i + 1] if res[-1] >= 0 else -(-res[-1] // ls[i + 1])
        elif (ls[i] == "+"):
            res.append(ls[i + 1])
        else:
            res.append(-ls[i + 1])
        i += 2
    return (sum(res))

#Delete Node in a Linked List
#solution:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    while (node.next.next):
        node.val = node.next.val
        node = node.next
    node.val = node.next.val
    node.next = None

#Lowest Common Ancestor of a Binary Search Tree
#solution1.1:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def lowestCommonAncestor(self, root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    while (True):
        if p == root or q == root:
            return (root)
        if (p.val < root.val) ^ (q.val < root.val):
            return (root)
        elif (p.val < root.val):
            root = root.left
        else:
            root = root.right
#solution1.2:
def lowestCommonAncestor(self, root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    while root:
        if root.val > p.val and root.val > q.val:
            root = root.left
        elif root.val < p.val and root.val < q.val:
            root = root.right
        else:
            return root

#Lowest Common Ancestor of a Binary Tree
#solution1: using stack, DFS
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def lowestCommonAncestor(self, root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    stack = []
    f1 = False
    first = None
    while True:
        while root:
            if (not f1) and (root == p or root == q):
                first = root
                stack.append((root, f1))
                f1 = True
                root = root.left
            elif not f1:
                stack.append((root, f1))
                root = root.left
            elif f1:
                if (root == p or root == q):
                    return (first)
                else:
                    stack.append((root, f1))
                    root = root.left
        node, f = stack.pop()
        if f1 and not f:
            first = node
        root = node.right
#solution2: using dictionary to keep father nodes
def lowestCommonAncestor(self, root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    stack = [root]
    parent = {root: None}
    while p not in parent or q not in parent:
        node = stack.pop()
        if node.left:
            parent[node.left] = node
            stack.append(node.left)
        if node.right:
            parent[node.right] = node
            stack.append(node.right)
    ancestors = set()
    while p:
        ancestors.add(p)
        p = parent[p]
    while q not in ancestors:
        q = parent[q]
    return q

#Product of Array Except Self
#solution1.1:
def productExceptSelf(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    length = len(nums)
    output = []
    tmp = 1
    for i in range(length):
        output.append(tmp)
        tmp = tmp * nums[i]
    tmp = 1
    for i in range(length - 1, -1, -1):
        output[i] *= tmp
        tmp = tmp * nums[i]
    return (output)
#solution1.2:
def productExceptSelf(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    length = len(nums)
    output = [1]
    for i in range(1, length):
        output.append(output[-1] * nums[i - 1])
    tmp = 1
    for i in range(length - 1, -1, -1):
        output[i] *= tmp
        tmp = tmp * nums[i]
    return (output)

#Valid Anagram
#solution:
def isAnagram(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    d = {}
    for i in s:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    for i in t:
        if i in d:
            d[i] -= 1
            if d[i] == 0:
                del d[i]
        else:
            return (False)
    if d:
        return (False)
    return (True)

#Binary Tree Paths
#solution:
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def binaryTreePaths(self, root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    def fun(root, path, ls):
        if not root:
            return
        if (not root.left) and (not root.right):
            ls.append(path)
            return
        if root.left:
            fun(root.left, path + "->" + str(root.left.val), ls)
        if root.right:
            fun(root.right, path + "->" + str(root.right.val), ls)
    if not root:
        return ([])
    ls = []
    fun(root, str(root.val), ls)
    return (ls)

#Ugly Number
#solution1:
def isUgly(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 0:
        return (False)
    if num == 1:
        return (True)
    while (num > 5):
        if num % 2 == 0:
            num = num // 2
        elif num % 3 == 0:
            num = num // 3
        elif num % 5 == 0:
            num = num // 5
        else:
            return (False)
    return (True)
#solution2:
def isUgly(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 0:
        return (False)
    for p in 2, 3, 5:
        while num % p == 0:
            num /= p
    return num == 1

#Single Number III
#solution1: Dictionary
def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    d = {}
    for n in nums:
        if n in d:
            del d[n]
        else:
            d[n] = 1
    return (list(d.keys()))
#solution2: Bit manipulation
def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    xor = 0
    a = 0
    b = 0
    for num in nums:
        xor ^= num
    mask = 1
    while (xor & mask == 0):
        mask = mask << 1
    for num in nums:
        if num & mask:
            a ^= num
        else:
            b ^= num
    return [a, b]

#Ugly Number II
#solution1:
def nthUglyNumber(self, n):
    """
    :type n: int
    :rtype: int
    """
    ls = [1]
    ls1 = []
    ls2 = []
    ls3 = []
    for _ in range(1, n):
        tmp = ls[-1]
        ls1.append(tmp * 2)
        ls2.append(tmp * 3)
        ls3.append(tmp * 5)
        mi = min(ls1[0], ls2[0], ls3[0])
        ls.append(mi)
        if ls1[0] == mi:
            ls1.pop(0)
        if ls2[0] == mi:
            ls2.pop(0)
        if ls3[0] == mi:
            ls3.pop(0)
    return (ls[-1])
#solution2: using only one list, but keep the index
def nthUglyNumber(self, n):
    """
    :type n: int
    :rtype: int
    """
    ugly = [1]
    i2, i3, i5 = 0, 0, 0
    while n > 1:
        u2, u3, u5 = 2 * ugly[i2], 3 * ugly[i3], 5 * ugly[i5]
        umin = min((u2, u3, u5))
        if umin == u2:
            i2 += 1
        if umin == u3:
            i3 += 1
        if umin == u5:
            i5 += 1
        ugly.append(umin)
        n -= 1
    return ugly[-1]

#Missing Number
#solution1:
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.append(None)
    i = 0
    j = len(nums)
    while (i < len(nums)):
        if (nums[i] == None):
            j = i
            i += 1
        elif (nums[i] == i):
            i += 1
        else:
            tmp = nums[nums[i]]
            nums[nums[i]] = nums[i]
            nums[i] = tmp
    return (j)
#solution2:
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    return (n * (n + 1) / 2 - sum(nums))
#solution3:
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    return sum(range(len(nums) + 1)) - sum(nums)

#Find Bad Version
#solution:
# @param version, an integer
# @return a bool
# def isBadVersion(version):
def firstBadVersion(self, n):
    """
    :type n: int
    :rtype: int
    """
    i, j = 1, n
    while (i < j):
        mid = (i + j) // 2
        if isBadVersion(mid):
            j = mid
        else:
            i = mid + 1
    return (i)

#Perfect Squares
#solution1: DP, time limit exceeded
def numSquares(self, n):
    """
    :type n: int
    :rtype: int
    """
    ls = [list(range(0, n + 1))]
    i = 2
    while (i ** 2 <= n):
        ls.append([0] * (n + 1))
        for j in range(1, n + 1):
            if j < i ** 2:
                ls[i - 1][j] = ls[i - 2][j]
            else:
                ls[i - 1][j] = min(ls[i - 2][j], ls[i - 1][j - (i ** 2)] + 1)
        i += 1
    return (ls[-1][-1])

#Move Zeroes
#solution:
def moveZeroes(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    if not nums:
        return
    i = 0
    while (i < len(nums)):
        if (not nums[i]):
            break
        else:
            i += 1
    j = i
    while (j < len(nums)):
        if nums[j]:
            break
        else:
            j += 1
    while (i < len(nums) and j < len(nums)):
        nums[i], nums[j] = nums[j], nums[i]
        while (i < len(nums)):
            if (not nums[i]):
                break
            else:
                i += 1
        while (j < len(nums)):
            if nums[j]:
                break
            else:
                j += 1
    return

#Nim Game
#solution1: DP, time limit exceeded
def canWinNim(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n <= 3:
        return (True)
    ls = [True, True, True]
    for i in xrange(3, n):
        if ls[i - 1] and ls[i - 2] and ls[i - 3]:
            ls.append(False)
        else:
            ls.append(True)
    return (ls[-1])
#solution2:
def canWinNim(self, n):
    """
    :type n: int
    :rtype: bool
    """
    return (bool(n & 3))

#Power of Three
#solution1:
def isPowerOfThree(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n <= 0:
        return (False)
    while (n != 1):
        if n % 3 == 0:
            n = n / 3
        else:
            return (False)
    return (True)
#solution2: recursive
def isPowerOfThree(self, n):
    """
    :type n: int
    :rtype: bool
    """
    def fun(n):
        if n == 1:
            return (True)
        else:
            a, b = divmod(n, 3)
            if b:
                return (False)
            return (fun(a))
    if n <= 0:
        return (False)
    return (fun(n))
#solution3: so smart!
def isPowerOfThree(self, n):
    """
    :type n: int
    :rtype: bool
    """
    return n > 0 and 1162261467 % n == 0

#Word Pattern
#solution:
def wordPattern(self, pattern, str):
    """
    :type pattern: str
    :type str: str
    :rtype: bool
    """
    ls = str.split()
    d = {}
    if len(ls) != len(pattern):
        return (False)
    for i in range(len(ls)):
        if pattern[i] in d:
            if d[pattern[i]] != ls[i]:
                return (False)
        else:
            d[pattern[i]] = ls[i]
    d = {}
    for i in range(len(ls)):
        if ls[i] in d:
            if d[ls[i]] != pattern[i]:
                return (False)
        else:
            d[ls[i]] = pattern[i]
    return (True)

#Longest Increasing Subsequence
#solution1: O(n^2)
def lengthOfLIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return (0)
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        ma = 1
        for j in range(i):
            if nums[j] < nums[i]:
                ma = max(ma, dp[j] + 1)
        dp[i] = ma
    return (max(dp))

#Maximum Product of Word Lengths
#solution1: time limit exceeded
def maxProduct(self, words):
    """
    :type words: List[str]
    :rtype: int
    """
    def fun(str1, str2):
        for s in str1:
            if s in str2:
                return (False)
        return (True)
    ma = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if fun(words[i], words[j]):
                ma = max(ma, len(words[i]) * len(words[j]))
    return (ma)
#solution2:
def maxProduct(self, words):
    """
    :type words: List[str]
    :rtype: int
    """
    sets = [set(x) for x in words]
    lens = [len(x) for x in words]
    ma = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if not (sets[i] & sets[j]):
                ma = max(ma, lens[i] * lens[j])
    return (ma)

#Odd Even Linked List
#solution1:
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def oddEvenList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    p = head
    if not p:
        return head
    q = head.next
    if not q:
        return head
    end = head
    while (end.next):
        end = end.next
    end.next = q
    p.next = q.next
    q.next = None
    end = end.next
    p = p.next
    while (p != q and p.next != q):
        qq = p.next
        end.next = qq
        p.next = qq.next
        qq.next = None
        end = end.next
        p = p.next
    return head
#solution2: more clear
def oddEvenList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    dummy1 = odd = ListNode(0)
    dummy2 = even = ListNode(0)
    while head:
        odd.next = head
        even.next = head.next
        odd = odd.next
        even = even.next
        head = head.next.next if even else None
    odd.next = dummy2.next
    return dummy1.next

#Number Complement
#solution:
def findComplement(self, num):
    """
    :type num: int
    :rtype: int
    """
    str = "{0:b}".format(num)
    str2 = "1" * len(str)
    return (int(str, 2) ^ int(str2, 2))

#Power of Four
#solution:
def isPowerOfFour(self, num):
    """
    :type num: int
    :rtype: bool
    """
    return (num in [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456,
            1073741824, 4294967296])

#Reverse String
#solution:
def reverseString(self, s):
    """
    :type s: str
    :rtype: str
    """
    return (s[::-1])

#Integer Break
#solution1: DP
def integerBreak(self, n):
    """
    :type n: int
    :rtype: int
    """
    dp = [1]
    for i in range(2, n + 1):
        left = 1
        right = i - 1
        ma = 1
        while (left <= right):
            ma = max(left * max(right, dp[right - 1]), ma)
            left += 1
            right -= 1
        dp.append(ma)
    return (dp[-1])

#Reverse Vowels of a String
#solution:
def reverseVowels(self, s):
    """
    :type s: str
    :rtype: str
    """
    ls1 = []
    ls2 = []
    for i, v in enumerate(s):
        if v in "aeiouAEIOU":   #be careful of this
            ls1.append(i)
            ls2.append(v)
    ls2.reverse()
    s = list(s)

    for i, v in enumerate(ls1):
        s[v] = ls2[i]
    return ("".join(s))

#Intersection of Two Arrays
#solution:
def intersection(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    return (list(set(nums1) & set(nums2)))

#Intersection of Two Arrays II
#solution:
def intersect(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    nums1.sort()
    nums2.sort()
    i = j = 0
    ls = []
    while (i < len(nums1) and j < len(nums2)):
        if nums1[i] == nums2[j]:
            ls.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1
    return (ls)

#Top K Frequent Elements
#solution:
def topKFrequent(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    d = {}
    for i in nums:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    ls = []
    for key, v in d.items():
        ls.append((v, key))
    ls.sort(reverse=True)
    return ([y for (x, y) in ls[:k]])

#Valid Perfect Square
#solution:
def isPerfectSquare(self, num):
    """
    :type num: int
    :rtype: bool
    """
    i, j = 1, num
    while (i < j):
        mid = (i + j) // 2
        if mid * mid == num:
            return (True)
        elif mid * mid > num:
            j = mid
        else:
            i = mid + 1
    if i == j:
        return (i * i == num)
    return (False)

#Guess Number Higher or Lower
#solution:
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):
def guessNumber(self, n):
    """
    :type n: int
    :rtype: int
    """
    i, j = 1, n
    while (i <= j):
        mid = (i + j) // 2
        tmp = guess(mid)
        if tmp == 0:
            return mid
        elif tmp > 0:
            i = mid + 1
        else:
            j = mid

#Hamming Distance
#solution1:
def hammingDistance(self, x, y):
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    tmp = x ^ y
    count = 0
    while (tmp):
        count += tmp & 1
        tmp = tmp >> 1
    return (count)
#solution2:
def hammingDistance(self, x, y):
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    x = x ^ y
    y = 0
    while x:
        y += 1
        x = x & (x - 1)
    return y

#Keyboard Row
#solution1:
def findWords(self, words):
    """
    :type words: List[str]
    :rtype: List[str]
    """
    def fun(word):
        tmp = 0
        if word[0] in "zxcvbnmZXCVBNM":
            tmp = 1
        elif word[0] in "asdfghjklASDFGHJKL":
            tmp = 2
        for w in word:
            if w in "zxcvbnmZXCVBNM":
                if tmp != 1:
                    return (False)
            elif w in "asdfghjklASDFGHJKL":
                if tmp != 2:
                    return (False)
            else:
                if tmp != 0:
                    return (False)
        return (True)
    return ([x for x in words if fun(x)])
#solution2: filter
def findWords(self, words):
    """
    :type words: List[str]
    :rtype: List[str]
    """
    row1 = set("qwertyuiopQWERTYUIOP")
    row2 = set("asdfghjklASDFGHJKL")
    row3 = set("ZXCVBNMzxcvbnm")
    return filter(lambda x: set(x).issubset(row1) or set(x).issubset(row2) or set(x).issubset(row3), words)

#Fizz Buzz
#solution1:
def fizzBuzz(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    ls = [str(x) for x in range(1, n + 1)]
    for x in range(3, n + 1, 3):
        ls[x - 1] = "Fizz"
    for x in range(5, n + 1, 5):
        ls[x - 1] = "Buzz"
    for x in range(15, n + 1, 15):
        ls[x - 1] = "FizzBuzz"
    return (ls)

#Next Greater Element
#solution:
def nextGreaterElement(self, findNums, nums):
    """
    :type findNums: List[int]
    :type nums: List[int]
    :rtype: List[int]
    """
    ls = [-1 for i in range(len(findNums))]
    for i, n in enumerate(findNums):
        j = nums.index(n) + 1
        while (j < len(nums)):
            if nums[j] > n:
                ls[i] = (nums[j])
                f = True
                break
            j += 1
    return (ls)

#Island Perimeter
#solution:
def islandPerimeter(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if not grid:
        return (0)
    if not grid[0]:
        return (0)
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]:
                if i - 1 < 0 or grid[i - 1][j] == 0:
                    count += 1
                if i + 1 >= len(grid) or grid[i + 1][j] == 0:
                    count += 1
                if j - 1 < 0 or grid[i][j - 1] == 0:
                    count += 1
                if j + 1 >= len(grid[0]) or grid[i][j + 1] == 0:
                    count += 1
    return (count)

#Max Consecutive Ones
#solution:
def findMaxConsecutiveOnes(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    ma = 0
    tmp = 0
    for i in nums:
        if i:
            tmp += 1
        else:
            ma = max(ma, tmp)
            tmp = 0
    ma = max(ma, tmp)
    return (ma)

#Complex Number Multiplication
#solution:
def complexNumberMultiply(self, a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    a1, a2 = a[0:-1].split('+')
    b1, b2 = b[0:-1].split('+')
    c1 = str(int(a1) * int(b1) - int(a2) * int(b2))
    c2 = str(int(a2) * int(b1) + int(b2) * int(a1))
    return (c1 + '+' + c2 + "i")

#Increasing Subsequences
#solution:
def findSubsequences(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    d = {()}
    for i in nums:
        d = d | {x + (i,) for x in d if (not x or i >= x[-1])}
    return ([x for x in d if len(x) >= 2])

#Elimination Game
#solution1:
def lastRemaining(self, n):
    """
    :type n: int
    :rtype: int
    """
    ls = list(range(1, n + 1))
    while (len(ls) > 1):
        ls = [ls[i] for i in range(1, len(ls), 2)]
        if len(ls) > 1:
            ls = [ls[i] for i in range(len(ls) % 2, len(ls), 2)]
        else:
            return (ls[0])
    return (ls[0])

#Find All Numbers Disappeared in an Array
#solution:
def findDisappearedNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    d = {}
    for i in range(1, len(nums) + 1):
        d[i] = 1
    for i in nums:
        if i in d:
            del d[i]
    return (d.keys())

#Longest Palindrome
#solution:
def longestPalindrome(self, s):
    """
    :type s: str
    :rtype: int
    """
    d = {}
    for i in s:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    count = 0
    length = 0
    for v in d.values():
        if v % 2 == 0:
            length += v
        else:
            length += v - 1
            count = 1
    return (length + count)

#Add Strings
#solution:
def addStrings(self, num1, num2):
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    if len(num1) > len(num2):
        num2 = "0" * (len(num1) - len(num2)) + num2
    else:
        num1 = "0" * (len(num2) - len(num1)) + num1
    res = ""
    tmp = 0
    i = len(num1) - 1
    while (i >= 0):
        this = int(num1[i]) + int(num2[i]) + tmp
        s, tmp = (this - 10, 1) if this >= 10 else (this, 0)
        res = str(s) + res
        i -= 1
    if tmp:
        res = "1" + res
    return (res)

#Perfect Number
#solution:
def checkPerfectNumber(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 1:
        return (False)
    s = 0
    for i in xrange(2, int(num ** 0.5) + 1):    #very important
        if num % i == 0:
            s += num // i
            if i != num // i:   #very important
                s += i
    return (s + 1 == num)

#Find All Duplicates in an Array
#solution1:
def findDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    d = {}
    for i in nums:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return ([k for k, v in d.items() if v == 2])
#solution2: O(1) space
def findDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    i = 0
    s = set()
    while (i < len(nums)):
        if nums[i] != i + 1:
            if nums[nums[i] - 1] == nums[i]:
                s.add(nums[i])
                i += 1
            else:
                tmp = nums[i]
                nums[i] = nums[nums[i] - 1]
                nums[tmp - 1] = tmp
        else:
            i += 1
    return (list(s))
#solution3: very smart!
def findDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    res = []
    for x in nums:
        if nums[abs(x) - 1] < 0:
            res.append(abs(x))
        else:
            nums[abs(x) - 1] *= -1
    return res

#Find the Difference
#solution:
def findTheDifference(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    d = {}
    for i in s:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    for i in t:
        if i in d:
            if d[i] == 1:
                del d[i]
            else:
                d[i] -= 1
        else:
            return (i)

#Minimum Moves to Equal Array Elements II
#solution:
def minMoves2(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    median = nums[len(nums) // 2]
    count = 0
    for i in nums:
        count += abs(median - i)
    return (count)

#Third Maximum Number
#solution:
def thirdMax(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums = list(set(nums))
    a = max(nums)
    nums.remove(a)
    if nums:
        b = max(nums)
        nums.remove(b)
        if nums:
            return (max(nums))
    return (a)

#Target Sum
#solution1: time limit exceeded
def findTargetSumWays(self, nums, S):
    """
    :type nums: List[int]
    :type S: int
    :rtype: int
    """
    def fun(nums, i, path, s, count):
        if i == len(nums):
            if path == s:
                count[0] += 1
            return
        fun(nums, i + 1, path + nums[i], s, count)
        fun(nums, i + 1, path - nums[i], s, count)
    count = [0]
    fun(nums, 0, 0, S, count)
    return (count[0])
#solution2: time limit exceeded
def findTargetSumWays(self, nums, S):
    """
    :type nums: List[int]
    :type S: int
    :rtype: int
    """
    ls = [sum(nums)]
    count = 1 if ls[0] == S else 0
    for i in nums:
        ls.extend([x - 2 * i for x in ls])
    return (ls.count(S))

#Range Sum Query - Immutable
#solution:
class NumArray(object):
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.accsum = [0]
        s = 0
        for i in nums:
            s += i
            self.accsum.append(s)
    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return (self.accsum[j + 1] - self.accsum[i])
        # Your NumArray object will be instantiated and called as such:
        # obj = NumArray(nums)
        # param_1 = obj.sumRange(i,j)

#Find the Duplicate Number
#solution:
def findDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    i, j = min(nums), max(nums)
    while (i < j):
        mid = (i + j) / 2.0
        c1 = c2 = 0
        for x in nums:
            if i <= x and x <= j:
                if x < mid:
                    c1 += 1
                elif x > mid:
                    c2 += 1
        if c1 > c2:
            j = int(mid)
        elif c1 < c2:
            i = int(math.ceil(mid))
        else:
            return (int(mid))
    return (i)

#Assign Cookies
#solution:
def findContentChildren(self, g, s):
    """
    :type g: List[int]
    :type s: List[int]
    :rtype: int
    """
    g.sort()
    s.sort()
    i = j = 0
    while (i < len(g) and j < len(s)):
        if g[i] <= s[j]:
            i += 1
            j += 1
        else:
            j += 1
    return (i)

#Counting Bits
#solution:
def countBits(self, num):
    """
    :type num: int
    :rtype: List[int]
    """
    dp = [0, 1]
    if num < 2:
        return (dp[:num + 1])
    i = j = 1
    while (i < num):
        while (j < i):
            dp.append(dp[i] + dp[j])
            if i + j == num:
                return (dp)
            j += 1
        j = 1
        i += i
        dp.append(1)
    return (dp)

#Number of Segments in a String
#solution:
def countSegments(self, s):
    """
    :type s: str
    :rtype: int
    """
    return (len(s.split()))

#Sum of Left Leaves
#solution:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def sumOfLeftLeaves(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    def fun(root, left, s):
        if not root:
            return
        if not root.left and not root.right:
            s[0] += root.val * left
        fun(root.left, 1, s)
        fun(root.right, 0, s)
    s = [0]
    fun(root, 0, s)
    return (s[0])

#Bulls and Cows
#solution:
def getHint(self, secret, guess):
    """
    :type secret: str
    :type guess: str
    :rtype: str
    """
    count = 0
    d = {}
    ls = []
    for i in range(len(secret)):
        if secret[i] == guess[i]:
            count += 1
        else:
            if secret[i] in d:
                d[secret[i]] += 1
            else:
                d[secret[i]] = 1
            ls.append(guess[i])
    count2 = 0
    for i in ls:
        if i in d:
            count2 += 1
            d[i] -= 1
            if not d[i]:
                del d[i]
    return (str(count) + 'A' + str(count2) + 'B')

#Integer Replacement
#solution1: time limit exceeded
def integerReplacement(self, n):
    """
    :type n: int
    :rtype: int
    """
    dp = [0, 0]
    i = 2
    while (i <= n):
        if i % 2 == 0:
            dp.append(dp[i // 2] + 1)
        else:
            dp.append(min(dp[i - 1], dp[(i + 1) // 2] + 1) + 1)
        i += 1
    return (dp[-1])
#solution2: recursive
def integerReplacement(self, n):
    """
    :type n: int
    :rtype: int
    """
    def fun(n):
        if n == 1:
            return (0)
        if n % 2 == 0:
            return (fun(n // 2) + 1)
        else:
            return (min(fun(n - 1), fun((n + 1) // 2) + 1) + 1)
    return (fun(n))

#First Unique Character in a String
#solution:
def firstUniqChar(self, s):
    """
    :type s: str
    :rtype: int
    """
    d = {}
    for i in s:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    for i, v in enumerate(s):
        if d[v] == 1:
            return (i)
    return (-1)

#Repeated Substring Pattern
#solution1:
def repeatedSubstringPattern(self, s):
    """
    :type s: str
    :rtype: bool
    """
    length = len(s)
    for i in range(length // 2):
        tmp = s[:i + 1]
        if length % (i + 1) == 0:
            if (tmp * (length // (i + 1)) == s):
                return (True)
    return (False)
#solution2: so smart!
def repeatedSubstringPattern(self, s):
    """
    :type s: str
    :rtype: bool
    """
    ss = (s * 2)[1:-1]
    return (s in ss)

#Minimum Moves to Equal Array Elements
#solution1.1:
def minMoves(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    mi = min(nums)
    count = 0
    for i in nums:
        count += i - mi
    return (count)
#solution1.2:
def minMoves(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    mi = nums[0]
    s = 0
    for i in nums:
        mi = min(mi, i)
        s += i
    return (s - mi * len(nums))
#solution1.3:
def minMoves(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    mi = nums[0]
    s = 0
    for i in nums:
        if i < mi:      #this can save time!
            mi = i
        s += i
    return (s - mi * len(nums))

#Path Sum III
#solution:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def pathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: int
    """
    def fun(root, ls, count, sum):
        if not root:
            return
        ls = [x + root.val for x in ls] + [root.val]
        for i in ls:
            if i == sum:
                count[0] += 1
        fun(root.left, ls, count, sum)
        fun(root.right, ls, count, sum)
    count = [0]
    fun(root, [], count, sum)
    return (count[0])

#Next Greater Element II
#solution1: time limit exceeded
def nextGreaterElements(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    length = len(nums)
    nums = nums + nums
    ls = []
    for i in range(length):
        ifFind = False
        for j in range(i + 1, i + length):
            if nums[j] > nums[i]:
                ls.append(nums[j])
                ifFind = True
                break
        if not ifFind:
            ls.append(-1)
    return (ls)
#solution2: stack
def nextGreaterElements(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    length = len(nums)
    stack = []
    ls = [-1] * length
    nums = nums + nums
    i = 0
    while (i < len(nums)):
        if not stack:
            stack.append((i, nums[i]))
            i += 1
        else:
            if stack[-1][1] < nums[i]:
                tmp = stack.pop()
                if tmp[0] < length:
                    ls[tmp[0]] = nums[i]
            else:
                stack.append((i, nums[i]))
                i += 1
    return (ls)

#Ransom Note
#solution1.1: Counter
def canConstruct(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    from collections import Counter
    c1 = Counter(ransomNote)
    c2 = Counter(magazine)
    for k, v in c1.items():
        if k not in c2 or v > c2[k]:
            return (False)
    return (True)
#solution1.2:
def canConstruct(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    from collections import Counter
    c1 = Counter(ransomNote)
    c2 = Counter(magazine)
    return (not c1 - c2)

#Can Place Flowers
#solution:
def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        if not n:
            return True
        for i in range(len(flowerbed)):
            if not flowerbed[i]:
                if (i==0 or flowerbed[i-1]==0) and (i==len(flowerbed)-1 or flowerbed[i+1]==0):
                    flowerbed[i] = 1
                    n-=1
                    if not n:
                        return True
        return False