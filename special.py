# 1. Two Sum
# use zip to sort two lists
# note: after sort, indexes are in wrong order; notice the order of return may be wrong
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        index = list(range(0,len(nums)))
        ls = sorted(zip(nums,index))
        new_index = [y for (x,y) in ls]
        new_nums = [x for (x,y) in ls]
        i = 0;
        j = len(new_nums)-1
        while(i<=j):
            if(new_nums[i]+new_nums[j] < target):
                i += 1
            elif(new_nums[i]+new_nums[j] > target):
                j -= 1
            else:                
                return([min(new_index[i],new_index[j]),max(new_index[i],new_index[j])])

# 605. Can Place Flowers
# note: use (or) and (or)
class Solution(object):
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