# 1. Two Sum
# use zip to sort two lists
# Note: after sort, indexes are in wrong order; notice the order of return may be wrong
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
