# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:30:59 2017

@author: Li Xiang
"""

num_polys = 5
num_sides = 6
num_colors = 3

c = num_colors
s = num_sides

#Wrong
#c*1 + (c**(s/2)-c)*2 + (c**s-c**(s/2))*s

c*1 + c*(c-1)*2 + (c**(s/2)-c)*s/2+(c**s-c**2-c**(s/2)+c)*s
