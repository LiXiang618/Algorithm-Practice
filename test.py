print("hello")
print(15 // 4)
print(15 % 4)

'doesn\'t'
"First Line.\nSecond Line."
print("First Line.\nSecond Line.")
print(r"First Line.\nSecond Line.")
print("""\
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""")
2*"Love"+"You"
"Love""You"
text = ('Put several strings within parentheses '
        'to have them joined together.')
text
word = "Python"
word[1:5]
word[:1]+word[1:]
word[-2:]
word[3:6]
word[2:111]+word[111:]
len(word)

squares = [1,4,9,16,25]
squares
4**3
squares + squares
squares.append(100)
squares
letters = ["a","b","c"]
len(letters)
letters = []

a,b = 0,1
while b< 10:
    print(b,end=",")
    a,b = b,a+b
print('\n')

x = 1
if(x<0):
    print("negative")
elif(x>0):
    print("positive")
else:
    print("0")

words = ['cat','dog','fish']
for w in words:
    print(w, len(w))

words.insert(0,"pet")
words.insert(len(words),"elephant")
words.append("fish")
words

range(5)
for i in range(5):
    print(i)
for i in range(0,10,3):
    print(i)

a = ['Mary','had','a','little','lamb']
for i in range(len(a)):
    print(i,a[i])

def fib(n):
    a,b = 0,1
    print(0,1,end=" ")
    for i in range(2,n):
        a,b = b, a+b
        print(b,end=" ")
    print()

fib(10)

for i in range(2,3):
    print(i)

def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)

def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    keys = sorted(keywords.keys())
    for kw in keys:
        print(kw, ":", keywords[kw])
cheeseshop("Limburger", "It's very runny, sir.",
           "It's really very, VERY runny, sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")

def concat(*args, sep="/"):
    return sep.join(args)
print(concat("a","b","c"))
sep = "*"
sep.join(["a","b","c"])

def parrot(voltage, state='a stiff', action='voom'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.", end=' ')
    print("E's", state, "!")
d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)

def make_incrementor(n):
    return lambda x: x + n
f = make_incrementor(42)
def make_incrementor2(n):
    return n
f = make_incrementor2(10)
f

pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair: pair[1])
pairs

a = list(range(10))
a = [1,2,3,4,5]
a.extend([1,2,3])

from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.append("Terry")
queue.popleft()

squares = list(map(lambda x: x**2, range(10)))
squares = [x**2 for x in range(10)]

[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
[(x,y) for x in range(100) for y in range(100) if x<y]

vec = [-4, -2, 0, 2, 4]
# create a new list with the values doubled
[x*2 for x in vec]
# filter the list to exclude negative numbers
[x for x in vec if x >= 0]
# apply a function to all the elements
[abs(x) for x in vec]
# call a method on each element
freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
[weapon.strip() for weapon in freshfruit]
# create a list of 2-tuples like (number, square)
[(x, x**2) for x in range(6)]

from math import pi
[str(round(pi, i)) for i in range(1, 6)]

matrix = [[1,2,3,4],
          [5,6,7,8],
          [9,10,11,12]]
[[row[i] for row in matrix] for i in range(4)]

list(zip(*matrix))

a = [1, 66.25, 333, 333, 1234.5]
del a[2:4]

t = [1,2,3,4,5]
t = [1,2,3,4,[1,2]]
t = 1,2,3,4,5
x,y,z,a,b = t

basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
a = set('abracadabra')
b = set('alacazam')
a
b
a - b
a | b
a & b
a ^ b
a = {x for x in 'abracadabra' if x not in 'abc'}
a

tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
list(tel.keys())
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
{x: x**2 for x in (2, 4, 6)}
dict(sape=4139, guido=4127, jack=4098)

knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k,v)
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i,v)
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    print('What is your {0}? It is {1}.'.format(q,a))

list(reversed(range(10)))
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
    print(f)

import math
raw_data = [56.2, float('NaN'), 51.7, 55.3, 52.5, float('NaN'), 47.8]
filtered_data = []
for value in raw_data:
    if not math.isnan(value):
        filtered_data.append(value)

string1, string2, string3 = '', 'Trondheim', 'Hammer Dance'
non_null = string1 or string2 or string3

import fibo
fibo.fib(100)
fibo.fib2(100)
from fibo import fib, fib2
from fibo import *

#python fibo.py 50

import sys
sys.path

import fibo, sys
dir(fibo)
dir(sys)

import builtins
dir(builtins)

str(1/7)
repr(1/7)

for x in range(1,11):
    print(repr(x).rjust(2),repr(x*x).rjust(3),end=' ')
    print(repr(x*x*x).rjust(4))
for x in range(1,11):
    print(repr(x).ljust(2),repr(x*x).ljust(3),end=' ')
    print(repr(x*x*x).ljust(4))
for x in range(1,11):
    print(repr(x).center(4),repr(x*x).center(4),end=' ')
    print(repr(x*x*x).center(4))
for x in range(1,11):
    print(repr(x).zfill(2),repr(x*x).zfill(3),end=' ')
    print(repr(x*x*x).zfill(4))

for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x * x, x * x * x))

print('We are the {} who say "{}!"'.format('knights', 'Ni'))
print('{1} and {0}'.format('spam', 'eggs'))
print('This {food} is {adjective}.'.format(food='spam', adjective='absolutely horrible'))
print('The story of {0}, {1}, and {other}.'.format('Bill', 'Manfred',other='Georg'))

#'!a' (apply ascii()), '!s' (apply str()) and '!r' (apply repr()) can be used to convert the value before it is formatted:
contents = 'eels'
print('My hovercraft is full of {}.'.format(contents))
print('My hovercraft is full of {!r}.'.format(contents))

import math
print('The value of PI is approximately {0:.13f}.'.format(math.pi))

table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
for name, phone in table.items():
    print('{0:10} ==> {1:10d}'.format(name, phone))

table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
print('Jack: {0[Jack]:d}; Sjoerd: {0[Sjoerd]:d}; ''Dcab: {0[Dcab]:d}'.format(table))
print('Jack: {Jack:d}; Sjoerd: {Sjoerd:d}; Dcab: {Dcab:d}'.format(**table))

import math
print('The value of PI is approximately %5.3f.' % math.pi)

f = open('file','r+')
f.read(5)
f.read()
f.read()
f.close()
f = open('file','r+')
f.readline()
f.readline()
f.readline()
f.close()
f = open('file','r+')
for line in f:
    print(line)
    f.close()
f = open('file','r+')
list(f)
f.close()
f = open('file','r+')
f.readlines()
f.close()

f = open('file','r+')
f.write("this is a test\n")
f.close()

f = open('file','a')
value = ('the answer',42)
s = str(value)
f.write(s)
f.tell()
f.close()

f = open('file','r')
f.tell()
f.close()
f = open('file','a')
f.tell()
f.close()

f = open("file",'rb+')
f.write(b'0123456789abcdef')
f.seek(5)
f.read(1)
f.seek(-2,2)
f.read(1)
f.close()

with open('file', 'r') as f:
    read_data = f.read()
    print(read_data)

import json
json.dumps([1,'simple','list'])
f = open('file','w')
json.dump([1,'simple','list'],f)
f.close()
f = open('file')
x = json.load(f)
f.close()

print(*[1,2,3])

[x,y,z,n] = [1,1,1,2]
ls = [[i,j,k] for i in range(0,x+1) for j in range(0,y+1) for k in range(0,z+1) if i+j+k!=n]
print(ls)

a = [1,2,3,4,5,6,5,4,3,2]
index = [i for i,j in enumerate(a) if j>3]
[a[i] for i in index]

x = "adfsdfDSDFS"
print("".join([e.lower() if e.isupper() else e.upper() for e in x ]))
x.swapcase()

sub = "abc"
s = "adsabcadaabcabc"
counter = 0
while sub in s:
    i = s.find(sub)
    s = s[:i]+s[i+1:]
    counter+=1

any(c.isupper() for c in "AsafAsdfa")
all(c.isupper() for c in "ADADF")
[i for i,j in enumerate("asdASDe") if j.isupper() ]

S, N = "abcdef", 3
for part in zip(*[iter(S)] * N):
    print(part)

for part in zip(iter(S),iter(S),iter(S)):
    print(part)

it = iter(S)
for part in zip(*[it],*[it],*[it]):
    print(part)

it1 = iter(S)
it2 = iter(S)
it3 = iter(S)
for part in zip(*[it1],*[it2],*[it3]):
    print(part)

sentinel = object()
iterators = [iter(ii) for ii in [it,it,it]]
while iterators:
    res = []
    for it in iterators:
        elem = next(it, sentinel)
        if elem is sentinel:
            break
        res.append(elem)
    print(res)

def zip2(*iterables):
    # zip('ABCD', 'xy') --> Ax By
    print("function begins")
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            print(elem)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)

S, N = "abcdef", 3
for part in zip2(*[iter(S)] * N):
    print("hello")

it = iter(S)
iterators = [it,it,it]
for it in iterators:
    elem  = next(it)
    print(elem)
    print(it)

it1 = iter(S)
it2 = iter(S)
it3 = iter(S)
iterators2 = [it1,it2,it3]
for it in iterators2:
    elem = next(it)
    print(elem)
    print(it)

a = set()
a.add(1)
a.update([2,3])
a.discard(0)
a.remove(0)
b = set([1,2,5])
a.union(b)
a.intersection(b)
a.difference(b)

eval("{0}+{1}".format(*[2,4,3]))
eval("{0}+{1}".format(*[2]))

a = complex("4+4j")
import cmath
cmath.polar(a)[1]
cmath.phase(a)
a = complex(5,5)
cmath.phase(a)/cmath.phase(complex(-1,0))*180

''.join(["1","2","3"])

divmod(177,10)
177//10
177%10

pow(2,4,11)

from itertools import product
list(product([1,2,3],repeat=3))
list(product([1,2,3],[1,2],[1,2,3]))
from itertools import permutations
list(permutations([1,4,3],2))
from itertools import combinations
list(combinations([1,2,3,4],2))
from itertools import combinations_with_replacement
list(combinations_with_replacement([1,2,3,4],2))
sorted([3,2,4,2])

from itertools import groupby
groups = []
uniquekeys = []
data = "asdfavseassssdasssdddd"
#data = sorted(data)
for k, g in groupby(data):
    groups.append(list(g))      # Store group iterator as a list
    uniquekeys.append(k)

gg = [k for k,g in groupby(data)]
kk = [list(g) for k,g in groupby(data)]

aa = [1,2,3]
bb = [1,2,3]
list(zip(aa,bb))

from collections import Counter
myList = [1,1,1,2,2,3,4,4,4,4,4]
print(Counter(myList))
print(Counter(myList).items())
print(Counter(myList).keys())
print(Counter(myList).values())
string = "assvcasdfbbbfffa"
d = Counter(string)
d = sorted(d.items())
d = sorted(d,key = lambda x:x[1],reverse = True)
for k,v in d:
    print(k,v)

myListC = Counter(myList)
myListC[2] +=1
myListC

from collections import defaultdict
d = defaultdict(list)
d['python'].append("awesome")
d['something-else'].append("not relevant")
d['python'].append("language")
if d['python']:
    print("yes")
for i in d.items():
    print(i)

d = {}
d['python'] = ["awesome"]
d["python"] = [*d["python"],"other"]

a = [1,2,3,4,5,3,2]

from collections import namedtuple
Point = namedtuple('Point','x,y')
pt1 = Point(2,3)
pt2 = Point(20,30)
pt1.x*pt2.x+pt1.y*pt2.y
print(pt1)

print("%.2f" % 0.1234)
print("")
print('{0:2d} {1:3d} {2:4d}'.format(1,2,3))
print('{0:.2f}'.format(0.1234))

["asdf a    asf ".split()]
a = [1,2,3,4,3]
a.index(2)

ordinary_dict = {}
ordinary_dict['b'] = 2
ordinary_dict['a'] = 3
from collections import OrderedDict
ordered_dict = OrderedDict()
ordered_dict['b'] = 2
ordered_dict['a'] = 3
ordered_dict['a'] = 1
ordered_dict.get('c',0)
ordered_dict.get('a',0)
ordered_dict.keys()
ordered_dict.values()
for k,v in ordered_dict.items():
    print(k,v)

string = "abc edc 123"
[int(x) for x in string.split() if x.isdigit()][0]
[x for x in string.split() if not x.isdigit()]

string.rpartition(" ")

from collections import deque
d = deque()
d.append(1)
d.appendleft(2)
d.extend([1,2,3,4])
d.extendleft([1,2,3])
d.rotate(3)
d.rotate(-3)
print(*d)

a = ["3"]
a.append("")
a

a = 1
print ("yes" if a==1 else "no")

import calendar
print (calendar.TextCalendar(firstweekday=0).formatyear(2015))
calendar.weekday(2015,8,5)
list(calendar.day_name)

#https://docs.python.org/2/library/datetime.html
from datetime import datetime as dt
fmt = '%a %d %b %Y %H:%M:%S %z'
print(int(abs((dt.strptime("Sun 10 May 2015 13:54:36 -0700", fmt)
         - dt.strptime("Sun 10 May 2015 13:54:36 +0000", fmt)).total_seconds())))

try:
    print(1/0)
except ZeroDivisionError as e:
    print("Error Code: ",e)

import re
re.compile(".*\+")

from math import pow


class complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return complex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return complex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        return complex(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)

    def __div__(self, other):
        try:
            return self.__mul__(complex(other.real, -1 * other.imag)).__mul__(complex(1.0 / (other.mod().real) ** 2, 0))
        except ZeroDivisionError as e:
            print(e)
            return None

    def mod(self):
        return complex(pow(self.real ** 2 + self.imag ** 2, 0.5), 0)

    def __str__(self, precision=2):
        return str(("%." + "%df" % precision) % float(self.real)) + ('+' if self.imag >= 0 else '-') + str(
            ("%." + "%df" % precision) % float(abs(self.imag))) + 'i'

A = complex(3,2)
B = complex(5,4)
print(A+B)

A = [1,2,3]
B = [4,5,6]
C = [7,8,9]
X = A+B+C
X = [A]+[B]+[C]
list(zip(*X))
print(*zip(*X))

X = []
X = X + [A]
X = X + [B]

import math
math.factorial(1000)

a = [1,2,3,4,3,2]
a.pop(1)
a.remove(2)
a.index(3,0,4)

list(map(int,["1","2"]))
a = [[1,2,3,4,5],[2,4,3,2,3],[4,3,4,5,6],[1,2,2,1,1]]
sorted(a,key = lambda x: x[2])
sorted(a)

from operator import itemgetter, attrgetter, methodcaller
a = [[1,2,3,4,5],[2,4,3,2,3],[4,3,4,5,6],[1,2,2,1,1]]
sorted(a,key = itemgetter(1,2))

X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]
[x for (y,x) in sorted(zip(Y,X))]
X[::-1]

s = "asEd23fD12"
s[::-1]
sorted(s,key = lambda x :x)
sorted(s,key = lambda x :str(x.isdigit()*(3+(x in "24680"))+x.isupper()*2+x.islower())+x)
sorted(s,key=lambda c: (c.isdigit() - c.islower(), c in '02468', c))


[int(x) for x in "12 3 -3 -22".split()]

x = "1"
y = "2"
x+y

l = list(range(10))
l = list(map(lambda x:x*x,l))
l = list(filter(lambda x: x>10 and x<80,l))

x = "abc@def.com"
x.count("@")
x.index("@")
all(map(lambda x: x in "abc",list(x.partition("@")[0])))
x.rpartition(".")[0].rpartition("@")[-1]
x.isalnum()

#ptrn = re.compile("^[a-zA-Z][\w-]*@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}$")

import re
print(bool(re.search(r"ly","similarly")))
print(bool(re.match(r"ly","similarly")))
print(bool(re.match(r"ly","lysimilarly")))

#-.2 +0.2 0.3
bool(re.match("^[\+-]?([0-9])*\.([0-9])+$" ,".1"))
# \d

s = re.split("-","+86-32-323-333")
list(map(lambda x: x.isdigit(),    re.split("[,\.]",".100,000,00.,000,")  ))
[x for x in s if x.isdigit()]

import re
m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
m.group(0,1,2,3,0,1,2,3)
m.groups()
m = re.match(r'(?P<user>\w+)@(?P<website>\w+)\.(?P<extension>\w+)','myname@hackerrank.com')
m.groupdict()

import re
re.findall(r'c(?=o)',"chocolate")
re.finditer(r'c(?=o)',"chocolate")
[(m.start(0), m.end(0)) for m in re.finditer(r'c(?=o)',"chocolate")]
[(m.start(0), m.end(0)) for m in re.finditer(r'c(?!o)',"chocolate")]
[(m.start(0), m.end(0)) for m in re.finditer(r'(?<=[a-z])[aeiou]',"he1o")]
[(m.start(0), m.end(0)) for m in re.finditer(r'(?<![a-z])[aeiou]',"he1o")]
[(m.start(0), m.end(0)) for m in re.finditer(r'(\w)(\w)y\2\1',"layal")]
[(m.start(0), m.end(0)) for m in re.finditer(r'([a-zA-Z0-9])(?=.*(\1){2,})',"..12345678910111213141516171820212223")]
[(m.start(0), m.end(0)) for m in re.finditer(r'([a-zA-Z0-9])\1+',"..12345678910111213141516171820212223")]
re.findall(r'([a-zA-Z0-9])(?=.*\1{2,})',"..12345678910111213141516171820212223")
re.findall(r'([a-zA-Z0-9])\1+',"..12345678910111213141516171820212223")

m = re.search(r'([a-zA-Z0-9])\1+', "12345678910111213141516171820212223")
m.groups()
m.group(0)
m.group(1)
print(m.group(1) if m else -1)

list(map(lambda x: x.group(),re.finditer(r'\w','http://www.hackerrank.com/')))
#[^a-z] not from a to z
re.findall(r"(?<=[^aeiouAEIOU])[aeiouAEIOU]{2,}(?=[^aeiouAEIOU])","rabcdeefgyYhFjkIoomnpOeorteeeeet")
c = "aeiouAEIOU"
re.findall(r"(?<=[^%s])[%s]{2,}(?=[^%s])" % (c,c,c),"rabcdeefgyYhFjkIoomnpOeorteeeeet")
re.findall(r"%s" % (c),"aeiouAEIOUasfafaf")

m = re.search(r'\d+',"1234,123")
m.start()
m.end()

k = "aa"
s = "aaadaa"
a = [(m.start(),m.end()) for m in re.finditer(r'aa',s)]
re.findall(k,s)
m = re.search(k,s)
bool(m)
s[6:]

ifprint = False
p = 0
while p < len(s):
    m = re.search(k, s[p:])
    if bool(m):
        print("({0}, {1})".format(m.start(), m.end() - 1))
        p = m.start() + 1
        ifprint = True
    else:
        break

if ifprint == False:
    print("(-1, -1)")

import re
def square(match):
    number = int(match.group(0))
    return str(number**2)

re.sub(r"\d+",square,"1 2 3 4 5 6 7 8 9 20")

html = """
<head>
<title>HTML</title>
</head>
<object type="application/x-flash"
  data="your-file.swf"
  width="0" height="0">
  <!-- <param name="movie"  value="your-file.swf" /> -->
  <param name="quality" value="high"/>
</object>
"""

re.sub("<!--.*?-->", "", html) #remove comment

re.sub("( && )|( \|\| )"," and ","asfnklhaosf ||| dafs &&sadfasf&")

string = "x&& &&& && && x || | ||\|| x"
re.sub(" && "," and ",string)
#x&& &&& and and x or | ||\|| x

import re
bool(re.match("(7|8|9)\d{9}","9F21300000"))

import email.utils
print(email.utils.parseaddr('DOSHI <DOSHI@hackerrank.com>'))
print(email.utils.formataddr(('DOSHI', 'DOSHI@hackerrank.com')))

bool(re.match("<[a-zA-Z][a-zA-Z0-9.-_]+@[a-zA-Z]+\.[a-zA-Z]{1,3}>", email))
bool(re.match("[.-_]","@"))
bool(re.match("[._-]","@"))

m = re.findall(r"(?<!^)#(?:[0-9a-fA-F]{3}){1,2}","#ddasdf #adeaaaasdff ")
m

from html.parser import HTMLParser
# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Found a start tag  :", tag)
        print(attrs)
    def handle_endtag(self, tag):
        print("Found an end tag   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Found an empty tag :", tag)
    def handle_comment(self, data):
        print("Comment  :", data)
    def handle_data(self, data):
        print("Data     :", data)

# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
parser.feed("<body data-modal-target class='1'><h1>HackerRank</h1><br /></body></html>")

html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'

s = "1312"
r = re.match(r"^.*([0-9]).*\1.*$",s)
bool(r)

print('\1')
print(r'\1')

s = "B1CDEF2354"
print('Valid'
      if all([re.search(r, s) for r in [r'[A-Za-z0-9]{10}',r'([A-Z].*){2}',r'([0-9].*){3}']])
         and not re.search(r'.*(.).*\1', s)
      else 'Invalid')

string = ""
ss = ["1234","1234","1234"]
list(zip(*ss))
for i in list(zip(*ss)):
    string += "".join(i)

raw_xml ='''
<feed xml:lang='en'>
    <title>HackerRank</title>
    <subtitle lang='en'>Programming challenges</subtitle>
    <link rel='alternate' type='text/html' href='http://hackerrank.com/'/>
    <updated>2013-12-25T12:00:00</updated>
</feed>
'''

xml = re.sub(r"\n"," ",raw_xml)

import xml.etree.ElementTree as etree
tree = etree.ElementTree(etree.fromstring(xml))
def traverse(node):
  return len(node.attrib) + sum(traverse(child) for child in node)
print(traverse(tree.getroot()))

import re
raw_xml = """
<data>
    <country name="Liechtenstein" size = "100">
        <rank>1</rank>
        <year>2008</year>
        <gdppc>141100</gdppc>
        <neighbor name="Austria" direction="E"/>
        <neighbor name="Switzerland" direction="W"/>
    </country>
    <country name="Singapore">
        <rank>4</rank>
        <year>2011</year>
        <gdppc>59900</gdppc>
        <neighbor name="Malaysia" direction="N"/>
    </country>
    <country name="Panama">
        <rank>68</rank>
        <year>2011</year>
        <gdppc>13600</gdppc>
        <neighbor name="Costa Rica" direction="W"/>
        <neighbor name="Colombia" direction="E"/>
    </country>
</data>
"""
xml = re.sub(r"\n"," ",raw_xml)
import xml.etree.ElementTree as ET
#tree = ET.parse('country_data.xml') parse from file
#root = tree.getroot() parse from file
root = ET.fromstring(xml)
root.tag
root.attrib
for child in root:
    print(child.tag, child.attrib)

root[0][1].text
for child in root:
    print(child.tag,child.attrib,child.text)
    for c in child:
        print(c.tag,c.attrib,c.text)

root[0].attrib
def attnum(node):
    return(len(node.attrib)+sum([attnum(x) for x in node]))
print(attnum(root))

max([1,2,3]+[1])
def max_depth(node):
    return(1+max([max_depth(x) for x in node]+[0]))
print(max_depth(root)-1)

def print_integers(values):
    def is_integer(value):
        try:
            return value == int(value)
        except:
            return False
    for v in values:
        if is_integer(v):
            print(v)

print_integers([1,2,3,"4", "parrot", 3.14])

def foo(x):
    return x * 2;
def bar(f):
    return lambda x: f(x) + 1

foobar = bar(foo)
foobar(5)

l = ["1","2","3"]

def wrapper(f):
    def func(ll):
        f([c+"d"for c in ll])
    return func

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

sort_phone(l)

x = [
 ['4', '21', '1', '14', '2008-10-24 15:42:58'],
 ['3', '22', '4', '2somename', '2008-10-24 15:22:03'],
 ['5', '21', '3', '19', '2008-10-24 15:45:45'],
 ['6', '21', '1', '1somename', '2008-10-24 15:45:49'],
 ['7', '22', '3', '2somename', '2008-10-24 15:45:51']
]

from operator import itemgetter

x.sort(key=itemgetter(1))

import numpy
a = numpy.array([1,2,3,4,5])
b = numpy.array([1,2,3,4,5],float)
print(b)

d1 = numpy.array([1,2,3,4,5])
print(d1.shape)
d2 = numpy.array([[1,2],[2,3],[3,4]])
print(d2.shape)
d2.shape = (2,3)
print(d2)
d2.shape = (6,1)
print(d2)
c = numpy.array([1,2,3,4,5,6])
c.shape = (2,3)
print(c)
numpy.reshape(c,(3,2))

my_array = numpy.array([[1,2,3],[4,5,6]])
print(numpy.transpose(my_array))
print(my_array.flatten())
numpy.transpose(my_array).flatten()

a = numpy.array([1,2,3])
b = numpy.array([4,5,6])
c = numpy.array([7,8,9])
print(numpy.concatenate((a,b,c)))

array1 = numpy.array([[1,2,3],[1,2,3]])
array2 = numpy.array([[4,5,6],[7,8,9]])
print(numpy.concatenate((array1,array2),axis = 1))

print(numpy.zeros((1,3)))
print(numpy.zeros((1,2),dtype=numpy.int))
print(numpy.ones((1,3)))
print(numpy.ones((1,2),dtype=numpy.int))

print(numpy.zeros(tuple([1,3])))

print(numpy.identity(3))
print(numpy.eye(8,7,k=1))
print(numpy.eye(8,7,k=-2))

my_array = numpy.array([1.1,2.2,3.3,4.4,5.5,6.6])
print(numpy.floor(my_array))
print(numpy.ceil(my_array))
print(numpy.rint(my_array))

my_array = numpy.array([[1,2],[3,4]])
print(numpy.sum(my_array,axis = 0))
print(numpy.sum(my_array,axis = 1))
print(numpy.sum(my_array,axis = None))
print(numpy.sum(my_array))
print(numpy.prod(my_array,axis = 0))
print(numpy.prod(my_array,axis = 1))
print(numpy.prod(my_array,axis = None))
print(numpy.prod(my_array))

my_array = numpy.array([[2,5],[3,7],[1,3],[4,0]])
print(numpy.min(my_array,axis = 0))
print(numpy.min(my_array,axis = 1))
print(numpy.min(my_array,axis = None))
print(numpy.min(my_array))

my_array = numpy.array([[1,2],[3,4]])
print(numpy.mean(my_array,axis = 0))
print(numpy.mean(my_array,axis = 1))
print(numpy.mean(my_array,axis = None))
print(numpy.mean(my_array))

print(numpy.var(my_array,axis = 0))
print(numpy.var(my_array,axis = 1))
print(numpy.var(my_array,axis = None))
print(numpy.var(my_array))

print(numpy.std(my_array,axis = 0))
print(numpy.std(my_array,axis = 1))
print(numpy.std(my_array,axis = None))
print(numpy.std(my_array))

A = numpy.array([ 1, 2 ])
B = numpy.array([ 3, 4 ])
print(numpy.dot(A,B))
print(numpy.cross(A,B))

A = numpy.array([[1,2],[3,4]])
B = numpy.array([[1,2],[3,4]])
A[0,:]
A[:,0]
numpy.dot(A,B)

print(numpy.inner(A,B))
print(numpy.outer(A,B))

print(numpy.poly([-1,1,1,10]))
print(numpy.poly([1,1]))
print(numpy.roots([1,0,-1]))
print(numpy.polyint([1,1,1]))#find the integration
print(numpy.polyder([1,1,1,1]))#find the derivative
print(numpy.polyval([1,-2,0,2],4))
print(numpy.polyfit([0,1,-1,2,-2],[0,1,1,4,4],2))#fit the value with specific order

p1 = numpy.poly1d([1, 2])
p2 = numpy.poly1d([9, 5, 4])
print(p1)
print(p2)
print(numpy.polyadd(p1,p2))
print(numpy.polysub(p1,p2))
print(numpy.polymul(p1,p2))
print(numpy.polydiv(p1,p2))

#https://docs.scipy.org/doc/numpy/reference/routines.linalg.html
print(numpy.linalg.det([[1 , 2], [2, 1]]))
vals, vecs = numpy.linalg.eig([[1 , 2], [2, 1]])
print(vals)
print(vecs)
print(numpy.linalg.inv([[1 , 2], [2, 1]]))

#sort two associated arraies
a = ['f','e','b','d','a']
b = list(range(0,len(a)))
aa = a
bb = b
ls1 = []
ls2 = []
while(len(aa)>0):
    ma = min(aa)
    for i in range(len(aa)):
        if(aa[i]==ma):
            ls1.append(aa.pop(i))
            ls2.append(bb.pop(i))
            break
ls1
ls2

a = [3,2,5]
b = [3,4,2]
[i>j for i,j in zip(a,b)]

format(1,'.3f')
a = [1,2,5,3,2,5,6]
[(i,j) for i,j in enumerate(a)]
[ i if(j>3) else 0 for i,j in enumerate(a)]
[ i for i,j in enumerate(a) if j>3]

if(2<1 and 0/0==0):
    print("Here")

d = {}
d[1] = []+["1"]
d[1] = d[1] + ["2"]

ord('a')
ord('a')-ord("A")
ord('A')
ord('a')
ord('b')

def myfun(x):
    if(x<10):
        return x
    else:
        return(myfun(sum([int(i) for i in str(x)])))
myfun(30866)
myfun(31234)
myfun(211)
30866%9
31234%9
211%9

int(3.6)#round down
int(3.2)+1 if (int(3.2)!=3.2) else int(3.2)

a = [0,None]
a = [1,2,3,3,3,32,2]
a.count(3)


carry, val = divmod(13,5)
carry
val

c = 1
c *= 2

#!/bin/python3

import sys
n,m = input().strip().split(' ')
n,m = [int(n),int(m)]
a = [int(a_temp) for a_temp in input().strip().split(' ')]
b = [int(b_temp) for b_temp in input().strip().split(' ')]
aa = a
c = 1
f = False
#lcm
while(max(aa)!=1):
    for i in range(2,max(aa)+1):
        if(any([x%i == 0 for x in aa])):
            aa = [int(x/i) if x%i == 0 else int(x) for x in aa]
            c *= i
            break
bb = b
d = 1
f = False
#gcd
while(1 not in bb):
    for i in range(2,min(bb)+1):
        if(all([x%i == 0 for x in bb])):
            bb = [int(x/i) for x in bb]
            d *= i
            break
        if(i==min(bb)):
            f = True
    if(f):
        break
i = 1
counter = 0
while(c*i<=d):
    if(d%(c*i) == 0):
        counter +=1
    i += 1
print(counter)

string = "123455"
string.find("5")

nums1 = [1,4,5,6,7,10]
nums2 = [2,3,3,4,5,7,9,9,9,9]
nums = []
i = j = 0
while(i<len(nums1) and j<len(nums2)):
    if(nums1[i]<=nums2[j]):
        nums.append(nums1[i])
        i += 1
    else:
        nums.append(nums2[j])
        j += 1
if(i<len(nums1)):
    nums += nums1[i:]
elif(j<len(nums2)):
    nums += nums2[j:]
nums

aa = [0,1,2,3]
bb = [0,1,2,3,4,5]
length = 2
aa[length:]
bb[0:-length]

s = "123"
s[2:2]
s[::-1]

ls = [['1','a'] for i in range(6)]
ls = [''.join(x) for x in ls]
ls
''.join([''.join(x) for x in ls])

import re
s = "   +234"
a = re.search(r"^ *[-+]?[0-9]+",s)
bool(a)
a.group()

s.strip()
s

n = 123456789
[(n%(10**(j+1)))//(10**j) for j in range(0,5)]

True + True

any([True,True])
a = "12344"
a[6:]
a[6]

[[0]*10]*5

d = 4
c = [-1,5,-3,3,2]
x = 1/2
y = c[d]
for i in range(d-1,-1,-1):
    y = y*x+c[i]
y

for i in range(10,0,-1):
    print(i)

for i in range(-1,-1,-1):
    print(i)
max(1,2,3,4)

ls = [(1,2),(3,4),(3,3)]
for (a,b) in ls:
    print(a,b)

divmod(0.7,1/2)
str2 = string[::-1]
num = 0
for i in range(len(str2)):
    num += 2**(len(str2)-i-1)*int(str2[i])
print(num)

def dtob1(n):
    string = ""
    while n>0:
        n,s = divmod(n,2)
        string += str(s)
    return(string[::-1])
dtob1(53)

def dtob2(n,precision=10):
    d = {}
    s = ""
    if(n*2>1):
        n = round(n*2-1,precision)
        s = "1"
    elif(n*2==1):
        return("1")
    else:
        n = round(n*2,precision)
        s = "0"
    string = s
    for _ in range(1,precision):
        if(n*2>1):
            n = round(n*2-1,precision)
            s = "1"
        elif(n*2==1):
            string += "1"
            break
        else:
            n = round(n*2,precision)
            s = "0"
        if(n in d):
            string = string[:d[n]]+"|" + string[d[n]:_] + "|"
            break
        else:
            d[n] = _
            string += s
    return(string)
dtob2(0.7,10)

def dtob(n,precision=10):
    a = int(n)
    b = n-int(n)
    return(dtob1(a)+"."+dtob2(b,precision))
dtob(53.7)
dtob(11.25)
dtob(2/3)
dtob(3/5)
dtob(3.2)
dtob(30.6)
dtob(99.9)

import math
math.atan(1)*4
math.pi
math.e
dtob(math.pi,15)
dtob(math.e,15)

s = "0b01001101011001010010110000100000011000010110111001100100001000000111100101101111011101010010110000100000011010000110000101110110011001010010000001110100011011110110111100100000011011010111010101100011011010000010000001100110011100100110010101100101001000000111010001101001011011010110010100101110"
len(s)
n = int(s, 2)
n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

bin(int.from_bytes("Let's begin a funny game!".encode(), 'big'))

1001100011001010111010000100111011100110010000001100010011001010110011101101001011011100010000001100001001000000110011001110101011011100110111001111001001000000110011101100001011011010110010100100001

s = set([(1,2,3),(1,2,3),(2,3,4)])
[[i,j,k] for (i,j,k) in s]

a = ["a","b","c"]
b = ["1","2","3"]
[i+j for i in a for j in b]

ls = []
ls.pop()

def fun(f,a,b,tol):
    if(f(a)*f(b)>0):
        return("f(a)f(b) not satisfied!")
    while((b-a)/2>tol):
        c = (a+b)/2
        if(f(c)==0):
            return(c)
        elif(f(a)*f(c)<0):
            b = c
        else:
            a = c
    return((a+b)/2)

f = lambda x: x**2-2
fun(f,0,2,0.5*10**-10)

for s in []:
    print(s)

list(set(["()","()","e"]))

n = [12,32,1,3,1]
n.index(min(n))

from heapq import heappush,heappop
def heapsort(iterable):
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]
heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])

h = [(4,"1"),(3,"3"),(5,"2"),(3,"3")]
from heapq import heapify
heapify(h)

#a = b = 0
#if(a=a+1 and b = b+2):
#    print(a,b)
i = 1
for i in range(10):
    i+=1
    print(i)

ls = [1,2,3,4,5,6,7,8,9]
i = 0
while(i<len(ls)):
    print("here")
    if(ls[i]%2==1):
        del ls[i]
    i+=1

ls = [1,2,3,3,3,4,4,4,5,5]
def fun(ls):
    i = 1
    last = ls[0]
    while(i<len(ls)):
        if(ls[i]==last):
            del ls[i]
        else:
            i+=1
            last = ls[i]
    return(ls)

ls
fun(ls)
ls

def fun2(ls):
    for i in range(len(ls)):
        ls[i] += 1
    return(ls)
ls = [1,2,3,3,3,4,4,4,5,5]
fun2(ls)
ls

torf = (1==1) is (1==2)
torf
2<<1

a = [1,2,3,4,5]
b = a
a
b.remove(3)
a

a = [1,2,3,4,5]
b = list(a)
a
b.remove(3)
a
b

ls = [1,5,4,3,2]

ls = [1,3,2]
j = len(ls)-1
i = j-1
ma = ls[j]
while(i>=0):
    if(ls[i]>=ma):
        ma = ls[i]
        i-=1
    else:
        while(ls[i]>=ls[j]):
            j-=1
        ls[i],ls[j] = ls[j],ls[i]
        ls = ls[:i+1]+sorted(ls[i+1:])
        break
if(i==-1):
    ls.sort()
print(ls)

t = (0,"")
#t[1] = "asd"

for i,v in enumerate("asdf"):
    print(i,v)

for i in [1,4,7]:
    print(i)

for x in range(1,5)+range(6,10):
    print(x)

a = 1
if ( not a):
    print(a)

"3" in "3"

def isok(i, j, n, b):
    # row
    d = {}
    this = b[i]
    print(this)
    for k in range(0, j):
        s = this[k]
        if (s != '.' and d and s in d):
            return (False)
        d[this[k]] = 1
    if (d and n in d):
        return (False)
    d[n] = 1
    print(d)
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
    this = [b[ci - 1][cj - 1], b[ci][cj - 1], b[ci + 1][cj - 1], b[ci - 1][cj], n, b[ci + 1][cj], b[ci - 1][cj + 1],
            b[ci][cj + 1], b[ci + 1][cj + 1]]
    for k in range(9):
        s = this[k]
        if (s != '.' and d and s in d):
            return (False)
        d[this[k]] = 1
    return (True)

#b = ["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]
#isok(0,1,1,b)

bb = [".19748632","783652419","426139875","357986241","264317598","198524367","975863124","832491756","641275983"]
isok(0,0,'1',bb)

a = b = 0
a+=1
a
b
aa = bb = []
aa.append(1)
aa
bb
s = ss = ""
s = s+"abc"
s
ss


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

x = "0"
y = "0"
int(add(x,y)) == int(x)+int(y)

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

x = "12314230"
y = "0"
int(multi(x,y)) == int(x)*int(y)

path = [1,2,3]
nums = [0]
path + nums
path.append(nums)
path

def fun(s):
    s = s[-1]
s = "123"
fun(s)
s

d = {1:3,2:4}
d
for key in d.values():
    print(key)

ls = [1,2,3,4,5]
for i in ls:
    print(i)
    ls.remove(i)

s = "12asd34"
sorted(s)

i = 6
j = 4
n = 8
[(i+x,j+x) for x in range(-min(i,j),min(n-i,n-j))]

[(i-x,j+x) for x in range(-min(n-i-1,j),min(i,n-j-1)+1)]

a = [(1,2,3)]
c,d = a.pop()[0:2]
a

float("Inf")

ls = [[1,2,3],[3,4,5]]
a = ls.pop(0)
a = [x.pop() for x in ls]
ls[0:2]
a

i = 10
for i in range(i,i-3,-1):
    print(i)
i

ls = [1,2,3,4,5]
ls[2:4] = [2]
ls

s = "asdf asdf asdfasf   sdf   "
s.split(" ")
s.rstrip()

import math
math.factorial(3)

if(0/0==0 or 3<4):
    print("Hello")

s = ""
s = 0+s

eval('0b10101')

a = 10
t = 0
def fun(a,t):
    if(a==0):
        t+=1
        return
    else:
        a-=1
        fun(a,t)

fun(a,t)
a
t

ls = [a]
def fun(ls):
    if(ls[0]==0):
        return
    else:
        ls[0]-=1
        fun(ls)

fun(ls)
ls

string = "/a/b/c//../e/././f/"
string.split("/")
string

ls = ["a","b"]
"/".join(ls)

ls = [[0]*10 for _ in range(10)]
ls
ls[0] = list(range(0,11))
ls

ls = [2,"d",None]
ls
ls = [[]]
not ls
ls = []
not ls

d = {'c':[1],'b':[2],'a':[3,4]}
max(d.values())
d['a']
ls = [1,2,3,-1,-2]
ls[ls.index(min(ls))]

ls = [[1,2],[3,4],[5]]
sum(ls,[])

s = "adfasf"
for i,j in enumerate(s,2):
    print(i,j)

a = 3
a -=6 >0
a

left = ["1","2","3"]
right = ['a','b']
[l+r for l in left for r in right]

for i in range(1,10) or [-1]:
    print(i)
for i in range(1,1) or [-1]:
    print(i)

product([1,2],[3,4])

ls = [1,2,3]
ls.insert(0,0)
ls

[[a+1,a+2] for a in ls]
ls.append(4) if 2>3 else 0

list(reversed(ls))

ls = [True,1]
ls = (True,1)
ls
min(float("Inf"),10)

[1,2,3]+[3]

num = 7
s = ""
while (num):
    num, tmp = divmod(num, 7)
    s = str(tmp) + s

if num > 0:
    s += "-"

nums = [1,3,2,3,1]
ls1 = [(nums[i],i,True) for i in range(len(nums))]
ls2 = [(2*nums[i],i,False) for i in range(len(nums))]
ls = ls1+ls2
ls.sort()
ls

ls = [1,2,3,4,5,6]
ls.index(3)

"sfw3a".isalnum()
"E".lower()

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

p = TreeNode(None)
if not p:
    print("hello")

"a".islo

s = "asfdaasdfassfdas"
s[-1000:3]
ls = [3,4,2,3]
ls[-100:2]

ls = ['aaa','abd',"bbc",'bcd',"aacc",'abcd','addd']
ls.sort()
ls.reverse()
ls.sort(key = lambda x :len(x))
ls.reverse()
ls

s = " asdf adfa   afdaf   adf  "
ls = s.split()
ls.reverse()
ls
" ".join(ls)

ord("A")
ord("B")
chr(65)

import math
math.factorial(24)

1%5
0%5

ls = [1]
bool(ls)

ls = [6,34, 30, 3, 5, 9]
ls.sort(key = lambda x: x//10)

d = {11:2,3:1,1:1,2:2}
list(d.keys())

timePoints = ["23:59","00:00"]
nums = [ int(x.split(":")[0]) + int(x.split(":")[1])/100 for x in timePoints]
nums+numss

"02".split(".")

"s"*(-1)

ls1 = [1,3,5]
ls2 = [2,4,6]
[(x,y) for x in ls1 for y in ls2]

s1 = "asdfa"
s2 = "asdfe"
ls = [x==y for x,y in zip(s1,s2)]

str(bin(10))
int('10',2)

n=1
b = str(bin(n))[2:][::-1]
b = b+"0"*(32-len(b))
int(b,2)

tmp = -1
for i in range(0,2147483647+1):
    tmp &=i
tmp

bin(100)

s = "asf"
t = "ddd"
list(zip(s,t))

s = "asfdasdf"
[s.find(i) for i in s]
s.find("a")
list(map(s.find,s))

n = 1500000
ls = [True]*n
for i in range(2,n):
    if not ls[i-1]:
        continue
    for j in range(i*i,n,i):
        ls[j-1] = False
count = 0
for i in range(2,n):
    if ls[i-1]:
        count+=1

min(1,2,3,4)

import  heapq
nums = [3,2,1,5,6,4]
heapq.heapify(nums)
nums

A0 = dict(zip(('a','b','c','d','e'),(1,2,3,4,5)))
A1 = range(10)
A2 = sorted([i for i in A1 if i in A0])
A3 = sorted([A0[s] for s in A0])
A4 = [i for i in A1 if i in A3]
A5 = {i:i*i for i in A1}
A6 = [[i,i*i] for i in A1]

ls = []
ls[-1]

s = "3+33-45+5"
for i,v in enumerate(s):
    print(i,v)

def fun(s):
    ls = []
    for i, v in enumerate(s):
        if v == "+" or v == "-":
            ls.append(i)
    ls.append(len(s))
    res = int(s[:ls[0]])
    tmp = ls[0]
    for i in ls:
        if s[tmp] == "+":
            res += int(s[tmp + 1:i])
        else:
            res -= int(s[tmp + 1:i])
        tmp = i
    return (res)

fun("3+3-2")

ls = [3,"+"]
ls[3:]

mask = 1
mask << 1

ls = [1,2,3,4,3]
ls.index(3)
ls[:-1]

any([True,True])

"asd asdf ad ".split()

max([1])
for i in set("asdfa"):
    print(i)

"{0:b}".format(10)
int('1010',2)

"asdf asdf d".split()

ls = []
ls[-1]
d = {1:2,2:3}
d.values()

10>>1

ls1 = [1,2,3,4,5]
ls2 = [5,4,3,2,1]
list(map(lambda x,y:x+y,ls1,ls2))
from functools import reduce
reduce((lambda x,y:x+y),ls1)
list(filter(lambda x:x>3,ls1))

[x for x in range(3,15,3)]

"asdf+sdf".split("+")

a = (1,2,3)
a+(1,)

import math
math.ceil(10.3)

s = set()
s.add(3)
s
import math
math.mean([1,2,3])
round(2.5)

arr = []

length = len(arr)
i,j = 0,length-1
while(i<j):
    mid = (i+j)//2
    if arr[mid] < k:
        i = mid+1
    elif arr[mid] >= k:
        j = mid

ls = [22,22,22,16,30,22,28,27,22,4,34,40,5,22,48,41,1,42,37,22,23,26,39,45,22,44,22,21,22,15,35,3,22,46,8,13,22,24,33,6,49,47,11,22,9,10,43,22,20,22]
ls.sort()
ls

from collections import Counter
c1 = Counter("123")
c2 = Counter("345")
for k,v in c1.items():
    print(k,v)

c1-c2

a = 11
b = 31

bin(-10)

ls = [1,2,3,4,5]
ls.remove(2)
ls

int('100', 2)
"{0:b}".format(10)
int('00010101',2)

"3"[3:]