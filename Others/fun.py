
def fun(s):
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

s = "1-(2+3-(4+(5-(1-(2+4-(5+6))))))"
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
        elif (tmp - 1 >= 0 and s[tmp - 1] == "-" and res[0] != "-"):
            s = s[:tmp] + res + s[i + 1:]
        else:
            if (res[0] == "-"):
                s = s[:tmp] + "0" + res + s[i + 1:]
            else:
                s = s[:tmp] + "0+" + res + s[i + 1:]
        i = tmp + len(res)
    else:
        i += 1
