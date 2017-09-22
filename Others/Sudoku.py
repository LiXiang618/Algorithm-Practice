def fun(i, j, b):
    if (i == 8 and j == 8):
        p = b[i][j] if b[i][j] != "." else "123456789"
        for n in p:
            if (isok(i, j, n, b)):
                b[i]= b[i][0:j]+n+b[i][j+1:]
                return (b)
        tmp = "." if p == "123456789" else p
        b[i] = b[i][0:j] + tmp + b[i][j + 1:]
        return ([])
    else:
        p = b[i][j] if b[i][j] != "." else "123456789"
        for n in p:
            if (isok(i, j, n, b)):
                b[i]= b[i][0:j] + n + b[i][j + 1:]
                if (j < 8):
                    res = fun(i, j + 1, b)
                    if (res):
                        return (res)
                else:
                    res = fun(i + 1, 0, b)
                    if (res):
                        return (res)
        tmp = "." if p == "123456789" else p
        b[i] = b[i][0:j] + tmp + b[i][j + 1:]
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
    this = [b[ci - 1][cj - 1], b[ci-1][cj], b[ci-1][cj + 1],
            b[ci][cj-1], b[ci][cj], b[ci][cj+1],
            b[ci + 1][cj - 1], b[ci+1][cj], b[ci + 1][cj + 1]]
    bias = (i%3)*3+(j%3)
    this[bias] = n

    for k in range(9):
        s = this[k]
        if (s != '.' and d and s in d):
            return (False)
        d[this[k]] = 1
    return (True)

board = ["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]

board = fun(0, 0, board)
print(board)