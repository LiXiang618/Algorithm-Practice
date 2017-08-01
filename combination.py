def fun(can, tar, p,res):
    if (tar == 0):
        if(list(p) not in res):
            res.append(list(p))
        return
    else:
        for i in range(len(can)):
            if (tar - can[i] >= 0):
                p.append(can[i])
                fun(can[(i + 1):], tar - can[i],p, res)
                p.pop()

candidates = [1,1,2,5,6,7,10]
target = 8
ls = []
fun(candidates,target,[],ls)
ls