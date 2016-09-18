
X = [
    [ 1, 0, 0 ],
    [ 1, 0, 1 ],
    [ 1, 1, 0 ],
    [ 1, 1, 1 ],
]

T = [ 0, 0, 0, 1 ]

W = [0, 0, 0]

N = 0.5

def sgn(value):
    return 1 if value > 0 else 0

def solve():
    i = 1
    while True:
        print "Cycle %s ----------------------------------" % i
        t = learn()
        i +=1
        if T == t:
            print "W0 = %s, W1 = %s, W2 = %s" % (W[0], W[1], W[2])
            break

def learn():
    r = []
    for i in range(0, len(X)):
        x = X[i]
        s = sgn(W[0] * x[0] + W[1] * x[1] + W[2] * x[2])
        r.append(s)
        if s == T[i]:
            print "... ok"
        else:
            for j in range(0, len(W)):
                w = W[j] + N * (T[i] - s) * x[j]
                W[j] = w
            print "... err => new W%s " % W
    return r

if __name__ == '__main__':
    solve()