from scipy import special as s
import extmath as e
import timeit

if 0:
    var = -0.5

    print(s.i1(var))
    print(e.bessi1(var))

    print(timeit.timeit("s.i0(var)", setup="from __main__ import s, var"))
    print(timeit.timeit("e.bessi0(var)", setup="from __main__ import e, var"))

    print("~~~~~~~~~~~~~~~~~~~")

if 0:
    var = -0.5

    print(s.j1(var))
    print(e.bessj1(var))

    print(timeit.timeit("s.i0(var)", setup="from __main__ import s, var"))
    print(timeit.timeit("e.bessi0(var)", setup="from __main__ import e, var"))

    print("~~~~~~~~~~~~~~~~~~~")

if 0:
    var = -0.5

    print(s.erfc(var))
    print(e.ErfC(var))

    print(timeit.timeit("s.erfc(var)", setup="from __main__ import s, var"))
    print(timeit.timeit("e.ErfC(var)", setup="from __main__ import e, var"))

    print("~~~~~~~~~~~~~~~~~~~")

if 1:
    var = -0.5

    print(s.gammaln(var))
    print(e.LnGamma(var))

    print(timeit.timeit("s.gammaln(var)", setup="from __main__ import s, var"))
    print(timeit.timeit("e.LnGamma(var)", setup="from __main__ import e, var"))

    print("~~~~~~~~~~~~~~~~~~~")

