"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable
EPS = 1e-6
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(x: float, y:float) -> float:
    """
    Multiplies

    Args:
        x (float): x
        y (float): y

    Returns:
        float: result of $x * y$
    """
    return x * y
# - id
def id(x: float) -> float:
    """
    Identity

    Args:
        x (float): x

    Returns:
        float: x
    """
    return x
# - add
def add(x: float, y: float) -> float:
    """
    Add two vars

    Args:
        x (float): x
        y (float): y

    Returns:
        float: result
    """
    return x + y
# - neg
def neg(x: float) -> float:
    """
    Take negative sign of input

    Args:
        x (float): x

    Returns:
        float: $-x$
    """
    return -x
# - lt
def lt(x: float, y: float) -> float:
    """
    Less than

    Args:
        x (float): x
        y (float): y

    Returns:
        float: 1.0 if x less than y
    """
    return x < y
# - eq
def eq(x: float, y: float) -> float:
    """
    Equal to

    Args:
        x (float): x
        y (float): y

    Returns:
        float: 1 if x = y
    """
    return x == y
# - max
def max(x: float, y: float) -> float:
    """
    Max of x and y

    Args:
        x (float): x
        y (float): y

    Returns:
        float: x if x > y else y
    """
    return x if x > y else y
# - is_close
def is_close(x: float, y: float) -> float:
    """
    Check if x and y are close enough (1e-2)

    Args:
        x (float): x
        y (float): y

    Returns:
        float: 1.0 if close enough
    """
    return abs(x - y) < 1e-2
# - sigmoid
def sigmoid(x: float) -> float:
    """
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$  

    Args:
        x (float): x

    Returns:
        float: sigmoid(x)
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
# - relu
def relu(x: float) -> float:
    """
    f(x) = x if x > 0 else 0

    Args:
        x (float): x

    Returns:
        float: relu(x)
    """
    return x if x > 0 else 0
# - log
def log(x: float) -> float:
    """
    log(x)

    Args:
        x (float): x

    Returns:
        float: log(x)
    """
    return math.log(x + EPS)
# - exp
def exp(x: float) -> float:
    """
    exp(x)

    Args:
        x (float): x

    Returns:
        float: exp(x)
    """
    return math.exp(x)
# - log_back
def log_back(x: float, d: float) -> float:
    """
    Computes the derivative of log times d

    Args:
        x (float): x
        d (float): d times

    Returns:
        float: result
    """
    return d * inv(x)
# - inv
def inv(x: float) -> float:
    """
    1 / x

    Args:
        x (float): x

    Returns:
        float: 1 / x
    """
    return 1 / (x)
# - inv_back
def inv_back(x: float, d: float) -> float:
    """
    d times derivative of 1 / x

    Args:
        x (float): x
        d (float): d times

    Returns:
        float: -d / x^2 
    """
    return -1 * d / (x ** 2)
# - relu_back
def relu_back(x: float, d: float) -> float:
    """
    d times derivative of relu(x)

    Args:
        x (float): x
        d (float): x

    Returns:
        float: 0 if x < 0, else d
    """
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order-map

    Args:
        fn (Callable[[float], float]): Function from one value to one value.

    Returns:
        Callable[[Iterable[float]], Iterable[float]]: A function that takes a list, applies `fn` to each element, and returns a new list
    """
    def apply(ls: Iterable[float]) -> Iterable[float]:
        """
        Using for loop to iter and apply fn to elements

        Args:
            ls (Iterable[float]): list

        Returns:
            Iterable[float]: result list
        """
        res = []
        for i in ls:
            res.append(fn(i))
        return res
    
    return apply
# - zipWith
def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith

    Args:
        fn (Callable[[float, float], float]): combine to values

    Returns:
        Callable[[Iterable[float], Iterable[float]], Iterable[float]]: Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by applying fn(x, y) on each pair of elements.
    """
    
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        """
        apply fn to ls1 and ls2

        Args:
            ls1 (Iterable[float]): list1
            ls2 (Iterable[float]): list2

        Returns:
            Iterable[float]: fn(list1, list2)
        """
        z = []
        for x, y in zip(ls1, ls2):
            z.append(fn(x, y))
        return z
    
    return apply
        
# - reduce
def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    """
    Higher order reduction

    Args:
        fn (Callable[[float, float], float]): fn
        start (float): start value

    Returns:
        Callable[[Iterable[float]], float]: Function that takes a list `ls` of elements $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2, fn(x_1, x_0)))`
    """
    def apply(ls: Iterable):
        """
        apply

        Args:
            ls (Iterable): list

        Returns:
            _type_: fn(x_n, fn(x_n-1, fn(...)))
        """
        res = start
        for i in ls:
            res = fn(i, res)
        return res
    
    return apply

#
# Use these to implement
# - negList : negate a list
def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate each element in "ls"

    Args:
        ls (Iterable[float]): input list

    Returns:
        Iterable[float]: negate list
    """
    neg_list = map(neg)
    return neg_list(ls)
    
# - addLists : add two lists together
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    add two lists together

    Args:
        ls1 (Iterable[float]): list1
        ls2 (Iterable[float]): list2

    Returns:
        Iterable[float]: result list
    """
    add_list = zipWith(add)
    return add_list(ls1, ls2)
# - sum: sum lists
def sum(ls: Iterable[float]) -> float:
    """
    Sum up a list using reduce and add

    Args:
        ls (Iterable[float]): input list

    Returns:
        float: sum of list
    """
    sum_list = reduce(add, 0.0)
    return sum_list(ls)
# - prod: take the product of lists
def prod(ls: Iterable[float]) -> float:
    """
    Product of a list

    Args:
        ls (Iterable[float]): input list

    Returns:
        float: product of a list
    """
    prod_list = reduce(mul, 1.0)
    return prod_list(ls)