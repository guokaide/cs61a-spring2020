""" Homework 1: Control """

from operator import add, sub

def a_plus_abs_b(a, b):
    """Return a+abs(b), but without calling abs.

    >>> a_plus_abs_b(2, 3)
    5
    >>> a_plus_abs_b(2, -3)
    5
    """
    if b < 0:
        f = sub
    else:
        f = add
    return f(a, b)

def two_of_three(a, b, c):
    """Return x*x + y*y, where x and y are the two largest members of the
    positive numbers a, b, and c.

    >>> two_of_three(1, 2, 3)
    13
    >>> two_of_three(5, 3, 1)
    34
    >>> two_of_three(10, 2, 8)
    164
    >>> two_of_three(5, 5, 5)
    50
    """
    return max(a*a+b*b, a*a+c*c, b*b+c*c)
    # Alternate solution
    return a**2 + b**2 + c**2 - min(a, b, c)**2

def largest_factor(n):
    """Return the largest factor of n that is smaller than n.

    >>> largest_factor(15) # factors are 1, 3, 5
    5
    >>> largest_factor(80) # factors are 1, 2, 4, 5, 8, 10, 16, 20, 40
    40
    >>> largest_factor(13) # factor is 1 since 13 is prime
    1
    """
    factor = n - 1
    while factor > 0:
        if n % factor == 0:
            return factor
        factor -= 1

def if_function(condition, true_result, false_result):
    """Return true_result if condition is a true value, and
    false_result otherwise.

    >>> if_function(True, 2, 3)
    2
    >>> if_function(False, 2, 3)
    3
    >>> if_function(3==2, 3+2, 3-2)
    1
    >>> if_function(3>2, 3+2, 3-2)
    5
    """
    if condition:
        return true_result
    else:
        return false_result


def with_if_statement():
    """
    >>> result = with_if_statement()
    2
    >>> print(result)
    None
    """
    if c():
        return t()
    else:
        return f()

def with_if_function():
    """
    >>> result = with_if_function()
    1
    2
    >>> print(result)
    None
    """
    return if_function(c(), t(), f())

def c():
    return False

def t():
    print(1)

def f():
    print(2)

def hailstone(n):
    """Print the hailstone sequence starting at n and return its
    length.

    >>> a = hailstone(10)
    10
    5
    16
    8
    4
    2
    1
    >>> a
    7
    """
    length = 1
    while n != 1:
        print(n)
        if n % 2 == 0:
            n = n // 2      # Integer division prevents "1.0" output
        else:
            n = 3 * n + 1
        length = length + 1
    print(n)                # n is now 1
    return length
	
""" Homework 2: Higher Order Functions"""

HW_SOURCE_FILE = 'hw02.py'

from operator import add, mul, sub

square = lambda x: x * x

identity = lambda x: x

triple = lambda x: 3 * x

increment = lambda x: x + 1

######################
# Required Questions #
######################
def product(n, term):
    """Return the product of the first n terms in a sequence.
    n    -- a positive integer
    term -- a function that takes one argument

    >>> product(3, identity)  # 1 * 2 * 3
    6
    >>> product(5, identity)  # 1 * 2 * 3 * 4 * 5
    120
    >>> product(3, square)    # 1^2 * 2^2 * 3^2
    36
    >>> product(5, square)    # 1^2 * 2^2 * 3^2 * 4^2 * 5^2
    14400
    >>> product(3, increment) # (1+1) * (2+1) * (3+1)
    24
    >>> product(3, triple)    # 1*3 * 2*3 * 3*3
    162
    """
    total, k = 1, 1
    while k <= n:
        total, k = term(k) * total, k + 1
    return total

def accumulate(combiner, base, n, term):
    """Return the result of combining the first n terms in a sequence and base.
    The terms to be combined are term(1), term(2), ..., term(n).  combiner is a
    two-argument commutative, associative function.

    >>> accumulate(add, 0, 5, identity)  # 0 + 1 + 2 + 3 + 4 + 5
    15
    >>> accumulate(add, 11, 5, identity) # 11 + 1 + 2 + 3 + 4 + 5
    26
    >>> accumulate(add, 11, 0, identity) # 11
    11
    >>> accumulate(add, 11, 3, square)   # 11 + 1^2 + 2^2 + 3^2
    25
    >>> accumulate(mul, 2, 3, square)    # 2 * 1^2 * 2^2 * 3^2
    72
    >>> accumulate(lambda x, y: x + y + 1, 2, 3, square)
    19
    >>> accumulate(lambda x, y: 2 * (x + y), 2, 3, square)
    58
    >>> accumulate(lambda x, y: (x + y) % 17, 19, 20, square)
    16
    """
    total, k = base, 1
    while k <= n:
        total, k = combiner(total, term(k)), k + 1
    return total

# Recursive solution
def accumulate2(combiner, base, n, term):
    if n == 0:
        return base
    return combiner(term(n), accumulate2(combiner, base, n-1, term))

# Alternative recursive solution using base to keep track of total
def accumulate3(combiner, base, n, term):
    if n == 0:
        return base
    return accumulate3(combiner, combiner(base, term(n)), n-1, term)

def summation_using_accumulate(n, term):
    """Returns the sum of term(1) + ... + term(n). The implementation
    uses accumulate.

    >>> summation_using_accumulate(5, square)
    55
    >>> summation_using_accumulate(5, triple)
    45
    >>> from construct_check import check
    >>> # ban iteration and recursion
    >>> check(HW_SOURCE_FILE, 'summation_using_accumulate',
    ...       ['Recursion', 'For', 'While'])
    True
    """
    return accumulate(add, 0, n, term)

def product_using_accumulate(n, term):
    """An implementation of product using accumulate.

    >>> product_using_accumulate(4, square)
    576
    >>> product_using_accumulate(6, triple)
    524880
    >>> from construct_check import check
    >>> # ban iteration and recursion
    >>> check(HW_SOURCE_FILE, 'product_using_accumulate',
    ...       ['Recursion', 'For', 'While'])
    True
    """
    return accumulate(mul, 1, n, term)

def compose1(f, g):
    """Return a function h, such that h(x) = f(g(x))."""
    def h(x):
        return f(g(x))
    return h

def make_repeater(f, n):
    """Return the function that computes the nth application of f.

    >>> add_three = make_repeater(increment, 3)
    >>> add_three(5)
    8
    >>> make_repeater(triple, 5)(1) # 3 * 3 * 3 * 3 * 3 * 1
    243
    >>> make_repeater(square, 2)(5) # square(square(5))
    625
    >>> make_repeater(square, 4)(5) # square(square(square(square(5))))
    152587890625
    >>> make_repeater(square, 0)(5) # Yes, it makes sense to apply the function zero times! 
    5
    """
    g = identity
    while n > 0:
        g = compose1(f, g)
        n = n - 1
    return g

# Alternative solutions
def make_repeater2(f, n):
    def h(x):
        k = 0
        while k < n:
            x, k = f(x), k + 1
        return x
    return h

def make_repeater3(f, n):
    if n == 0:
        return lambda x: x
    return lambda x: f(make_repeater3(f, n - 1)(x))

def make_repeater4(f, n):
    if n == 0:
        return lambda x: x
    return compose1(f, make_repeater4(f, n - 1))

def make_repeater5(f, n):
    return accumulate(compose1, lambda x: x, n, lambda k: f)


##########################
# Just for fun Questions #
##########################

def zero(f):
    return lambda x: x

def successor(n):
    return lambda f: lambda x: f(n(f)(x))

def one(f):
    """Church numeral 1: same as successor(zero)"""
    return lambda x: f(x)

def two(f):
    """Church numeral 2: same as successor(successor(zero))"""
    return lambda x: f(f(x))

three = successor(two)

def church_to_int(n):
    """Convert the Church numeral n to a Python integer.

    >>> church_to_int(zero)
    0
    >>> church_to_int(one)
    1
    >>> church_to_int(two)
    2
    >>> church_to_int(three)
    3
    """
    return n(lambda x: x + 1)(0)

def add_church(m, n):
    """Return the Church numeral for m + n, for Church numerals m and n.

    >>> church_to_int(add_church(two, three))
    5
    """
    return lambda f: lambda x: m(f)(n(f)(x))

def mul_church(m, n):
    """Return the Church numeral for m * n, for Church numerals m and n.

    >>> four = successor(three)
    >>> church_to_int(mul_church(two, three))
    6
    >>> church_to_int(mul_church(three, four))
    12
    """
    return lambda f: m(n(f))

def pow_church(m, n):
    """Return the Church numeral m ** n, for Church numerals m and n.

    >>> church_to_int(pow_church(two, three))
    8
    >>> church_to_int(pow_church(three, two))
    9
    """
    return n(m)
	
	
""" Homework 3"""
HW_SOURCE_FILE = 'hw03.py'

#############
# Questions #
#############

def num_sevens(n):
    """Returns the number of times 7 appears as a digit of n.

    >>> num_sevens(3)
    0
    >>> num_sevens(7)
    1
    >>> num_sevens(7777777)
    7
    >>> num_sevens(2637)
    1
    >>> num_sevens(76370)
    2
    >>> num_sevens(12345)
    0
    >>> from construct_check import check
    >>> # ban all assignment statements
    >>> check(HW_SOURCE_FILE, 'num_sevens',
    ...       ['Assign', 'AugAssign'])
    True
    """
    if n % 10 == 7:
        return 1 + num_sevens(n // 10)
    elif n < 10:
        return 0
    else:
        return num_sevens(n // 10)

def pingpong(n):
    """Return the nth element of the ping-pong sequence.

    >>> pingpong(7)
    7
    >>> pingpong(8)
    6
    >>> pingpong(15)
    1
    >>> pingpong(21)
    -1
    >>> pingpong(22)
    0
    >>> pingpong(30)
    6
    >>> pingpong(68)
    2
    >>> pingpong(69)
    1
    >>> pingpong(70)
    0
    >>> pingpong(71)
    1
    >>> pingpong(72)
    0
    >>> pingpong(100)
    2
    >>> from construct_check import check
    >>> # ban assignment statements
    >>> check(HW_SOURCE_FILE, 'pingpong', ['Assign', 'AugAssign'])
    True
    """
    def helper(result, i, step):
        if i == n:
            return result
        elif i % 7 == 0 or num_sevens(i) > 0:
            return helper(result - step, i + 1, -step)
        else:
            return helper(result + step, i + 1, step)
    return helper(1, 1, 1)

# Alternate solution 1
def pingpong_next(x, i, step):
    if i == n:
        return x
    return pingpong_next(x + step, i + 1, next_dir(step, i+1))

def next_dir(step, i):
    if i % 7 == 0 or num_sevens(i) > 0:
        return -step
    return step

return pingpong_next(1, 1, 1)

# Alternate solution 2
def pingpong(n):
    if n <= 7:
        return n
    return direction(n) + pingpong(n-1)

def direction(n):
    if n < 7:
        return 1
    if (n-1) % 7 == 0 or num_sevens(n-1) > 0:
        return -1 * direction(n-1)
    return direction(n-1)

def count_change(amount):
    """Return the number of ways to make change for amount.

    >>> count_change(7)
    6
    >>> count_change(10)
    14
    >>> count_change(20)
    60
    >>> count_change(100)
    9828
    >>> from construct_check import check
    >>> # ban iteration
    >>> check(HW_SOURCE_FILE, 'count_change', ['While', 'For'])
    True
    """
    def constrained_count(amount, smallest_coin):
        if amount == 0:
            return 1
        if smallest_coin > amount:
            return 0
        without_coin = constrained_count(amount, smallest_coin * 2)
        with_coin = constrained_count(amount - smallest_coin, smallest_coin)
        return without_coin + with_coin
    return constrained_count(amount, 1)


def flatten(lst):
    """Returns a flattened version of lst.

    >>> flatten([1, 2, 3])     # normal list
    [1, 2, 3]
    >>> x = [1, [2, 3], 4]      # deep list
    >>> flatten(x)
    [1, 2, 3, 4]
    >>> x # Ensure x is not mutated
    [1, [2, 3], 4]
    >>> x = [[1, [1, 1]], 1, [1, 1]] # deep list
    >>> flatten(x)
    [1, 1, 1, 1, 1, 1]
    >>> x
    [[1, [1, 1]], 1, [1, 1]]
    """
    if not lst:
        return []
    elif type(lst[0]) == list:
        return flatten(lst[0]) + flatten(lst[1:])
    else:
        return [lst[0]] + flatten(lst[1:])

###################
# Extra Questions #
###################

def print_move(origin, destination):
    """Print instructions to move a disk."""
    print("Move the top disk from rod", origin, "to rod", destination)

def move_stack(n, start, end):
    """Print the moves required to move n disks on the start pole to the end
    pole without violating the rules of Towers of Hanoi.

    n -- number of disks
    start -- a pole position, either 1, 2, or 3
    end -- a pole position, either 1, 2, or 3

    There are exactly three poles, and start and end must be different. Assume
    that the start pole has at least n disks of increasing size, and the end
    pole is either empty or has a top disk larger than the top n start disks.

    >>> move_stack(1, 1, 3)
    Move the top disk from rod 1 to rod 3
    >>> move_stack(2, 1, 3)
    Move the top disk from rod 1 to rod 2
    Move the top disk from rod 1 to rod 3
    Move the top disk from rod 2 to rod 3
    >>> move_stack(3, 1, 3)
    Move the top disk from rod 1 to rod 3
    Move the top disk from rod 1 to rod 2
    Move the top disk from rod 3 to rod 2
    Move the top disk from rod 1 to rod 3
    Move the top disk from rod 2 to rod 1
    Move the top disk from rod 2 to rod 3
    Move the top disk from rod 1 to rod 3
    """
    assert 1 <= start <= 3 and 1 <= end <= 3 and start != end, "Bad start/end"
    if n == 1:
        print_move(start, end)
    else:
        other = 6 - start - end
        move_stack(n-1, start, other)
        print_move(start, end)
        move_stack(n-1, other, end)

from operator import sub, mul

def make_anonymous_factorial():
    """Return the value of an expression that computes factorial.

    >>> make_anonymous_factorial()(5)
    120
    >>> from construct_check import check
    >>> # ban any assignments or recursion
    >>> check(HW_SOURCE_FILE, 'make_anonymous_factorial', ['Assign', 'AugAssign', 'FunctionDef', 'Recursion'])
    True
    """
    return (lambda f: lambda k: f(f, k))(lambda f, k: k if k == 1 else mul(k, f(f, sub(k, 1))))
    # Alternate solution:
    #   return (lambda f: f(f))(lambda f: lambda x: 1 if x == 0 else x * f(f)(x - 1))
	
""" Homework 4"""
this_file = __file__

def intersection(st, ave):
    """Represent an intersection using the Cantor pairing function."""
    return (st+ave)*(st+ave+1)//2 + ave

def street(inter):
    return w(inter) - avenue(inter)

def avenue(inter):
    return inter - (w(inter) ** 2 + w(inter)) // 2

w = lambda z: int(((8*z+1)**0.5-1)/2)

def taxicab(a, b):
    """Return the taxicab distance between two intersections.

    >>> times_square = intersection(46, 7)
    >>> ess_a_bagel = intersection(51, 3)
    >>> taxicab(times_square, ess_a_bagel)
    9
    >>> taxicab(ess_a_bagel, times_square)
    9
    """
    return abs(street(a)-street(b)) + abs(avenue(a)-avenue(b))

# Mobiles

def mobile(left, right):
    """Construct a mobile from a left side and a right side."""
    assert is_side(left), "left must be a side"
    assert is_side(right), "right must be a side"
    return ['mobile', left, right]

def is_mobile(m):
    """Return whether m is a mobile."""
    return type(m) == list and len(m) == 3 and m[0] == 'mobile'

def left(m):
    """Select the left side of a mobile."""
    assert is_mobile(m), "must call left on a mobile"
    return m[1]

def right(m):
    """Select the right side of a mobile."""
    assert is_mobile(m), "must call right on a mobile"
    return m[2]

def side(length, mobile_or_weight):
    """Construct a side: a length of rod with a mobile or weight at the end."""
    assert is_mobile(mobile_or_weight) or is_weight(mobile_or_weight)
    return ['side', length, mobile_or_weight]

def is_side(s):
    """Return whether s is a side."""
    return type(s) == list and len(s) == 3 and s[0] == 'side'

def length(s):
    """Select the length of a side."""
    assert is_side(s), "must call length on a side"
    return s[1]

def end(s):
    """Select the mobile or weight hanging at the end of a side."""
    assert is_side(s), "must call end on a side"
    return s[2]

def weight(size):
    """Construct a weight of some size."""
    assert size > 0
    return ['weight', size]

def size(w):
    """Select the size of a weight."""
    assert is_weight(w), 'must call size on a weight'
    return w[1]

def is_weight(w):
    """Whether w is a weight."""
    return type(w) == list and len(w) == 2 and w[0] == 'weight'

def examples():
    t = mobile(side(1, weight(2)),
               side(2, weight(1)))
    u = mobile(side(5, weight(1)),
               side(1, mobile(side(2, weight(3)),
                              side(3, weight(2)))))
    v = mobile(side(4, t), side(2, u))
    return (t, u, v)

def total_weight(m):
    """Return the total weight of m, a weight or mobile.

    >>> t, u, v = examples()
    >>> total_weight(t)
    3
    >>> total_weight(u)
    6
    >>> total_weight(v)
    9
    """
    if is_weight(m):
        return size(m)
    else:
        assert is_mobile(m), "must get total weight of a mobile or a weight"
        return total_weight(end(left(m))) + total_weight(end(right(m)))

def balanced(m):
    """Return whether m is balanced.

    >>> t, u, v = examples()
    >>> balanced(t)
    True
    >>> balanced(v)
    True
    >>> w = mobile(side(3, t), side(2, u))
    >>> balanced(w)
    False
    >>> balanced(mobile(side(1, v), side(1, w)))
    False
    >>> balanced(mobile(side(1, w), side(1, v)))
    False
    """
    if is_weight(m):
        return True
    else:
        left_end, right_end = end(left(m)), end(right(m))
        torque_left = length(left(m)) * total_weight(left_end)
        torque_right = length(right(m)) * total_weight(right_end)
        return balanced(left_end) and balanced(right_end) and torque_left == torque_right

def totals_tree(m):
    """Return a tree representing the mobile with its total weight at the root.

    >>> t, u, v = examples()
    >>> print_tree(totals_tree(t))
    3
      2
      1
    >>> print_tree(totals_tree(u))
    6
      1
      5
        3
        2
    >>> print_tree(totals_tree(v))
    9
      3
        2
        1
      6
        1
        5
          3
          2
    """
    if is_weight(m):
        return tree(size(m))
    else:
        branches = [totals_tree(end(f(m))) for f in [left, right]]
        return tree(sum([label(b) for b in branches]), branches)

def replace_leaf(t, old, new):
    """Returns a new tree where every leaf value equal to old has
    been replaced with new.

    >>> yggdrasil = tree('odin',
    ...                  [tree('balder',
    ...                        [tree('thor'),
    ...                         tree('freya')]),
    ...                   tree('frigg',
    ...                        [tree('thor')]),
    ...                   tree('thor',
    ...                        [tree('sif'),
    ...                         tree('thor')]),
    ...                   tree('thor')])
    >>> laerad = copy_tree(yggdrasil) # copy yggdrasil for testing purposes
    >>> print_tree(replace_leaf(yggdrasil, 'thor', 'freya'))
    odin
      balder
        freya
        freya
      frigg
        freya
      thor
        sif
        freya
      freya
    >>> laerad == yggdrasil # Make sure original tree is unmodified
    True
    """
    if is_leaf(t) and label(t) == old:
        return tree(new)
    else:
        bs = [replace_leaf(b, old, new) for b in branches(t)]
        return tree(label(t), bs)

def make_fib():
    """Returns a function that returns the next Fibonacci number
    every time it is called.

    >>> fib = make_fib()
    >>> fib()
    0
    >>> fib()
    1
    >>> fib()
    1
    >>> fib()
    2
    >>> fib()
    3
    >>> fib2 = make_fib()
    >>> fib() + sum([fib2() for _ in range(5)])
    12
    >>> from construct_check import check
    >>> # Do not use lists in your implementation
    >>> check(this_file, 'make_fib', ['List'])
    True
    """
    cur, next = 0, 1
    def fib():
        nonlocal cur, next
        result = cur
        cur, next = next, cur + next
        return result
    return fib

def make_withdraw(balance, password):
    """Return a password-protected withdraw function.

    >>> w = make_withdraw(100, 'hax0r')
    >>> w(25, 'hax0r')
    75
    >>> error = w(90, 'hax0r')
    >>> error
    'Insufficient funds'
    >>> error = w(25, 'hwat')
    >>> error
    'Incorrect password'
    >>> new_bal = w(25, 'hax0r')
    >>> new_bal
    50
    >>> w(75, 'a')
    'Incorrect password'
    >>> w(10, 'hax0r')
    40
    >>> w(20, 'n00b')
    'Incorrect password'
    >>> w(10, 'hax0r')
    "Your account is locked. Attempts: ['hwat', 'a', 'n00b']"
    >>> w(10, 'l33t')
    "Your account is locked. Attempts: ['hwat', 'a', 'n00b']"
    >>> type(w(10, 'l33t')) == str
    True
    """
    attempts = []
    def withdraw(amount, password_attempt):
        nonlocal balance
        if len(attempts) == 3:
            return 'Your account is locked. Attempts: ' + str(attempts)
        if password_attempt != password:
            attempts.append(password_attempt)
            return 'Incorrect password'
        if amount > balance:
            return 'Insufficient funds'
        balance = balance - amount
        return balance
    return withdraw

def make_joint(withdraw, old_password, new_password):
    """Return a password-protected withdraw function that has joint access to
    the balance of withdraw.

    >>> w = make_withdraw(100, 'hax0r')
    >>> w(25, 'hax0r')
    75
    >>> make_joint(w, 'my', 'secret')
    'Incorrect password'
    >>> j = make_joint(w, 'hax0r', 'secret')
    >>> w(25, 'secret')
    'Incorrect password'
    >>> j(25, 'secret')
    50
    >>> j(25, 'hax0r')
    25
    >>> j(100, 'secret')
    'Insufficient funds'

    >>> j2 = make_joint(j, 'secret', 'code')
    >>> j2(5, 'code')
    20
    >>> j2(5, 'secret')
    15
    >>> j2(5, 'hax0r')
    10

    >>> j2(25, 'password')
    'Incorrect password'
    >>> j2(5, 'secret')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    >>> j(5, 'secret')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    >>> w(5, 'hax0r')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    >>> make_joint(w, 'hax0r', 'hello')
    "Your account is locked. Attempts: ['my', 'secret', 'password']"
    """
    error = withdraw(0, old_password)
    if type(error) == str:
        return error
    def joint(amount, password_attempt):
        if password_attempt == new_password:
            return withdraw(amount, old_password)
        return withdraw(amount, password_attempt)
    return joint



## Tree Methods ##

def tree(label, branches=[]):
    """Construct a tree with the given label value and a list of branches."""
    for branch in branches:
        assert is_tree(branch), 'branches must be trees'
    return [label] + list(branches)

def label(tree):
    """Return the label value of a tree."""
    return tree[0]

def branches(tree):
    """Return the list of branches of the given tree."""
    return tree[1:]

def is_tree(tree):
    """Returns True if the given tree is a tree, and False otherwise."""
    if type(tree) != list or len(tree) < 1:
        return False
    for branch in branches(tree):
        if not is_tree(branch):
            return False
    return True

def is_leaf(tree):
    """Returns True if the given tree's list of branches is empty, and False
    otherwise.
    """
    return not branches(tree)

def print_tree(t, indent=0):
    """Print a representation of this tree in which each node is
    indented by two spaces times its depth from the root.

    >>> print_tree(tree(1))
    1
    >>> print_tree(tree(1, [tree(2)]))
    1
      2
    >>> numbers = tree(1, [tree(2), tree(3, [tree(4), tree(5)]), tree(6, [tree(7)])])
    >>> print_tree(numbers)
    1
      2
      3
        4
        5
      6
        7
    """
    print('  ' * indent + str(label(t)))
    for b in branches(t):
        print_tree(b, indent + 1)

def copy_tree(t):
    """Returns a copy of t. Only for testing purposes.

    >>> t = tree(5)
    >>> copy = copy_tree(t)
    >>> t = tree(6)
    >>> print_tree(copy)
    5
    """
    return tree(label(t), [copy_tree(b) for b in branches(t)])

###################
# Extra Questions #
###################

def interval(a, b):
    """Construct an interval from a to b."""
    assert a <= b, 'Lower bound cannot be greater than upper bound'
    return [a, b]

def lower_bound(x):
    """Return the lower bound of interval x."""
    return x[0]

def upper_bound(x):
    """Return the upper bound of interval x."""
    return x[1]

def str_interval(x):
    """Return a string representation of interval x.
    """
    return '{0} to {1}'.format(lower_bound(x), upper_bound(x))

def add_interval(x, y):
    """Return an interval that contains the sum of any value in interval x and
    any value in interval y."""
    lower = lower_bound(x) + lower_bound(y)
    upper = upper_bound(x) + upper_bound(y)
    return interval(lower, upper)

def mul_interval(x, y):
    """Return the interval that contains the product of any value in x and any
    value in y."""
    p1 = x[0] * y[0]
    p2 = x[0] * y[1]
    p3 = x[1] * y[0]
    p4 = x[1] * y[1]
    return [min(p1, p2, p3, p4), max(p1, p2, p3, p4)]

def sub_interval(x, y):
    """Return the interval that contains the difference between any value in x
    and any value in y."""
    negative_y = interval(-upper_bound(y), -lower_bound(y))
    return add_interval(x, negative_y)

def div_interval(x, y):
    """Return the interval that contains the quotient of any value in x divided by
    any value in y. Division is implemented as the multiplication of x by the
    reciprocal of y."""
    assert not (lower_bound(y) <= 0 <= upper_bound(y)), 'Divide by zero'
    reciprocal_y = interval(1/upper_bound(y), 1/lower_bound(y))
    return mul_interval(x, reciprocal_y)

def par1(r1, r2):
    return div_interval(mul_interval(r1, r2), add_interval(r1, r2))

def par2(r1, r2):
    one = interval(1, 1)
    rep_r1 = div_interval(one, r1)
    rep_r2 = div_interval(one, r2)
    return div_interval(one, add_interval(rep_r1, rep_r2))

def multiple_references_explanation():
    return """The multiple reference problem exists.  The true value
    within a particular interval is fixed (though unknown).  Nested
    combinations that refer to the same interval twice may assume two different
    true values for the same interval, which is an error that results in
    intervals that are larger than they should be.

    Consider the case of i * i, where i is an interval from -1 to 1.  No value
    within this interval, when squared, will give a negative result.  However,
    our mul_interval function will allow us to choose 1 from the first
    reference to i and -1 from the second, giving an erroneous lower bound of
    -1.

    Hence, a program like par2 is better than par1 because it never combines
    the same interval more than once.
    """

def quadratic(x, a, b, c):
    """Return the interval that is the range of the quadratic defined by
    coefficients a, b, and c, for domain interval x.

    >>> str_interval(quadratic(interval(0, 2), -2, 3, -1))
    '-3 to 0.125'
    >>> str_interval(quadratic(interval(1, 3), 2, -3, 1))
    '0 to 10'
    """
    extremum = -b / (2*a)
    f = lambda x: a * x * x + b * x + c
    l, u, e = map(f, (lower_bound(x), upper_bound(x), extremum))
    if extremum >= lower_bound(x) and extremum <= upper_bound(x):
        return interval(min(l, u, e), max(l, u, e))
    else:
        return interval(min(l, u), max(l, u))
		
""" Homework 5"""


def make_counter():
    """Return a counter function.

    >>> c = make_counter()
    >>> c('a')
    1
    >>> c('a')
    2
    >>> c('b')
    1
    >>> c('a')
    3
    >>> c2 = make_counter()
    >>> c2('b')
    1
    >>> c2('b')
    2
    >>> c('b') + c2('b')
    5
    """
    totals = {}
    def counter(key):
        totals[key] = totals.get(key, 0) + 1
        return totals[key]
    return counter


class VendingMachine:
    """A vending machine that vends some product for some price.

    >>> v = VendingMachine('candy', 10)
    >>> v.vend()
    'Machine is out of stock.'
    >>> v.deposit(15)
    'Machine is out of stock. Here is your $15.'
    >>> v.restock(2)
    'Current candy stock: 2'
    >>> v.vend()
    'You must deposit $10 more.'
    >>> v.deposit(7)
    'Current balance: $7'
    >>> v.vend()
    'You must deposit $3 more.'
    >>> v.deposit(5)
    'Current balance: $12'
    >>> v.vend()
    'Here is your candy and $2 change.'
    >>> v.deposit(10)
    'Current balance: $10'
    >>> v.vend()
    'Here is your candy.'
    >>> v.deposit(15)
    'Machine is out of stock. Here is your $15.'

    >>> w = VendingMachine('soda', 2)
    >>> w.restock(3)
    'Current soda stock: 3'
    >>> w.restock(3)
    'Current soda stock: 6'
    >>> w.deposit(2)
    'Current balance: $2'
    >>> w.vend()
    'Here is your soda.'
    """
    def __init__(self, product, price):
        self.product = product
        self.price = price
        self.stock = 0
        self.balance = 0

    def restock(self, n):
        self.stock += n
        return 'Current {0} stock: {1}'.format(self.product, self.stock)

    def deposit(self, n):
        if self.stock == 0:
            return 'Machine is out of stock. Here is your ${0}.'.format(n)
        self.balance += n
        return 'Current balance: ${0}'.format(self.balance)

    def vend(self):
        if self.stock == 0:
            return 'Machine is out of stock.'
        difference = self.price - self.balance
        if difference > 0:
            return 'You must deposit ${0} more.'.format(difference)
        message = 'Here is your {0}'.format(self.product)
        if difference != 0:
            message += ' and ${0} change'.format(-difference)
        self.balance = 0
        self.stock -= 1
        return message + '.'

def preorder(t):
    """Return a list of the entries in this tree in the order that they
    would be visited by a preorder traversal (see problem description).

    >>> numbers = Tree(1, [Tree(2), Tree(3, [Tree(4), Tree(5)]), Tree(6, [Tree(7)])])
    >>> preorder(numbers)
    [1, 2, 3, 4, 5, 6, 7]
    >>> preorder(Tree(2, [Tree(4, [Tree(6)])]))
    [2, 4, 6]
    """
    if t.branches == []:
        return [t.label]
    flattened_children = []
    for child in t.branches:
        flattened_children += preorder(child)
    return [t.label] + flattened_children

# Alternate solution
from functools import reduce

def preorder(t):
    return reduce(add, [preorder(child) for child in t.branches], [t.label])

def store_digits(n):
    """Stores the digits of a positive number n in a linked list.

    >>> s = store_digits(1)
    >>> s
    Link(1)
    >>> store_digits(2345)
    Link(2, Link(3, Link(4, Link(5))))
    >>> store_digits(876)
    Link(8, Link(7, Link(6)))
    """
    result = Link.empty
    while n > 0:
        result = Link(n % 10, result)
        n //= 10
    return result

def generate_paths(t, x):
    """Yields all possible paths from the root of t to a node with the label x
    as a list.

    >>> t1 = Tree(1, [Tree(2, [Tree(3), Tree(4, [Tree(6)]), Tree(5)]), Tree(5)])
    >>> print(t1)
    1
      2
        3
        4
          6
        5
      5
    >>> next(generate_paths(t1, 6))
    [1, 2, 4, 6]
    >>> path_to_5 = generate_paths(t1, 5)
    >>> sorted(list(path_to_5))
    [[1, 2, 5], [1, 5]]

    >>> t2 = Tree(0, [Tree(2, [t1])])
    >>> print(t2)
    0
      2
        1
          2
            3
            4
              6
            5
          5
    >>> path_to_2 = generate_paths(t2, 2)
    >>> sorted(list(path_to_2))
    [[0, 2], [0, 2, 1, 2]]
    """

    if t.label == x:
    	yield [x]

    for b in t.branches:
    	for path in generate_paths(b, x):

    		yield [t.label] + path


def remove_all(link , value):
    """Remove all the nodes containing value in link. Assume that the
    first element is never removed.

    >>> l1 = Link(0, Link(2, Link(2, Link(3, Link(1, Link(2, Link(3)))))))
    >>> print(l1)
    <0 2 2 3 1 2 3>
    >>> remove_all(l1, 2)
    >>> print(l1)
    <0 3 1 3>
    >>> remove_all(l1, 3)
    >>> print(l1)
    <0 1>
    >>> remove_all(l1, 3)
    >>> print(l1)
    <0 1>
    """
    if link is Link.empty or link.rest is Link.empty:
        return
    if link.rest.first == value:
        link.rest = link.rest.rest
        remove_all(link, value)
    else:
        remove_all(link.rest, value)
    
    # alternate solution
    if link is not Link.empty and link.rest is not Link.empty:
        remove_all(link.rest, value)
        if link.rest.first == value:
            link.rest = link.rest.rest

    # Video Walkthrough: https://youtu.be/hdO9Ry8d5FU?t=39m33s
def deep_map(f, link):
    """Return a Link with the same structure as link but with fn mapped over
    its elements. If an element is an instance of a linked list, recursively
    apply f inside that linked list as well.

    >>> s = Link(1, Link(Link(2, Link(3)), Link(4)))
    >>> print(deep_map(lambda x: x * x, s))
    <1 <4 9> 16>
    >>> print(s) # unchanged
    <1 <2 3> 4>
    >>> print(deep_map(lambda x: 2 * x, Link(s, Link(Link(Link(5))))))
    <<2 <4 6> 8> <<10>>>
    """
    if link is Link.empty:
        return link
    if isinstance(link.first, Link):
        first = deep_map(f, link.first)
    else:
        first = f(link.first)
    return Link(first, deep_map(f, link.rest))
class Mint:
    """A mint creates coins by stamping on years.

    The update method sets the mint's stamp to Mint.current_year.

    >>> mint = Mint()
    >>> mint.year
    2019
    >>> dime = mint.create(Dime)
    >>> dime.year
    2019
    >>> Mint.current_year = 2100  # Time passes
    >>> nickel = mint.create(Nickel)
    >>> nickel.year     # The mint has not updated its stamp yet
    2019
    >>> nickel.worth()  # 5 cents + (81 - 50 years)
    36
    >>> mint.update()   # The mint's year is updated to 2100
    >>> Mint.current_year = 2175     # More time passes
    >>> mint.create(Dime).worth()    # 10 cents + (75 - 50 years)
    35
    >>> Mint().create(Dime).worth()  # A new mint has the current year
    10
    >>> dime.worth()     # 10 cents + (156 - 50 years)
    116
    >>> Dime.cents = 20  # Upgrade all dimes!
    >>> dime.worth()     # 20 cents + (156 - 50 years)
    126

    >>> m = Mint()
    >>> n = m.create(Nickel)
    >>> n.worth()
    5
    >>> n.year = 2019
    >>> n.worth()
    113

    """
    current_year = 2019

    def __init__(self):
        self.update()

    def create(self, kind):
        return kind(self.year)

    def update(self):
        self.year = Mint.current_year

class Coin:
    def __init__(self, year):
        self.year = year

    def worth(self):
        return self.cents + max(0, Mint.current_year - self.year - 50)

class Nickel(Coin):
    cents = 5

class Dime(Coin):
    cents = 10

## Link Class ##

class Link:
    """A linked list.

    >>> s = Link(1)
    >>> s.first
    1
    >>> s.rest is Link.empty
    True
    >>> s = Link(2, Link(3, Link(4)))
    >>> s.first = 5
    >>> s.rest.first = 6
    >>> s.rest.rest = Link.empty
    >>> s                                    # Displays the contents of repr(s)
    Link(5, Link(6))
    >>> s.rest = Link(7, Link(Link(8, Link(9))))
    >>> s
    Link(5, Link(7, Link(Link(8, Link(9)))))
    >>> print(s)                             # Prints str(s)
    <5 7 <8 9>>
    """
    empty = ()

    def __init__(self, first, rest=empty):
        assert rest is Link.empty or isinstance(rest, Link)
        self.first = first
        self.rest = rest

    def __repr__(self):
        if self.rest is not Link.empty:
            rest_repr = ', ' + repr(self.rest)
        else:
            rest_repr = ''
        return 'Link(' + repr(self.first) + rest_repr + ')'

    def __str__(self):
        string = '<'
        while self.rest is not Link.empty:
            string += str(self.first) + ' '
            self = self.rest
        return string + str(self.first) + '>'

## Tree ADT ##

class Tree:
    """
    >>> t = Tree(3, [Tree(2, [Tree(5)]), Tree(4)])
    >>> t.label
    3
    >>> t.branches[0].label
    2
    >>> t.branches[1].is_leaf()
    True
    """
    def __init__(self, label, branches=[]):
        for b in branches:
            assert isinstance(b, Tree)
        self.label = label
        self.branches = list(branches)

    def is_leaf(self):
        return not self.branches

    def map(self, fn):
        """
        Apply a function `fn` to each node in the tree and mutate the tree.

        >>> t1 = Tree(1)
        >>> t1.map(lambda x: x + 2)
        >>> t1.map(lambda x : x * 4)
        >>> t1.label
        12
        >>> t2 = Tree(3, [Tree(2, [Tree(5)]), Tree(4)])
        >>> t2.map(lambda x: x * x)
        >>> t2
        Tree(9, [Tree(4, [Tree(25)]), Tree(16)])
        """
        self.label = fn(self.label)
        for b in self.branches:
            b.map(fn)

    def __contains__(self, e):
        """
        Determine whether an element exists in the tree.

        >>> t1 = Tree(1)
        >>> 1 in t1
        True
        >>> 8 in t1
        False
        >>> t2 = Tree(3, [Tree(2, [Tree(5)]), Tree(4)])
        >>> 6 in t2
        False
        >>> 5 in t2
        True
        """
        if self.label == e:
            return True
        for b in self.branches:
            if e in b:
                return True
        return False

    def __repr__(self):
        if self.branches:
            branch_str = ', ' + repr(self.branches)
        else:
            branch_str = ''
        return 'Tree({0}{1})'.format(self.label, branch_str)

    def __str__(self):
        def print_tree(t, indent=0):
            tree_str = '  ' * indent + str(t.label) + "\n"
            for b in t.branches:
                tree_str += print_tree(b, indent + 1)
            return tree_str
        return print_tree(self).rstrip()

