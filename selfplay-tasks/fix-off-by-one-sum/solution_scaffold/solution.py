def sum_to_n(n: int) -> int:
    total = 0
    for i in range(1, n):  # BUG: off by one — excludes n itself
        total += i
    return total
