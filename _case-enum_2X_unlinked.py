

# The case enumeration how many blocks on average are destroyed by 2X on unlinked trap functions.


def crossover(x, y, split1, split2):
    assert len(x) is len(y)
    N = len(x)
    min_split = min(split1, split2)
    max_split = max(split1, split2)
    # from the minimum split value to the maximum split value, the values are swapped
    y1 = [x[i] for i in range(0, min_split)] + [y[i] for i in range(min_split, max_split)] + [x[i] for i in range(max_split, N)]
    y2 = [y[i] for i in range(0, min_split)] + [x[i] for i in range(min_split, max_split)] + [y[i] for i in range(max_split, N)]
    return y1, y2

def count_intact(x):
    intact = 0
    for i in range(10):
        if x[i] == x[i+10] == x[i+20] == x[i+30]:
            intact += 1
    return intact / 10

l = 40
x1 = [0 for _ in range(40)]
x2 = [1 for _ in range(40)]
intact = 0
for split1 in range(40):
    for split2 in range(40):
        y1, y2 = crossover(x1, x2, split1, split2)
        intact += count_intact(y1) # y2 would work too
print(intact / (40*40))