def printInfo(iters, prev_x, df, df2, cur_x, prev_step_size):
    print("Iteration :", iters)
    print("previous x :", prev_x)
    print("gradient df:", df)
    print("second derivative df2:", df2)
    print("delta x :", df / df2)
    print("current x :", cur_x)
    print("step size :", prev_step_size)
    print()

current = 3    # Algorithm starts at x=3
precision = 0.00001
previous_step_size = 1
max_iters = 10000   # Maximum number of iterations
iters = 0   # Iteration counter

df = lambda x: 9*x**2 - 24*x + 5    # First derivative
df2 = lambda x: 18*x - 24     # Second derivative

while (previous_step_size > precision) & (iters < max_iters) :
    previous = current
    current -= df(previous) / df2(previous)
    previous_step_size = abs(current - previous)
    iters +=1
    printInfo(iters, previous, df(previous), df2(previous), current, previous_step_size)

if iters >= max_iters:
    print("local minimum not found")
    print("value found after max iterations : ", current)
else:
    print("The local minimum occurs at:", current)
    print("Number of iterations used:", iters)