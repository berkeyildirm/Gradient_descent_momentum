def printInfo(iters, prev_x, df, cur_x, prev_step_size, velocity):
    print("Iteration :", iters)
    print("previous x :", prev_x)
    print("gradient df:", df)
    print("current x :", cur_x)
    print("step size :", prev_step_size)
    print("velocity :", velocity)
    print()


cur_x = 3    # Algorithm starts at x=3
gamma = 0.001   # Step size
precision = 0.0001
previous_step_size = 1
max_iters = 10000   # Maximum iterations
iters = 0   # Iteration counter
momentum = 0.9
velocity = 0

df = lambda x: 9*x**2 - 24*x + 5    # First derivative


while (previous_step_size > precision) & (iters < max_iters) :
    prev_x = cur_x
    velocity = momentum * velocity + gamma * df(prev_x)
    cur_x -=  velocity
    previous_step_size = abs(cur_x - prev_x)
    iters +=1
    printInfo(iters, prev_x, df(prev_x), cur_x, previous_step_size, velocity)

if iters >= max_iters:
    print("local minimum not found")
    print("value found after max iterations: ", cur_x)
else:
    print("The local minimum occurs at:", cur_x)
    print("Number of iterations used:", iters)
   