import matplotlib.pyplot as plt

# Create toy data.
xdata = [0.0,1.0,2.0,3.0,4.0,5.0]
ydata = [0.1,0.9,2.2,2.8,3.9,5.1]
sigma = [1.0,1.0,1.0,1.0,1.0,1.0]

# Initial guess and learning rate.
x0    = [0.5, 1.5]
alpha = 0.01

# Store initial guess for plotting.
A = [x0[0]]
B = [x0[1]]

# Iteration
for i in range(50):
    a = x0[0]
    b = x0[1]
    # Compute gradients of chi2 w.r.t a and b.
    # Do the math first before trying to understand
    # this piece of code!
    grad_a = 0.0
    grad_b = 0.0
    for n in range(len(xdata)):
        grad_a = grad_a - 2.0*(ydata[n] - a - b*xdata[n])*1.0/(sigma[n]*sigma[n])
        grad_b = grad_b - 2.0*(ydata[n] - a - b*xdata[n])*xdata[n]/(sigma[n]*sigma[n])
    # Update parameters.
    a_new = a - alpha*grad_a
    b_new = b - alpha*grad_b
    x0    = [a_new,b_new]

    # Store parameters for plotting.
    A.append(a_new)
    B.append(b_new)

# Plot route of gradient descent.
plt.figure(1)
plt.plot(A, B, 'o-', ms=6, lw=3, color='blue')
plt.plot([0.0], [1.0], 'x', ms=12, markeredgewidth=3, color='orange')
plt.xlim(-0.05,0.55)
plt.ylim(0.75,1.55)
plt.xlabel(r'$a$', fontsize=24)
plt.ylabel(r'$b$', fontsize=24)
plt.savefig('example-gradient-descent.png')
plt.show()
