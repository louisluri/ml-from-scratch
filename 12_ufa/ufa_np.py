import numpy as np
import matplotlib.pyplot as plt

def mse_loss(preds, targets):
    """returns the loss and the gradient to be passed back"""
    n = len(targets)
    loss = np.sum(((preds - targets)**2) / n)
    grad = 2 * (preds - targets) / n
    return loss, grad

class ufa:
    def __init__(self, hidden_dim):
        self.w1 = np.random.randn(hidden_dim, 1) * np.sqrt(2/1)
        self.w2 = np.random.randn(1, hidden_dim) * np.sqrt(2/hidden_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

        self.w1g = np.zeros_like(self.w1)
        self.b1g = np.zeros_like(self.b1)
        self.w2g = np.zeros_like(self.w2)
        self.b2g = np.zeros_like(self.b2)

        self.x = None
        self.x1 = None
        self.x2 = None

    def forward(self, x):
        self.x = x
        self.x1 = (self.w1 @ x) + self.b1
        self.x2 = np.maximum(0, self.x1) # Correct way to ReLU
        return (self.w2 @ self.x2) + self.b2
    
    def __call__(self, x):
        return self.forward(x.reshape(1, -1)).flatten()
    
    def backward(self, grad):
        grad = grad.reshape(1, -1)
        self.b2g += np.sum(grad, axis=-1, keepdims=True)
        self.w2g += grad @ self.x2.T

        grad = self.w2.T @ grad
        grad[self.x1 < 0] = 0.0

        self.b1g += np.sum(grad, axis=-1, keepdims=True)
        self.w1g += grad @ self.x.T

    def step(self, lr):
        self.w1g = np.clip(self.w1g, -1, 1)
        self.w2g = np.clip(self.w2g, -1, 1)
        self.w1 -= lr * self.w1g
        self.b1 -= lr * self.b1g
        self.w2 -= lr * self.w2g
        self.b2 -= lr * self.b2g

    def zero_grad(self):
        self.w1g = np.zeros_like(self.w1)
        self.b1g = np.zeros_like(self.b1)
        self.w2g = np.zeros_like(self.w2)
        self.b2g = np.zeros_like(self.b2)

x = np.linspace(-10, 10, 100)
y = np.sin(x)/x #np.abs(x)

plt.ion()
fig, ax = plt.subplots(facecolor="#996996")
ax.plot(x, y, label='Function', color='blue', linewidth=2)

ax.relim()
ax.autoscale_view()
ax.set_autoscalex_on(False)
ax.set_autoscaley_on(False)

m = ufa(1024)
approx, = ax.plot(x, m(x), label='Approximation', color='red', linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle='--', alpha=0.7)
#plt.legend(loc="upper right")

lr_start = 1e-2
lr_end = 3e-3
num_epochs = 100000

m.zero_grad()
for epoch in range(num_epochs):
    if not plt.fignum_exists(fig.number):
        break
    
    lr = lr_start * (lr_end / lr_start) ** (epoch / num_epochs) # Exponential decay

    preds = m(x)
    loss, grad = mse_loss(preds, y)
    
    m.backward(grad)
    m.step(lr)
    m.zero_grad()
    
    if (epoch+1) % 50 == 0:
        approx.set_ydata(preds)
        plt.title(f"Epoch: {epoch+1} | Loss: {loss:.2e} | lr: {lr:.2e}")
        plt.draw()
        plt.pause(0.000001)

        if(epoch+1) % 1000 == 0: print(f"Epoch {epoch+1}, Loss: {loss:.2e}, lr: {lr:.2e}")

plt.ioff()
plt.show()