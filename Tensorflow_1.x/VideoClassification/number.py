global_step = 29
learning_rate = 1e-1
max_epochs = 30

epochs = max_epochs // 3

def where(condition, x1, x2):
    if condition: return x1
    return x2

learning_rate = where(
    global_step < epochs, 
    learning_rate, 

    where(
        global_step < (epochs * 2), 
        learning_rate * 0.1, 
        learning_rate * 0.01
    )
)

print(epochs, learning_rate)