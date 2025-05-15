
def train_manual(train_set, val_set, epochs, lr):
    n_params = len(train_set[0][0])

    weights = np.random.normal(-1, 1, n_params)
    bias = 1

    train_loss = []
    # val_loss = []

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch", leave=False):
        running_loss = 0.0
        for x, y in train_set[:100]:
            # make linear prediction
            y_pred = 0.0
            for i in range(n_params):
                y_pred += weights[i] * x[i]

            y_pred = y_pred + bias
            # calculate MSE
            loss = (y_pred - y) ** 2
            running_loss += loss

            # update weights
            for i in range(n_params):
                weight_grad = x[i] * 2 * (y_pred - y)
                weights[i] = weights[i] - lr * weight_grad

            # update bias
            bias_grad = 2 * (y - y_pred)
            bias = bias - lr * bias_grad

        train_loss.append(running_loss / len(train_set))

        # make sample prediction
        x, y = train_set[0]
        y_pred = 0.0
        for i in range(n_params):
            y_pred += weights[i] * x[i]
        y_pred += y_pred + bias
        print(f"Prediction: {y_pred}, True Value: {y}")

    return train_loss

 