
## Sentiment experiments

Running `model.py` creates a model, evaluates it, and exports the parameters.

With current settings, the model created should be as follows:

    Train on 8000 samples, validate on 2000 samples
    Epoch 1/5
    8000/8000 [==============================] - 327s 41ms/step - loss: 0.4954 - acc: 0.7474 - val_loss: 0.3467 - val_acc: 0.8565
    Epoch 2/5
    8000/8000 [==============================] - 334s 42ms/step - loss: 0.2794 - acc: 0.8841 - val_loss: 0.3062 - val_acc: 0.8705
    Epoch 3/5
    8000/8000 [==============================] - 339s 42ms/step - loss: 0.1517 - acc: 0.9486 - val_loss: 0.3239 - val_acc: 0.8705
    Epoch 4/5
    8000/8000 [==============================] - 399s 50ms/step - loss: 0.0552 - acc: 0.9861 - val_loss: 0.3728 - val_acc: 0.8665
    Epoch 5/5
    8000/8000 [==============================] - 308s 39ms/step - loss: 0.0166 - acc: 0.9988 - val_loss: 0.3962 - val_acc: 0.8740


Running `evaluate.py` splits the training data into sentences and evaluate the sentences according to the model.