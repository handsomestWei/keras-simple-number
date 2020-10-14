

from keras import Sequential
from keras.layers import Dense, Dropout

# 3层全连接模型


def build_simple_fnn_model(input_size, units, output_size):
    model = Sequential()

    model.add(Dense(units=units,
                    input_shape=(input_size,),
                    use_bias=True,
                    kernel_initializer='random_normal',
                    activation='sigmoid'))
    model.add(Dropout(rate=0.5, seed=0.2))

    model.add(Dense(units,
                    use_bias=True,
                    kernel_initializer='random_normal',
                    activation='relu'))
    model.add(Dropout(rate=0.4, seed=0.3))

    model.add(Dense(units,
                    use_bias=True,
                    kernel_initializer='random_normal',
                    activation='tanh'))
    model.add(Dropout(rate=0.3, seed=0.4))

    model.add(Dense(output_size,
                    activation='softmax'))

    return model
