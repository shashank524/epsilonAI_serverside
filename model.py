import keras.layers as layers
from keras.models import Sequential

def load_model():
    """
    Returns the spatiotemporal autoencoder model for abnormal event detection in videos.
    """
    model = Sequential()
    model.add(layers.Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1),
                            padding='valid', input_shape=(227, 227, 10, 1), activation='tanh'))
    model.add(layers.Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1),
                            padding='valid', activation='tanh'))

    # Add ConvLSTM layers
    convlstm_params = {
        'filters': [64, 32, 64],
        'kernel_size': (3, 3),
        'strides': 1,
        'padding': 'same',
        'dropout': [0.4, 0.3, 0.5],
        'return_sequences': True
    }
    for i in range(len(convlstm_params['filters'])):
        model.add(layers.ConvLSTM2D(filters=convlstm_params['filters'][i],
                                    kernel_size=convlstm_params['kernel_size'],
                                    strides=convlstm_params['strides'],
                                    padding=convlstm_params['padding'],
                                    dropout=convlstm_params['dropout'][i],
                                    return_sequences=convlstm_params['return_sequences']))

    # Add Conv3DTranspose layers
    model.add(layers.Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1),
                                     padding='valid', activation='tanh'))
    model.add(layers.Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1),
                                     padding='valid', activation='tanh'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model
