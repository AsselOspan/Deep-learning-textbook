"""Та же сеть 2–2–2 в TensorFlow/Keras, тензоры и параметры float64."""
import tensorflow as tf


def run():
    x = tf.constant([[1., 2.]], dtype=tf.float64)
    target = tf.constant([[0.2, 0.8]], dtype=tf.float64)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2,), dtype='float64'),
        tf.keras.layers.Dense(2, activation='sigmoid', dtype='float64'),
        tf.keras.layers.Dense(2, activation='sigmoid', dtype='float64'),
    ])
    # В Keras строки kernel соответствуют входам, столбцы — выходам.
    model.layers[0].set_weights([
        tf.constant([[0.20, 0.30], [-0.10, 0.05]], tf.float64).numpy(),
        tf.constant([0.20, -0.30], tf.float64).numpy(),
    ])
    model.layers[1].set_weights([
        tf.constant([[0.40, -0.10], [-0.20, 0.30]], tf.float64).numpy(),
        tf.constant([0.10, -0.10], tf.float64).numpy(),
    ])
    with tf.GradientTape() as tape:
        before = 0.5 * tf.reduce_sum(tf.square(model(x) - target))
    parameters = model.trainable_variables
    gradients = tape.gradient(before, parameters)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)
    optimizer.apply_gradients(zip(gradients, parameters))
    after = 0.5 * tf.reduce_sum(tf.square(model(x) - target))
    return float(before), float(after), [g.numpy() for g in gradients]


if __name__ == '__main__':
    before, after, gradients = run()
    print(f'TensorFlow {tf.__version__}: {before:.12f} → {after:.12f}')
    for name, grad in zip(('kernel1', 'b1', 'kernel2', 'b2'), gradients):
        print(name, grad)
