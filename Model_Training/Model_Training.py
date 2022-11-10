import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ViTConfig:
    """Store all the hyperparameters here"""
    num_classes: int = 2 #Can be changed to multi-classed classification
    input_shape: Tuple[int, int, int] = (3, 3, 21) #depends on the size of the image we want
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 256
    num_epochs: int = 100
    image_size: int = 72  
    patch_size: int = 6  
    projection_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 8
    mlp_head_units: Tuple[int, int] = (2048, 1024)

    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ] 
        self.mlp_head_units = list(self.mlp_head_units)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
        

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
def create_vit_classifier(config: ViTConfig):
    inputs = layers.Input(shape=config.input_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)
    augmented = inputs # TODO: remember to change
    # Create patches.
    patches = Patches(config.patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(config.num_patches, config.projection_dim)(patches)

    attention_score_dict = {}
    # Create multiple layers of the Transformer block.
    for _ in range(config.transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        MHA_layer = layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.projection_dim,
            dropout=0.1
        )
        attention_output, attention_scores = MHA_layer(x1, x1, return_attention_scores=True)
        # Save attention scores
        attention_score_dict[MHA_layer.name] = attention_scores
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=config.transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=config.mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(config.num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=[logits, attention_score_dict])
    return model

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================
def run_experiment(model, config: ViTConfig):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=30,
        epochs=20,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


if __name__ == "__main__":
    #Import the data generated from Data_Pre_Processing
    import numpy as np
    ##Process the data separately and load them since processing the data takes quite some time and we don't want to process it evertime we run the model
    x_test = np.load('/content/drive/MyDrive/Research/Data_project/x_test3.npy')
    x_train  = np.load('/content/drive/MyDrive/Research/Data_project/x_train3.npy')
    y_test  = np.load('/content/drive/MyDrive/Research/Data_project/y_test3.npy')
    y_train  = np.load('/content/drive/MyDrive/Research/Data_project/y_train3.npy')

    config = ViTConfig()
    image_size = config.image_size

    #Augment the dataset by random flip and rotation
    more_data = keras.Sequential(
        [layers.Normalization(),layers.Resizing(image_size, image_size),layers.RandomZoom(height_factor=0.2, width_factor=0.2),],
        name="more_data",)
    more_data.layers[0].adapt(x_train)


    vit = create_vit_classifier(config)
    history = run_experiment(vit, config)

    #Save the trained model
    vit.save('path')




