# %%
import numpy as np
import tensorflow as tf
import cv2
from Model_Training import ViTConfig


def attention_rollout_map(image, attention_score_dict):
    """
    Taken from https://keras.io/examples/vision/probing_vits/
    Computes attention rollout mask
    """

    # Stack the individual attention matrices from individual Transformer blocks.
    attn_mat = tf.stack([attention_score_dict[k] for k in attention_score_dict.keys()])
    attn_mat = tf.squeeze(attn_mat, axis=1)

    # Average the attention weights across all heads.
    attn_mat = tf.reduce_mean(attn_mat, axis=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_attn = tf.eye(attn_mat.shape[1])
    aug_attn_mat = attn_mat + residual_attn
    aug_attn_mat = aug_attn_mat / tf.reduce_sum(aug_attn_mat, axis=-1)[..., None]
    aug_attn_mat = aug_attn_mat.numpy()

    # Recursively multiply the weight matrices.
    joint_attentions = np.zeros(aug_attn_mat.shape)
    joint_attentions[0] = aug_attn_mat[0]

    for n in range(1, aug_attn_mat.shape[0]):
        joint_attentions[n] = np.matmul(aug_attn_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_attn_mat.shape[-1]))
    mask = v[0].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), image.size)[..., np.newaxis]
    masked_image = (mask * image).astype("uint8")
    return masked_image, mask


# def attention_heatmap(image, attention_score_dict, config: ViTConfig):
#     """
#     Taken from https://keras.io/examples/vision/probing_vits/
#     Get attention weights of every head in the last layer
#     """
#     patch_size = config.patch_size
#     num_heads = config.num_heads

#     # Sort the Transformer blocks in order of their depth.
#     attention_score_list = list(attention_score_dict.keys())
#     attention_score_list.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)

#     # Process the attention maps for overlay.
#     w_featmap = image.shape[2] // patch_size
#     h_featmap = image.shape[1] // patch_size
#     attention_scores = attention_score_dict[attention_score_list[0]]

#     # Taking the representations from CLS token.
#     attentions = attention_scores[0, :, 0, :].reshape(num_heads, -1)

#     # Reshape the attention scores to resemble mini patches.
#     attentions = attentions.reshape(num_heads, w_featmap, h_featmap)
#     attentions = attentions.transpose((1, 2, 0))

#     # Resize the attention patches to 224x224 (224: 14x16).
#     attentions = tf.image.resize(
#         attentions, size=(h_featmap * patch_size, w_featmap * patch_size)
#     )
#     return attentions

# %%
if __name__ == "__main__":
    """
    Adapted from https://keras.io/examples/vision/probing_vits/
    """
    from Model_Training import create_vit_classifier
    import tensorflow.keras as keras
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Turn GPU off. Somehow not working.
    import requests
    from PIL import Image
    from io import BytesIO
    import matplotlib.pyplot as plt

    RESOLUTION = 72

    crop_layer = keras.layers.CenterCrop(RESOLUTION, RESOLUTION)
    norm_layer = keras.layers.Normalization(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
    )
    rescale_layer = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)


    def preprocess_image(image, model_type, size=RESOLUTION):
        # Turn the image into a numpy array and add batch dim.
        image = np.array(image)
        image = tf.expand_dims(image, 0)

        # If model type is vit rescale the image to [-1, 1].
        if model_type == "original_vit":
            image = rescale_layer(image)

        # Resize the image using bicubic interpolation.
        resize_size = int((256 / 224) * size)
        image = tf.image.resize(image, (resize_size, resize_size), method="bicubic")

        # Crop the image.
        image = crop_layer(image)

        # If model type is DeiT or DINO normalize the image.
        if model_type != "original_vit":
            image = norm_layer(image)

        return image.numpy()


    def load_image_from_url(url, model_type):
        # Credit: Willi Gierke
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        preprocessed_image = preprocess_image(image, model_type)
        return image, preprocessed_image


    img_url = "https://dl.fbaipublicfiles.com/dino/img.png"
    image, preprocessed_image = load_image_from_url(img_url, model_type="original_vit")

    plt.imshow(image)
    plt.title("Original image")
    plt.axis("off")
    plt.show()
  
    config = ViTConfig(input_shape=(RESOLUTION,RESOLUTION,3))
    vit = create_vit_classifier(config)
    predictions, attention_score_dict = vit(preprocessed_image)

    # Test attention rollout
    masked_image, mask = attention_rollout_map(image, attention_score_dict)
    plt.imshow(masked_image)
    plt.title("Masked image")
    plt.axis("off")
    plt.show()

    # Test attention heatmap
    # attentions = attention_heatmap(preprocessed_image, attention_score_dict, config)
    

# %%
