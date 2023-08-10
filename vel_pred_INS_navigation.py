import csv
import tensorflow as tf
import numpy as np

class IMUDataPreprocessor:
    def __init__(self, window_size, orientation_source='training'):
        self.window_size = window_size
        self.orientation_source = orientation_source

    def _rotate_body_to_navigation(self, imu_samples, orientations):
        # Assuming imu_samples shape: (batch_size, window_size, num_features)
        # Assuming orientations shape: (batch_size, 4) (quaternions)

        # Rotate imu_samples from body frame (ωb, αb) to navigation frame (ωn, αn)
        # Implement the rotation logic using quaternions.


        # For example (using random rotations, you should replace this with your implementation):
        rotated_samples = imu_samples + np.random.randn(*imu_samples.shape)

        return rotated_samples

    def prepare_dataset(self, imu_samples, orientations=None):
      # Assuming imu_samples shape: (batch_size, num_timesteps, num_features)

      dataset_X, dataset_gt_vel = [], []

      # Slide the window over the IMU samples to create the datasets
      for t in range(self.window_size - 1, imu_samples.shape[1] + 1):
        window_start = t - self.window_size
        window_end = t


        window_X = imu_samples[:, window_start:window_end, :]

        if self.orientation_source == 'training':
          # Use provided orientations for training
            if orientations is not None and t <= orientations.shape[1]:
              window_rotated_X = self._rotate_body_to_navigation(window_X, orientations[:, t - 1])
            else:
              raise ValueError("Missing orientations for sliding window index: {}".format(t))
        else:
          # Use device orientation estimated from IMU for testing
          window_rotated_X = self._rotate_body_to_navigation(window_X, imu_samples[:, t - 1])

        dataset_X.append(window_rotated_X)

        # Assuming ground truth velocities are present in the IMU data (adjust the indices as per your data)
        window_gt_vel = imu_samples[:, window_end - 1, :3]  # Assuming 3D velocity (x, y, z)
        dataset_gt_vel.append(window_gt_vel)

      dataset_X = np.concatenate(dataset_X, axis=1)  # Concatenate along the second axis
      dataset_gt_vel = np.stack(dataset_gt_vel, axis=1)

      return dataset_X, dataset_gt_vel


class SpatialEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialEmbedding, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(64, activation='relu')

    def call(self, inputs):
        # Assuming inputs shape: (batch_size, window_size, num_features)

        # Apply 1D convolution followed by batch normalization and dense layer
        spatial_embedded = self.conv1d(inputs)
        spatial_embedded = self.batch_norm(spatial_embedded)
        spatial_embedded = self.dense(spatial_embedded)
        print(spatial_embedded.shape)
        return spatial_embedded

class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_timesteps, num_features, window_size):
        super(TemporalEmbedding, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.window_size = window_size
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True)
        )
        self.positional_encoding = tf.keras.layers.Dense(32, activation='relu')
        self.dense_reshape = tf.keras.layers.Dense(6, activation='relu')  # Add dense layer for reshaping

    def call(self, inputs):
        # Assuming inputs shape: (batch_size, num_timesteps, num_features).

        # Apply bidirectional LSTM to exploit temporal information
        temporal_embedded = self.bidirectional_lstm(inputs)
        print('step1',temporal_embedded)

        # Add positional encoding using a trainable neural network
        positional_encoding = self.positional_encoding(tf.ones_like(temporal_embedded))

        # Calculate the number of windows that can be created
        num_windows = self.num_timesteps - self.window_size + 1

        # Tile the positional encoding tensor to match the temporal_embedded tensor's shape
        positional_encoding = tf.tile(positional_encoding, [1, num_windows, 1])

        # Reshape the positional encoding to match the temporal_embedded shape
        positional_encoding_reshaped = tf.reshape(positional_encoding, [-1, num_windows, 32])

        print("Temporal embedded shape before concatenation:", temporal_embedded.shape)

        # Concatenate temporal and positional embeddings
        temporal_embedded = tf.concat([temporal_embedded, positional_encoding], axis=-1)
        print('step2 after concat',temporal_embedded)
        # Apply dense layer for reshaping
        temporal_embedded = self.dense_reshape(temporal_embedded)
        print('step3 after dense layer',temporal_embedded.shape)
        return temporal_embedded


class LocalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim):
        super(LocalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_q = tf.keras.layers.Conv1D(48, kernel_size=1, strides=1, padding="valid") # 48 = num_heads * head_dim
        self.conv_k = tf.keras.layers.Conv1D(48, kernel_size=1, strides=1, padding="valid")
        self.conv_v = tf.keras.layers.Conv1D(48, kernel_size=1, strides=1, padding="valid")

    def call(self, x):
        print('input attention', x.shape)
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        print("q shape:", q.shape)
        print("k shape:", k.shape)
        print("v shape:", v.shape)

        q = tf.reshape(q, (batch_size, 100 , 8, 6))
        k = tf.reshape(k, (batch_size, 100, 8,6))
        v = tf.reshape(v, (batch_size, 100, 8,6)) #batch_size, seq_length, self.num_heads, self.head_dim
        print("q shape after:", q.shape)
        print("k shape after:", k.shape)
        print("v shape after:", v.shape)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_scores, v)
        output = tf.reshape(output, (tf.shape(output)[0], tf.shape(output)[1], 48))

        return output



class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim):
        super(GlobalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_q = tf.keras.layers.Conv1D(48, kernel_size=1, strides=1, padding="valid")
        self.conv_k = tf.keras.layers.Conv1D(48, kernel_size=1, strides=1, padding="valid")
        self.conv_v = tf.keras.layers.Conv1D(48, kernel_size=1, strides=1, padding="valid")

    def call(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        q = tf.reshape(q, (tf.shape(q)[0], tf.shape(q)[1], 8, 6))
        k = tf.reshape(k, (tf.shape(k)[0], tf.shape(k)[1], 8, 6))
        v = tf.reshape(v, (tf.shape(v)[0], tf.shape(v)[1], 8, 6))

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_scores, v)
        output = tf.reshape(output, (tf.shape(output)[0], tf.shape(output)[1], 48))

        return output



class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim #input_dim // num_heads

        self.query = tf.keras.layers.Dense(input_dim)
        self.key = tf.keras.layers.Dense(input_dim)
        self.value = tf.keras.layers.Dense(input_dim)
        self.fc_out = tf.keras.layers.Dense(input_dim)

    def call(self, x):
        batch_size, seq_len, _ = x.shape

        # Split input into multiple heads and transpose for attention computation
        query = tf.reshape(self.query(x), (batch_size, seq_len, self.num_heads, self.head_dim))
        key = tf.reshape(self.key(x), (batch_size, seq_len, self.num_heads, self.head_dim))
        value = tf.reshape(self.value(x), (batch_size, seq_len, self.num_heads, self.head_dim))

        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # Compute scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # Apply attention to values and concatenate heads
        attention_output = tf.matmul(attention_probs, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, -1))

        # Apply fully connected layer
        output = self.fc_out(attention_output)

        return output


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_heads):
        super(BottleneckBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = tf.keras.layers.Dense(input_dim)
        self.key = tf.keras.layers.Dense(input_dim)
        self.value = tf.keras.layers.Dense(input_dim)
        self.fc_out = tf.keras.layers.Dense(input_dim)

    def call(self, x):
        batch_size, seq_len, _ = x.shape
        print("Input shape:", x.shape)
        tf.debugging.assert_shapes([(x, (batch_size, seq_len, _))])

        # Split input into multiple heads and transpose for attention computation
        query = tf.reshape(self.query(x), (batch_size, seq_len, self.num_heads, self.head_dim))
        key = tf.reshape(self.key(x), (batch_size, seq_len, self.num_heads, self.head_dim))
        value = tf.reshape(self.value(x), (batch_size, seq_len, self.num_heads, self.head_dim))
        print("Query shape:", query.shape)
        print("Key shape:", key.shape)
        print("Value shape:", value.shape)

        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        tf.debugging.assert_shapes([(query, (batch_size, self.num_heads, seq_len, self.head_dim))])
        tf.debugging.assert_shapes([(key, (batch_size, self.num_heads, seq_len, self.head_dim))])
        tf.debugging.assert_shapes([(value, (batch_size, self.num_heads, seq_len, self.head_dim))])

        # Compute scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # Apply attention to values and concatenate heads
        attention_output = tf.matmul(attention_probs, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, -1))

        # Apply fully connected layer
        output = self.fc_out(attention_output)
        print("Attention output shape:", attention_output.shape)
        print("Output shape:", output.shape)
        tf.debugging.assert_shapes([(attention_output, (batch_size, seq_len, -1))])
        tf.debugging.assert_shapes([(output, (batch_size, seq_len, -1))])
        return output



class ModifiedBottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_heads, head_dim, use_local_attention, use_global_attention):
        super(ModifiedBottleneckBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.local_self_attention = LocalSelfAttention(num_heads, self.head_dim)
        self.global_self_attention = GlobalSelfAttention(num_heads, self.head_dim)
        self.conv1x1 = tf.keras.layers.Conv1D(input_dim, kernel_size=1, activation="relu")


    def call(self, x):
        local_attention_output = self.local_self_attention(x)
        global_attention_output = self.global_self_attention(x)

        print("local_attention_output shape:", local_attention_output.shape)
        print("global_attention_output shape:", global_attention_output.shape)

        attention_output = local_attention_output + global_attention_output
        conv_output = self.conv1x1(attention_output)
        return conv_output


class SpatialEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_heads, num_layers, head_dim, use_local_attention, use_global_attention):
        super(SpatialEncoder, self).__init__()
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.layers = [ModifiedBottleneckBlock(input_dim, num_heads, head_dim, use_local_attention, use_global_attention) for _ in range(num_layers)]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        print('shape output encoder:', x.shape)
        return x





class TemporalDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, head_dim):
        super(TemporalDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layers = [self._build_layer() for _ in range(num_layers)]

    def _build_layer(self):
        masked_self_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.head_dim)
        feed_forward = tf.keras.layers.Dense(self.head_dim, activation="relu")
        multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.head_dim)
        return [masked_self_attention, feed_forward, multi_head_attention]

    def call(self, x, encoder_output):
        for i in range(self.num_layers):
            masked_self_attention, feed_forward, multi_head_attention = self.layers[i]
            # Apply attention and feed-forward layers to each window independently
            x = masked_self_attention(x, x)
            x = feed_forward(x)
            print('encoder output',encoder_output.shape)
            #encoder_output = tf.reshape(encoder_output, (batch_size, 100, 8, 6))
            # Concatenate the output of self-attention and feed-forward sub-layers
            fused_x = tf.concat([x, encoder_output], axis=-1)
            #fused_x = encoder_output
            # Pass through the multi-head attention sub-layer over encoder's output
            fused_x = multi_head_attention(fused_x, encoder_output)

            # Apply position-wise fully connected feed-forward sub-layer
            x = tf.keras.layers.Dense(self.head_dim, activation="relu")(fused_x)

        return x




class CTIN(tf.keras.Model):
      def __init__(self, num_timesteps, num_features, window_size, num_layers, num_heads, head_dim):
        super(CTIN, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.window_size = window_size

        self.spatial_embedding = tf.keras.layers.Conv1D(head_dim, kernel_size=1, strides=1, padding="valid")
        self.temporal_embedding = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(head_dim, return_sequences=True)
        )

        self.spatial_encoder = SpatialEncoder(input_dim=head_dim, num_heads=num_heads, num_layers=num_layers, head_dim=head_dim, use_local_attention=True, use_global_attention=True)
        self.temporal_decoder = TemporalDecoder(num_layers, num_heads, head_dim)

        self.concat_dense = tf.keras.layers.Dense(head_dim, activation="relu")  # Add dense layer for reshaping


        self.velocity_mlp = tf.keras.layers.Dense(2)
        self.covariance_mlp = tf.keras.layers.Dense(4)

        self.local_self_attention = LocalSelfAttention(num_heads, head_dim)
        self.global_self_attention = GlobalSelfAttention(num_heads, head_dim)

        self.conv1x1 = tf.keras.layers.Conv1D(head_dim, kernel_size=1, activation="relu")




      def call(self, inputs):
        spatial_embedded = self.spatial_embedding(inputs)
        batch_size = tf.shape(inputs)[0]
        num_windows = self.num_timesteps - self.window_size + 1
        # Tile spatial_embedded to match the time dimension of temporal_embedded
        spatial_embedded = tf.tile(spatial_embedded, [1, self.num_timesteps, 1])
        print("spatial_embedded shape:", spatial_embedded.shape)

        temporal_embedded = self.temporal_embedding(inputs)
        print("temporal embedded shape",temporal_embedded.shape)

        # Apply dense layer for reshaping the embeddings
        reshaped_temporal_embedded = self.concat_dense(temporal_embedded)
        print("temporal embedded shape after reshaping",reshaped_temporal_embedded.shape)

        # Add spatial and temporal embeddings
        embedded_inputs = tf.concat([spatial_embedded, reshaped_temporal_embedded], axis=1)
        print("embedded inputs",embedded_inputs.shape)


        spatial_encoded = self.spatial_encoder(spatial_embedded)
        print('spatial encoded', spatial_encoded.shape)
        print('spatial embedded', spatial_embedded.shape)
        temporal_decoded = self.temporal_decoder(spatial_embedded, spatial_encoded)  # Pass encoder's output as well
        print('Temporal decoded', temporal_decoded.shape)

        velocity_pred = self.velocity_mlp(temporal_decoded)
        covariance_pred = self.covariance_mlp(temporal_decoded)

        return velocity_pred, covariance_pred




# Define the parameters
batch_size = 16
num_samples = 100
num_timesteps = 10
num_features = 6
window_size = 5
num_layers = 4
num_heads = 8
head_dim = 6
num_epochs = 5
learning_rate = 0.001

# Generate some random IMU samples and orientations
imu_samples = np.random.randn(num_samples, num_timesteps, num_features)
orientations = np.random.randn(num_samples, num_timesteps, 4)  # Assuming quaternion

# Generate random ground truth velocities
ground_truth_velocities = np.random.randn(num_samples, num_timesteps - window_size + 1, 3)

# Create the CTIN model
ctin_model = CTIN(num_timesteps, num_features, window_size, num_layers, num_heads, head_dim)

dataset_X = imu_samples
dataset_gt_vel = ground_truth_velocities

# Convert the dataset into a TensorFlow Dataset and create batches
dataset = tf.data.Dataset.from_tensor_slices((dataset_X, dataset_gt_vel))
dataset = dataset.batch(batch_size)
print(dataset)

loss_fn = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0

    for batch in dataset:
        inputs, targets = batch

        with tf.GradientTape() as tape:
            velocity_pred, covariance_pred = ctin_model(inputs)
            loss = loss_fn(targets, velocity_pred)

        gradients = tape.gradient(loss, ctin_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ctin_model.trainable_variables))

        total_loss += loss.numpy()
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

print("Training finished!")

# Evaluate the model on the validation or test dataset
eval_results = ctin_model.evaluate(validation_inputs, validation_targets)
print("Evaluation results:", eval_results)

import matplotlib.pyplot as plt

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



