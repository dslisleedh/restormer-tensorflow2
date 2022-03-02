import tensorflow as tf
from einops import rearrange
from einops.layers.keras import Rearrange


class WithBaisLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, dim, epsilon=1e-5):
        super(WithBaisLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.dim = dim

        self.gamma = tf.Variable(tf.ones(shape=self.dim))
        self.beta = tf.Variable(tf.zeros(shape=self.dim))

    def call(self, inputs, **kwargs):
        b, h, w, c = inputs.get_shape().as_list()
        inputs = rearrange(inputs,
                           'b h w c -> b (h w) c'
                           )
        mu, sigma = tf.nn.moments(inputs,
                                  axes=-1,
                                  keepdims=True
                                  )
        inputs = ((inputs - mu) / tf.sqrt(sigma+self.epsilon)) * self.gamma + self.beta
        inputs = rearrange(inputs,
                           'b (h w) c -> b h w c',
                           h=h, w=w
                           )
        return inputs


class BiasFreeLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, dim, epsilon=1e-5):
        super(BiasFreeLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.dim = dim

        self.gamma = tf.Variable(tf.ones(shape=self.dim))

    def call(self, inputs, **kwargs):
        b, h, w, c = inputs.get_shape().as_list()
        inputs = rearrange(inputs,
                           'b h w c -> b (h w) c'
                           )
        mu, sigma = tf.nn.moments(inputs,
                                  axes=-1,
                                  keepdims=True
                                  )
        inputs = ((inputs - mu) / tf.sqrt(sigma+self.epsilon)) * self.gamma
        inputs = rearrange(inputs,
                           'b (h w) c -> b h w c',
                           h=h, w=w
                           )
        return inputs


class GatedDconvFFN(tf.keras.layers.Layer):
    def __init__(self, dims, e, bias:bool):
        super(GatedDconvFFN, self).__init__()
        self.dims = dims
        self.e = e
        self.bias = bias

        self.projection_in = tf.keras.layers.Conv2D(int(self.dims * self.e) * 2,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='VALID',
                                                    use_bias=self.bias
                                                    )
        self.deconv = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                                      strides=1,
                                                      padding='SAME',
                                                      use_bias=self.bias
                                                      )
        self.projetion_out = tf.keras.layers.Conv2D(int(self.dims),
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='VALID',
                                                    use_bias=self.bias
                                                    )

    def call(self, inputs, **kwargs):
        inputs = self.projection_in(inputs)
        inputs = self.deconv(inputs)
        x1, x2 = tf.split(inputs,
                          axis=-1,
                          num_or_size_splits=2
                          )
        inputs = tf.multiply(tf.nn.gelu(x1), x2)
        inputs = self.projetion_out(inputs)
        return inputs


class MultiDconvHeadTransposedAttention(tf.keras.layers.Layer):
    def __init__(self, dims, num_heads, bias: bool):
        super(MultiDconvHeadTransposedAttention, self).__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.temperature = tf.Variable(tf.ones(1, 1, self.num_heads))
        self.bias = bias

        self.qkv = tf.keras.layers.Conv2D(self.dims * 3,
                                          kernel_size=1,
                                          strides=1,
                                          padding='VALID',
                                          use_bias=self.bias
                                          )
        self.qkv_dconv = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                                         strides=1,
                                                         padding='SAME',
                                                         use_bias=self.bias
                                                         )
        self.projection_out = tf.keras.layers.Conv2D(self.dims,
                                                     kernel_size=1,
                                                     padding='VALID',
                                                     use_bias=self.bias
                                                     )

    def call(self, inputs, *args, **kwargs):
        b, h, w, c = inputs.get_shape().as_list()

        qkv = self.qkv_dconv(self.qkv(inputs))
        q, k, v = tf.split(qkv,
                           num_or_size_splits=3,
                           axis=-1
                           )

        q = rearrange(q,
                      'b h w (heads c) -> b heads c (h w)',
                      heads=self.num_heads
                      )
        q = tf.keras.utils.normalize(q,
                                     axis=-1
                                     )
        k = rearrange(k,
                      'b h w (heads c) -> b heads c (h w)',
                      heads=self.num_heads
                      )
        k = tf.keras.utils.normalize(k,
                                     axis=-1,
                                     )
        v = rearrange(v,
                      'b h w (heads c) -> b heads c (h w)',
                      heads=self.num_heads
                      )

        attn = q @ tf.transpose(k, perm=[0, 1, 3, 2]) * self.temperature
        attn = tf.nn.softmax(attn, axis=-1)

        out = (attn @ v)
        out = rearrange(out,
                        'b heads c (h w) -> b h w (heads c)',
                        heads=self.num_heads, h=h, w=w
                        )
        out = self.projection_out(out)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dims, num_heads, e, bias, layernorm_type):
        super(TransformerBlock, self).__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.e = e
        self.bias = bias
        self.layernorm_type = layernorm_type

        if self.layernorm_type == 'withbias':
            self.ln1 = WithBaisLayerNormalization(self.dims)
            self.ln2 = WithBaisLayerNormalization(self.dims)
        else:
            self.ln1 = BiasFreeLayerNormalization(self.dims)
            self.ln2 = BiasFreeLayerNormalization(self.dims)
        self.attn = MultiDconvHeadTransposedAttention(self.dims, self.num_heads, self.bias)
        self.ffn = GatedDconvFFN(self.dims, self.e, self.bias)

    def call(self, inputs, *args, **kwargs):
        inputs = inputs + self.attn(self.ln1(inputs))
        inputs = inputs + self.ffn(self.ln2(inputs))
        return inputs


class OverlapPatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, dims=48, bias=False):
        super(OverlapPatchEmbedding, self).__init__()
        self.dims = dims
        self.bias = bias

        self.forward = tf.keras.layers.Conv2D(self.dims,
                                              kernel_size=3,
                                              strides=1,
                                              padding='SAME',
                                              use_bias=self.bias
                                              )

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class Downsample(tf.keras.layers.Layer):
    def __init__(self, dims):
        super(Downsample, self).__init__()
        self.dims = dims

        self.forward = tf.keras.layers.Conv2D(self.dims // 2,
                                              kernel_size=3,
                                              strides=1,
                                              padding='SAME',
                                              use_bias=False
                                              )

    def call(self, inputs, *args, **kwargs):
        return tf.nn.space_to_depth(self.forward(inputs),
                                    block_size=2
                                    )


class Upsample(tf.keras.layers.Layer):
    def __init__(self, dims):
        super(Upsample, self).__init__()
        self.dims = dims

        self.forward = tf.keras.layers.Conv2D(self.dims * 2,
                                              kernel_size=3,
                                              strides=1,
                                              padding='SAME',
                                              use_bias=False
                                              )

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(self.forward(inputs),
                                    block_size=2
                                    )
