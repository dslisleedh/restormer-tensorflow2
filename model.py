from layers import *
import tensorflow as tf
import tensorflow_addons as tfa
from einops import rearrange
from einops.layers.keras import Rearrange


class Restormer(tf.keras.models.Model):
    def __init__(self,
                 output_channels=3,
                 dims=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 num_heads=[1, 2, 4, 8],
                 e=2.66,
                 bias=False,
                 layernorm_type='withbias'
                 ):
        super(Restormer, self).__init__()
        self.output_channel = output_channels
        self.dims = dims
        self.num_blocks = num_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.num_heads = num_heads
        self.e = e
        self.bias = bias
        self.layernorm_type = layernorm_type

        self.patch_embedding = OverlapPatchEmbedding(self.dims, self.bias)

        ### Encoder
        self.encoder_l1 = tf.keras.Sequential([
            TransformerBlock(self.dims,
                             self.num_heads[0],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_blocks[0])
        ])
        self.down_1to2 = Downsample(self.dims)
        self.encoder_l2 = tf.keras.Sequential([
            TransformerBlock(int(self.dims * 2 ** 1),
                             self.num_heads[1],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_blocks[1])
        ])
        self.down_2to3 = Downsample(int(self.dims * 2 ** 1))
        self.encoder_l3 = tf.keras.Sequential([
            TransformerBlock(int(self.dims * 2 ** 2),
                             self.num_heads[2],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_blocks[2])
        ])
        self.down_3to4 = Downsample(int(self.dims * 2 ** 2))
        ### Latent
        self.latent = tf.keras.Sequential([
            TransformerBlock(int(self.dims * 2 ** 3),
                             self.num_heads[3],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_blocks[3])
        ])
        ### Decoder
        self.up_4to3 = Upsample(int(self.dims * 2 ** 3))
        self.reduce_channel_4to3 = tf.keras.layers.Conv2D(filters=int(self.dims * 2 ** 2),
                                                          kernel_size=1,
                                                          strides=1,
                                                          padding='VALID',
                                                          use_bias=self.bias
                                                          )
        self.decoder_l3 = tf.keras.Sequential([
            TransformerBlock(int(self.dims * 2 ** 2),
                             self.num_heads[2],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_blocks[2])
        ])
        self.up_3to2 = Upsample(int(self.dims * 2 ** 2))
        self.reduce_channel_3to2 = tf.keras.layers.Conv2D(filters=int(self.dims * 2 ** 1),
                                                          kernel_size=1,
                                                          strides=1,
                                                          padding='VALID',
                                                          use_bias=self.bias
                                                          )
        self.decoder_l2 = tf.keras.Sequential([
            TransformerBlock(int(self.dims * 2 ** 1),
                             self.num_heads[1],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_blocks[1])
        ])
        self.up_2to1 = Upsample(int(self.dims * 2 ** 1))
        self.decoder_l1 = tf.keras.Sequential([
            TransformerBlock(int(self.dims * 2 ** 1),
                             self.num_heads[0],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_blocks[0])
        ])
        ### Refinement
        self.refinement = tf.keras.Sequential([
            TransformerBlock(int(self.dims * 2 ** 1),
                             self.num_heads[0],
                             self.e,
                             self.bias,
                             self.layernorm_type
                             ) for _ in range(self.num_refinement_blocks)
        ])
        self.out_channel = tf.keras.layers.Conv2D(filters=3,
                                                  strides=1,
                                                  kernel_size=1,
                                                  padding='SAME',
                                                  use_bias=self.bias
                                                  )

    def call(self, inputs, training=None, mask=None):
        in_encoder_l1 = self.patch_embedding(inputs)
        out_encoder_l1 = self.encoder_l1(in_encoder_l1)

        in_encoder_l2 = self.down_1to2(out_encoder_l1)
        out_encoder_l2 = self.encoder_l2(in_encoder_l2)

        in_encoder_l3 = self.down_2to3(out_encoder_l2)
        out_encoder_l3 = self.encoder_l3(in_encoder_l3)

        latent = self.latent(self.down_3to4(out_encoder_l3))

        in_decoder_l3 = self.reduce_channel_4to3(tf.concat([self.up_4to3(latent), out_encoder_l3],
                                                           axis=-1
                                                           )
                                                 )
        out_decoder_l3 = self.decoder_l3(in_decoder_l3)

        in_decoder_l2 = self.reduce_channel_3to2(tf.concat([self.up_3to2(out_decoder_l3), out_encoder_l2],
                                                           axis=-1
                                                           )
                                                 )
        out_decoder_l2 = self.decoder_l2(in_decoder_l2)

        in_decoder_l1 = tf.concat([self.up_2to1(out_decoder_l2), out_encoder_l1],
                                  axis=-1
                                  )
        out_decoder_l1 = self.decoder_l1(in_decoder_l1)

        refined = self.refinement(out_decoder_l1)

        out = self.out_channel(refined) + inputs

        return out