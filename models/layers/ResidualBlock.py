from tensorflow.keras.layers import Layer


class ResidualBlock(Layer):
    def __init__(self,
                 block_contents,
                 multiplier,
                 **kwargs):
        super(ResidualBlock, self).__init__()
        self.block_contents = block_contents
        self.multiplier = multiplier

    def call(self, inputs):
        print(self.name)
        output = inputs
        print(inputs.shape)
        for block in self.block_contents:
            for layer in block:
                print(layer.name)
                out = layer(output)
            output = output + out
        return output
