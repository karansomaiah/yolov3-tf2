# self.conv_blocks = {
#     # block 1 (after first two convolutional inputs)
#     "block1": {
#         "input": "conv2",
#         "design": [(32, 1, 1, 1), (64, 3, 3, 1)],
#         "residual": True,
#     },
#     # block 2
#     "block2": {
#         "input": "conv3",
#         "design": [(64, 1, 1, 1), (128, 3, 3, 1)] * 2,
#         "residual": True,
#     },
#     # block 3
#     "block3": {
#         "input": "conv4",
#         "design": [(128, 1, 1, 1), (256, 3, 3, 1)] * 8,
#         "residual": True,
#     },
#     # block 4
#     "block4": {
#         "input": "conv5",
#         "design": [(256, 1, 1, 1), (512, 3, 3, 1)] * 8,
#         "residual": True,
#     },
#     # block 5
#     "block5": {
#         "input": "conv6",
#         "design": [(512, 1, 1, 1), (1024, 3, 3, 1)] * 4,
#         "residual": True,
#     },
# }
