
from nn_activations import sin, sin_L1, cos, cos_L1, tanh, tanh_L1, sigmoid, sigmoid_L1, fixedswish, swish_L3, \
                           asinh_X_cos, tanh_X_cos_L1L2, x_D_exp_p_expn, sin_O_tanh_L2, fixedswish_O_erf_L2, tanh_O_asinh_L0L1, \
                           fixedswish_S_erf_L4, cos_L0, nx_S_sin_L3, cos_X_erf_L1L4, sin_L0L1, \
                           sin_D_expn_p_1_L1, sigmoid_X_tanh, erf_O_fixedswish_L1L2, x_X_sigmoid_L1L3, exp_p_expn_min_atan_L1L2L4, \
                           tanh_O_fixedswish_L1, tanh_L0, tanh_O_fixedswish_L0, sin_L0, tanh_O_fixedswish_L0L1, asinh_O_sin_L0L1L2, \
                           atan_O_sin_L1, sin_O_fixedswish, sin_X_sigmoid_L2, sigmoid_L0L1, neg_O_tanh_L0L2, fixedswish_L1, atan_O_fixedswish_L1L2, \
                           asinh_O_fixedswish_L0L2, cos_X_atan_X_sigmoid_L0L4L7, atan_X_neg_O_sigmoid



# klein_gordon
config = [
           [5, [[], []], 32, [sin, 0]],         # BO-FcNet-1
           # [5, [[], []], 48, [sin_L1, 1]],    # BO-FcNet-2
           # [10, [[], []], 48, [sin_L1, 1]],   # BO-FcNet-3
           # [9, [[0, 2, 4, 6], [2, 4, 6, 8]], 50, [tanh, 0]],   # BO-FcResNet-1
           # [8, [[0, 2, 4, 6], [2, 4, 6, 7]], 40, [sin, 0]],    # BO-FcResNet-2
           # [8, [[0, 2, 4, 6], [2, 4, 6, 7]], 34, [sin, 0]],    # BO-FcResNet-3
           # [6, [[], []], 32, [cos_X_erf_L1L4, 2]],                            # rand-1
           # [8, [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], 40, [sin_L0L1, 2]],  # rand-2
           # [7, [[0, 1, 2, 4, 5], [1, 2, 3, 5, 6]], 48, [sin, 0]],             # rand-3
           # [6, [[0], [1]], 48, [tanh_L0, 1]],                                            # evo-w/o-1
           # [9, [[0, 1, 3, 4, 5, 7], [1, 3, 4, 5, 7, 8]], 32, [sin_L0, 1]]                # evo-w/o-2
           # [9, [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]], 50, [sin_L0L1, 2]]  # evo-w/o-3
           # [6, [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], 48, [asinh_X_cos, 0]],                  # evo-w/-1
           # [9, [[0, 1, 2, 3, 7], [1, 2, 3, 5, 8]], 44, [tanh_X_cos_L1L2, 2]],              # evo-w/-2
           # [7, [[0, 1, 2, 4, 5], [1, 2, 4, 5, 6]], 50, [cos_X_atan_X_sigmoid_L0L4L7, 3]],  # evo-w/-3
        ]


#burgers
# config = [
#             [11, [[], []], 20, [fixedswish, 0]],    # BO-FcNet-1
#             [3, [[], []], 24, [tanh_L1, 1]],        # BO-FcNet-2
#             [11, [[], []], 32, [sin_L1, 1]],        # BO-FcNet-3
#             [3, [[0], [2]], 34, [tanh_L1, 1]],              # BO-FcResNet-1
#             [6, [[0,2,4], [2,4,5]], 22, [tanh_L1, 1]],      # BO-FcResNet-2
#             [3, [[0], [2]], 20, [tanh_L1, 1]],              # BO-FcResNet-3
#             [8, [[0, 4], [4, 6]], 28, [atan_O_sin_L1, 1]],                         # rand-1
#             [6, [[], []], 34, [sin_O_fixedswish, 0]],                              # rand-2
#             [8, [[0, 1, 2, 3, 4, 6], [1, 2, 3, 4, 5, 7]], 26, [sigmoid_L0L1, 2]],  # rand-3
#             [8, [[0], [2]], 22, [neg_O_tanh_L0L2, 2]],      # evo-w/o-1
#             [10, [[], []], 42, [fixedswish_L1, 1]],         # evo-w/o-2
#             [6, [[0], [5]], 32, [tanh_L1, 1]]               # evo-w/o-3
#             [8, [[3], [5]], 40, [sin_X_sigmoid_L2, 1]],         # evo-w/-1
#             [5, [[0], [3]], 20, [sigmoid_L0L1, 2]],             # evo-w/-2
#             [8, [[0], [3]], 42, [asinh_O_fixedswish_L0L2, 2]],  # evo-w/-3
#           ]



#lame
# config = [  [7, [[], []], 50, [fixedswish, 0]],    # BO-FcNet-1
#             [10, [[], []], 50, [swish_L3, 1]],     # BO-FcNet-2
#             [7, [[], []], 46, [swish_L3, 1]],      # BO-FcNet-3
#             [9, [[0, 2, 4, 6], [2, 4, 6, 8]], 50, [tanh_L1, 1]],           # BO-FcResNet-1
#             [11, [[0, 2, 4, 6, 8], [2, 4, 6, 8, 10]], 50, [tanh_L1, 1]],   # BO-FcResNet-2
#             [11, [[0, 2, 4, 6, 8], [2, 4, 6, 8, 10]], 42, [tanh_L1, 1]],   # BO-FcResNet-3
#             [9, [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]], 42, [fixedswish_O_erf_L2, 1]],  # rand-1
#             [9, [[0, 2, 4, 6], [2, 4, 6, 8]], 42, [exp_p_expn_min_atan_L1L2L4, 3]],                   # rand-2
#             [9, [[1, 2, 5], [2, 5, 8]], 26, [atan_X_neg_O_sigmoid, 0]],                               # rand-3
#             [11, [[0, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 10]], 22, [tanh_O_fixedswish_L0, 1] ],      # evo-w/o-1
#             [7, [[0, 1, 3, 4, 5], [1, 3, 4, 5, 6]], 32, [tanh_L0, 1] ],                                             # evo-w/o-2
#             [7, [[0, 2, 4], [2, 4, 6]], 50, [tanh_L1, 1] ]                                                          # evo-w/o-3
#             [10, [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 9]], 42, [x_D_exp_p_expn, 0]],                  # evo-w/-1
#             [8, [[0, 1, 3, 4, 6], [1, 3, 4, 6, 7]], 48, [sin_O_tanh_L2, 1]],                                      # evo-w/-2
#             [11, [[0, 1, 2, 3, 4, 6, 7, 8, 9], [1, 2, 3, 4, 6, 7, 8, 9, 10]], 48, [atan_O_fixedswish_L1L2, 2]],   # evo-w/-3
#            ]


