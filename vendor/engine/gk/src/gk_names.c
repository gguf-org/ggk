// Human-readable names for the op enums.
//
// Keyed by enum value rather than written in order, so an op added in the
// middle of the enum cannot silently shift every name after it. Anything
// missing reads back as "unknown" instead of running off the table.

#include "gk_impl.h"

#define GK_NAME(op) [op] = #op

static const char * const g_op_name[GK_OP_COUNT] = {
    [GK_OP_NONE]                = "none",

    [GK_OP_DUP]                 = "dup",
    [GK_OP_ADD]                 = "add",
    [GK_OP_ADD1]                = "add1",
    [GK_OP_ACC]                 = "acc",
    [GK_OP_SUB]                 = "sub",
    [GK_OP_MUL]                 = "mul",
    [GK_OP_DIV]                 = "div",
    [GK_OP_SQR]                 = "sqr",
    [GK_OP_SQRT]                = "sqrt",
    [GK_OP_LOG]                 = "log",
    [GK_OP_SIN]                 = "sin",
    [GK_OP_COS]                 = "cos",
    [GK_OP_SUM]                 = "sum",
    [GK_OP_SUM_ROWS]            = "sum_rows",
    [GK_OP_MEAN]                = "mean",
    [GK_OP_ARGMAX]              = "argmax",
    [GK_OP_COUNT_EQUAL]         = "count_equal",
    [GK_OP_REPEAT]              = "repeat",
    [GK_OP_CONCAT]              = "concat",
    [GK_OP_FILL]                = "fill",

    [GK_OP_NORM]                = "norm",
    [GK_OP_RMS_NORM]            = "rms_norm",
    [GK_OP_GROUP_NORM]          = "group_norm",
    [GK_OP_L2_NORM]             = "l2_norm",

    [GK_OP_MUL_MAT]             = "mul_mat",
    [GK_OP_MUL_MAT_ID]          = "mul_mat_id",
    [GK_OP_OUT_PROD]            = "out_prod",

    [GK_OP_SCALE]               = "scale",
    [GK_OP_SET]                 = "set",
    [GK_OP_CPY]                 = "cpy",
    [GK_OP_CONT]                = "cont",
    [GK_OP_RESHAPE]             = "reshape",
    [GK_OP_VIEW]                = "view",
    [GK_OP_PERMUTE]             = "permute",
    [GK_OP_TRANSPOSE]           = "transpose",
    [GK_OP_GET_ROWS]            = "get_rows",
    [GK_OP_SET_ROWS]            = "set_rows",
    [GK_OP_DIAG]                = "diag",
    [GK_OP_DIAG_MASK_INF]       = "diag_mask_inf",
    [GK_OP_DIAG_MASK_ZERO]      = "diag_mask_zero",
    [GK_OP_SOFT_MAX]            = "soft_max",
    [GK_OP_ROPE]                = "rope",
    [GK_OP_CLAMP]               = "clamp",

    [GK_OP_IM2COL]              = "im2col",
    [GK_OP_COL2IM_1D]           = "col2im_1d",
    [GK_OP_CONV_TRANSPOSE_1D]   = "conv_transpose_1d",
    [GK_OP_CONV_TRANSPOSE_2D]   = "conv_transpose_2d",
    [GK_OP_POOL_1D]             = "pool_1d",
    [GK_OP_POOL_2D]             = "pool_2d",
    [GK_OP_UPSCALE]             = "upscale",
    [GK_OP_PAD]                 = "pad",
    [GK_OP_PAD_REFLECT_1D]      = "pad_reflect_1d",
    [GK_OP_ROLL]                = "roll",
    [GK_OP_ARANGE]              = "arange",
    [GK_OP_TIMESTEP_EMBEDDING]  = "timestep_embedding",
    [GK_OP_ARGSORT]             = "argsort",
    [GK_OP_TOP_K]               = "top_k",
    [GK_OP_LEAKY_RELU]          = "leaky_relu",

    [GK_OP_FLASH_ATTN_EXT]      = "flash_attn_ext",
    [GK_OP_SSM_CONV]            = "ssm_conv",
    [GK_OP_SSM_SCAN]            = "ssm_scan",
    [GK_OP_WIN_PART]            = "win_part",
    [GK_OP_WIN_UNPART]          = "win_unpart",
    [GK_OP_GET_REL_POS]         = "get_rel_pos",
    [GK_OP_ADD_REL_POS]         = "add_rel_pos",
    [GK_OP_RWKV_WKV6]           = "rwkv_wkv6",
    [GK_OP_RWKV_WKV7]           = "rwkv_wkv7",
    [GK_OP_GATED_LINEAR_ATTN]   = "gated_linear_attn",
    [GK_OP_SOLVE_TRI]           = "solve_tri",
    [GK_OP_CUMSUM]              = "cumsum",
    [GK_OP_TRI]                 = "tri",

    [GK_OP_UNARY]               = "unary",
    [GK_OP_GLU]                 = "glu",

    [GK_OP_MAP_CUSTOM1]         = "map_custom1",
    [GK_OP_MAP_CUSTOM2]         = "map_custom2",
    [GK_OP_MAP_CUSTOM3]         = "map_custom3",
    [GK_OP_CUSTOM]              = "custom",

    [GK_OP_ADD_ID]              = "add_id",
    [GK_OP_REPEAT_BACK]         = "repeat_back",
    [GK_OP_SILU_BACK]           = "silu_back",
    [GK_OP_RMS_NORM_BACK]       = "rms_norm_back",
    [GK_OP_GET_ROWS_BACK]       = "get_rows_back",
    [GK_OP_SOFT_MAX_BACK]       = "soft_max_back",
    [GK_OP_ROPE_BACK]           = "rope_back",
    [GK_OP_IM2COL_BACK]         = "im2col_back",
    [GK_OP_IM2COL_3D]           = "im2col_3d",
    [GK_OP_CONV_2D]             = "conv_2d",
    [GK_OP_CONV_3D]             = "conv_3d",
    [GK_OP_CONV_2D_DW]          = "conv_2d_dw",
    [GK_OP_POOL_2D_BACK]        = "pool_2d_back",
    [GK_OP_FLASH_ATTN_BACK]     = "flash_attn_back",
    [GK_OP_GATED_DELTA_NET]     = "gated_delta_net",
    [GK_OP_LIGHTNING_INDEXER]   = "lightning_indexer",
    [GK_OP_DSV4_HC_COMB]        = "dsv4_hc_comb",
    [GK_OP_DSV4_HC_PRE]         = "dsv4_hc_pre",
    [GK_OP_DSV4_HC_POST]        = "dsv4_hc_post",
    [GK_OP_CROSS_ENTROPY_LOSS]  = "cross_entropy_loss",
    [GK_OP_CROSS_ENTROPY_LOSS_BACK] = "cross_entropy_loss_back",
    [GK_OP_OPT_STEP_ADAMW]      = "opt_step_adamw",
    [GK_OP_OPT_STEP_SGD]        = "opt_step_sgd",
};

// Infix/prefix forms used when a graph is printed as an expression.
static const char * const g_op_symbol[GK_OP_COUNT] = {
    [GK_OP_NONE]     = "none",
    [GK_OP_DUP]      = "x",
    [GK_OP_ADD]      = "x+y",
    [GK_OP_ADD1]     = "x+y",
    [GK_OP_ACC]      = "view(x,nb,offset)+=y",
    [GK_OP_SUB]      = "x-y",
    [GK_OP_MUL]      = "x*y",
    [GK_OP_DIV]      = "x/y",
    [GK_OP_SQR]      = "x^2",
    [GK_OP_SQRT]     = "sqrt(x)",
    [GK_OP_LOG]      = "log(x)",
    [GK_OP_SIN]      = "sin(x)",
    [GK_OP_COS]      = "cos(x)",
    [GK_OP_SUM]      = "sum(x)",
    [GK_OP_SUM_ROWS] = "sum_rows(x)",
    [GK_OP_MEAN]     = "mean(x)",
    [GK_OP_REPEAT]   = "repeat(x)",
    [GK_OP_CONCAT]   = "concat(x,y)",
    [GK_OP_MUL_MAT]  = "X*Y",
    [GK_OP_SCALE]    = "x*v",
    [GK_OP_CPY]      = "x->y",
    [GK_OP_CONT]     = "cont(x)",
    [GK_OP_RESHAPE]  = "reshape(x)",
    [GK_OP_VIEW]     = "view(x)",
    [GK_OP_PERMUTE]  = "permute(x)",
    [GK_OP_TRANSPOSE]= "transpose(x)",
    [GK_OP_GET_ROWS] = "get_rows(x)",
    [GK_OP_SOFT_MAX] = "soft_max(x)",
    [GK_OP_ROPE]     = "rope(x)",
    [GK_OP_CLAMP]    = "clamp(x)",
    [GK_OP_UNARY]    = "unary(x)",
    [GK_OP_GLU]      = "glu(x)",
};

static const char * const g_unary_op_name[GK_UNARY_OP_COUNT] = {
    [GK_UNARY_OP_EXPM1]       = "expm1",
    [GK_UNARY_OP_SOFTPLUS]    = "softplus",
    [GK_UNARY_OP_ABS]         = "abs",
    [GK_UNARY_OP_SGN]         = "sgn",
    [GK_UNARY_OP_NEG]         = "neg",
    [GK_UNARY_OP_STEP]        = "step",
    [GK_UNARY_OP_TANH]        = "tanh",
    [GK_UNARY_OP_ELU]         = "elu",
    [GK_UNARY_OP_RELU]        = "relu",
    [GK_UNARY_OP_SIGMOID]     = "sigmoid",
    [GK_UNARY_OP_GELU]        = "gelu",
    [GK_UNARY_OP_GELU_QUICK]  = "gelu_quick",
    [GK_UNARY_OP_GELU_ERF]    = "gelu_erf",
    [GK_UNARY_OP_SILU]        = "silu",
    [GK_UNARY_OP_HARDSWISH]   = "hardswish",
    [GK_UNARY_OP_HARDSIGMOID] = "hardsigmoid",
    [GK_UNARY_OP_EXP]         = "exp",
    [GK_UNARY_OP_XIELU]       = "xielu",
    [GK_UNARY_OP_FLOOR]       = "floor",
    [GK_UNARY_OP_CEIL]        = "ceil",
    [GK_UNARY_OP_ROUND]       = "round",
    [GK_UNARY_OP_TRUNC]       = "trunc",
};

static const char * const g_glu_op_name[GK_GLU_OP_COUNT] = {
    [GK_GLU_OP_REGLU]       = "reglu",
    [GK_GLU_OP_GEGLU]       = "geglu",
    [GK_GLU_OP_SWIGLU]      = "swiglu",
    [GK_GLU_OP_SWIGLU_OAI]  = "swiglu_oai",
    [GK_GLU_OP_GEGLU_ERF]   = "geglu_erf",
    [GK_GLU_OP_GEGLU_QUICK] = "geglu_quick",
};

static const char * gk_lookup(const char * const * table, int n, int i) {
    if (i < 0 || i >= n || table[i] == NULL) {
        return "unknown";
    }
    return table[i];
}

const char * gk_op_name(enum gk_op op) {
    return gk_lookup(g_op_name, GK_OP_COUNT, (int) op);
}

const char * gk_op_symbol(enum gk_op op) {
    return gk_lookup(g_op_symbol, GK_OP_COUNT, (int) op);
}

const char * gk_unary_op_name(enum gk_unary_op op) {
    return gk_lookup(g_unary_op_name, GK_UNARY_OP_COUNT, (int) op);
}

const char * gk_glu_op_name(enum gk_glu_op op) {
    return gk_lookup(g_glu_op_name, GK_GLU_OP_COUNT, (int) op);
}
