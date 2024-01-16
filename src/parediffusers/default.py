DEFAULT_UNET_CONFIG = {
	"in_channels": 4,
	"out_channels": 4,
	"sample_size": 96,
	"center_input_sample": False,
	"flip_sin_to_cos": True,
	"freq_shift": 0,
	"down_block_types": [
		"CrossAttnDownBlock2D",
		"CrossAttnDownBlock2D",
		"CrossAttnDownBlock2D",
		"DownBlock2D",
	],
	"mid_block_type": "UNetMidBlock2DCrossAttn",
	"up_block_types": [
		"UpBlock2D",
		"CrossAttnUpBlock2D",
		"CrossAttnUpBlock2D",
		"CrossAttnUpBlock2D",
	],
	"only_cross_attention": False,
	"block_out_channels": [320, 640, 1280, 1280],
	"layers_per_block": 2,
	"downsample_padding": 1,
	"mid_block_scale_factor": 1,
	"dropout": 0.0,
	"act_fn": "silu",
	"norm_num_groups": 32,
	"norm_eps": 1e-5,
	"cross_attention_dim": 1024,
	"transformer_layers_per_block": 1,
	"reverse_transformer_layers_per_block": None,
	"encoder_hid_dim": None,
	"encoder_hid_dim_type": None,
	"attention_head_dim": [5, 10, 20, 20],
	"num_attention_heads": None,
	"dual_cross_attention": False,
	"use_linear_projection": True,
	"class_embed_type": None,
	"addition_embed_type": None,
	"addition_time_embed_dim": None,
	"num_class_embeds": None,
	"upcast_attention": False,
	"resnet_time_scale_shift": "default",
	"resnet_skip_time_act": False,
	"resnet_out_scale_factor": 1.0,
	"time_embedding_type": "positional",
	"time_embedding_dim": None,
	"time_embedding_act_fn": None,
	"timestep_post_act": None,
	"time_cond_proj_dim": None,
	"conv_in_kernel": 3,
	"conv_out_kernel": 3,
	"projection_class_embeddings_input_dim": None,
	"attention_type": "default",
	"class_embeddings_concat": False,
	"mid_block_only_cross_attention": None,
	"cross_attention_norm": None,
	"addition_embed_type_num_heads": 64,
	"_use_default_values": [
		"timestep_post_act",
		"addition_time_embed_dim",
		"conv_out_kernel",
		"encoder_hid_dim",
		"transformer_layers_per_block",
		"conv_in_kernel",
		"time_cond_proj_dim",
		"resnet_out_scale_factor",
		"only_cross_attention",
		"num_class_embeds",
		"upcast_attention",
		"mid_block_only_cross_attention",
		"time_embedding_type",
		"encoder_hid_dim_type",
		"num_attention_heads",
		"class_embed_type",
		"attention_type",
		"reverse_transformer_layers_per_block",
		"cross_attention_norm",
		"mid_block_type",
		"resnet_time_scale_shift",
		"addition_embed_type",
		"dropout",
		"class_embeddings_concat",
		"addition_embed_type_num_heads",
		"resnet_skip_time_act",
		"projection_class_embeddings_input_dim",
		"time_embedding_dim",
		"time_embedding_act_fn",
	],
	"_class_name": "UNet2DConditionModel",
	"_diffusers_version": "0.8.0",
	"_name_or_path": "stabilityai/stable-diffusion-2",	
}