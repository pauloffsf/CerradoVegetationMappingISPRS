
import tensorflow as tf
from tensorflow.keras import layers
from seg_models.keras_vision_transformer import swin_layers
from seg_models.keras_vision_transformer import transformer_layers

'''
Swin Transformer implementation from https://github.com/yingkaisha/keras-vision-transformer

'''


def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):
	'''
    Stacked Swin Transformers that share the same token size.
	Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
	*Dropout is turned off.
	'''
	# Turn-off dropouts
	mlp_drop_rate = 0 # Droupout after each MLP layer
	attn_drop_rate = 0 # Dropout after Swin-Attention
	proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
	drop_path_rate = 0 # Drop-path within skip-connections

	qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
	qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor

	if shift_window:
		shift_size = window_size // 2
	else:
		shift_size = 0
    
	for i in range(stack_num):
		if i % 2 == 0:
			shift_size_temp = 0
		else:
			shift_size_temp = shift_size

		X = swin_layers.SwinTransformerBlock(dim=embed_dim,
											num_patch=num_patch, 
											num_heads=num_heads, 
											window_size=window_size, 
											shift_size=shift_size_temp, 
											num_mlp=num_mlp, 
											qkv_bias=qkv_bias, 
											qk_scale=qk_scale,
											mlp_drop=mlp_drop_rate,
											attn_drop=attn_drop_rate, 
											proj_drop=proj_drop_rate, 
											drop_path_prob=drop_path_rate, 
											name='name{}'.format(i))(X)
	return X

def SwinTransformerStack(input_shape = None, input_tensor = None, classes=None):
	
	patch_size = (4,4)
	input_size = input_tensor.shape.as_list()[1:]
	num_patch_x = input_size[0]//patch_size[0]
	num_patch_y = input_size[1]//patch_size[1]
	

	#Hyperparameters:
	num_blocks = 3
	stack_num= 5
	num_filters= 256
	num_patch= (num_patch_x,num_patch_y)
	num_heads= 4
	window_size= 4
	num_mlp= 512
	shift_window=True
	name = 'TransformerStack'


	c = input_tensor
	# Patch extraction
	c = transformer_layers.patch_extract(patch_size)(c)
	#Patch Embedding:
	c = transformer_layers.patch_embedding(num_patch_x*num_patch_y, num_filters)(c)
	for i in range(num_blocks):
		c = swin_transformer_stack(c,
									stack_num = stack_num,
									embed_dim = num_filters,
									num_patch=(num_patch_x, num_patch_y), 
									num_heads=num_heads,
									window_size=window_size, 
									num_mlp=num_mlp, 
									shift_window=shift_window, 
									name='{}_swin_stack{}'.format(name, i+1))

	c = tf.reshape(c, shape=(-1, input_size[0], input_size[1], int(num_filters/(patch_size[0]*patch_size[1]))))
	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="SwinTransformerStack")

	return model


def SwinTransformerDoubleStack(input_shape = None, input_tensor = None, classes=None):
	#Double Transformer stack, one for S1 and another for S2 Fusion at Classification
	patch_size = (4,4)
	input_size = input_tensor[0].shape.as_list()[1:]
	num_patch_x = input_size[0]//patch_size[0]
	num_patch_y = input_size[1]//patch_size[1]
	

	#Hyperparameters:
	num_blocks = 3
	stack_num= 5
	num_filters= 256
	num_patch= (num_patch_x,num_patch_y)
	num_heads= 4
	window_size= 4
	num_mlp= 512
	shift_window=True
	name = 'TransformerDoubleStack'


	c = input_tensor[0]
	# Patch extraction
	c = transformer_layers.patch_extract(patch_size)(c)
	#Patch Embedding:
	c = transformer_layers.patch_embedding(num_patch_x*num_patch_y, num_filters)(c)
	for i in range(num_blocks):
		c = swin_transformer_stack(c,
									stack_num = stack_num,
									embed_dim = num_filters,
									num_patch=(num_patch_x, num_patch_y), 
									num_heads=num_heads,
									window_size=window_size, 
									num_mlp=num_mlp, 
									shift_window=shift_window, 
									name='{}_swin_stack{}'.format(name, i+1))

	c = tf.reshape(c, shape=(-1, input_size[0], input_size[1], int(num_filters/(patch_size[0]*patch_size[1]))))

	c0=input_tensor[1]
	C0=transformer_layers.patch_extract(patch_size)(c0)
	c0 = transformer_layers.patch_embedding(num_patch_x*num_patch_y, num_filters)(c0)
	for i in range(num_blocks):
		c0 = swin_transformer_stack(c0,
									stack_num = stack_num,
									embed_dim = num_filters,
									num_patch=(num_patch_x, num_patch_y), 
									num_heads=num_heads,
									window_size=window_size, 
									num_mlp=num_mlp, 
									shift_window=shift_window, 
									name='{}_swin_stack{}'.format(name, i+1))

	c0 = tf.reshape(c0, shape=(-1, input_size[0], input_size[1], int(num_filters/(patch_size[0]*patch_size[1]))))

	c = layers.concatenate([c, c0], axis=-1)
	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="SwinTransformerStack")

	return model


def SwinUnet(input_shape = None, input_tensor = None, classes=None):

    patch_size = (4,4)								# Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    num_heads = [4, 8, 8, 8]						# number of attention heads per down/upsampling level
    window_size = [4,2,2,2]							# the size of attention window per down/upsampling level
    stack_num_down = 2         						# number of Swin Transformers per downsampling level
    stack_num_up = 2 								# number of Swin Transformers per upsampling level
    depth = 4										# the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level
    num_mlp = 512									# number of MLP nodes within the Transformer
    shift_window = True 							# Apply window shifting, i.e., Swin-MSA
    name = 'SwinUnet'
    filter_num_begin = 128
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    c = input_tensor
    c = transformer_layers.patch_extract(patch_size)(c)
    c = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim)(c)
    
    # The first Swin Transformer stack
    c = swin_transformer_stack(c, 
                               stack_num=stack_num_down, 
                               embed_dim=embed_dim, 
                               num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], 
                               window_size=window_size[0], 
                               num_mlp=num_mlp, 
                               shift_window=shift_window, 
                               name='{}_swin_down0'.format(name))
    X_skip.append(c)


    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        c = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(c)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        c = swin_transformer_stack(c, 
                                   stack_num=stack_num_down, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], 
                                   window_size=window_size[i+1], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_down{}'.format(name, i+1))
        
        # Store tensors for concat
        X_skip.append(c)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    c = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        c = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                               embed_dim=embed_dim, 
                                               upsample_rate=2, 
                                               return_vector=True)(c)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        c = layers.concatenate([c, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        c = layers.Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(c)
        
        # Swin Transformer stack
        c = swin_transformer_stack(c, 
                                  stack_num=stack_num_up, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i], 
                                   window_size=window_size[i], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_up{}'.format(name, i))
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    
    c = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                           embed_dim=embed_dim, 
                                           upsample_rate=patch_size[0], 
                                           return_vector=False)(c)

    outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
    outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
    model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="SwinUnet")
    return model


def SwinUnetDouble(input_shape = None, input_tensor = None, classes=None):

	c = input_tensor[0]


	

	outputs = layers.Conv2D(classes, kernel_size=1, name="final_conv")(c)
	outputs = layers.Activation(activation="sigmoid", name="output")(outputs)
	model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="SwinUnet")
	return model
