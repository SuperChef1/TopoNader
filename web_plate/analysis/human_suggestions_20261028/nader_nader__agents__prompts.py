definition_BDAG = """\n###BlockDefinition###
Each block starts with "##block_name##", and each line has a description. You can use the "index:operation" method to represent the operation, and use the "index1->index2" to describe the calculation graph. Noting that the output node can have only one input.
The following is a list of available operations:
    Conv2d(out_channels,kernel_size,stride,dilation,groups) Two-dimensional convolution operation, 'out_channels' represents the output dimension; 'kernel_size' represents the convolution kernel size; 'stride' represents the step size, default: 1; 'dilation' is the hole convolution size, default: 1; 'groups' groups number of the channels, default:1.
    Linear(out_channels) Linear fully connected layer, 'out_channels' represents the output dimension.
    AvgPool2d(kernel_size,stride) Two-dimensional average pooling operation, 'kernel_size' represents the kernel size, 'stride' represents the step size.
    MaxPool2d(kernel_size,stride) Two-dimensional maximum pooling operation, 'kernel_size' represents the kernel size, 'stride' represents the step size.
    AdaptiveMaxPool2d(output_size) Two-dimensional maximum pooling operation pools the input feature map into a feature map with a length and width of output_size. For example, AdaptiveMaxPool2d(output_size=1) pools a feature map of the shape of (B,C,H,W) into (B,C,1,1) shape.
    AdaptiveAvgPool2d(output_size) Two-dimensional average pooling operation.
    Add Tensor-by-element addition operation, the input tensor shape for this operation must conform to the pytorch broadcasting rule.
    Mul Tensor-by-element multiplication operation, the input tensor shape for this operation must conform to the pytorch broadcasting rule.
    multiply Matrix multiplication operation, the two tensor shapes entered for this operation must conform to the pytorch tensor multiplication rule.
    concat(dim) Tensor concating operation, all tensors input to this operation are concated in the dim dimension. The sizes of the concated tensors dimensions other than the dim dimension should be consistent. For example, concat(dim=1) concates all input tensors in the 1 dimension.
    mean(dim) Average the tensor in dim dimension. For example, mean(dim=1) pools a input tensor of shape (B,L,D) into the output tensor of shape (B,1,D) by average in the dimension 1.
    max(dim) Maximize the tensor in dim dimension. For example, max(dim=2) pools a input tensor of shape (B,L,D) into the output tensor of shape (B,L,2) by max in the dimension 2.
    sum(dim) Sum the tensor in dim dimension. For example, sum(dim=0) pools a input tensor of shape (B,L,D) into the output tensor of shape (0,L,D) by sum in the dimension 0.
    softmax(dim) Apply a softmax operation at dim dimension. For example, softmax(dim=1) calculate the softmax of input tensor with shape (B,L,D) and the output tensor's shape is (B,L,D).
The activation functions that can be used are: ReLU, GELU, Sigmoid.
The normalization methods that can be used are:
    BN: Batch normalization
    LN: Layer normalization.
The tensor can be transformed by using the following operations:
    permute(*dims) rearranges the tensor dimensions, 'dims' is the order of the new dimensions, for example: permute(0, 2, 3, 1) changes the tensor shape from (B,C,H,W) to (B,H,W,C).
    repeat(*sizes) repeats the tensor along the specified dimensions. 'sizes' is a list containing the number of repetitions along each dimension. For example: repeat(1,3,2,4) repeats the tensor 1 times in the first dimension, 3 times in the second dimension, 2 times in the third dimenstion, 4 times in the forth dimenstion.
    reshape(*shape) changes the shape of the tensor to the specified shape; 'shape' is an array representing the new shape; you can use -1 as the size of a dimension to automatically calculate the size of the dimension to ensure that the total number of elements remains unchanged; for example: reshape(B,H,W,C) means changing the shape of the tensor to (B,H,W,C).
Variables you may use include:
    input: input feature map, the shape is (B,C,H,W);
    output: output feature map, the shape is (B,dim,H,W);
    C: the number of channels of the input feature map;
    dim: the number of channels of the output feature map;
    H: the height of the input feature map;
    W: the width of the input feature map
You can use the basic +,-,x, / operations.
"""

definition_BDAG_DARTS = """\n###BlockDefinition###
Each block starts with "##block_name##", and each line has a description. You can use the "index:operation" method to represent the operation, and use the "index1->index2" to describe the calculation graph. Noting that the output node can have only one input.
The following is a list of available operations:
    Conv2d(out_channels,kernel_size,stride,dilation,groups) Two-dimensional convolution operation, 'out_channels' represents the output dimension; 'kernel_size' represents the convolution kernel size; 'stride' represents the step size, default: 1; 'dilation' is the hole convolution size, default: 1; 'groups' groups number of the channels, default:1.
    Linear(out_channels) Linear fully connected layer, 'out_channels' represents the output dimension.
    AvgPool2d(kernel_size,stride) Two-dimensional average pooling operation, 'kernel_size' represents the kernel size, 'stride' represents the step size.
    MaxPool2d(kernel_size,stride) Two-dimensional maximum pooling operation, 'kernel_size' represents the kernel size, 'stride' represents the step size.
    AdaptiveMaxPool2d(output_size) Two-dimensional maximum pooling operation pools the input feature map into a feature map with a length and width of output_size. For example, AdaptiveMaxPool2d(output_size=1) pools a feature map of the shape of (B,C,H,W) into (B,C,1,1) shape.
    AdaptiveAvgPool2d(output_size) Two-dimensional average pooling operation.
    Add Tensor-by-element addition operation, the input tensor shape for this operation must conform to the pytorch broadcasting rule.
    Mul Tensor-by-element multiplication operation, the input tensor shape for this operation must conform to the pytorch broadcasting rule.
    multiply Matrix multiplication operation, the two tensor shapes entered for this operation must conform to the pytorch tensor multiplication rule.
    concat(dim) Tensor concating operation, all tensors input to this operation are concated in the dim dimension. The sizes of the concated tensors dimensions other than the dim dimension should be consistent. For example, concat(dim=1) concates all input tensors in the 1 dimension.
    mean(dim) Average the tensor in dim dimension. For example, mean(dim=1) pools a input tensor of shape (B,L,D) into the output tensor of shape (B,1,D) by average in the dimension 1.
    max(dim) Maximize the tensor in dim dimension. For example, max(dim=2) pools a input tensor of shape (B,L,D) into the output tensor of shape (B,L,2) by max in the dimension 2.
    sum(dim) Sum the tensor in dim dimension. For example, sum(dim=0) pools a input tensor of shape (B,L,D) into the output tensor of shape (0,L,D) by sum in the dimension 0.
    softmax(dim) Apply a softmax operation at dim dimension. For example, softmax(dim=1) calculate the softmax of input tensor with shape (B,L,D) and the output tensor's shape is (B,L,D).
    DropPath Apply a DropPath operation. A random subset of layers is skipped during each training iteration.
The activation functions that can be used are: ReLU, GELU, Sigmoid.
The normalization methods that can be used are:
    BN: Batch normalization
    LN: Layer normalization.
The tensor can be transformed by using the following operations:
    permute(*dims) rearranges the tensor dimensions, 'dims' is the order of the new dimensions, for example: permute(0, 2, 3, 1) changes the tensor shape from (B,dim,H,W) to (B,H,W,dim).
    repeat(*sizes) repeats the tensor along the specified dimensions. 'sizes' is a list containing the number of repetitions along each dimension. For example: repeat(1,3,2,4) repeats the tensor 1 times in the first dimension, 3 times in the second dimension, 2 times in the third dimenstion, 4 times in the forth dimenstion.
    reshape(*shape) changes the shape of the tensor to the specified shape; 'shape' is an array representing the new shape; you can use -1 as the size of a dimension to automatically calculate the size of the dimension to ensure that the total number of elements remains unchanged; for example: reshape(B,H,W,dim) means changing the shape of the tensor to (B,H,W,dim).
Variables you may use include:
    dim: the number of channels of the output feature map;
    H: the height of the input feature map;
    W: the width of the input feature map
    input1: input feature map from the output of the cell before the previous one, the shape is (B,dim,H,W);
    input2: input feature map from the output of the previous cell, the shape is (B,dim,H,W);
    output: output feature map;
You can use the basic +,-,x, / operations.
"""

prompt_generate_stem_downsample = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning. 
You need to generate the input block into its corresponding stem block and downsample block.
Please ensure that the number of input channels and output channels of the generated blocks are C and dim respectively.
When outputting, you only need to output blocks that meet the defined rules, and do not output other irrelevant information.
{definition_BDAG}
\n###examples###
{{examples}}
###constraint###
1. The height and width of the output feature map of the stem block must be 1/4 of the input feature map.
2. The height and width of the output feature map of the downsample block must be 1/2 of the input feature map.
{{experience}}
\n###input###
{{input}}
\n###output###
"""

prompt_generate_downsample_darts = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning. 
You need to generate a optimal downsample block for the input block.
Please ensure that the number of output channels of the generated block is dim.
When outputting, you must need to output blocks that meet the defined rules, and do not output other irrelevant information.
Note that each block has two inputs: the output from the block before the previous one and the output from the previous block. Please make effective use of both inputs.
{definition_BDAG_DARTS}
\n###examples###
{{examples}}
\n###constraint###
The height and width of the output feature map of the downsample block must be 1/2 of the input feature map.
\n{{experience}}
\n###input###
{{input}}
\n###output###
When outputting, you only need to output the block that meet the defined rules, and do not output other irrelevant information.
Please first think step by step how to modify the modified block to improve the performance of the overall model, and then give the modified block.
"""

prompt_generate_stem_downsample_nas_bench_201 = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning. 
You need to generate the input block into its corresponding stem block and downsample block.
Please ensure that the number of input channels and output channels of the generated blocks are C and dim respectively.
When outputting, you only need to output blocks that meet the defined rules, and do not output other irrelevant information.
{definition_BDAG}
\n###examples###
{{examples}}
###constraint###
1. The height and width of the output feature map of the stem block must be same with the input feature map.
2. The height and width of the output feature map of the downsample block must be 1/2 of the input feature map.
{{experience}}
\n###input###
{{input}}
\n###output###
"""

prompt_generate_specific_block = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning. 
You need to generate the specified model according to the input prompts.
Please ensure that the number of input channels and output channels of the generated block are both C.
When outputting, you only need to output blocks that meet the defined rules, and do not output other irrelevant information.
{definition_BDAG}
\n###examples###
{{examples}}
\n###input###
{{input}}
\n###output###
"""

prompt_develop_experience="""###Constraint1###
Refer to the following suggestions to help you generate a block that better meets block definition.
{experience}
"""

prompt_research_experience="""###Constraint2###
Refer to the following suggestions to help you generate a block that has better performance.
{experience}
"""

prompt_modify_block = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning.
Please make reasonable modifications to the specified block based on the characteristics of the block and the proposal.
Please ensure that the number of input channels and output channels of the generated block are both C.
Note that structures in the modified block that unrelated to the proposal should be kept as original as possible.
{definition_BDAG}

###proposal###
{{proposal}}

###block###
{{block}}

{{dev_expe}}
{{res_expe}}
###output###
When outputting, you only need to output the block that meet the defined rules, and do not output other irrelevant information.
\n###output###
"""

prompt_modify_block_darts = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning.
Please make reasonable modifications to the specified block based on the characteristics of the block and the proposal.
Please ensure that the number of output channels of the generated block is dim.
Note that each block has two inputs: the output from the block before the previous one and the output from the previous block. Please make effective use of both inputs.
{definition_BDAG_DARTS}

###proposal###
{{proposal}}

###block###
{{block}}

{{dev_expe}}
{{res_expe}}
###output###
When outputting, you only need to output the block that meet the defined rules, and do not output other irrelevant information.
Please first think step by step how to modify the modified block to improve the performance of the overall model, and then give the modified block.
"""

prompt_modify_block2 = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning.
Please make appropriate modifications to the block based on the proposal, and modify it three times in different ways to generate three new blocks.
Please ensure that the number of input channels and output channels of the generated block are both C.
Note that some inspirations' targets are multiple blocks.
Note that structures in the modified blocks that unrelated to the proposal should be kept as original as possible.
{definition_BDAG}
\n###proposal###
{{proposal}}
\n###blocks###
{{blocks}}
\n###output###
"""


prompt_check_proposals = """###Instruction###
You are an expert who is proficient in various model structures of deep learning. 
Below are the basic block definition of computer vision models, a block to be improved and candidate proposals.
You need to determine which of the candidate proposals can guide the improvement of the block.
Please return the serial numbers of useful proposals.
{definition_BDAG}
\n###block###
{block}
\n###candidate proposals###
{proposals}
\n
{format_output}
"""

'''
你是一位精通神经架构设计的专家.
现在需要用一个有向无环图来描述一个神经网络,下面是有向无环图的定义.

现在有一个不完全满足上述定义的一个网络和改网络定义错误的简单提示(网络依次包括三种block:base block, stem block和downsample block):
{}

Error Reason: {}
下面请你分析网络设计错误的原因,并基于此给出一句通用的设计提示,来提示用户能准确的设计出完全符合要求的网络:
'''
prompt_reflector_develop_allfailed_base = f"""###Instruction###
You are an expert who is proficient in neural architecture design. 
The structure of neural networks is now described in terms of directed acyclic graphs. The following is the definition of directed acyclic graphs.
{definition_BDAG}

###input###
Now there is a network that does not fully meet the above definition and a hint of reason:
{{block}}
Error reason: {{error}}

###output###
Please analyze the reason of network design errors, and based on this, give a general design tip to prompt users to accurately design a network that fully meets the requirements. The tip should be wrapped in <tip> and </tip>.
"""

prompt_reflector_develop_allfailed_stem = f"""###Instruction###
You are an expert who is proficient in neural architecture design. 
The structure of neural networks is now described in terms of directed acyclic graphs. The following is the definition of directed acyclic graphs.
{definition_BDAG}

###input###
Now there is a network that does not fully meet the above definition and a hint of reason:
{{block}}
Error reason: {{error}}

###output###
Please analyze the reason of network design errors, and based on this, give a general design tip to prompt users to accurately design a network that fully meets the requirements. The tip should be wrapped in <tip> and </tip>.
"""

prompt_reflector_develop_allfailed_downsample = f"""###Instruction###
You are an expert who is proficient in neural architecture design. 
The structure of neural networks is now described in terms of directed acyclic graphs. The following is the definition of directed acyclic graphs.
{definition_BDAG}

###input###
Now there is a network that does not fully meet the above definition and a hint of reason:
{{block}}
Error reason: {{error}}

###output###
Please analyze the reason of network design errors, and based on this, give a general design tip to prompt users to accurately design a network that fully meets the requirements. The tip should be wrapped in <tip> and </tip>.
"""

prompt_reflector_develop_allfailed_all = f"""###Instruction###
You are an expert who is proficient in neural architecture design. 
The structure of neural networks is now described in terms of directed acyclic graphs. The following is the definition of directed acyclic graphs.
{definition_BDAG}

###input###
Now there is a network that does not fully meet the above definition and a hint of reason. The network consists of three types of blocks in order :base block, stem block, and downsample block.
{{block}}
Error reason: {{error}}

###output###
Please analyze the reason of network design errors, and based on this, give a general design tip to prompt users to accurately design a network that fully meets the requirements. The tip should be wrapped in <tip> and </tip>.
"""



'''
research team
'''

# proposer
PROMPT_PROPOSER_INIT = f"""###Instruction###
You are a computer vision research expert, and you have deep insights into neural arcgitecture design.
You will be given a block to be improved and several candidate inspirations, you need to compare the candidate inspirations and rank them according their usefulness for guiding the improvement of the block.

###block###
The following is the block to be improved and candidate inspirations. The neural architecture of the block of the model is described in the form of a computational graph.
{{block}}

###candidate inspirations###
The following are the candidate inspirations. Each inspiration is given in the form of 'inspiration_index:inspiration'.
{{inspirations}}

###output###
Please rank the all candidate inspirations in descending order according to their usefulness. You response should wrap all inspiration_index of the inspirations with <response> and </response>, and use ',' to separate different indexes.
"""

# proposer without reader
PROMPT_PROPOSE_INSPIRATION_INIT = f"""###Instruction###
You are a computer vision research expert, and you have deep insights into neural arcgitecture design.
You will be given a block to be improved, you need to propose 10 inspirations that can imporves the performace of the block.

###block###
The following is the block to be improved. The neural architecture of the block of the model is described in the form of a computational graph.
{{block}}

###Constrain###
1. The inspirations must be relevant to the neural network structure.
2. The inspirations must be a sentence no more than 50 words.
3. The inspirations must be diverse and innovative.

###output###
Please response the inspirations. You response should wrap each inspiration with <response> and </response>.
"""

MOGD = """###Model Optimization Graph Definition###
The following is the definition of the model optimization graph:
The optimization path graph of the model is a tree structure.
The node represents the model, describing the structure of the basic block and the accuracy on the test dataset. It is defined in the format of <model_name><block>...</block><acc>...</acc></model_name>. 'model_name' is the name of model. The block is a directed acyclic graph that describes the block calculation process.
Proposals are defined in the form of name:content.
The edge represents the process of obtaining another model from one model through proposal. For example: model_1--proposal_1-->model_2 means that model model_2 was obtained by modifing model model_1 according to the proposal proposal_1."""

PROMPT_PROPOSER_MOG = f""""###Instruction###
You are a computer vision research expert, and you have deep insights into neural arcgitecture design. You need to select three pairs of model-inspirations based on the given model optimization graph and candidate inspirations.

{MOGD}

###model optimization graph###
The following is the current model optimization graph:
{{block}}

###candidate inspirations###
The following are the candidate inspirations. Each inspiration is given in the form of 'inspiration_index:inspiration'.
{{inspirations}}

###Constrain###
The new model modified according to your choice should outperform all existing models in the model optimization graph on the test set.
When you choose the model to be modified and the corresponding proposal, you need to pay attention to the following points:
1. You need to carefully observe and analysis the structure of each model in the model optimization graph.
2. You need to carefully observe and analysis each proposal and the corresponding utility in the model optimization graph.
2. You need to carefully observe and analysis the modification path of the model in the model optimization graph.
3. You should infer the combination of the model to be modified and the corresponding proposal that is most likely to get the best new model based on the modification path in the model optimization graph and candidate inspirations.
4. The suboptimal performing models in the model optimization graph may also achieve the better performance through modification by proposal. You also need to pay attention to and analyze the potential of poorly performing models when making decisions.


###output###
You need to output the three pairs of model-inspirations.
Each pair has the following format <proposal><model>model_name</model><inspiration>inspiration_index</inspiration></proposal>
"""

PROMPT_RESEARCH_REFLECTION_INSPIRATION = f"""###Instruction###
You are a computer vision research expert, and you have deep insights into neural arcgitecture design.
You will be provided with two basic neural network block described in the form of computational graphs. The second model is modified from the first model. Please describe what needs to be done to change the first block into the second block and analyze what effects will be produced.

###input###
The following are the computational graphs of the two blocks:
{{input}}
###output###
Please summary your response in one sentence wrapped in <response> and </response>. In the summary, don't describe the subject, just describe what happened.
"""

# eval develop
PROMPT_EVAL_DEVELOP = f"""###Instruction###
You are a computer vision research expert, your task is assess whether the AI assistant has properly modified the block according to inspiration.

The following are the rules for describing blocks:
{definition_BDAG}

###input###
The following are the original block, inspiration and the block modified by the AI assistant based on inspiration:
#original block#
{{raw_block}}
#inspiration#
{{inspiration}}
#modified block#
{{pred_block}}

Here is a correctly modified block:
{{ref_block}}

###output###
Please assess whether the AI assistant has properly modified the block according to inspiration.
Please think step by step and give the answer 'yes' or 'no' wrap in <answer> and </answer> in the end.
"""


# research reflector
prompt_reflector_research_failed = f"""###Instruction###
You are an expert who is proficient in neural architecture design.
You will be given a raw model and a modified model and their corresponding accuracy on test dataset and you need to analyze why the accuracy of the raw model decreases after modification, and give a suggestion to avoid this error in the next modification.
The structure of neural networks is now described in terms of directed acyclic graphs. The following is the definition of directed acyclic graphs.
{definition_BDAG}

###Input###
Their is the raw model and its accuracy:
{{raw}}
Accuracy: {{raw_acc}}
Their is the modified model and its accuracy:
{{new}}
Accuracy: {{new_acc}}

###Constrain###
1. The suggestion must be relevant to the neural network structure.
2. The suggestion must be a sentence no more than 50 words.
3. The suggestion must be must be general.

###Output###
Please think step by step about the reasons why the accuracy of the model decreases after modification and give a suggestion to avoid this error in the next modification.
The suggestion should be wrapped in <suggestion> and </suggestion>.
"""
