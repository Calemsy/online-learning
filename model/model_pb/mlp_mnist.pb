
B
input_xPlaceholder*
dtype0*
shape:����������
A
input_yPlaceholder*
dtype0*
shape:���������
K
truncated_normal/shapeConst*
valueB"  �  *
dtype0
B
truncated_normal/meanConst*
valueB
 *    *
dtype0
D
truncated_normal/stddevConst*
dtype0*
valueB
 *���=
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
X
w1
VariableV2*
shared_name *
dtype0*
	container *
shape:
��
r
	w1/AssignAssignw1truncated_normal*
T0*
_class
	loc:@w1*
validate_shape(*
use_locking(
7
w1/readIdentityw1*
T0*
_class
	loc:@w1
;
zerosConst*
valueB	�*    *
dtype0
W
b1
VariableV2*
shape:	�*
shared_name *
dtype0*
	container 
g
	b1/AssignAssignb1zeros*
validate_shape(*
use_locking(*
T0*
_class
	loc:@b1
7
b1/readIdentityb1*
T0*
_class
	loc:@b1
Q
MatMulMatMulinput_xw1/read*
transpose_b( *
T0*
transpose_a( 
&
addAddV2MatMulb1/read*
T0

ReluReluadd*
T0
M
truncated_normal_1/shapeConst*
valueB"�  
   *
dtype0
D
truncated_normal_1/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
W
w2
VariableV2*
shared_name *
dtype0*
	container *
shape:	�

t
	w2/AssignAssignw2truncated_normal_1*
use_locking(*
T0*
_class
	loc:@w2*
validate_shape(
7
w2/readIdentityw2*
T0*
_class
	loc:@w2
<
zeros_1Const*
valueB
*    *
dtype0
V
b2
VariableV2*
shared_name *
dtype0*
	container *
shape
:

i
	b2/AssignAssignb2zeros_1*
validate_shape(*
use_locking(*
T0*
_class
	loc:@b2
7
b2/readIdentityb2*
T0*
_class
	loc:@b2
P
MatMul_1MatMulReluw2/read*
transpose_b( *
T0*
transpose_a( 
)
logitsAddMatMul_1b2/read*
T0
:
ArgMax/dimensionConst*
value	B :*
dtype0
R
ArgMaxArgMaxlogitsArgMax/dimension*

Tidx0*
T0*
output_type0	
A
ExpandDims/dimConst*
dtype0*
valueB :
���������
E

ExpandDims
ExpandDimsArgMaxExpandDims/dim*

Tdim0*
T0	
=
CastCastinput_y*

SrcT0*
Truncate( *

DstT0	
I
EqualEqual
ExpandDimsCast*
incompatible_shape_error(*
T0	
=
Cast_1CastEqual*

SrcT0
*
Truncate( *

DstT0
:
ConstConst*
dtype0*
valueB"       
E
accuracyMeanCast_1Const*
T0*

Tidx0*
	keep_dims( 
@
Variable/initial_valueConst*
value	B : *
dtype0
T
Variable
VariableV2*
shared_name *
dtype0*
	container *
shape: 
�
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
?
Cast_2Castinput_y*

SrcT0*
Truncate( *

DstT0
=
one_hot/on_valueConst*
valueB
 *  �?*
dtype0
>
one_hot/off_valueConst*
valueB
 *    *
dtype0
7
one_hot/depthConst*
dtype0*
value	B :

u
one_hotOneHotCast_2one_hot/depthone_hot/on_valueone_hot/off_value*
T0*
TI0*
axis���������
[
9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientone_hot*
T0
S
)softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0
T
*softmax_cross_entropy_with_logits_sg/ShapeShapelogits*
T0*
out_type0
U
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
dtype0*
value	B :
V
,softmax_cross_entropy_with_logits_sg/Shape_1Shapelogits*
T0*
out_type0
T
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0
�
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0
�
0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N
]
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
dtype0*
valueB:
�
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
T0*
Index0
k
4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
dtype0*
valueB:
���������
Z
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0
�
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
N*

Tidx0*
T0
�
,softmax_cross_entropy_with_logits_sg/ReshapeReshapelogits+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0
U
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0
�
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0
V
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0
�
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0
�
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
N*
T0*

axis 
_
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0
�
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
T0*
Index0
m
6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
���������*
dtype0
\
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0
�
-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*

Tidx0
�
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0
�
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0
V
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
value	B :
�
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0
`
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0
�
1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N
�
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
T0*
Index0
�
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0
5
Const_1Const*
valueB: *
dtype0
k
lossMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_1*
T0*

Tidx0*
	keep_dims( 
8
gradients/ShapeConst*
dtype0*
valueB 
@
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
O
!gradients/loss_grad/Reshape/shapeConst*
valueB:*
dtype0
p
gradients/loss_grad/ReshapeReshapegradients/Fill!gradients/loss_grad/Reshape/shape*
T0*
Tshape0
k
gradients/loss_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
s
gradients/loss_grad/TileTilegradients/loss_grad/Reshapegradients/loss_grad/Shape*

Tmultiples0*
T0
m
gradients/loss_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
D
gradients/loss_grad/Shape_2Const*
valueB *
dtype0
G
gradients/loss_grad/ConstConst*
valueB: *
dtype0
~
gradients/loss_grad/ProdProdgradients/loss_grad/Shape_1gradients/loss_grad/Const*
T0*

Tidx0*
	keep_dims( 
I
gradients/loss_grad/Const_1Const*
valueB: *
dtype0
�
gradients/loss_grad/Prod_1Prodgradients/loss_grad/Shape_2gradients/loss_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
G
gradients/loss_grad/Maximum/yConst*
dtype0*
value	B :
j
gradients/loss_grad/MaximumMaximumgradients/loss_grad/Prod_1gradients/loss_grad/Maximum/y*
T0
h
gradients/loss_grad/floordivFloorDivgradients/loss_grad/Prodgradients/loss_grad/Maximum*
T0
f
gradients/loss_grad/CastCastgradients/loss_grad/floordiv*

SrcT0*
Truncate( *

DstT0
c
gradients/loss_grad/truedivRealDivgradients/loss_grad/Tilegradients/loss_grad/Cast*
T0
�
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0
�
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/loss_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0
R
gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0
u
Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*
T0*

Tdim0
�
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0
�
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0
�
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0
w
Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0
�
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0
�
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
�
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul
�
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
k
Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapelogits*
T0*
out_type0
�
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0
G
gradients/logits_grad/ShapeShapeMatMul_1*
T0*
out_type0
H
gradients/logits_grad/Shape_1Shapeb2/read*
T0*
out_type0
�
+gradients/logits_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/logits_grad/Shapegradients/logits_grad/Shape_1*
T0
�
gradients/logits_grad/SumSumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape+gradients/logits_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
w
gradients/logits_grad/ReshapeReshapegradients/logits_grad/Sumgradients/logits_grad/Shape*
T0*
Tshape0
�
gradients/logits_grad/Sum_1SumCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape-gradients/logits_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
}
gradients/logits_grad/Reshape_1Reshapegradients/logits_grad/Sum_1gradients/logits_grad/Shape_1*
T0*
Tshape0
p
&gradients/logits_grad/tuple/group_depsNoOp^gradients/logits_grad/Reshape ^gradients/logits_grad/Reshape_1
�
.gradients/logits_grad/tuple/control_dependencyIdentitygradients/logits_grad/Reshape'^gradients/logits_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/logits_grad/Reshape
�
0gradients/logits_grad/tuple/control_dependency_1Identitygradients/logits_grad/Reshape_1'^gradients/logits_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/logits_grad/Reshape_1
�
gradients/MatMul_1_grad/MatMulMatMul.gradients/logits_grad/tuple/control_dependencyw2/read*
T0*
transpose_a( *
transpose_b(
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu.gradients/logits_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
i
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0
B
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0
E
gradients/add_grad/Shape_1Shapeb1/read*
T0*
out_type0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyw1/read*
T0*
transpose_a( *
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMulinput_x+gradients/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
E
train_step/learning_rateConst*
valueB
 *��L=*
dtype0
�
)train_step/update_w1/ApplyGradientDescentApplyGradientDescentw1train_step/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_class
	loc:@w1*
use_locking( 
�
)train_step/update_b1/ApplyGradientDescentApplyGradientDescentb1train_step/learning_rate-gradients/add_grad/tuple/control_dependency_1*
T0*
_class
	loc:@b1*
use_locking( 
�
)train_step/update_w2/ApplyGradientDescentApplyGradientDescentw2train_step/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
_class
	loc:@w2*
use_locking( 
�
)train_step/update_b2/ApplyGradientDescentApplyGradientDescentb2train_step/learning_rate0gradients/logits_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b2
�
train_step/updateNoOp*^train_step/update_b1/ApplyGradientDescent*^train_step/update_b2/ApplyGradientDescent*^train_step/update_w1/ApplyGradientDescent*^train_step/update_w2/ApplyGradientDescent
k
train_step/valueConst^train_step/update*
_class
loc:@Variable*
value	B :*
dtype0
l

train_step	AssignAddVariabletrain_step/value*
use_locking( *
T0*
_class
loc:@Variable
:
gradients_1/ShapeConst*
dtype0*
valueB 
B
gradients_1/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0
Q
#gradients_1/loss_grad/Reshape/shapeConst*
valueB:*
dtype0
v
gradients_1/loss_grad/ReshapeReshapegradients_1/Fill#gradients_1/loss_grad/Reshape/shape*
T0*
Tshape0
m
gradients_1/loss_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
y
gradients_1/loss_grad/TileTilegradients_1/loss_grad/Reshapegradients_1/loss_grad/Shape*

Tmultiples0*
T0
o
gradients_1/loss_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
F
gradients_1/loss_grad/Shape_2Const*
valueB *
dtype0
I
gradients_1/loss_grad/ConstConst*
valueB: *
dtype0
�
gradients_1/loss_grad/ProdProdgradients_1/loss_grad/Shape_1gradients_1/loss_grad/Const*

Tidx0*
	keep_dims( *
T0
K
gradients_1/loss_grad/Const_1Const*
valueB: *
dtype0
�
gradients_1/loss_grad/Prod_1Prodgradients_1/loss_grad/Shape_2gradients_1/loss_grad/Const_1*

Tidx0*
	keep_dims( *
T0
I
gradients_1/loss_grad/Maximum/yConst*
value	B :*
dtype0
p
gradients_1/loss_grad/MaximumMaximumgradients_1/loss_grad/Prod_1gradients_1/loss_grad/Maximum/y*
T0
n
gradients_1/loss_grad/floordivFloorDivgradients_1/loss_grad/Prodgradients_1/loss_grad/Maximum*
T0
j
gradients_1/loss_grad/CastCastgradients_1/loss_grad/floordiv*

SrcT0*
Truncate( *

DstT0
i
gradients_1/loss_grad/truedivRealDivgradients_1/loss_grad/Tilegradients_1/loss_grad/Cast*
T0
�
Egradients_1/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0
�
Ggradients_1/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients_1/loss_grad/truedivEgradients_1/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0
T
gradients_1/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0
w
Dgradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
@gradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsGgradients_1/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0
�
9gradients_1/softmax_cross_entropy_with_logits_sg_grad/mulMul@gradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0
�
@gradients_1/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0
�
9gradients_1/softmax_cross_entropy_with_logits_sg_grad/NegNeg@gradients_1/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0
y
Fgradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
Bgradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsGgradients_1/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeFgradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0
�
;gradients_1/softmax_cross_entropy_with_logits_sg_grad/mul_1MulBgradients_1/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_19gradients_1/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0
m
Cgradients_1/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapelogits*
T0*
out_type0
�
Egradients_1/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshape9gradients_1/softmax_cross_entropy_with_logits_sg_grad/mulCgradients_1/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0
I
gradients_1/logits_grad/ShapeShapeMatMul_1*
T0*
out_type0
J
gradients_1/logits_grad/Shape_1Shapeb2/read*
T0*
out_type0
�
-gradients_1/logits_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/logits_grad/Shapegradients_1/logits_grad/Shape_1*
T0
�
gradients_1/logits_grad/SumSumEgradients_1/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape-gradients_1/logits_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
}
gradients_1/logits_grad/ReshapeReshapegradients_1/logits_grad/Sumgradients_1/logits_grad/Shape*
T0*
Tshape0
�
gradients_1/logits_grad/Sum_1SumEgradients_1/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape/gradients_1/logits_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
!gradients_1/logits_grad/Reshape_1Reshapegradients_1/logits_grad/Sum_1gradients_1/logits_grad/Shape_1*
T0*
Tshape0
�
 gradients_1/MatMul_1_grad/MatMulMatMulgradients_1/logits_grad/Reshapew2/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_1/MatMul_1_grad/MatMul_1MatMulRelugradients_1/logits_grad/Reshape*
T0*
transpose_a(*
transpose_b( 
[
gradients_1/Relu_grad/ReluGradReluGrad gradients_1/MatMul_1_grad/MatMulRelu*
T0
D
gradients_1/add_grad/ShapeShapeMatMul*
T0*
out_type0
G
gradients_1/add_grad/Shape_1Shapeb1/read*
T0*
out_type0
�
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*
T0
�
gradients_1/add_grad/SumSumgradients_1/Relu_grad/ReluGrad*gradients_1/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
T0*
Tshape0
�
gradients_1/add_grad/Sum_1Sumgradients_1/Relu_grad/ReluGrad,gradients_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
T0*
Tshape0
~
gradients_1/MatMul_grad/MatMulMatMulgradients_1/add_grad/Reshapew1/read*
transpose_b(*
T0*
transpose_a( 
�
 gradients_1/MatMul_grad/MatMul_1MatMulinput_xgradients_1/add_grad/Reshape*
transpose_b( *
T0*
transpose_a(
;
d-w1Identity gradients_1/MatMul_grad/MatMul_1*
T0
:
gradients_2/ShapeConst*
valueB *
dtype0
B
gradients_2/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
T0*

index_type0
Q
#gradients_2/loss_grad/Reshape/shapeConst*
valueB:*
dtype0
v
gradients_2/loss_grad/ReshapeReshapegradients_2/Fill#gradients_2/loss_grad/Reshape/shape*
T0*
Tshape0
m
gradients_2/loss_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
y
gradients_2/loss_grad/TileTilegradients_2/loss_grad/Reshapegradients_2/loss_grad/Shape*

Tmultiples0*
T0
o
gradients_2/loss_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
F
gradients_2/loss_grad/Shape_2Const*
dtype0*
valueB 
I
gradients_2/loss_grad/ConstConst*
valueB: *
dtype0
�
gradients_2/loss_grad/ProdProdgradients_2/loss_grad/Shape_1gradients_2/loss_grad/Const*
T0*

Tidx0*
	keep_dims( 
K
gradients_2/loss_grad/Const_1Const*
valueB: *
dtype0
�
gradients_2/loss_grad/Prod_1Prodgradients_2/loss_grad/Shape_2gradients_2/loss_grad/Const_1*

Tidx0*
	keep_dims( *
T0
I
gradients_2/loss_grad/Maximum/yConst*
value	B :*
dtype0
p
gradients_2/loss_grad/MaximumMaximumgradients_2/loss_grad/Prod_1gradients_2/loss_grad/Maximum/y*
T0
n
gradients_2/loss_grad/floordivFloorDivgradients_2/loss_grad/Prodgradients_2/loss_grad/Maximum*
T0
j
gradients_2/loss_grad/CastCastgradients_2/loss_grad/floordiv*

SrcT0*
Truncate( *

DstT0
i
gradients_2/loss_grad/truedivRealDivgradients_2/loss_grad/Tilegradients_2/loss_grad/Cast*
T0
�
Egradients_2/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0
�
Ggradients_2/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients_2/loss_grad/truedivEgradients_2/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0
T
gradients_2/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0
w
Dgradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
@gradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsGgradients_2/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0
�
9gradients_2/softmax_cross_entropy_with_logits_sg_grad/mulMul@gradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0
�
@gradients_2/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0
�
9gradients_2/softmax_cross_entropy_with_logits_sg_grad/NegNeg@gradients_2/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0
y
Fgradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
dtype0*
valueB :
���������
�
Bgradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsGgradients_2/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeFgradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0
�
;gradients_2/softmax_cross_entropy_with_logits_sg_grad/mul_1MulBgradients_2/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_19gradients_2/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0
m
Cgradients_2/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapelogits*
T0*
out_type0
�
Egradients_2/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshape9gradients_2/softmax_cross_entropy_with_logits_sg_grad/mulCgradients_2/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0
I
gradients_2/logits_grad/ShapeShapeMatMul_1*
T0*
out_type0
J
gradients_2/logits_grad/Shape_1Shapeb2/read*
T0*
out_type0
�
-gradients_2/logits_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/logits_grad/Shapegradients_2/logits_grad/Shape_1*
T0
�
gradients_2/logits_grad/SumSumEgradients_2/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape-gradients_2/logits_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
}
gradients_2/logits_grad/ReshapeReshapegradients_2/logits_grad/Sumgradients_2/logits_grad/Shape*
T0*
Tshape0
�
gradients_2/logits_grad/Sum_1SumEgradients_2/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape/gradients_2/logits_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
!gradients_2/logits_grad/Reshape_1Reshapegradients_2/logits_grad/Sum_1gradients_2/logits_grad/Shape_1*
T0*
Tshape0
�
 gradients_2/MatMul_1_grad/MatMulMatMulgradients_2/logits_grad/Reshapew2/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_2/MatMul_1_grad/MatMul_1MatMulRelugradients_2/logits_grad/Reshape*
transpose_b( *
T0*
transpose_a(
[
gradients_2/Relu_grad/ReluGradReluGrad gradients_2/MatMul_1_grad/MatMulRelu*
T0
D
gradients_2/add_grad/ShapeShapeMatMul*
T0*
out_type0
G
gradients_2/add_grad/Shape_1Shapeb1/read*
T0*
out_type0
�
*gradients_2/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_grad/Shapegradients_2/add_grad/Shape_1*
T0
�
gradients_2/add_grad/SumSumgradients_2/Relu_grad/ReluGrad*gradients_2/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients_2/add_grad/ReshapeReshapegradients_2/add_grad/Sumgradients_2/add_grad/Shape*
T0*
Tshape0
�
gradients_2/add_grad/Sum_1Sumgradients_2/Relu_grad/ReluGrad,gradients_2/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients_2/add_grad/Reshape_1Reshapegradients_2/add_grad/Sum_1gradients_2/add_grad/Shape_1*
T0*
Tshape0
9
d-b1Identitygradients_2/add_grad/Reshape_1*
T0
:
gradients_3/ShapeConst*
dtype0*
valueB 
B
gradients_3/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*
T0*

index_type0
Q
#gradients_3/loss_grad/Reshape/shapeConst*
valueB:*
dtype0
v
gradients_3/loss_grad/ReshapeReshapegradients_3/Fill#gradients_3/loss_grad/Reshape/shape*
T0*
Tshape0
m
gradients_3/loss_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
y
gradients_3/loss_grad/TileTilegradients_3/loss_grad/Reshapegradients_3/loss_grad/Shape*

Tmultiples0*
T0
o
gradients_3/loss_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
F
gradients_3/loss_grad/Shape_2Const*
dtype0*
valueB 
I
gradients_3/loss_grad/ConstConst*
dtype0*
valueB: 
�
gradients_3/loss_grad/ProdProdgradients_3/loss_grad/Shape_1gradients_3/loss_grad/Const*

Tidx0*
	keep_dims( *
T0
K
gradients_3/loss_grad/Const_1Const*
valueB: *
dtype0
�
gradients_3/loss_grad/Prod_1Prodgradients_3/loss_grad/Shape_2gradients_3/loss_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
I
gradients_3/loss_grad/Maximum/yConst*
value	B :*
dtype0
p
gradients_3/loss_grad/MaximumMaximumgradients_3/loss_grad/Prod_1gradients_3/loss_grad/Maximum/y*
T0
n
gradients_3/loss_grad/floordivFloorDivgradients_3/loss_grad/Prodgradients_3/loss_grad/Maximum*
T0
j
gradients_3/loss_grad/CastCastgradients_3/loss_grad/floordiv*

SrcT0*
Truncate( *

DstT0
i
gradients_3/loss_grad/truedivRealDivgradients_3/loss_grad/Tilegradients_3/loss_grad/Cast*
T0
�
Egradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0
�
Ggradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients_3/loss_grad/truedivEgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0
T
gradients_3/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0
w
Dgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
@gradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsGgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0
�
9gradients_3/softmax_cross_entropy_with_logits_sg_grad/mulMul@gradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0
�
@gradients_3/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0
�
9gradients_3/softmax_cross_entropy_with_logits_sg_grad/NegNeg@gradients_3/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0
y
Fgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
Bgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsGgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeFgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0
�
;gradients_3/softmax_cross_entropy_with_logits_sg_grad/mul_1MulBgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_19gradients_3/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0
m
Cgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapelogits*
T0*
out_type0
�
Egradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshape9gradients_3/softmax_cross_entropy_with_logits_sg_grad/mulCgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0
I
gradients_3/logits_grad/ShapeShapeMatMul_1*
T0*
out_type0
J
gradients_3/logits_grad/Shape_1Shapeb2/read*
T0*
out_type0
�
-gradients_3/logits_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/logits_grad/Shapegradients_3/logits_grad/Shape_1*
T0
�
gradients_3/logits_grad/SumSumEgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape-gradients_3/logits_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
}
gradients_3/logits_grad/ReshapeReshapegradients_3/logits_grad/Sumgradients_3/logits_grad/Shape*
T0*
Tshape0
�
gradients_3/logits_grad/Sum_1SumEgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape/gradients_3/logits_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
!gradients_3/logits_grad/Reshape_1Reshapegradients_3/logits_grad/Sum_1gradients_3/logits_grad/Shape_1*
T0*
Tshape0
�
 gradients_3/MatMul_1_grad/MatMulMatMulgradients_3/logits_grad/Reshapew2/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_3/MatMul_1_grad/MatMul_1MatMulRelugradients_3/logits_grad/Reshape*
transpose_b( *
T0*
transpose_a(
=
d-w2Identity"gradients_3/MatMul_1_grad/MatMul_1*
T0
:
gradients_4/ShapeConst*
valueB *
dtype0
B
gradients_4/grad_ys_0Const*
dtype0*
valueB
 *  �?
]
gradients_4/FillFillgradients_4/Shapegradients_4/grad_ys_0*
T0*

index_type0
Q
#gradients_4/loss_grad/Reshape/shapeConst*
valueB:*
dtype0
v
gradients_4/loss_grad/ReshapeReshapegradients_4/Fill#gradients_4/loss_grad/Reshape/shape*
T0*
Tshape0
m
gradients_4/loss_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
y
gradients_4/loss_grad/TileTilegradients_4/loss_grad/Reshapegradients_4/loss_grad/Shape*

Tmultiples0*
T0
o
gradients_4/loss_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
F
gradients_4/loss_grad/Shape_2Const*
dtype0*
valueB 
I
gradients_4/loss_grad/ConstConst*
valueB: *
dtype0
�
gradients_4/loss_grad/ProdProdgradients_4/loss_grad/Shape_1gradients_4/loss_grad/Const*

Tidx0*
	keep_dims( *
T0
K
gradients_4/loss_grad/Const_1Const*
valueB: *
dtype0
�
gradients_4/loss_grad/Prod_1Prodgradients_4/loss_grad/Shape_2gradients_4/loss_grad/Const_1*

Tidx0*
	keep_dims( *
T0
I
gradients_4/loss_grad/Maximum/yConst*
value	B :*
dtype0
p
gradients_4/loss_grad/MaximumMaximumgradients_4/loss_grad/Prod_1gradients_4/loss_grad/Maximum/y*
T0
n
gradients_4/loss_grad/floordivFloorDivgradients_4/loss_grad/Prodgradients_4/loss_grad/Maximum*
T0
j
gradients_4/loss_grad/CastCastgradients_4/loss_grad/floordiv*

SrcT0*
Truncate( *

DstT0
i
gradients_4/loss_grad/truedivRealDivgradients_4/loss_grad/Tilegradients_4/loss_grad/Cast*
T0
�
Egradients_4/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0
�
Ggradients_4/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients_4/loss_grad/truedivEgradients_4/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0
T
gradients_4/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0
w
Dgradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
@gradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsGgradients_4/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0
�
9gradients_4/softmax_cross_entropy_with_logits_sg_grad/mulMul@gradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0
�
@gradients_4/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0
�
9gradients_4/softmax_cross_entropy_with_logits_sg_grad/NegNeg@gradients_4/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0
y
Fgradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
Bgradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsGgradients_4/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeFgradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0
�
;gradients_4/softmax_cross_entropy_with_logits_sg_grad/mul_1MulBgradients_4/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_19gradients_4/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0
m
Cgradients_4/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapelogits*
T0*
out_type0
�
Egradients_4/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshape9gradients_4/softmax_cross_entropy_with_logits_sg_grad/mulCgradients_4/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0
I
gradients_4/logits_grad/ShapeShapeMatMul_1*
T0*
out_type0
J
gradients_4/logits_grad/Shape_1Shapeb2/read*
T0*
out_type0
�
-gradients_4/logits_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/logits_grad/Shapegradients_4/logits_grad/Shape_1*
T0
�
gradients_4/logits_grad/SumSumEgradients_4/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape-gradients_4/logits_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
}
gradients_4/logits_grad/ReshapeReshapegradients_4/logits_grad/Sumgradients_4/logits_grad/Shape*
T0*
Tshape0
�
gradients_4/logits_grad/Sum_1SumEgradients_4/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape/gradients_4/logits_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
!gradients_4/logits_grad/Reshape_1Reshapegradients_4/logits_grad/Sum_1gradients_4/logits_grad/Shape_1*
T0*
Tshape0
<
d-b2Identity!gradients_4/logits_grad/Reshape_1*
T0
N
initNoOp^Variable/Assign
^b1/Assign
^b2/Assign
^w1/Assign
^w2/Assign

init_1NoOp

ws_initNoOp^init^init_1"�