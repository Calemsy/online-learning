
A
input_xPlaceholder*
shape:���������'*
dtype0
A
input_yPlaceholder*
dtype0*
shape:���������
K
truncated_normal/shapeConst*
dtype0*
valueB"      
B
truncated_normal/meanConst*
valueB
 *    *
dtype0
D
truncated_normal/stddevConst*
valueB
 *���=*
dtype0
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
W
w
VariableV2*
shared_name *
dtype0*
	container *
shape:
��
o
w/AssignAssignwtruncated_normal*
use_locking(*
T0*
_class

loc:@w*
validate_shape(
4
w/readIdentityw*
T0*
_class

loc:@w
=
CastCastinput_x*

SrcT0*
Truncate( *

DstT0
7
GatherV2/axisConst*
value	B : *
dtype0
o
GatherV2GatherV2w/readCastGatherV2/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0
?
Sum/reduction_indicesConst*
value	B :*
dtype0
Q
SumSumGatherV2Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
M
truncated_normal_1/shapeConst*
dtype0*
valueB"      
D
truncated_normal_1/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
dtype0*
seed2 
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
W
v
VariableV2*
dtype0*
	container *
shape:
��*
shared_name 
q
v/AssignAssignvtruncated_normal_1*
use_locking(*
T0*
_class

loc:@v*
validate_shape(
4
v/readIdentityv*
T0*
_class

loc:@v
9
GatherV2_1/axisConst*
value	B : *
dtype0
s

GatherV2_1GatherV2v/readCastGatherV2_1/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0
B
Reshape/shapeConst*
valueB"   ����*
dtype0
D
ReshapeReshape
GatherV2_1Reshape/shape*
T0*
Tshape0
M
truncated_normal_2/shapeConst*
valueB"p     *
dtype0
D
truncated_normal_2/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_2/stddevConst*
dtype0*
valueB
 *���=
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*
seed2 *

seed 
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
X
w1
VariableV2*
shape:
��*
shared_name *
dtype0*
	container 
t
	w1/AssignAssignw1truncated_normal_2*
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
valueB	�*    *
dtype0
W
b1
VariableV2*
dtype0*
	container *
shape:	�*
shared_name 
g
	b1/AssignAssignb1zeros*
use_locking(*
T0*
_class
	loc:@b1*
validate_shape(
7
b1/readIdentityb1*
T0*
_class
	loc:@b1
Q
MatMulMatMulReshapew1/read*
T0*
transpose_a( *
transpose_b( 
$
AddAddMatMulb1/read*
T0

TanhTanhAdd*
T0
M
truncated_normal_3/shapeConst*
dtype0*
valueB"   �   
D
truncated_normal_3/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_3/stddevConst*
dtype0*
valueB
 *���=
~
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
dtype0*
seed2 *

seed 
e
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0
S
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0
X
w2
VariableV2*
shared_name *
dtype0*
	container *
shape:
��
t
	w2/AssignAssignw2truncated_normal_3*
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
=
zeros_1Const*
dtype0*
valueB	�*    
W
b2
VariableV2*
shared_name *
dtype0*
	container *
shape:	�
i
	b2/AssignAssignb2zeros_1*
use_locking(*
T0*
_class
	loc:@b2*
validate_shape(
7
b2/readIdentityb2*
T0*
_class
	loc:@b2
P
MatMul_1MatMulTanhw2/read*
transpose_a( *
transpose_b( *
T0
(
Add_1AddMatMul_1b2/read*
T0

Tanh_1TanhAdd_1*
T0
M
truncated_normal_4/shapeConst*
valueB"�   @   *
dtype0
D
truncated_normal_4/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_4/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0
S
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0
W
w3
VariableV2*
shared_name *
dtype0*
	container *
shape:	�@
t
	w3/AssignAssignw3truncated_normal_4*
use_locking(*
T0*
_class
	loc:@w3*
validate_shape(
7
w3/readIdentityw3*
T0*
_class
	loc:@w3
<
zeros_2Const*
dtype0*
valueB@*    
V
b3
VariableV2*
dtype0*
	container *
shape
:@*
shared_name 
i
	b3/AssignAssignb3zeros_2*
use_locking(*
T0*
_class
	loc:@b3*
validate_shape(
7
b3/readIdentityb3*
T0*
_class
	loc:@b3
R
MatMul_2MatMulTanh_1w3/read*
transpose_a( *
transpose_b( *
T0
(
Add_2AddMatMul_2b3/read*
T0

Tanh_2TanhAdd_2*
T0
M
truncated_normal_5/shapeConst*
valueB"@      *
dtype0
D
truncated_normal_5/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_5/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0
S
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0
V
w4
VariableV2*
shared_name *
dtype0*
	container *
shape
:@
t
	w4/AssignAssignw4truncated_normal_5*
use_locking(*
T0*
_class
	loc:@w4*
validate_shape(
7
w4/readIdentityw4*
T0*
_class
	loc:@w4
<
zeros_3Const*
valueB*    *
dtype0
V
b4
VariableV2*
dtype0*
	container *
shape
:*
shared_name 
i
	b4/AssignAssignb4zeros_3*
validate_shape(*
use_locking(*
T0*
_class
	loc:@b4
7
b4/readIdentityb4*
T0*
_class
	loc:@b4
R
MatMul_3MatMulTanh_2w4/read*
T0*
transpose_a( *
transpose_b( 
(
Add_3AddMatMul_3b4/read*
T0

Tanh_3TanhAdd_3*
T0
$
add_4AddV2Tanh_3Sum*
T0
"
logitsIdentityadd_4*
T0
]
(logistic_loss/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
K
logistic_loss/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
logistic_loss/zeros_likeFill(logistic_loss/zeros_like/shape_as_tensorlogistic_loss/zeros_like/Const*
T0*

index_type0
T
logistic_loss/GreaterEqualGreaterEqualadd_4logistic_loss/zeros_like*
T0
d
logistic_loss/SelectSelectlogistic_loss/GreaterEqualadd_4logistic_loss/zeros_like*
T0
(
logistic_loss/NegNegadd_4*
T0
_
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negadd_4*
T0
1
logistic_loss/mulMuladd_4input_y*
T0
J
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0
9
logistic_loss/ExpExplogistic_loss/Select_1*
T0
8
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0
E
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0
:
ConstConst*
dtype0*
valueB"       
H
lossMeanlogistic_lossConst*

Tidx0*
	keep_dims( *
T0
!
loss_1Identityloss*
T0
?
Cast_1Castinput_y*
Truncate( *

DstT0*

SrcT0
"
SigmoidSigmoidadd_4*
T0
7

auc/Cast/xConst*
valueB
 *    *
dtype0
S
%auc/assert_greater_equal/GreaterEqualGreaterEqualSigmoid
auc/Cast/x*
T0
S
auc/assert_greater_equal/ConstConst*
valueB"       *
dtype0
�
auc/assert_greater_equal/AllAll%auc/assert_greater_equal/GreaterEqualauc/assert_greater_equal/Const*

Tidx0*
	keep_dims( 
k
%auc/assert_greater_equal/Assert/ConstConst*
dtype0*.
value%B# Bpredictions must be in [0, 1]
{
'auc/assert_greater_equal/Assert/Const_1Const*<
value3B1 B+Condition x >= y did not hold element-wise:*
dtype0
`
'auc/assert_greater_equal/Assert/Const_2Const*
dtype0*!
valueB Bx (Sigmoid:0) = 
c
'auc/assert_greater_equal/Assert/Const_3Const*$
valueB By (auc/Cast/x:0) = *
dtype0
�
2auc/assert_greater_equal/Assert/AssertGuard/SwitchSwitchauc/assert_greater_equal/Allauc/assert_greater_equal/All*
T0


4auc/assert_greater_equal/Assert/AssertGuard/switch_tIdentity4auc/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0

}
4auc/assert_greater_equal/Assert/AssertGuard/switch_fIdentity2auc/assert_greater_equal/Assert/AssertGuard/Switch*
T0

f
3auc/assert_greater_equal/Assert/AssertGuard/pred_idIdentityauc/assert_greater_equal/All*
T0

o
0auc/assert_greater_equal/Assert/AssertGuard/NoOpNoOp5^auc/assert_greater_equal/Assert/AssertGuard/switch_t
�
>auc/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentity4auc/assert_greater_equal/Assert/AssertGuard/switch_t1^auc/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*G
_class=
;9loc:@auc/assert_greater_equal/Assert/AssertGuard/switch_t
�
9auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const5^auc/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0
�
9auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const5^auc/assert_greater_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x >= y did not hold element-wise:*
dtype0
�
9auc/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const5^auc/assert_greater_equal/Assert/AssertGuard/switch_f*!
valueB Bx (Sigmoid:0) = *
dtype0
�
9auc/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const5^auc/assert_greater_equal/Assert/AssertGuard/switch_f*$
valueB By (auc/Cast/x:0) = *
dtype0
�
2auc/assert_greater_equal/Assert/AssertGuard/AssertAssert9auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch9auc/assert_greater_equal/Assert/AssertGuard/Assert/data_09auc/assert_greater_equal/Assert/AssertGuard/Assert/data_19auc/assert_greater_equal/Assert/AssertGuard/Assert/data_2;auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_19auc/assert_greater_equal/Assert/AssertGuard/Assert/data_4;auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T

2*
	summarize
�
9auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitchauc/assert_greater_equal/All3auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*/
_class%
#!loc:@auc/assert_greater_equal/All
�
;auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1SwitchSigmoid3auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*
_class
loc:@Sigmoid
�
;auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switch
auc/Cast/x3auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*
_class
loc:@auc/Cast/x
�
@auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1Identity4auc/assert_greater_equal/Assert/AssertGuard/switch_f3^auc/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*G
_class=
;9loc:@auc/assert_greater_equal/Assert/AssertGuard/switch_f
�
1auc/assert_greater_equal/Assert/AssertGuard/MergeMerge@auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1>auc/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N
9
auc/Cast_1/xConst*
valueB
 *  �?*
dtype0
L
auc/assert_less_equal/LessEqual	LessEqualSigmoidauc/Cast_1/x*
T0
P
auc/assert_less_equal/ConstConst*
valueB"       *
dtype0
{
auc/assert_less_equal/AllAllauc/assert_less_equal/LessEqualauc/assert_less_equal/Const*

Tidx0*
	keep_dims( 
h
"auc/assert_less_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0
x
$auc/assert_less_equal/Assert/Const_1Const*
dtype0*<
value3B1 B+Condition x <= y did not hold element-wise:
]
$auc/assert_less_equal/Assert/Const_2Const*
dtype0*!
valueB Bx (Sigmoid:0) = 
b
$auc/assert_less_equal/Assert/Const_3Const*
dtype0*&
valueB By (auc/Cast_1/x:0) = 
x
/auc/assert_less_equal/Assert/AssertGuard/SwitchSwitchauc/assert_less_equal/Allauc/assert_less_equal/All*
T0

y
1auc/assert_less_equal/Assert/AssertGuard/switch_tIdentity1auc/assert_less_equal/Assert/AssertGuard/Switch:1*
T0

w
1auc/assert_less_equal/Assert/AssertGuard/switch_fIdentity/auc/assert_less_equal/Assert/AssertGuard/Switch*
T0

`
0auc/assert_less_equal/Assert/AssertGuard/pred_idIdentityauc/assert_less_equal/All*
T0

i
-auc/assert_less_equal/Assert/AssertGuard/NoOpNoOp2^auc/assert_less_equal/Assert/AssertGuard/switch_t
�
;auc/assert_less_equal/Assert/AssertGuard/control_dependencyIdentity1auc/assert_less_equal/Assert/AssertGuard/switch_t.^auc/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*D
_class:
86loc:@auc/assert_less_equal/Assert/AssertGuard/switch_t
�
6auc/assert_less_equal/Assert/AssertGuard/Assert/data_0Const2^auc/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0
�
6auc/assert_less_equal/Assert/AssertGuard/Assert/data_1Const2^auc/assert_less_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x <= y did not hold element-wise:*
dtype0
�
6auc/assert_less_equal/Assert/AssertGuard/Assert/data_2Const2^auc/assert_less_equal/Assert/AssertGuard/switch_f*!
valueB Bx (Sigmoid:0) = *
dtype0
�
6auc/assert_less_equal/Assert/AssertGuard/Assert/data_4Const2^auc/assert_less_equal/Assert/AssertGuard/switch_f*&
valueB By (auc/Cast_1/x:0) = *
dtype0
�
/auc/assert_less_equal/Assert/AssertGuard/AssertAssert6auc/assert_less_equal/Assert/AssertGuard/Assert/Switch6auc/assert_less_equal/Assert/AssertGuard/Assert/data_06auc/assert_less_equal/Assert/AssertGuard/Assert/data_16auc/assert_less_equal/Assert/AssertGuard/Assert/data_28auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_16auc/assert_less_equal/Assert/AssertGuard/Assert/data_48auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T

2*
	summarize
�
6auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitchauc/assert_less_equal/All0auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*,
_class"
 loc:@auc/assert_less_equal/All
�
8auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1SwitchSigmoid0auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*
_class
loc:@Sigmoid
�
8auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switchauc/Cast_1/x0auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*
_class
loc:@auc/Cast_1/x
�
=auc/assert_less_equal/Assert/AssertGuard/control_dependency_1Identity1auc/assert_less_equal/Assert/AssertGuard/switch_f0^auc/assert_less_equal/Assert/AssertGuard/Assert*
T0
*D
_class:
86loc:@auc/assert_less_equal/Assert/AssertGuard/switch_f
�
.auc/assert_less_equal/Assert/AssertGuard/MergeMerge=auc/assert_less_equal/Assert/AssertGuard/control_dependency_1;auc/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N
�

auc/Cast_2CastCast_12^auc/assert_greater_equal/Assert/AssertGuard/Merge/^auc/assert_less_equal/Assert/AssertGuard/Merge*
Truncate( *

DstT0
*

SrcT0
F
auc/Reshape/shapeConst*
valueB"����   *
dtype0
I
auc/ReshapeReshapeSigmoidauc/Reshape/shape*
T0*
Tshape0
H
auc/Reshape_1/shapeConst*
valueB"   ����*
dtype0
P
auc/Reshape_1Reshape
auc/Cast_2auc/Reshape_1/shape*
T0
*
Tshape0
�
	auc/ConstConst*�
value�B��"���ֳϩ�;ϩ$<��v<ϩ�<C��<���<�=ϩ$=	?9=C�M=}ib=��v=�Ʌ=��=2_�=ϩ�=l��=	?�=���=C��=��=}i�=��=���=�� >��>G�
>�>�9>2_>��>ϩ$>�)>l�.>�4>	?9>Wd>>��C>��H>C�M>��R>�X>.D]>}ib>ˎg>�l>h�q>��v>$|>���>Q7�>�Ʌ>�\�>G�>>��><��>�9�>�̗>2_�>��>���>(�>ϩ�>v<�>ϩ>�a�>l��>��>��>b��>	?�>�ѻ>Wd�>���>���>M�>���>�A�>C��>�f�>���>9��>��>���>.D�>���>}i�>$��>ˎ�>r!�>��>�F�>h��>l�>���>^��>$�>���>�� ?��?Q7?��?��?L?�\?�	?G�
?�8?�?B�?�?�]?<�?��?�9?7�?��?�?2_?��?��?-;?��?�� ?("?{`#?ϩ$?#�%?v<'?ʅ(?�)?q+?�a,?�-?l�.?�=0?�1?g�2?�4?c5?b�6?��7?	?9?]�:?��;?=?Wd>?��??��@?R@B?��C?��D?MF?�eG?��H?H�I?�AK?�L?C�M?�O?�fP?>�Q?��R?�BT?9�U?��V?�X?3hY?��Z?��[?.D]?��^?��_?) a?}ib?вc?$�d?xEf?ˎg?�h?r!j?�jk?�l?m�m?�Fo?�p?h�q?�"s?lt?c�u?��v?
Hx?^�y?��z?$|?Ym}?��~? �?*
dtype0
@
auc/ExpandDims/dimConst*
valueB:*
dtype0
P
auc/ExpandDims
ExpandDims	auc/Constauc/ExpandDims/dim*

Tdim0*
T0
>
	auc/stackConst*
dtype0*
valueB"      
F
auc/TileTileauc/ExpandDims	auc/stack*

Tmultiples0*
T0
G
auc/transpose/permConst*
valueB"       *
dtype0
Q
auc/transpose	Transposeauc/Reshapeauc/transpose/perm*
T0*
Tperm0
I
auc/Tile_1/multiplesConst*
valueB"�      *
dtype0
R

auc/Tile_1Tileauc/transposeauc/Tile_1/multiples*

Tmultiples0*
T0
5
auc/GreaterGreater
auc/Tile_1auc/Tile*
T0
)
auc/LogicalNot
LogicalNotauc/Greater
I
auc/Tile_2/multiplesConst*
valueB"�      *
dtype0
R

auc/Tile_2Tileauc/Reshape_1auc/Tile_2/multiples*
T0
*

Tmultiples0
*
auc/LogicalNot_1
LogicalNot
auc/Tile_2
}
$auc/true_positives/Initializer/zerosConst*%
_class
loc:@auc/true_positives*
valueB�*    *
dtype0
�
auc/true_positives
VariableV2*
dtype0*
	container *
shape:�*
shared_name *%
_class
loc:@auc/true_positives
�
auc/true_positives/AssignAssignauc/true_positives$auc/true_positives/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@auc/true_positives
g
auc/true_positives/readIdentityauc/true_positives*
T0*%
_class
loc:@auc/true_positives
5
auc/LogicalAnd
LogicalAnd
auc/Tile_2auc/Greater
J

auc/Cast_3Castauc/LogicalAnd*

SrcT0
*
Truncate( *

DstT0
C
auc/Sum/reduction_indicesConst*
value	B :*
dtype0
[
auc/SumSum
auc/Cast_3auc/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
z
auc/AssignAdd	AssignAddauc/true_positivesauc/Sum*
use_locking( *
T0*%
_class
loc:@auc/true_positives

%auc/false_negatives/Initializer/zerosConst*&
_class
loc:@auc/false_negatives*
valueB�*    *
dtype0
�
auc/false_negatives
VariableV2*
dtype0*
	container *
shape:�*
shared_name *&
_class
loc:@auc/false_negatives
�
auc/false_negatives/AssignAssignauc/false_negatives%auc/false_negatives/Initializer/zeros*
T0*&
_class
loc:@auc/false_negatives*
validate_shape(*
use_locking(
j
auc/false_negatives/readIdentityauc/false_negatives*
T0*&
_class
loc:@auc/false_negatives
:
auc/LogicalAnd_1
LogicalAnd
auc/Tile_2auc/LogicalNot
L

auc/Cast_4Castauc/LogicalAnd_1*
Truncate( *

DstT0*

SrcT0

E
auc/Sum_1/reduction_indicesConst*
value	B :*
dtype0
_
	auc/Sum_1Sum
auc/Cast_4auc/Sum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0
�
auc/AssignAdd_1	AssignAddauc/false_negatives	auc/Sum_1*
use_locking( *
T0*&
_class
loc:@auc/false_negatives
}
$auc/true_negatives/Initializer/zerosConst*
dtype0*%
_class
loc:@auc/true_negatives*
valueB�*    
�
auc/true_negatives
VariableV2*
shared_name *%
_class
loc:@auc/true_negatives*
dtype0*
	container *
shape:�
�
auc/true_negatives/AssignAssignauc/true_negatives$auc/true_negatives/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@auc/true_negatives*
validate_shape(
g
auc/true_negatives/readIdentityauc/true_negatives*
T0*%
_class
loc:@auc/true_negatives
@
auc/LogicalAnd_2
LogicalAndauc/LogicalNot_1auc/LogicalNot
L

auc/Cast_5Castauc/LogicalAnd_2*

SrcT0
*
Truncate( *

DstT0
E
auc/Sum_2/reduction_indicesConst*
value	B :*
dtype0
_
	auc/Sum_2Sum
auc/Cast_5auc/Sum_2/reduction_indices*

Tidx0*
	keep_dims( *
T0
~
auc/AssignAdd_2	AssignAddauc/true_negatives	auc/Sum_2*
use_locking( *
T0*%
_class
loc:@auc/true_negatives

%auc/false_positives/Initializer/zerosConst*&
_class
loc:@auc/false_positives*
valueB�*    *
dtype0
�
auc/false_positives
VariableV2*
shared_name *&
_class
loc:@auc/false_positives*
dtype0*
	container *
shape:�
�
auc/false_positives/AssignAssignauc/false_positives%auc/false_positives/Initializer/zeros*
T0*&
_class
loc:@auc/false_positives*
validate_shape(*
use_locking(
j
auc/false_positives/readIdentityauc/false_positives*
T0*&
_class
loc:@auc/false_positives
=
auc/LogicalAnd_3
LogicalAndauc/LogicalNot_1auc/Greater
L

auc/Cast_6Castauc/LogicalAnd_3*

SrcT0
*
Truncate( *

DstT0
E
auc/Sum_3/reduction_indicesConst*
value	B :*
dtype0
_
	auc/Sum_3Sum
auc/Cast_6auc/Sum_3/reduction_indices*

Tidx0*
	keep_dims( *
T0
�
auc/AssignAdd_3	AssignAddauc/false_positives	auc/Sum_3*
use_locking( *
T0*&
_class
loc:@auc/false_positives
6
	auc/add/yConst*
valueB
 *�7�5*
dtype0
=
auc/addAddV2auc/true_positives/read	auc/add/y*
T0
N
	auc/add_1AddV2auc/true_positives/readauc/false_negatives/read*
T0
8
auc/add_2/yConst*
valueB
 *�7�5*
dtype0
3
	auc/add_2AddV2	auc/add_1auc/add_2/y*
T0
/
auc/divRealDivauc/add	auc/add_2*
T0
N
	auc/add_3AddV2auc/false_positives/readauc/true_negatives/read*
T0
8
auc/add_4/yConst*
valueB
 *�7�5*
dtype0
3
	auc/add_4AddV2	auc/add_3auc/add_4/y*
T0
B
	auc/div_1RealDivauc/false_positives/read	auc/add_4*
T0
E
auc/strided_slice/stackConst*
dtype0*
valueB: 
H
auc/strided_slice/stack_1Const*
valueB:�*
dtype0
G
auc/strided_slice/stack_2Const*
dtype0*
valueB:
�
auc/strided_sliceStridedSlice	auc/div_1auc/strided_slice/stackauc/strided_slice/stack_1auc/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask 
G
auc/strided_slice_1/stackConst*
dtype0*
valueB:
I
auc/strided_slice_1/stack_1Const*
valueB: *
dtype0
I
auc/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_1StridedSlice	auc/div_1auc/strided_slice_1/stackauc/strided_slice_1/stack_1auc/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask 
?
auc/subSubauc/strided_sliceauc/strided_slice_1*
T0
G
auc/strided_slice_2/stackConst*
valueB: *
dtype0
J
auc/strided_slice_2/stack_1Const*
valueB:�*
dtype0
I
auc/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_2StridedSliceauc/divauc/strided_slice_2/stackauc/strided_slice_2/stack_1auc/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
G
auc/strided_slice_3/stackConst*
valueB:*
dtype0
I
auc/strided_slice_3/stack_1Const*
valueB: *
dtype0
I
auc/strided_slice_3/stack_2Const*
dtype0*
valueB:
�
auc/strided_slice_3StridedSliceauc/divauc/strided_slice_3/stackauc/strided_slice_3/stack_1auc/strided_slice_3/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
E
	auc/add_5AddV2auc/strided_slice_2auc/strided_slice_3*
T0
:
auc/truediv/yConst*
valueB
 *   @*
dtype0
9
auc/truedivRealDiv	auc/add_5auc/truediv/y*
T0
-
auc/MulMulauc/subauc/truediv*
T0
9
auc/Const_1Const*
valueB: *
dtype0
L
	auc/valueSumauc/Mulauc/Const_1*
T0*

Tidx0*
	keep_dims( 
8
auc/add_6/yConst*
valueB
 *�7�5*
dtype0
7
	auc/add_6AddV2auc/AssignAddauc/add_6/y*
T0
;
	auc/add_7AddV2auc/AssignAddauc/AssignAdd_1*
T0
8
auc/add_8/yConst*
valueB
 *�7�5*
dtype0
3
	auc/add_8AddV2	auc/add_7auc/add_8/y*
T0
3
	auc/div_2RealDiv	auc/add_6	auc/add_8*
T0
=
	auc/add_9AddV2auc/AssignAdd_3auc/AssignAdd_2*
T0
9
auc/add_10/yConst*
valueB
 *�7�5*
dtype0
5

auc/add_10AddV2	auc/add_9auc/add_10/y*
T0
:
	auc/div_3RealDivauc/AssignAdd_3
auc/add_10*
T0
G
auc/strided_slice_4/stackConst*
valueB: *
dtype0
J
auc/strided_slice_4/stack_1Const*
valueB:�*
dtype0
I
auc/strided_slice_4/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_4StridedSlice	auc/div_3auc/strided_slice_4/stackauc/strided_slice_4/stack_1auc/strided_slice_4/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
G
auc/strided_slice_5/stackConst*
dtype0*
valueB:
I
auc/strided_slice_5/stack_1Const*
valueB: *
dtype0
I
auc/strided_slice_5/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_5StridedSlice	auc/div_3auc/strided_slice_5/stackauc/strided_slice_5/stack_1auc/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask
C
	auc/sub_1Subauc/strided_slice_4auc/strided_slice_5*
T0
G
auc/strided_slice_6/stackConst*
valueB: *
dtype0
J
auc/strided_slice_6/stack_1Const*
valueB:�*
dtype0
I
auc/strided_slice_6/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_6StridedSlice	auc/div_2auc/strided_slice_6/stackauc/strided_slice_6/stack_1auc/strided_slice_6/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask 
G
auc/strided_slice_7/stackConst*
dtype0*
valueB:
I
auc/strided_slice_7/stack_1Const*
valueB: *
dtype0
I
auc/strided_slice_7/stack_2Const*
dtype0*
valueB:
�
auc/strided_slice_7StridedSlice	auc/div_2auc/strided_slice_7/stackauc/strided_slice_7/stack_1auc/strided_slice_7/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
F

auc/add_11AddV2auc/strided_slice_6auc/strided_slice_7*
T0
<
auc/truediv_1/yConst*
dtype0*
valueB
 *   @
>
auc/truediv_1RealDiv
auc/add_11auc/truediv_1/y*
T0
3
	auc/Mul_1Mul	auc/sub_1auc/truediv_1*
T0
9
auc/Const_2Const*
valueB: *
dtype0
R
auc/update_opSum	auc/Mul_1auc/Const_2*
T0*

Tidx0*
	keep_dims( 
-
	auc_valueIdentityauc/update_op*
T0
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
dtype0*
valueB
 *  �?
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
V
!gradients/loss_grad/Reshape/shapeConst*
valueB"      *
dtype0
p
gradients/loss_grad/ReshapeReshapegradients/Fill!gradients/loss_grad/Reshape/shape*
T0*
Tshape0
N
gradients/loss_grad/ConstConst*
dtype0*
valueB"      
s
gradients/loss_grad/TileTilegradients/loss_grad/Reshapegradients/loss_grad/Const*

Tmultiples0*
T0
H
gradients/loss_grad/Const_1Const*
valueB
 *  �D*
dtype0
f
gradients/loss_grad/truedivRealDivgradients/loss_grad/Tilegradients/loss_grad/Const_1*
T0
S
-gradients/logistic_loss_grad/tuple/group_depsNoOp^gradients/loss_grad/truediv
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentitygradients/loss_grad/truediv.^gradients/logistic_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/loss_grad/truediv
�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identitygradients/loss_grad/truediv.^gradients/logistic_loss_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/loss_grad/truediv
k
$gradients/logistic_loss/sub_grad/NegNeg5gradients/logistic_loss_grad/tuple/control_dependency*
T0
�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp%^gradients/logistic_loss/sub_grad/Neg6^gradients/logistic_loss_grad/tuple/control_dependency
�
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity5gradients/logistic_loss_grad/tuple/control_dependency2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/loss_grad/truediv
�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity$gradients/logistic_loss/sub_grad/Neg2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss/sub_grad/Neg
�
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0
u
&gradients/logistic_loss/Log1p_grad/addAddV2(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
l
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*
T0
�
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0
s
>gradients/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
a
4gradients/logistic_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
.gradients/logistic_loss/Select_grad/zeros_likeFill>gradients/logistic_loss/Select_grad/zeros_like/shape_as_tensor4gradients/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0
�
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1
O
&gradients/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
S
(gradients/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0
z
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1input_y*
T0
�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0
z
&gradients/logistic_loss/mul_grad/Mul_1Muladd_4;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0
�
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
�
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape
�
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1
o
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
u
@gradients/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
c
6gradients/logistic_loss/Select_1_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    
�
0gradients/logistic_loss/Select_1_grad/zeros_likeFill@gradients/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor6gradients/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0
�
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0
�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
�
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
�
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1
t
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0
�
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
N*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
D
gradients/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
C
gradients/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
T0
�
gradients/add_4_grad/SumSumgradients/AddN*gradients/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0*
Tshape0
�
gradients/add_4_grad/Sum_1Sumgradients/AddN,gradients/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
z
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
�
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_4_grad/Reshape
�
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1
j
gradients/Tanh_3_grad/TanhGradTanhGradTanh_3-gradients/add_4_grad/tuple/control_dependency*
T0
D
gradients/Sum_grad/ShapeShapeGatherV2*
T0*
out_type0
n
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/addAddV2Sum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
p
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0
u
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0
u
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*+
_class!
loc:@gradients/Sum_grad/Shape*

Tidx0
t
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N
s
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/ReshapeReshape/gradients/add_4_grad/tuple/control_dependency_1 gradients/Sum_grad/DynamicStitch*
T0*
Tshape0
s
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*

Tmultiples0
b
-gradients/Add_3_grad/BroadcastGradientArgs/s0Const*
valueB"      *
dtype0
b
-gradients/Add_3_grad/BroadcastGradientArgs/s1Const*
valueB"      *
dtype0
�
*gradients/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/Add_3_grad/BroadcastGradientArgs/s0-gradients/Add_3_grad/BroadcastGradientArgs/s1*
T0
_
*gradients/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients/Add_3_grad/SumSumgradients/Tanh_3_grad/TanhGrad*gradients/Add_3_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
W
"gradients/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
|
gradients/Add_3_grad/ReshapeReshapegradients/Add_3_grad/Sum"gradients/Add_3_grad/Reshape/shape*
T0*
Tshape0
m
%gradients/Add_3_grad/tuple/group_depsNoOp^gradients/Add_3_grad/Reshape^gradients/Tanh_3_grad/TanhGrad
�
-gradients/Add_3_grad/tuple/control_dependencyIdentitygradients/Tanh_3_grad/TanhGrad&^gradients/Add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Tanh_3_grad/TanhGrad
�
/gradients/Add_3_grad/tuple/control_dependency_1Identitygradients/Add_3_grad/Reshape&^gradients/Add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_3_grad/Reshape
p
gradients/GatherV2_grad/ShapeConst*
dtype0	*
_class

loc:@w*%
valueB	"              
�
gradients/GatherV2_grad/CastCastgradients/GatherV2_grad/Shape*

SrcT0	*
_class

loc:@w*
Truncate( *

DstT0
C
gradients/GatherV2_grad/SizeSizeCast*
T0*
out_type0
P
&gradients/GatherV2_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
"gradients/GatherV2_grad/ExpandDims
ExpandDimsgradients/GatherV2_grad/Size&gradients/GatherV2_grad/ExpandDims/dim*
T0*

Tdim0
Y
+gradients/GatherV2_grad/strided_slice/stackConst*
valueB:*
dtype0
[
-gradients/GatherV2_grad/strided_slice/stack_1Const*
valueB: *
dtype0
[
-gradients/GatherV2_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
%gradients/GatherV2_grad/strided_sliceStridedSlicegradients/GatherV2_grad/Cast+gradients/GatherV2_grad/strided_slice/stack-gradients/GatherV2_grad/strided_slice/stack_1-gradients/GatherV2_grad/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask 
M
#gradients/GatherV2_grad/concat/axisConst*
dtype0*
value	B : 
�
gradients/GatherV2_grad/concatConcatV2"gradients/GatherV2_grad/ExpandDims%gradients/GatherV2_grad/strided_slice#gradients/GatherV2_grad/concat/axis*
T0*
N*

Tidx0
z
gradients/GatherV2_grad/ReshapeReshapegradients/Sum_grad/Tilegradients/GatherV2_grad/concat*
T0*
Tshape0
m
!gradients/GatherV2_grad/Reshape_1ReshapeCast"gradients/GatherV2_grad/ExpandDims*
T0*
Tshape0
�
gradients/MatMul_3_grad/MatMulMatMul-gradients/Add_3_grad/tuple/control_dependencyw4/read*
T0*
transpose_a( *
transpose_b(
�
 gradients/MatMul_3_grad/MatMul_1MatMulTanh_2-gradients/Add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
�
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul
�
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1
m
gradients/Tanh_2_grad/TanhGradTanhGradTanh_20gradients/MatMul_3_grad/tuple/control_dependency*
T0
b
-gradients/Add_2_grad/BroadcastGradientArgs/s0Const*
dtype0*
valueB"   @   
b
-gradients/Add_2_grad/BroadcastGradientArgs/s1Const*
dtype0*
valueB"   @   
�
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/Add_2_grad/BroadcastGradientArgs/s0-gradients/Add_2_grad/BroadcastGradientArgs/s1*
T0
X
*gradients/Add_2_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients/Add_2_grad/SumSumgradients/Tanh_2_grad/TanhGrad*gradients/Add_2_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
W
"gradients/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
|
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sum"gradients/Add_2_grad/Reshape/shape*
T0*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Tanh_2_grad/TanhGrad
�
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Tanh_2_grad/TanhGrad&^gradients/Add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Tanh_2_grad/TanhGrad
�
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_2_grad/Reshape
�
gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyw3/read*
T0*
transpose_a( *
transpose_b(
�
 gradients/MatMul_2_grad/MatMul_1MatMulTanh_1-gradients/Add_2_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1
m
gradients/Tanh_1_grad/TanhGradTanhGradTanh_10gradients/MatMul_2_grad/tuple/control_dependency*
T0
b
-gradients/Add_1_grad/BroadcastGradientArgs/s0Const*
valueB"   �   *
dtype0
b
-gradients/Add_1_grad/BroadcastGradientArgs/s1Const*
dtype0*
valueB"   �   
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/Add_1_grad/BroadcastGradientArgs/s0-gradients/Add_1_grad/BroadcastGradientArgs/s1*
T0
X
*gradients/Add_1_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients/Add_1_grad/SumSumgradients/Tanh_1_grad/TanhGrad*gradients/Add_1_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
W
"gradients/Add_1_grad/Reshape/shapeConst*
valueB"   �   *
dtype0
|
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sum"gradients/Add_1_grad/Reshape/shape*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Tanh_1_grad/TanhGrad
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Tanh_1_grad/TanhGrad&^gradients/Add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Tanh_1_grad/TanhGrad
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyw2/read*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMulTanh-gradients/Add_1_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
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
gradients/Tanh_grad/TanhGradTanhGradTanh0gradients/MatMul_1_grad/tuple/control_dependency*
T0
`
+gradients/Add_grad/BroadcastGradientArgs/s0Const*
valueB"      *
dtype0
`
+gradients/Add_grad/BroadcastGradientArgs/s1Const*
valueB"      *
dtype0
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/Add_grad/BroadcastGradientArgs/s0+gradients/Add_grad/BroadcastGradientArgs/s1*
T0
V
(gradients/Add_grad/Sum/reduction_indicesConst*
dtype0*
valueB: 
�
gradients/Add_grad/SumSumgradients/Tanh_grad/TanhGrad(gradients/Add_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
U
 gradients/Add_grad/Reshape/shapeConst*
dtype0*
valueB"      
v
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sum gradients/Add_grad/Reshape/shape*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Tanh_grad/TanhGrad
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Tanh_grad/TanhGrad$^gradients/Add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Tanh_grad/TanhGrad
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape
�
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyw1/read*
T0*
transpose_a( *
transpose_b(
�
gradients/MatMul_grad/MatMul_1MatMulReshape+gradients/Add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
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
J
gradients/Reshape_grad/ShapeShape
GatherV2_1*
T0*
out_type0
�
gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0
r
gradients/GatherV2_1_grad/ShapeConst*
_class

loc:@v*%
valueB	"              *
dtype0	
�
gradients/GatherV2_1_grad/CastCastgradients/GatherV2_1_grad/Shape*

SrcT0	*
_class

loc:@v*
Truncate( *

DstT0
E
gradients/GatherV2_1_grad/SizeSizeCast*
T0*
out_type0
R
(gradients/GatherV2_1_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
$gradients/GatherV2_1_grad/ExpandDims
ExpandDimsgradients/GatherV2_1_grad/Size(gradients/GatherV2_1_grad/ExpandDims/dim*

Tdim0*
T0
[
-gradients/GatherV2_1_grad/strided_slice/stackConst*
valueB:*
dtype0
]
/gradients/GatherV2_1_grad/strided_slice/stack_1Const*
valueB: *
dtype0
]
/gradients/GatherV2_1_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
'gradients/GatherV2_1_grad/strided_sliceStridedSlicegradients/GatherV2_1_grad/Cast-gradients/GatherV2_1_grad/strided_slice/stack/gradients/GatherV2_1_grad/strided_slice/stack_1/gradients/GatherV2_1_grad/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask
O
%gradients/GatherV2_1_grad/concat/axisConst*
value	B : *
dtype0
�
 gradients/GatherV2_1_grad/concatConcatV2$gradients/GatherV2_1_grad/ExpandDims'gradients/GatherV2_1_grad/strided_slice%gradients/GatherV2_1_grad/concat/axis*
T0*
N*

Tidx0
�
!gradients/GatherV2_1_grad/ReshapeReshapegradients/Reshape_grad/Reshape gradients/GatherV2_1_grad/concat*
T0*
Tshape0
q
#gradients/GatherV2_1_grad/Reshape_1ReshapeCast$gradients/GatherV2_1_grad/ExpandDims*
T0*
Tshape0
E
train_step/learning_rateConst*
dtype0*
valueB
 *��L=
x
train_step/update_w/mulMulgradients/GatherV2_grad/Reshapetrain_step/learning_rate*
T0*
_class

loc:@w
�
train_step/update_w/ScatterSub
ScatterSubw!gradients/GatherV2_grad/Reshape_1train_step/update_w/mul*
T0*
_class

loc:@w*
use_locking( *
Tindices0
z
train_step/update_v/mulMul!gradients/GatherV2_1_grad/Reshapetrain_step/learning_rate*
T0*
_class

loc:@v
�
train_step/update_v/ScatterSub
ScatterSubv#gradients/GatherV2_1_grad/Reshape_1train_step/update_v/mul*
use_locking( *
Tindices0*
T0*
_class

loc:@v
�
)train_step/update_w1/ApplyGradientDescentApplyGradientDescentw1train_step/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@w1
�
)train_step/update_b1/ApplyGradientDescentApplyGradientDescentb1train_step/learning_rate-gradients/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b1
�
)train_step/update_w2/ApplyGradientDescentApplyGradientDescentw2train_step/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@w2
�
)train_step/update_b2/ApplyGradientDescentApplyGradientDescentb2train_step/learning_rate/gradients/Add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b2
�
)train_step/update_w3/ApplyGradientDescentApplyGradientDescentw3train_step/learning_rate2gradients/MatMul_2_grad/tuple/control_dependency_1*
T0*
_class
	loc:@w3*
use_locking( 
�
)train_step/update_b3/ApplyGradientDescentApplyGradientDescentb3train_step/learning_rate/gradients/Add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b3
�
)train_step/update_w4/ApplyGradientDescentApplyGradientDescentw4train_step/learning_rate2gradients/MatMul_3_grad/tuple/control_dependency_1*
T0*
_class
	loc:@w4*
use_locking( 
�
)train_step/update_b4/ApplyGradientDescentApplyGradientDescentb4train_step/learning_rate/gradients/Add_3_grad/tuple/control_dependency_1*
T0*
_class
	loc:@b4*
use_locking( 
�

train_stepNoOp*^train_step/update_b1/ApplyGradientDescent*^train_step/update_b2/ApplyGradientDescent*^train_step/update_b3/ApplyGradientDescent*^train_step/update_b4/ApplyGradientDescent^train_step/update_v/ScatterSub^train_step/update_w/ScatterSub*^train_step/update_w1/ApplyGradientDescent*^train_step/update_w2/ApplyGradientDescent*^train_step/update_w3/ApplyGradientDescent*^train_step/update_w4/ApplyGradientDescent
:
gradients_1/ShapeConst*
valueB *
dtype0
B
gradients_1/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0
X
#gradients_1/loss_grad/Reshape/shapeConst*
valueB"      *
dtype0
v
gradients_1/loss_grad/ReshapeReshapegradients_1/Fill#gradients_1/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_1/loss_grad/ConstConst*
dtype0*
valueB"      
y
gradients_1/loss_grad/TileTilegradients_1/loss_grad/Reshapegradients_1/loss_grad/Const*

Tmultiples0*
T0
J
gradients_1/loss_grad/Const_1Const*
valueB
 *  �D*
dtype0
l
gradients_1/loss_grad/truedivRealDivgradients_1/loss_grad/Tilegradients_1/loss_grad/Const_1*
T0
U
&gradients_1/logistic_loss/sub_grad/NegNeggradients_1/loss_grad/truediv*
T0
w
*gradients_1/logistic_loss/Log1p_grad/add/xConst^gradients_1/loss_grad/truediv*
valueB
 *  �?*
dtype0
y
(gradients_1/logistic_loss/Log1p_grad/addAddV2*gradients_1/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_1/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_1/logistic_loss/Log1p_grad/add*
T0
�
(gradients_1/logistic_loss/Log1p_grad/mulMulgradients_1/loss_grad/truediv/gradients_1/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_1/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
c
6gradients_1/logistic_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
0gradients_1/logistic_loss/Select_grad/zeros_likeFill@gradients_1/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_1/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_1/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_1/loss_grad/truediv0gradients_1/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_1/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_1/logistic_loss/Select_grad/zeros_likegradients_1/loss_grad/truediv*
T0
Q
(gradients_1/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_1/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/mul_grad/Shape*gradients_1/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_1/logistic_loss/mul_grad/MulMul&gradients_1/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_1/logistic_loss/mul_grad/SumSum&gradients_1/logistic_loss/mul_grad/Mul8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*gradients_1/logistic_loss/mul_grad/ReshapeReshape&gradients_1/logistic_loss/mul_grad/Sum(gradients_1/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_1/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_1/logistic_loss/sub_grad/Neg*
T0
�
(gradients_1/logistic_loss/mul_grad/Sum_1Sum(gradients_1/logistic_loss/mul_grad/Mul_1:gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients_1/logistic_loss/mul_grad/Reshape_1Reshape(gradients_1/logistic_loss/mul_grad/Sum_1*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_1/logistic_loss/Exp_grad/mulMul(gradients_1/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_1/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
e
8gradients_1/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_1/logistic_loss/Select_1_grad/zeros_likeFillBgradients_1/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_1/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_1/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_1/logistic_loss/Exp_grad/mul2gradients_1/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_1/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_1/logistic_loss/Select_1_grad/zeros_like&gradients_1/logistic_loss/Exp_grad/mul*
T0
f
&gradients_1/logistic_loss/Neg_grad/NegNeg.gradients_1/logistic_loss/Select_1_grad/Select*
T0
�
gradients_1/AddNAddN,gradients_1/logistic_loss/Select_grad/Select*gradients_1/logistic_loss/mul_grad/Reshape0gradients_1/logistic_loss/Select_1_grad/Select_1&gradients_1/logistic_loss/Neg_grad/Neg*
N*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select
F
gradients_1/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_1/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_1/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_4_grad/Shapegradients_1/add_4_grad/Shape_1*
T0
�
gradients_1/add_4_grad/SumSumgradients_1/AddN,gradients_1/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
z
gradients_1/add_4_grad/ReshapeReshapegradients_1/add_4_grad/Sumgradients_1/add_4_grad/Shape*
T0*
Tshape0
�
gradients_1/add_4_grad/Sum_1Sumgradients_1/AddN.gradients_1/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_1/add_4_grad/Reshape_1Reshapegradients_1/add_4_grad/Sum_1gradients_1/add_4_grad/Shape_1*
T0*
Tshape0
F
gradients_1/Sum_grad/ShapeShapeGatherV2*
T0*
out_type0
r
gradients_1/Sum_grad/SizeConst*
dtype0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :
�
gradients_1/Sum_grad/addAddV2Sum/reduction_indicesgradients_1/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
�
gradients_1/Sum_grad/modFloorModgradients_1/Sum_grad/addgradients_1/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
t
gradients_1/Sum_grad/Shape_1Const*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
valueB *
dtype0
y
 gradients_1/Sum_grad/range/startConst*
dtype0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B : 
y
 gradients_1/Sum_grad/range/deltaConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_grad/rangeRange gradients_1/Sum_grad/range/startgradients_1/Sum_grad/Size gradients_1/Sum_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
x
gradients_1/Sum_grad/Fill/valueConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_grad/FillFillgradients_1/Sum_grad/Shape_1gradients_1/Sum_grad/Fill/value*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*

index_type0
�
"gradients_1/Sum_grad/DynamicStitchDynamicStitchgradients_1/Sum_grad/rangegradients_1/Sum_grad/modgradients_1/Sum_grad/Shapegradients_1/Sum_grad/Fill*
N*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
w
gradients_1/Sum_grad/Maximum/yConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_grad/MaximumMaximum"gradients_1/Sum_grad/DynamicStitchgradients_1/Sum_grad/Maximum/y*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
�
gradients_1/Sum_grad/floordivFloorDivgradients_1/Sum_grad/Shapegradients_1/Sum_grad/Maximum*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
�
gradients_1/Sum_grad/ReshapeReshape gradients_1/add_4_grad/Reshape_1"gradients_1/Sum_grad/DynamicStitch*
T0*
Tshape0
y
gradients_1/Sum_grad/TileTilegradients_1/Sum_grad/Reshapegradients_1/Sum_grad/floordiv*

Tmultiples0*
T0
r
gradients_1/GatherV2_grad/ShapeConst*
_class

loc:@w*%
valueB	"              *
dtype0	
�
gradients_1/GatherV2_grad/CastCastgradients_1/GatherV2_grad/Shape*

SrcT0	*
_class

loc:@w*
Truncate( *

DstT0
E
gradients_1/GatherV2_grad/SizeSizeCast*
T0*
out_type0
R
(gradients_1/GatherV2_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
$gradients_1/GatherV2_grad/ExpandDims
ExpandDimsgradients_1/GatherV2_grad/Size(gradients_1/GatherV2_grad/ExpandDims/dim*

Tdim0*
T0
[
-gradients_1/GatherV2_grad/strided_slice/stackConst*
dtype0*
valueB:
]
/gradients_1/GatherV2_grad/strided_slice/stack_1Const*
valueB: *
dtype0
]
/gradients_1/GatherV2_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
'gradients_1/GatherV2_grad/strided_sliceStridedSlicegradients_1/GatherV2_grad/Cast-gradients_1/GatherV2_grad/strided_slice/stack/gradients_1/GatherV2_grad/strided_slice/stack_1/gradients_1/GatherV2_grad/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask
O
%gradients_1/GatherV2_grad/concat/axisConst*
value	B : *
dtype0
�
 gradients_1/GatherV2_grad/concatConcatV2$gradients_1/GatherV2_grad/ExpandDims'gradients_1/GatherV2_grad/strided_slice%gradients_1/GatherV2_grad/concat/axis*

Tidx0*
T0*
N
�
!gradients_1/GatherV2_grad/ReshapeReshapegradients_1/Sum_grad/Tile gradients_1/GatherV2_grad/concat*
T0*
Tshape0
q
#gradients_1/GatherV2_grad/Reshape_1ReshapeCast$gradients_1/GatherV2_grad/ExpandDims*
T0*
Tshape0
E
d-w/strided_slice/stackConst*
valueB: *
dtype0
G
d-w/strided_slice/stack_1Const*
dtype0*
valueB:
G
d-w/strided_slice/stack_2Const*
dtype0*
valueB:
�
d-w/strided_sliceStridedSlicegradients_1/GatherV2_grad/Castd-w/strided_slice/stackd-w/strided_slice/stack_1d-w/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
�
	d-w/inputUnsortedSegmentSum!gradients_1/GatherV2_grad/Reshape#gradients_1/GatherV2_grad/Reshape_1d-w/strided_slice*
Tnumsegments0*
Tindices0*
T0
#
d-wIdentity	d-w/input*
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
X
#gradients_2/loss_grad/Reshape/shapeConst*
valueB"      *
dtype0
v
gradients_2/loss_grad/ReshapeReshapegradients_2/Fill#gradients_2/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_2/loss_grad/ConstConst*
dtype0*
valueB"      
y
gradients_2/loss_grad/TileTilegradients_2/loss_grad/Reshapegradients_2/loss_grad/Const*
T0*

Tmultiples0
J
gradients_2/loss_grad/Const_1Const*
dtype0*
valueB
 *  �D
l
gradients_2/loss_grad/truedivRealDivgradients_2/loss_grad/Tilegradients_2/loss_grad/Const_1*
T0
U
&gradients_2/logistic_loss/sub_grad/NegNeggradients_2/loss_grad/truediv*
T0
w
*gradients_2/logistic_loss/Log1p_grad/add/xConst^gradients_2/loss_grad/truediv*
dtype0*
valueB
 *  �?
y
(gradients_2/logistic_loss/Log1p_grad/addAddV2*gradients_2/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_2/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_2/logistic_loss/Log1p_grad/add*
T0
�
(gradients_2/logistic_loss/Log1p_grad/mulMulgradients_2/loss_grad/truediv/gradients_2/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_2/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
c
6gradients_2/logistic_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
0gradients_2/logistic_loss/Select_grad/zeros_likeFill@gradients_2/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_2/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_2/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_2/loss_grad/truediv0gradients_2/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_2/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_2/logistic_loss/Select_grad/zeros_likegradients_2/loss_grad/truediv*
T0
Q
(gradients_2/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_2/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_2/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_2/logistic_loss/mul_grad/Shape*gradients_2/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_2/logistic_loss/mul_grad/MulMul&gradients_2/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_2/logistic_loss/mul_grad/SumSum&gradients_2/logistic_loss/mul_grad/Mul8gradients_2/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*gradients_2/logistic_loss/mul_grad/ReshapeReshape&gradients_2/logistic_loss/mul_grad/Sum(gradients_2/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_2/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_2/logistic_loss/sub_grad/Neg*
T0
�
(gradients_2/logistic_loss/mul_grad/Sum_1Sum(gradients_2/logistic_loss/mul_grad/Mul_1:gradients_2/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients_2/logistic_loss/mul_grad/Reshape_1Reshape(gradients_2/logistic_loss/mul_grad/Sum_1*gradients_2/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_2/logistic_loss/Exp_grad/mulMul(gradients_2/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_2/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_2/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_2/logistic_loss/Select_1_grad/zeros_likeFillBgradients_2/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_2/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_2/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_2/logistic_loss/Exp_grad/mul2gradients_2/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_2/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_2/logistic_loss/Select_1_grad/zeros_like&gradients_2/logistic_loss/Exp_grad/mul*
T0
f
&gradients_2/logistic_loss/Neg_grad/NegNeg.gradients_2/logistic_loss/Select_1_grad/Select*
T0
�
gradients_2/AddNAddN,gradients_2/logistic_loss/Select_grad/Select*gradients_2/logistic_loss/mul_grad/Reshape0gradients_2/logistic_loss/Select_1_grad/Select_1&gradients_2/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_2/logistic_loss/Select_grad/Select*
N
F
gradients_2/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_2/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_2/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_4_grad/Shapegradients_2/add_4_grad/Shape_1*
T0
�
gradients_2/add_4_grad/SumSumgradients_2/AddN,gradients_2/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_2/add_4_grad/ReshapeReshapegradients_2/add_4_grad/Sumgradients_2/add_4_grad/Shape*
T0*
Tshape0
�
gradients_2/add_4_grad/Sum_1Sumgradients_2/AddN.gradients_2/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_2/add_4_grad/Reshape_1Reshapegradients_2/add_4_grad/Sum_1gradients_2/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_2/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_2/add_4_grad/Reshape*
T0
a
,gradients_2/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients_2/Add_3_grad/SumSum gradients_2/Tanh_3_grad/TanhGrad,gradients_2/Add_3_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_2/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_2/Add_3_grad/ReshapeReshapegradients_2/Add_3_grad/Sum$gradients_2/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_2/MatMul_3_grad/MatMulMatMul gradients_2/Tanh_3_grad/TanhGradw4/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_2/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_2/Tanh_3_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
_
 gradients_2/Tanh_2_grad/TanhGradTanhGradTanh_2 gradients_2/MatMul_3_grad/MatMul*
T0
Z
,gradients_2/Add_2_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_2/Add_2_grad/SumSum gradients_2/Tanh_2_grad/TanhGrad,gradients_2/Add_2_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_2/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
�
gradients_2/Add_2_grad/ReshapeReshapegradients_2/Add_2_grad/Sum$gradients_2/Add_2_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_2/MatMul_2_grad/MatMulMatMul gradients_2/Tanh_2_grad/TanhGradw3/read*
transpose_a( *
transpose_b(*
T0
�
"gradients_2/MatMul_2_grad/MatMul_1MatMulTanh_1 gradients_2/Tanh_2_grad/TanhGrad*
transpose_a(*
transpose_b( *
T0
_
 gradients_2/Tanh_1_grad/TanhGradTanhGradTanh_1 gradients_2/MatMul_2_grad/MatMul*
T0
Z
,gradients_2/Add_1_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_2/Add_1_grad/SumSum gradients_2/Tanh_1_grad/TanhGrad,gradients_2/Add_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_2/Add_1_grad/Reshape/shapeConst*
valueB"   �   *
dtype0
�
gradients_2/Add_1_grad/ReshapeReshapegradients_2/Add_1_grad/Sum$gradients_2/Add_1_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_2/MatMul_1_grad/MatMulMatMul gradients_2/Tanh_1_grad/TanhGradw2/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_2/MatMul_1_grad/MatMul_1MatMulTanh gradients_2/Tanh_1_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
[
gradients_2/Tanh_grad/TanhGradTanhGradTanh gradients_2/MatMul_1_grad/MatMul*
T0
X
*gradients_2/Add_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_2/Add_grad/SumSumgradients_2/Tanh_grad/TanhGrad*gradients_2/Add_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
W
"gradients_2/Add_grad/Reshape/shapeConst*
valueB"      *
dtype0
|
gradients_2/Add_grad/ReshapeReshapegradients_2/Add_grad/Sum"gradients_2/Add_grad/Reshape/shape*
T0*
Tshape0
�
gradients_2/MatMul_grad/MatMulMatMulgradients_2/Tanh_grad/TanhGradw1/read*
transpose_b(*
T0*
transpose_a( 
�
 gradients_2/MatMul_grad/MatMul_1MatMulReshapegradients_2/Tanh_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
L
gradients_2/Reshape_grad/ShapeShape
GatherV2_1*
T0*
out_type0
�
 gradients_2/Reshape_grad/ReshapeReshapegradients_2/MatMul_grad/MatMulgradients_2/Reshape_grad/Shape*
T0*
Tshape0
t
!gradients_2/GatherV2_1_grad/ShapeConst*
_class

loc:@v*%
valueB	"              *
dtype0	
�
 gradients_2/GatherV2_1_grad/CastCast!gradients_2/GatherV2_1_grad/Shape*

SrcT0	*
_class

loc:@v*
Truncate( *

DstT0
G
 gradients_2/GatherV2_1_grad/SizeSizeCast*
T0*
out_type0
T
*gradients_2/GatherV2_1_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
&gradients_2/GatherV2_1_grad/ExpandDims
ExpandDims gradients_2/GatherV2_1_grad/Size*gradients_2/GatherV2_1_grad/ExpandDims/dim*

Tdim0*
T0
]
/gradients_2/GatherV2_1_grad/strided_slice/stackConst*
valueB:*
dtype0
_
1gradients_2/GatherV2_1_grad/strided_slice/stack_1Const*
valueB: *
dtype0
_
1gradients_2/GatherV2_1_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
)gradients_2/GatherV2_1_grad/strided_sliceStridedSlice gradients_2/GatherV2_1_grad/Cast/gradients_2/GatherV2_1_grad/strided_slice/stack1gradients_2/GatherV2_1_grad/strided_slice/stack_11gradients_2/GatherV2_1_grad/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0
Q
'gradients_2/GatherV2_1_grad/concat/axisConst*
value	B : *
dtype0
�
"gradients_2/GatherV2_1_grad/concatConcatV2&gradients_2/GatherV2_1_grad/ExpandDims)gradients_2/GatherV2_1_grad/strided_slice'gradients_2/GatherV2_1_grad/concat/axis*
T0*
N*

Tidx0
�
#gradients_2/GatherV2_1_grad/ReshapeReshape gradients_2/Reshape_grad/Reshape"gradients_2/GatherV2_1_grad/concat*
T0*
Tshape0
u
%gradients_2/GatherV2_1_grad/Reshape_1ReshapeCast&gradients_2/GatherV2_1_grad/ExpandDims*
T0*
Tshape0
E
d-v/strided_slice/stackConst*
dtype0*
valueB: 
G
d-v/strided_slice/stack_1Const*
valueB:*
dtype0
G
d-v/strided_slice/stack_2Const*
valueB:*
dtype0
�
d-v/strided_sliceStridedSlice gradients_2/GatherV2_1_grad/Castd-v/strided_slice/stackd-v/strided_slice/stack_1d-v/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
�
	d-v/inputUnsortedSegmentSum#gradients_2/GatherV2_1_grad/Reshape%gradients_2/GatherV2_1_grad/Reshape_1d-v/strided_slice*
T0*
Tnumsegments0*
Tindices0
#
d-vIdentity	d-v/input*
T0
:
gradients_3/ShapeConst*
valueB *
dtype0
B
gradients_3/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*
T0*

index_type0
X
#gradients_3/loss_grad/Reshape/shapeConst*
dtype0*
valueB"      
v
gradients_3/loss_grad/ReshapeReshapegradients_3/Fill#gradients_3/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_3/loss_grad/ConstConst*
dtype0*
valueB"      
y
gradients_3/loss_grad/TileTilegradients_3/loss_grad/Reshapegradients_3/loss_grad/Const*

Tmultiples0*
T0
J
gradients_3/loss_grad/Const_1Const*
dtype0*
valueB
 *  �D
l
gradients_3/loss_grad/truedivRealDivgradients_3/loss_grad/Tilegradients_3/loss_grad/Const_1*
T0
U
&gradients_3/logistic_loss/sub_grad/NegNeggradients_3/loss_grad/truediv*
T0
w
*gradients_3/logistic_loss/Log1p_grad/add/xConst^gradients_3/loss_grad/truediv*
valueB
 *  �?*
dtype0
y
(gradients_3/logistic_loss/Log1p_grad/addAddV2*gradients_3/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_3/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_3/logistic_loss/Log1p_grad/add*
T0
�
(gradients_3/logistic_loss/Log1p_grad/mulMulgradients_3/loss_grad/truediv/gradients_3/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_3/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
c
6gradients_3/logistic_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
0gradients_3/logistic_loss/Select_grad/zeros_likeFill@gradients_3/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_3/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_3/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_3/loss_grad/truediv0gradients_3/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_3/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_3/logistic_loss/Select_grad/zeros_likegradients_3/loss_grad/truediv*
T0
Q
(gradients_3/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_3/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_3/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_3/logistic_loss/mul_grad/Shape*gradients_3/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_3/logistic_loss/mul_grad/MulMul&gradients_3/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_3/logistic_loss/mul_grad/SumSum&gradients_3/logistic_loss/mul_grad/Mul8gradients_3/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*gradients_3/logistic_loss/mul_grad/ReshapeReshape&gradients_3/logistic_loss/mul_grad/Sum(gradients_3/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_3/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_3/logistic_loss/sub_grad/Neg*
T0
�
(gradients_3/logistic_loss/mul_grad/Sum_1Sum(gradients_3/logistic_loss/mul_grad/Mul_1:gradients_3/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients_3/logistic_loss/mul_grad/Reshape_1Reshape(gradients_3/logistic_loss/mul_grad/Sum_1*gradients_3/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_3/logistic_loss/Exp_grad/mulMul(gradients_3/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_3/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_3/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_3/logistic_loss/Select_1_grad/zeros_likeFillBgradients_3/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_3/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_3/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_3/logistic_loss/Exp_grad/mul2gradients_3/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_3/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_3/logistic_loss/Select_1_grad/zeros_like&gradients_3/logistic_loss/Exp_grad/mul*
T0
f
&gradients_3/logistic_loss/Neg_grad/NegNeg.gradients_3/logistic_loss/Select_1_grad/Select*
T0
�
gradients_3/AddNAddN,gradients_3/logistic_loss/Select_grad/Select*gradients_3/logistic_loss/mul_grad/Reshape0gradients_3/logistic_loss/Select_1_grad/Select_1&gradients_3/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_3/logistic_loss/Select_grad/Select*
N
F
gradients_3/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_3/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_3/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_4_grad/Shapegradients_3/add_4_grad/Shape_1*
T0
�
gradients_3/add_4_grad/SumSumgradients_3/AddN,gradients_3/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_3/add_4_grad/ReshapeReshapegradients_3/add_4_grad/Sumgradients_3/add_4_grad/Shape*
T0*
Tshape0
�
gradients_3/add_4_grad/Sum_1Sumgradients_3/AddN.gradients_3/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_3/add_4_grad/Reshape_1Reshapegradients_3/add_4_grad/Sum_1gradients_3/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_3/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_3/add_4_grad/Reshape*
T0
a
,gradients_3/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients_3/Add_3_grad/SumSum gradients_3/Tanh_3_grad/TanhGrad,gradients_3/Add_3_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_3/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_3/Add_3_grad/ReshapeReshapegradients_3/Add_3_grad/Sum$gradients_3/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_3/MatMul_3_grad/MatMulMatMul gradients_3/Tanh_3_grad/TanhGradw4/read*
transpose_a( *
transpose_b(*
T0
�
"gradients_3/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_3/Tanh_3_grad/TanhGrad*
transpose_b( *
T0*
transpose_a(
_
 gradients_3/Tanh_2_grad/TanhGradTanhGradTanh_2 gradients_3/MatMul_3_grad/MatMul*
T0
Z
,gradients_3/Add_2_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_3/Add_2_grad/SumSum gradients_3/Tanh_2_grad/TanhGrad,gradients_3/Add_2_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_3/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
�
gradients_3/Add_2_grad/ReshapeReshapegradients_3/Add_2_grad/Sum$gradients_3/Add_2_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_3/MatMul_2_grad/MatMulMatMul gradients_3/Tanh_2_grad/TanhGradw3/read*
transpose_a( *
transpose_b(*
T0
�
"gradients_3/MatMul_2_grad/MatMul_1MatMulTanh_1 gradients_3/Tanh_2_grad/TanhGrad*
transpose_a(*
transpose_b( *
T0
_
 gradients_3/Tanh_1_grad/TanhGradTanhGradTanh_1 gradients_3/MatMul_2_grad/MatMul*
T0
Z
,gradients_3/Add_1_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_3/Add_1_grad/SumSum gradients_3/Tanh_1_grad/TanhGrad,gradients_3/Add_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_3/Add_1_grad/Reshape/shapeConst*
valueB"   �   *
dtype0
�
gradients_3/Add_1_grad/ReshapeReshapegradients_3/Add_1_grad/Sum$gradients_3/Add_1_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_3/MatMul_1_grad/MatMulMatMul gradients_3/Tanh_1_grad/TanhGradw2/read*
transpose_b(*
T0*
transpose_a( 
�
"gradients_3/MatMul_1_grad/MatMul_1MatMulTanh gradients_3/Tanh_1_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
[
gradients_3/Tanh_grad/TanhGradTanhGradTanh gradients_3/MatMul_1_grad/MatMul*
T0
X
*gradients_3/Add_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_3/Add_grad/SumSumgradients_3/Tanh_grad/TanhGrad*gradients_3/Add_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
W
"gradients_3/Add_grad/Reshape/shapeConst*
valueB"      *
dtype0
|
gradients_3/Add_grad/ReshapeReshapegradients_3/Add_grad/Sum"gradients_3/Add_grad/Reshape/shape*
T0*
Tshape0
�
gradients_3/MatMul_grad/MatMulMatMulgradients_3/Tanh_grad/TanhGradw1/read*
T0*
transpose_a( *
transpose_b(
�
 gradients_3/MatMul_grad/MatMul_1MatMulReshapegradients_3/Tanh_grad/TanhGrad*
transpose_a(*
transpose_b( *
T0
;
d-w1Identity gradients_3/MatMul_grad/MatMul_1*
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
X
#gradients_4/loss_grad/Reshape/shapeConst*
dtype0*
valueB"      
v
gradients_4/loss_grad/ReshapeReshapegradients_4/Fill#gradients_4/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_4/loss_grad/ConstConst*
valueB"      *
dtype0
y
gradients_4/loss_grad/TileTilegradients_4/loss_grad/Reshapegradients_4/loss_grad/Const*
T0*

Tmultiples0
J
gradients_4/loss_grad/Const_1Const*
dtype0*
valueB
 *  �D
l
gradients_4/loss_grad/truedivRealDivgradients_4/loss_grad/Tilegradients_4/loss_grad/Const_1*
T0
U
&gradients_4/logistic_loss/sub_grad/NegNeggradients_4/loss_grad/truediv*
T0
w
*gradients_4/logistic_loss/Log1p_grad/add/xConst^gradients_4/loss_grad/truediv*
valueB
 *  �?*
dtype0
y
(gradients_4/logistic_loss/Log1p_grad/addAddV2*gradients_4/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_4/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_4/logistic_loss/Log1p_grad/add*
T0
�
(gradients_4/logistic_loss/Log1p_grad/mulMulgradients_4/loss_grad/truediv/gradients_4/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_4/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
c
6gradients_4/logistic_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
0gradients_4/logistic_loss/Select_grad/zeros_likeFill@gradients_4/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_4/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_4/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_4/loss_grad/truediv0gradients_4/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_4/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_4/logistic_loss/Select_grad/zeros_likegradients_4/loss_grad/truediv*
T0
Q
(gradients_4/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_4/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_4/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_4/logistic_loss/mul_grad/Shape*gradients_4/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_4/logistic_loss/mul_grad/MulMul&gradients_4/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_4/logistic_loss/mul_grad/SumSum&gradients_4/logistic_loss/mul_grad/Mul8gradients_4/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_4/logistic_loss/mul_grad/ReshapeReshape&gradients_4/logistic_loss/mul_grad/Sum(gradients_4/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_4/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_4/logistic_loss/sub_grad/Neg*
T0
�
(gradients_4/logistic_loss/mul_grad/Sum_1Sum(gradients_4/logistic_loss/mul_grad/Mul_1:gradients_4/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
,gradients_4/logistic_loss/mul_grad/Reshape_1Reshape(gradients_4/logistic_loss/mul_grad/Sum_1*gradients_4/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_4/logistic_loss/Exp_grad/mulMul(gradients_4/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_4/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_4/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_4/logistic_loss/Select_1_grad/zeros_likeFillBgradients_4/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_4/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_4/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_4/logistic_loss/Exp_grad/mul2gradients_4/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_4/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_4/logistic_loss/Select_1_grad/zeros_like&gradients_4/logistic_loss/Exp_grad/mul*
T0
f
&gradients_4/logistic_loss/Neg_grad/NegNeg.gradients_4/logistic_loss/Select_1_grad/Select*
T0
�
gradients_4/AddNAddN,gradients_4/logistic_loss/Select_grad/Select*gradients_4/logistic_loss/mul_grad/Reshape0gradients_4/logistic_loss/Select_1_grad/Select_1&gradients_4/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_4/logistic_loss/Select_grad/Select*
N
F
gradients_4/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_4/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_4/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_4_grad/Shapegradients_4/add_4_grad/Shape_1*
T0
�
gradients_4/add_4_grad/SumSumgradients_4/AddN,gradients_4/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_4/add_4_grad/ReshapeReshapegradients_4/add_4_grad/Sumgradients_4/add_4_grad/Shape*
T0*
Tshape0
�
gradients_4/add_4_grad/Sum_1Sumgradients_4/AddN.gradients_4/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_4/add_4_grad/Reshape_1Reshapegradients_4/add_4_grad/Sum_1gradients_4/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_4/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_4/add_4_grad/Reshape*
T0
a
,gradients_4/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients_4/Add_3_grad/SumSum gradients_4/Tanh_3_grad/TanhGrad,gradients_4/Add_3_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_4/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_4/Add_3_grad/ReshapeReshapegradients_4/Add_3_grad/Sum$gradients_4/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_4/MatMul_3_grad/MatMulMatMul gradients_4/Tanh_3_grad/TanhGradw4/read*
transpose_a( *
transpose_b(*
T0
�
"gradients_4/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_4/Tanh_3_grad/TanhGrad*
transpose_b( *
T0*
transpose_a(
_
 gradients_4/Tanh_2_grad/TanhGradTanhGradTanh_2 gradients_4/MatMul_3_grad/MatMul*
T0
Z
,gradients_4/Add_2_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_4/Add_2_grad/SumSum gradients_4/Tanh_2_grad/TanhGrad,gradients_4/Add_2_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_4/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
�
gradients_4/Add_2_grad/ReshapeReshapegradients_4/Add_2_grad/Sum$gradients_4/Add_2_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_4/MatMul_2_grad/MatMulMatMul gradients_4/Tanh_2_grad/TanhGradw3/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_4/MatMul_2_grad/MatMul_1MatMulTanh_1 gradients_4/Tanh_2_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
_
 gradients_4/Tanh_1_grad/TanhGradTanhGradTanh_1 gradients_4/MatMul_2_grad/MatMul*
T0
Z
,gradients_4/Add_1_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_4/Add_1_grad/SumSum gradients_4/Tanh_1_grad/TanhGrad,gradients_4/Add_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_4/Add_1_grad/Reshape/shapeConst*
valueB"   �   *
dtype0
�
gradients_4/Add_1_grad/ReshapeReshapegradients_4/Add_1_grad/Sum$gradients_4/Add_1_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_4/MatMul_1_grad/MatMulMatMul gradients_4/Tanh_1_grad/TanhGradw2/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_4/MatMul_1_grad/MatMul_1MatMulTanh gradients_4/Tanh_1_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
[
gradients_4/Tanh_grad/TanhGradTanhGradTanh gradients_4/MatMul_1_grad/MatMul*
T0
X
*gradients_4/Add_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_4/Add_grad/SumSumgradients_4/Tanh_grad/TanhGrad*gradients_4/Add_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
W
"gradients_4/Add_grad/Reshape/shapeConst*
valueB"      *
dtype0
|
gradients_4/Add_grad/ReshapeReshapegradients_4/Add_grad/Sum"gradients_4/Add_grad/Reshape/shape*
T0*
Tshape0
7
d-b1Identitygradients_4/Add_grad/Reshape*
T0
:
gradients_5/ShapeConst*
valueB *
dtype0
B
gradients_5/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_5/FillFillgradients_5/Shapegradients_5/grad_ys_0*
T0*

index_type0
X
#gradients_5/loss_grad/Reshape/shapeConst*
dtype0*
valueB"      
v
gradients_5/loss_grad/ReshapeReshapegradients_5/Fill#gradients_5/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_5/loss_grad/ConstConst*
valueB"      *
dtype0
y
gradients_5/loss_grad/TileTilegradients_5/loss_grad/Reshapegradients_5/loss_grad/Const*

Tmultiples0*
T0
J
gradients_5/loss_grad/Const_1Const*
dtype0*
valueB
 *  �D
l
gradients_5/loss_grad/truedivRealDivgradients_5/loss_grad/Tilegradients_5/loss_grad/Const_1*
T0
U
&gradients_5/logistic_loss/sub_grad/NegNeggradients_5/loss_grad/truediv*
T0
w
*gradients_5/logistic_loss/Log1p_grad/add/xConst^gradients_5/loss_grad/truediv*
valueB
 *  �?*
dtype0
y
(gradients_5/logistic_loss/Log1p_grad/addAddV2*gradients_5/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_5/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_5/logistic_loss/Log1p_grad/add*
T0
�
(gradients_5/logistic_loss/Log1p_grad/mulMulgradients_5/loss_grad/truediv/gradients_5/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_5/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
c
6gradients_5/logistic_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
0gradients_5/logistic_loss/Select_grad/zeros_likeFill@gradients_5/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_5/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_5/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_5/loss_grad/truediv0gradients_5/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_5/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_5/logistic_loss/Select_grad/zeros_likegradients_5/loss_grad/truediv*
T0
Q
(gradients_5/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_5/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_5/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_5/logistic_loss/mul_grad/Shape*gradients_5/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_5/logistic_loss/mul_grad/MulMul&gradients_5/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_5/logistic_loss/mul_grad/SumSum&gradients_5/logistic_loss/mul_grad/Mul8gradients_5/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_5/logistic_loss/mul_grad/ReshapeReshape&gradients_5/logistic_loss/mul_grad/Sum(gradients_5/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_5/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_5/logistic_loss/sub_grad/Neg*
T0
�
(gradients_5/logistic_loss/mul_grad/Sum_1Sum(gradients_5/logistic_loss/mul_grad/Mul_1:gradients_5/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients_5/logistic_loss/mul_grad/Reshape_1Reshape(gradients_5/logistic_loss/mul_grad/Sum_1*gradients_5/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_5/logistic_loss/Exp_grad/mulMul(gradients_5/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_5/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_5/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_5/logistic_loss/Select_1_grad/zeros_likeFillBgradients_5/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_5/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_5/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_5/logistic_loss/Exp_grad/mul2gradients_5/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_5/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_5/logistic_loss/Select_1_grad/zeros_like&gradients_5/logistic_loss/Exp_grad/mul*
T0
f
&gradients_5/logistic_loss/Neg_grad/NegNeg.gradients_5/logistic_loss/Select_1_grad/Select*
T0
�
gradients_5/AddNAddN,gradients_5/logistic_loss/Select_grad/Select*gradients_5/logistic_loss/mul_grad/Reshape0gradients_5/logistic_loss/Select_1_grad/Select_1&gradients_5/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_5/logistic_loss/Select_grad/Select*
N
F
gradients_5/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_5/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_5/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_4_grad/Shapegradients_5/add_4_grad/Shape_1*
T0
�
gradients_5/add_4_grad/SumSumgradients_5/AddN,gradients_5/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_5/add_4_grad/ReshapeReshapegradients_5/add_4_grad/Sumgradients_5/add_4_grad/Shape*
T0*
Tshape0
�
gradients_5/add_4_grad/Sum_1Sumgradients_5/AddN.gradients_5/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_5/add_4_grad/Reshape_1Reshapegradients_5/add_4_grad/Sum_1gradients_5/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_5/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_5/add_4_grad/Reshape*
T0
a
,gradients_5/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients_5/Add_3_grad/SumSum gradients_5/Tanh_3_grad/TanhGrad,gradients_5/Add_3_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_5/Add_3_grad/Reshape/shapeConst*
dtype0*
valueB"      
�
gradients_5/Add_3_grad/ReshapeReshapegradients_5/Add_3_grad/Sum$gradients_5/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_5/MatMul_3_grad/MatMulMatMul gradients_5/Tanh_3_grad/TanhGradw4/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_5/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_5/Tanh_3_grad/TanhGrad*
transpose_b( *
T0*
transpose_a(
_
 gradients_5/Tanh_2_grad/TanhGradTanhGradTanh_2 gradients_5/MatMul_3_grad/MatMul*
T0
Z
,gradients_5/Add_2_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_5/Add_2_grad/SumSum gradients_5/Tanh_2_grad/TanhGrad,gradients_5/Add_2_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_5/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
�
gradients_5/Add_2_grad/ReshapeReshapegradients_5/Add_2_grad/Sum$gradients_5/Add_2_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_5/MatMul_2_grad/MatMulMatMul gradients_5/Tanh_2_grad/TanhGradw3/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_5/MatMul_2_grad/MatMul_1MatMulTanh_1 gradients_5/Tanh_2_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
_
 gradients_5/Tanh_1_grad/TanhGradTanhGradTanh_1 gradients_5/MatMul_2_grad/MatMul*
T0
Z
,gradients_5/Add_1_grad/Sum/reduction_indicesConst*
dtype0*
valueB: 
�
gradients_5/Add_1_grad/SumSum gradients_5/Tanh_1_grad/TanhGrad,gradients_5/Add_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_5/Add_1_grad/Reshape/shapeConst*
valueB"   �   *
dtype0
�
gradients_5/Add_1_grad/ReshapeReshapegradients_5/Add_1_grad/Sum$gradients_5/Add_1_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_5/MatMul_1_grad/MatMulMatMul gradients_5/Tanh_1_grad/TanhGradw2/read*
transpose_b(*
T0*
transpose_a( 
�
"gradients_5/MatMul_1_grad/MatMul_1MatMulTanh gradients_5/Tanh_1_grad/TanhGrad*
transpose_b( *
T0*
transpose_a(
=
d-w2Identity"gradients_5/MatMul_1_grad/MatMul_1*
T0
:
gradients_6/ShapeConst*
dtype0*
valueB 
B
gradients_6/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_6/FillFillgradients_6/Shapegradients_6/grad_ys_0*
T0*

index_type0
X
#gradients_6/loss_grad/Reshape/shapeConst*
valueB"      *
dtype0
v
gradients_6/loss_grad/ReshapeReshapegradients_6/Fill#gradients_6/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_6/loss_grad/ConstConst*
valueB"      *
dtype0
y
gradients_6/loss_grad/TileTilegradients_6/loss_grad/Reshapegradients_6/loss_grad/Const*

Tmultiples0*
T0
J
gradients_6/loss_grad/Const_1Const*
valueB
 *  �D*
dtype0
l
gradients_6/loss_grad/truedivRealDivgradients_6/loss_grad/Tilegradients_6/loss_grad/Const_1*
T0
U
&gradients_6/logistic_loss/sub_grad/NegNeggradients_6/loss_grad/truediv*
T0
w
*gradients_6/logistic_loss/Log1p_grad/add/xConst^gradients_6/loss_grad/truediv*
valueB
 *  �?*
dtype0
y
(gradients_6/logistic_loss/Log1p_grad/addAddV2*gradients_6/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_6/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_6/logistic_loss/Log1p_grad/add*
T0
�
(gradients_6/logistic_loss/Log1p_grad/mulMulgradients_6/loss_grad/truediv/gradients_6/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_6/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
c
6gradients_6/logistic_loss/Select_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    
�
0gradients_6/logistic_loss/Select_grad/zeros_likeFill@gradients_6/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_6/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_6/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_6/loss_grad/truediv0gradients_6/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_6/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_6/logistic_loss/Select_grad/zeros_likegradients_6/loss_grad/truediv*
T0
Q
(gradients_6/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_6/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_6/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_6/logistic_loss/mul_grad/Shape*gradients_6/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_6/logistic_loss/mul_grad/MulMul&gradients_6/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_6/logistic_loss/mul_grad/SumSum&gradients_6/logistic_loss/mul_grad/Mul8gradients_6/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_6/logistic_loss/mul_grad/ReshapeReshape&gradients_6/logistic_loss/mul_grad/Sum(gradients_6/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_6/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_6/logistic_loss/sub_grad/Neg*
T0
�
(gradients_6/logistic_loss/mul_grad/Sum_1Sum(gradients_6/logistic_loss/mul_grad/Mul_1:gradients_6/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients_6/logistic_loss/mul_grad/Reshape_1Reshape(gradients_6/logistic_loss/mul_grad/Sum_1*gradients_6/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_6/logistic_loss/Exp_grad/mulMul(gradients_6/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_6/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_6/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_6/logistic_loss/Select_1_grad/zeros_likeFillBgradients_6/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_6/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_6/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_6/logistic_loss/Exp_grad/mul2gradients_6/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_6/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_6/logistic_loss/Select_1_grad/zeros_like&gradients_6/logistic_loss/Exp_grad/mul*
T0
f
&gradients_6/logistic_loss/Neg_grad/NegNeg.gradients_6/logistic_loss/Select_1_grad/Select*
T0
�
gradients_6/AddNAddN,gradients_6/logistic_loss/Select_grad/Select*gradients_6/logistic_loss/mul_grad/Reshape0gradients_6/logistic_loss/Select_1_grad/Select_1&gradients_6/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_6/logistic_loss/Select_grad/Select*
N
F
gradients_6/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_6/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_6/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/add_4_grad/Shapegradients_6/add_4_grad/Shape_1*
T0
�
gradients_6/add_4_grad/SumSumgradients_6/AddN,gradients_6/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
z
gradients_6/add_4_grad/ReshapeReshapegradients_6/add_4_grad/Sumgradients_6/add_4_grad/Shape*
T0*
Tshape0
�
gradients_6/add_4_grad/Sum_1Sumgradients_6/AddN.gradients_6/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_6/add_4_grad/Reshape_1Reshapegradients_6/add_4_grad/Sum_1gradients_6/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_6/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_6/add_4_grad/Reshape*
T0
a
,gradients_6/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients_6/Add_3_grad/SumSum gradients_6/Tanh_3_grad/TanhGrad,gradients_6/Add_3_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_6/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_6/Add_3_grad/ReshapeReshapegradients_6/Add_3_grad/Sum$gradients_6/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_6/MatMul_3_grad/MatMulMatMul gradients_6/Tanh_3_grad/TanhGradw4/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_6/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_6/Tanh_3_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
_
 gradients_6/Tanh_2_grad/TanhGradTanhGradTanh_2 gradients_6/MatMul_3_grad/MatMul*
T0
Z
,gradients_6/Add_2_grad/Sum/reduction_indicesConst*
dtype0*
valueB: 
�
gradients_6/Add_2_grad/SumSum gradients_6/Tanh_2_grad/TanhGrad,gradients_6/Add_2_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_6/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
�
gradients_6/Add_2_grad/ReshapeReshapegradients_6/Add_2_grad/Sum$gradients_6/Add_2_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_6/MatMul_2_grad/MatMulMatMul gradients_6/Tanh_2_grad/TanhGradw3/read*
transpose_a( *
transpose_b(*
T0
�
"gradients_6/MatMul_2_grad/MatMul_1MatMulTanh_1 gradients_6/Tanh_2_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
_
 gradients_6/Tanh_1_grad/TanhGradTanhGradTanh_1 gradients_6/MatMul_2_grad/MatMul*
T0
Z
,gradients_6/Add_1_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_6/Add_1_grad/SumSum gradients_6/Tanh_1_grad/TanhGrad,gradients_6/Add_1_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_6/Add_1_grad/Reshape/shapeConst*
dtype0*
valueB"   �   
�
gradients_6/Add_1_grad/ReshapeReshapegradients_6/Add_1_grad/Sum$gradients_6/Add_1_grad/Reshape/shape*
T0*
Tshape0
9
d-b2Identitygradients_6/Add_1_grad/Reshape*
T0
:
gradients_7/ShapeConst*
valueB *
dtype0
B
gradients_7/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_7/FillFillgradients_7/Shapegradients_7/grad_ys_0*
T0*

index_type0
X
#gradients_7/loss_grad/Reshape/shapeConst*
dtype0*
valueB"      
v
gradients_7/loss_grad/ReshapeReshapegradients_7/Fill#gradients_7/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_7/loss_grad/ConstConst*
valueB"      *
dtype0
y
gradients_7/loss_grad/TileTilegradients_7/loss_grad/Reshapegradients_7/loss_grad/Const*

Tmultiples0*
T0
J
gradients_7/loss_grad/Const_1Const*
valueB
 *  �D*
dtype0
l
gradients_7/loss_grad/truedivRealDivgradients_7/loss_grad/Tilegradients_7/loss_grad/Const_1*
T0
U
&gradients_7/logistic_loss/sub_grad/NegNeggradients_7/loss_grad/truediv*
T0
w
*gradients_7/logistic_loss/Log1p_grad/add/xConst^gradients_7/loss_grad/truediv*
valueB
 *  �?*
dtype0
y
(gradients_7/logistic_loss/Log1p_grad/addAddV2*gradients_7/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_7/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_7/logistic_loss/Log1p_grad/add*
T0
�
(gradients_7/logistic_loss/Log1p_grad/mulMulgradients_7/loss_grad/truediv/gradients_7/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_7/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
c
6gradients_7/logistic_loss/Select_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    
�
0gradients_7/logistic_loss/Select_grad/zeros_likeFill@gradients_7/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_7/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_7/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_7/loss_grad/truediv0gradients_7/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_7/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_7/logistic_loss/Select_grad/zeros_likegradients_7/loss_grad/truediv*
T0
Q
(gradients_7/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_7/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_7/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_7/logistic_loss/mul_grad/Shape*gradients_7/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_7/logistic_loss/mul_grad/MulMul&gradients_7/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_7/logistic_loss/mul_grad/SumSum&gradients_7/logistic_loss/mul_grad/Mul8gradients_7/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*gradients_7/logistic_loss/mul_grad/ReshapeReshape&gradients_7/logistic_loss/mul_grad/Sum(gradients_7/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_7/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_7/logistic_loss/sub_grad/Neg*
T0
�
(gradients_7/logistic_loss/mul_grad/Sum_1Sum(gradients_7/logistic_loss/mul_grad/Mul_1:gradients_7/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients_7/logistic_loss/mul_grad/Reshape_1Reshape(gradients_7/logistic_loss/mul_grad/Sum_1*gradients_7/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_7/logistic_loss/Exp_grad/mulMul(gradients_7/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_7/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_7/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_7/logistic_loss/Select_1_grad/zeros_likeFillBgradients_7/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_7/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_7/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_7/logistic_loss/Exp_grad/mul2gradients_7/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_7/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_7/logistic_loss/Select_1_grad/zeros_like&gradients_7/logistic_loss/Exp_grad/mul*
T0
f
&gradients_7/logistic_loss/Neg_grad/NegNeg.gradients_7/logistic_loss/Select_1_grad/Select*
T0
�
gradients_7/AddNAddN,gradients_7/logistic_loss/Select_grad/Select*gradients_7/logistic_loss/mul_grad/Reshape0gradients_7/logistic_loss/Select_1_grad/Select_1&gradients_7/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_7/logistic_loss/Select_grad/Select*
N
F
gradients_7/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_7/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_7/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/add_4_grad/Shapegradients_7/add_4_grad/Shape_1*
T0
�
gradients_7/add_4_grad/SumSumgradients_7/AddN,gradients_7/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
z
gradients_7/add_4_grad/ReshapeReshapegradients_7/add_4_grad/Sumgradients_7/add_4_grad/Shape*
T0*
Tshape0
�
gradients_7/add_4_grad/Sum_1Sumgradients_7/AddN.gradients_7/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_7/add_4_grad/Reshape_1Reshapegradients_7/add_4_grad/Sum_1gradients_7/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_7/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_7/add_4_grad/Reshape*
T0
a
,gradients_7/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients_7/Add_3_grad/SumSum gradients_7/Tanh_3_grad/TanhGrad,gradients_7/Add_3_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_7/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_7/Add_3_grad/ReshapeReshapegradients_7/Add_3_grad/Sum$gradients_7/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_7/MatMul_3_grad/MatMulMatMul gradients_7/Tanh_3_grad/TanhGradw4/read*
transpose_a( *
transpose_b(*
T0
�
"gradients_7/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_7/Tanh_3_grad/TanhGrad*
transpose_a(*
transpose_b( *
T0
_
 gradients_7/Tanh_2_grad/TanhGradTanhGradTanh_2 gradients_7/MatMul_3_grad/MatMul*
T0
Z
,gradients_7/Add_2_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_7/Add_2_grad/SumSum gradients_7/Tanh_2_grad/TanhGrad,gradients_7/Add_2_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_7/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
�
gradients_7/Add_2_grad/ReshapeReshapegradients_7/Add_2_grad/Sum$gradients_7/Add_2_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_7/MatMul_2_grad/MatMulMatMul gradients_7/Tanh_2_grad/TanhGradw3/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_7/MatMul_2_grad/MatMul_1MatMulTanh_1 gradients_7/Tanh_2_grad/TanhGrad*
transpose_b( *
T0*
transpose_a(
=
d-w3Identity"gradients_7/MatMul_2_grad/MatMul_1*
T0
:
gradients_8/ShapeConst*
dtype0*
valueB 
B
gradients_8/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_8/FillFillgradients_8/Shapegradients_8/grad_ys_0*
T0*

index_type0
X
#gradients_8/loss_grad/Reshape/shapeConst*
valueB"      *
dtype0
v
gradients_8/loss_grad/ReshapeReshapegradients_8/Fill#gradients_8/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_8/loss_grad/ConstConst*
dtype0*
valueB"      
y
gradients_8/loss_grad/TileTilegradients_8/loss_grad/Reshapegradients_8/loss_grad/Const*
T0*

Tmultiples0
J
gradients_8/loss_grad/Const_1Const*
dtype0*
valueB
 *  �D
l
gradients_8/loss_grad/truedivRealDivgradients_8/loss_grad/Tilegradients_8/loss_grad/Const_1*
T0
U
&gradients_8/logistic_loss/sub_grad/NegNeggradients_8/loss_grad/truediv*
T0
w
*gradients_8/logistic_loss/Log1p_grad/add/xConst^gradients_8/loss_grad/truediv*
valueB
 *  �?*
dtype0
y
(gradients_8/logistic_loss/Log1p_grad/addAddV2*gradients_8/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_8/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_8/logistic_loss/Log1p_grad/add*
T0
�
(gradients_8/logistic_loss/Log1p_grad/mulMulgradients_8/loss_grad/truediv/gradients_8/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_8/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
c
6gradients_8/logistic_loss/Select_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    
�
0gradients_8/logistic_loss/Select_grad/zeros_likeFill@gradients_8/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_8/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_8/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_8/loss_grad/truediv0gradients_8/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_8/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_8/logistic_loss/Select_grad/zeros_likegradients_8/loss_grad/truediv*
T0
Q
(gradients_8/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_8/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_8/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_8/logistic_loss/mul_grad/Shape*gradients_8/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_8/logistic_loss/mul_grad/MulMul&gradients_8/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_8/logistic_loss/mul_grad/SumSum&gradients_8/logistic_loss/mul_grad/Mul8gradients_8/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*gradients_8/logistic_loss/mul_grad/ReshapeReshape&gradients_8/logistic_loss/mul_grad/Sum(gradients_8/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_8/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_8/logistic_loss/sub_grad/Neg*
T0
�
(gradients_8/logistic_loss/mul_grad/Sum_1Sum(gradients_8/logistic_loss/mul_grad/Mul_1:gradients_8/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
,gradients_8/logistic_loss/mul_grad/Reshape_1Reshape(gradients_8/logistic_loss/mul_grad/Sum_1*gradients_8/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_8/logistic_loss/Exp_grad/mulMul(gradients_8/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_8/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_8/logistic_loss/Select_1_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    
�
2gradients_8/logistic_loss/Select_1_grad/zeros_likeFillBgradients_8/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_8/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_8/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_8/logistic_loss/Exp_grad/mul2gradients_8/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_8/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_8/logistic_loss/Select_1_grad/zeros_like&gradients_8/logistic_loss/Exp_grad/mul*
T0
f
&gradients_8/logistic_loss/Neg_grad/NegNeg.gradients_8/logistic_loss/Select_1_grad/Select*
T0
�
gradients_8/AddNAddN,gradients_8/logistic_loss/Select_grad/Select*gradients_8/logistic_loss/mul_grad/Reshape0gradients_8/logistic_loss/Select_1_grad/Select_1&gradients_8/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_8/logistic_loss/Select_grad/Select*
N
F
gradients_8/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_8/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_8/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_8/add_4_grad/Shapegradients_8/add_4_grad/Shape_1*
T0
�
gradients_8/add_4_grad/SumSumgradients_8/AddN,gradients_8/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
z
gradients_8/add_4_grad/ReshapeReshapegradients_8/add_4_grad/Sumgradients_8/add_4_grad/Shape*
T0*
Tshape0
�
gradients_8/add_4_grad/Sum_1Sumgradients_8/AddN.gradients_8/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_8/add_4_grad/Reshape_1Reshapegradients_8/add_4_grad/Sum_1gradients_8/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_8/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_8/add_4_grad/Reshape*
T0
a
,gradients_8/Add_3_grad/Sum/reduction_indicesConst*
valueB"       *
dtype0
�
gradients_8/Add_3_grad/SumSum gradients_8/Tanh_3_grad/TanhGrad,gradients_8/Add_3_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
Y
$gradients_8/Add_3_grad/Reshape/shapeConst*
dtype0*
valueB"      
�
gradients_8/Add_3_grad/ReshapeReshapegradients_8/Add_3_grad/Sum$gradients_8/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_8/MatMul_3_grad/MatMulMatMul gradients_8/Tanh_3_grad/TanhGradw4/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_8/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_8/Tanh_3_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
_
 gradients_8/Tanh_2_grad/TanhGradTanhGradTanh_2 gradients_8/MatMul_3_grad/MatMul*
T0
Z
,gradients_8/Add_2_grad/Sum/reduction_indicesConst*
valueB: *
dtype0
�
gradients_8/Add_2_grad/SumSum gradients_8/Tanh_2_grad/TanhGrad,gradients_8/Add_2_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_8/Add_2_grad/Reshape/shapeConst*
valueB"   @   *
dtype0
�
gradients_8/Add_2_grad/ReshapeReshapegradients_8/Add_2_grad/Sum$gradients_8/Add_2_grad/Reshape/shape*
T0*
Tshape0
9
d-b3Identitygradients_8/Add_2_grad/Reshape*
T0
:
gradients_9/ShapeConst*
valueB *
dtype0
B
gradients_9/grad_ys_0Const*
dtype0*
valueB
 *  �?
]
gradients_9/FillFillgradients_9/Shapegradients_9/grad_ys_0*
T0*

index_type0
X
#gradients_9/loss_grad/Reshape/shapeConst*
valueB"      *
dtype0
v
gradients_9/loss_grad/ReshapeReshapegradients_9/Fill#gradients_9/loss_grad/Reshape/shape*
T0*
Tshape0
P
gradients_9/loss_grad/ConstConst*
dtype0*
valueB"      
y
gradients_9/loss_grad/TileTilegradients_9/loss_grad/Reshapegradients_9/loss_grad/Const*

Tmultiples0*
T0
J
gradients_9/loss_grad/Const_1Const*
valueB
 *  �D*
dtype0
l
gradients_9/loss_grad/truedivRealDivgradients_9/loss_grad/Tilegradients_9/loss_grad/Const_1*
T0
U
&gradients_9/logistic_loss/sub_grad/NegNeggradients_9/loss_grad/truediv*
T0
w
*gradients_9/logistic_loss/Log1p_grad/add/xConst^gradients_9/loss_grad/truediv*
dtype0*
valueB
 *  �?
y
(gradients_9/logistic_loss/Log1p_grad/addAddV2*gradients_9/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_9/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_9/logistic_loss/Log1p_grad/add*
T0
�
(gradients_9/logistic_loss/Log1p_grad/mulMulgradients_9/loss_grad/truediv/gradients_9/logistic_loss/Log1p_grad/Reciprocal*
T0
u
@gradients_9/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
c
6gradients_9/logistic_loss/Select_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    
�
0gradients_9/logistic_loss/Select_grad/zeros_likeFill@gradients_9/logistic_loss/Select_grad/zeros_like/shape_as_tensor6gradients_9/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
,gradients_9/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_9/loss_grad/truediv0gradients_9/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_9/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_9/logistic_loss/Select_grad/zeros_likegradients_9/loss_grad/truediv*
T0
Q
(gradients_9/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
U
*gradients_9/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_9/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_9/logistic_loss/mul_grad/Shape*gradients_9/logistic_loss/mul_grad/Shape_1*
T0
g
&gradients_9/logistic_loss/mul_grad/MulMul&gradients_9/logistic_loss/sub_grad/Neginput_y*
T0
�
&gradients_9/logistic_loss/mul_grad/SumSum&gradients_9/logistic_loss/mul_grad/Mul8gradients_9/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*gradients_9/logistic_loss/mul_grad/ReshapeReshape&gradients_9/logistic_loss/mul_grad/Sum(gradients_9/logistic_loss/mul_grad/Shape*
T0*
Tshape0
g
(gradients_9/logistic_loss/mul_grad/Mul_1Muladd_4&gradients_9/logistic_loss/sub_grad/Neg*
T0
�
(gradients_9/logistic_loss/mul_grad/Sum_1Sum(gradients_9/logistic_loss/mul_grad/Mul_1:gradients_9/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
,gradients_9/logistic_loss/mul_grad/Reshape_1Reshape(gradients_9/logistic_loss/mul_grad/Sum_1*gradients_9/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_9/logistic_loss/Exp_grad/mulMul(gradients_9/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
w
Bgradients_9/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
e
8gradients_9/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
2gradients_9/logistic_loss/Select_1_grad/zeros_likeFillBgradients_9/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor8gradients_9/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
.gradients_9/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_9/logistic_loss/Exp_grad/mul2gradients_9/logistic_loss/Select_1_grad/zeros_like*
T0
�
0gradients_9/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_9/logistic_loss/Select_1_grad/zeros_like&gradients_9/logistic_loss/Exp_grad/mul*
T0
f
&gradients_9/logistic_loss/Neg_grad/NegNeg.gradients_9/logistic_loss/Select_1_grad/Select*
T0
�
gradients_9/AddNAddN,gradients_9/logistic_loss/Select_grad/Select*gradients_9/logistic_loss/mul_grad/Reshape0gradients_9/logistic_loss/Select_1_grad/Select_1&gradients_9/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_9/logistic_loss/Select_grad/Select*
N
F
gradients_9/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
E
gradients_9/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients_9/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_9/add_4_grad/Shapegradients_9/add_4_grad/Shape_1*
T0
�
gradients_9/add_4_grad/SumSumgradients_9/AddN,gradients_9/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
z
gradients_9/add_4_grad/ReshapeReshapegradients_9/add_4_grad/Sumgradients_9/add_4_grad/Shape*
T0*
Tshape0
�
gradients_9/add_4_grad/Sum_1Sumgradients_9/AddN.gradients_9/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_9/add_4_grad/Reshape_1Reshapegradients_9/add_4_grad/Sum_1gradients_9/add_4_grad/Shape_1*
T0*
Tshape0
]
 gradients_9/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_9/add_4_grad/Reshape*
T0
a
,gradients_9/Add_3_grad/Sum/reduction_indicesConst*
dtype0*
valueB"       
�
gradients_9/Add_3_grad/SumSum gradients_9/Tanh_3_grad/TanhGrad,gradients_9/Add_3_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Y
$gradients_9/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_9/Add_3_grad/ReshapeReshapegradients_9/Add_3_grad/Sum$gradients_9/Add_3_grad/Reshape/shape*
T0*
Tshape0
�
 gradients_9/MatMul_3_grad/MatMulMatMul gradients_9/Tanh_3_grad/TanhGradw4/read*
T0*
transpose_a( *
transpose_b(
�
"gradients_9/MatMul_3_grad/MatMul_1MatMulTanh_2 gradients_9/Tanh_3_grad/TanhGrad*
T0*
transpose_a(*
transpose_b( 
=
d-w4Identity"gradients_9/MatMul_3_grad/MatMul_1*
T0
;
gradients_10/ShapeConst*
dtype0*
valueB 
C
gradients_10/grad_ys_0Const*
valueB
 *  �?*
dtype0
`
gradients_10/FillFillgradients_10/Shapegradients_10/grad_ys_0*
T0*

index_type0
Y
$gradients_10/loss_grad/Reshape/shapeConst*
dtype0*
valueB"      
y
gradients_10/loss_grad/ReshapeReshapegradients_10/Fill$gradients_10/loss_grad/Reshape/shape*
T0*
Tshape0
Q
gradients_10/loss_grad/ConstConst*
dtype0*
valueB"      
|
gradients_10/loss_grad/TileTilegradients_10/loss_grad/Reshapegradients_10/loss_grad/Const*

Tmultiples0*
T0
K
gradients_10/loss_grad/Const_1Const*
valueB
 *  �D*
dtype0
o
gradients_10/loss_grad/truedivRealDivgradients_10/loss_grad/Tilegradients_10/loss_grad/Const_1*
T0
W
'gradients_10/logistic_loss/sub_grad/NegNeggradients_10/loss_grad/truediv*
T0
y
+gradients_10/logistic_loss/Log1p_grad/add/xConst^gradients_10/loss_grad/truediv*
valueB
 *  �?*
dtype0
{
)gradients_10/logistic_loss/Log1p_grad/addAddV2+gradients_10/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
r
0gradients_10/logistic_loss/Log1p_grad/Reciprocal
Reciprocal)gradients_10/logistic_loss/Log1p_grad/add*
T0
�
)gradients_10/logistic_loss/Log1p_grad/mulMulgradients_10/loss_grad/truediv0gradients_10/logistic_loss/Log1p_grad/Reciprocal*
T0
v
Agradients_10/logistic_loss/Select_grad/zeros_like/shape_as_tensorConst*
valueB"      *
dtype0
d
7gradients_10/logistic_loss/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
1gradients_10/logistic_loss/Select_grad/zeros_likeFillAgradients_10/logistic_loss/Select_grad/zeros_like/shape_as_tensor7gradients_10/logistic_loss/Select_grad/zeros_like/Const*
T0*

index_type0
�
-gradients_10/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqualgradients_10/loss_grad/truediv1gradients_10/logistic_loss/Select_grad/zeros_like*
T0
�
/gradients_10/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual1gradients_10/logistic_loss/Select_grad/zeros_likegradients_10/loss_grad/truediv*
T0
R
)gradients_10/logistic_loss/mul_grad/ShapeShapeadd_4*
T0*
out_type0
V
+gradients_10/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
9gradients_10/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients_10/logistic_loss/mul_grad/Shape+gradients_10/logistic_loss/mul_grad/Shape_1*
T0
i
'gradients_10/logistic_loss/mul_grad/MulMul'gradients_10/logistic_loss/sub_grad/Neginput_y*
T0
�
'gradients_10/logistic_loss/mul_grad/SumSum'gradients_10/logistic_loss/mul_grad/Mul9gradients_10/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
+gradients_10/logistic_loss/mul_grad/ReshapeReshape'gradients_10/logistic_loss/mul_grad/Sum)gradients_10/logistic_loss/mul_grad/Shape*
T0*
Tshape0
i
)gradients_10/logistic_loss/mul_grad/Mul_1Muladd_4'gradients_10/logistic_loss/sub_grad/Neg*
T0
�
)gradients_10/logistic_loss/mul_grad/Sum_1Sum)gradients_10/logistic_loss/mul_grad/Mul_1;gradients_10/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
-gradients_10/logistic_loss/mul_grad/Reshape_1Reshape)gradients_10/logistic_loss/mul_grad/Sum_1+gradients_10/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
u
'gradients_10/logistic_loss/Exp_grad/mulMul)gradients_10/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
x
Cgradients_10/logistic_loss/Select_1_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB"      
f
9gradients_10/logistic_loss/Select_1_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0
�
3gradients_10/logistic_loss/Select_1_grad/zeros_likeFillCgradients_10/logistic_loss/Select_1_grad/zeros_like/shape_as_tensor9gradients_10/logistic_loss/Select_1_grad/zeros_like/Const*
T0*

index_type0
�
/gradients_10/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual'gradients_10/logistic_loss/Exp_grad/mul3gradients_10/logistic_loss/Select_1_grad/zeros_like*
T0
�
1gradients_10/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual3gradients_10/logistic_loss/Select_1_grad/zeros_like'gradients_10/logistic_loss/Exp_grad/mul*
T0
h
'gradients_10/logistic_loss/Neg_grad/NegNeg/gradients_10/logistic_loss/Select_1_grad/Select*
T0
�
gradients_10/AddNAddN-gradients_10/logistic_loss/Select_grad/Select+gradients_10/logistic_loss/mul_grad/Reshape1gradients_10/logistic_loss/Select_1_grad/Select_1'gradients_10/logistic_loss/Neg_grad/Neg*
N*
T0*@
_class6
42loc:@gradients_10/logistic_loss/Select_grad/Select
G
gradients_10/add_4_grad/ShapeShapeTanh_3*
T0*
out_type0
F
gradients_10/add_4_grad/Shape_1ShapeSum*
T0*
out_type0
�
-gradients_10/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_10/add_4_grad/Shapegradients_10/add_4_grad/Shape_1*
T0
�
gradients_10/add_4_grad/SumSumgradients_10/AddN-gradients_10/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
}
gradients_10/add_4_grad/ReshapeReshapegradients_10/add_4_grad/Sumgradients_10/add_4_grad/Shape*
T0*
Tshape0
�
gradients_10/add_4_grad/Sum_1Sumgradients_10/AddN/gradients_10/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
!gradients_10/add_4_grad/Reshape_1Reshapegradients_10/add_4_grad/Sum_1gradients_10/add_4_grad/Shape_1*
T0*
Tshape0
_
!gradients_10/Tanh_3_grad/TanhGradTanhGradTanh_3gradients_10/add_4_grad/Reshape*
T0
b
-gradients_10/Add_3_grad/Sum/reduction_indicesConst*
dtype0*
valueB"       
�
gradients_10/Add_3_grad/SumSum!gradients_10/Tanh_3_grad/TanhGrad-gradients_10/Add_3_grad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0
Z
%gradients_10/Add_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_10/Add_3_grad/ReshapeReshapegradients_10/Add_3_grad/Sum%gradients_10/Add_3_grad/Reshape/shape*
T0*
Tshape0
:
d-b4Identitygradients_10/Add_3_grad/Reshape*
T0
�
initNoOp
^b1/Assign
^b2/Assign
^b3/Assign
^b4/Assign	^v/Assign	^w/Assign
^w1/Assign
^w2/Assign
^w3/Assign
^w4/Assign
�
init_1NoOp^auc/false_negatives/Assign^auc/false_positives/Assign^auc/true_negatives/Assign^auc/true_positives/Assign

ws_initNoOp^init^init_1"�