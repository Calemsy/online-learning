
A
input_xPlaceholder*
dtype0*
shape:���������'
A
input_yPlaceholder*
dtype0*
shape:���������
K
truncated_normal/shapeConst*
valueB"      *
dtype0
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
V
w0
VariableV2*
shared_name *
dtype0*
	container *
shape
:
r
	w0/AssignAssignw0truncated_normal*
validate_shape(*
use_locking(*
T0*
_class
	loc:@w0
7
w0/readIdentityw0*
T0*
_class
	loc:@w0
M
truncated_normal_1/shapeConst*
valueB"      *
dtype0
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
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
dtype0*
seed2 *

seed 
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
W
w
VariableV2*
shared_name *
dtype0*
	container *
shape:
��
q
w/AssignAssignwtruncated_normal_1*
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
M
truncated_normal_2/shapeConst*
valueB"      *
dtype0
D
truncated_normal_2/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_2/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
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
v/AssignAssignvtruncated_normal_2*
T0*
_class

loc:@v*
validate_shape(*
use_locking(
4
v/readIdentityv*
T0*
_class

loc:@v
=
CastCastinput_x*

SrcT0*
Truncate( *

DstT0
U
embedding_lookup/axisConst*
_class

loc:@w*
value	B : *
dtype0
�
embedding_lookupGatherV2w/readCastembedding_lookup/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_class

loc:@w
@
embedding_lookup/IdentityIdentityembedding_lookup*
T0
?
Sum/reduction_indicesConst*
value	B :*
dtype0
b
SumSumembedding_lookup/IdentitySum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
#
addAddV2Sumw0/read*
T0
W
embedding_lookup_1/axisConst*
dtype0*
_class

loc:@v*
value	B : 
�
embedding_lookup_1GatherV2v/readCastembedding_lookup_1/axis*
Taxis0*

batch_dims *
Tindices0*
Tparams0*
_class

loc:@v
D
embedding_lookup_1/IdentityIdentityembedding_lookup_1*
T0
A
Sum_1/reduction_indicesConst*
value	B :*
dtype0
h
Sum_1Sumembedding_lookup_1/IdentitySum_1/reduction_indices*
T0*

Tidx0*
	keep_dims( 
 
SquareSquareSum_1*
T0
8
Square_1Squareembedding_lookup_1/Identity*
T0
A
Sum_2/reduction_indicesConst*
value	B :*
dtype0
U
Sum_2SumSquare_1Sum_2/reduction_indices*

Tidx0*
	keep_dims( *
T0
"
subSubSquareSum_2*
T0
J
Sum_3/reduction_indicesConst*
valueB :
���������*
dtype0
P
Sum_3SumsubSum_3/reduction_indices*

Tidx0*
	keep_dims(*
T0
2
mul/xConst*
valueB
 *   ?*
dtype0
!
mulMulmul/xSum_3*
T0
!
add_1AddV2addmul*
T0
"
logitsIdentityadd_1*
T0
5
logistic_loss/zeros_like	ZerosLikeadd_1*
T0
T
logistic_loss/GreaterEqualGreaterEqualadd_1logistic_loss/zeros_like*
T0
d
logistic_loss/SelectSelectlogistic_loss/GreaterEqualadd_1logistic_loss/zeros_like*
T0
(
logistic_loss/NegNegadd_1*
T0
_
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negadd_1*
T0
1
logistic_loss/mulMuladd_1input_y*
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
Cast_1Castinput_y*

SrcT0*
Truncate( *

DstT0
"
SigmoidSigmoidadd_1*
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
%auc/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0
{
'auc/assert_greater_equal/Assert/Const_1Const*
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:
`
'auc/assert_greater_equal/Assert/Const_2Const*!
valueB Bx (Sigmoid:0) = *
dtype0
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
9auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const5^auc/assert_greater_equal/Assert/AssertGuard/switch_f*
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:
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
$auc/assert_less_equal/Assert/Const_1Const*<
value3B1 B+Condition x <= y did not hold element-wise:*
dtype0
]
$auc/assert_less_equal/Assert/Const_2Const*!
valueB Bx (Sigmoid:0) = *
dtype0
b
$auc/assert_less_equal/Assert/Const_3Const*&
valueB By (auc/Cast_1/x:0) = *
dtype0
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
6auc/assert_less_equal/Assert/AssertGuard/Assert/data_2Const2^auc/assert_less_equal/Assert/AssertGuard/switch_f*
dtype0*!
valueB Bx (Sigmoid:0) = 
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
N*
T0

�

auc/Cast_2CastCast_12^auc/assert_greater_equal/Assert/AssertGuard/Merge/^auc/assert_less_equal/Assert/AssertGuard/Merge*

SrcT0*
Truncate( *

DstT0

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
dtype0*
valueB"   ����
P
auc/Reshape_1Reshape
auc/Cast_2auc/Reshape_1/shape*
T0
*
Tshape0
8
	auc/ShapeShapeauc/Reshape*
T0*
out_type0
E
auc/strided_slice/stackConst*
valueB: *
dtype0
G
auc/strided_slice/stack_1Const*
valueB:*
dtype0
G
auc/strided_slice/stack_2Const*
valueB:*
dtype0
�
auc/strided_sliceStridedSlice	auc/Shapeauc/strided_slice/stackauc/strided_slice/stack_1auc/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
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
5
auc/stack/0Const*
value	B :*
dtype0
O
	auc/stackPackauc/stack/0auc/strided_slice*
T0*

axis *
N
F
auc/TileTileauc/ExpandDims	auc/stack*
T0*

Tmultiples0
G
auc/transpose/permConst*
valueB"       *
dtype0
Q
auc/transpose	Transposeauc/Reshapeauc/transpose/perm*
Tperm0*
T0
I
auc/Tile_1/multiplesConst*
valueB"�      *
dtype0
R

auc/Tile_1Tileauc/transposeauc/Tile_1/multiples*
T0*

Tmultiples0
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
VariableV2*
shared_name *%
_class
loc:@auc/true_positives*
dtype0*
	container *
shape:�
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
auc/Cast_3auc/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
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
VariableV2*
shape:�*
shared_name *&
_class
loc:@auc/false_negatives*
dtype0*
	container 
�
auc/false_negatives/AssignAssignauc/false_negatives%auc/false_negatives/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@auc/false_negatives*
validate_shape(
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
auc/Cast_4auc/Sum_1/reduction_indices*
T0*

Tidx0*
	keep_dims( 
�
auc/AssignAdd_1	AssignAddauc/false_negatives	auc/Sum_1*
T0*&
_class
loc:@auc/false_negatives*
use_locking( 
}
$auc/true_negatives/Initializer/zerosConst*
dtype0*%
_class
loc:@auc/true_negatives*
valueB�*    
�
auc/true_negatives
VariableV2*%
_class
loc:@auc/true_negatives*
dtype0*
	container *
shape:�*
shared_name 
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
auc/Cast_5auc/Sum_2/reduction_indices*
T0*

Tidx0*
	keep_dims( 
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
VariableV2*
shape:�*
shared_name *&
_class
loc:@auc/false_positives*
dtype0*
	container 
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
auc/Sum_3/reduction_indicesConst*
dtype0*
value	B :
_
	auc/Sum_3Sum
auc/Cast_6auc/Sum_3/reduction_indices*
T0*

Tidx0*
	keep_dims( 
�
auc/AssignAdd_3	AssignAddauc/false_positives	auc/Sum_3*
T0*&
_class
loc:@auc/false_positives*
use_locking( 
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
G
auc/strided_slice_1/stackConst*
valueB: *
dtype0
J
auc/strided_slice_1/stack_1Const*
dtype0*
valueB:�
I
auc/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
auc/strided_slice_1StridedSlice	auc/div_1auc/strided_slice_1/stackauc/strided_slice_1/stack_1auc/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask 
G
auc/strided_slice_2/stackConst*
dtype0*
valueB:
I
auc/strided_slice_2/stack_1Const*
valueB: *
dtype0
I
auc/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_2StridedSlice	auc/div_1auc/strided_slice_2/stackauc/strided_slice_2/stack_1auc/strided_slice_2/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0
A
auc/subSubauc/strided_slice_1auc/strided_slice_2*
T0
G
auc/strided_slice_3/stackConst*
valueB: *
dtype0
J
auc/strided_slice_3/stack_1Const*
valueB:�*
dtype0
I
auc/strided_slice_3/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_3StridedSliceauc/divauc/strided_slice_3/stackauc/strided_slice_3/stack_1auc/strided_slice_3/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
G
auc/strided_slice_4/stackConst*
valueB:*
dtype0
I
auc/strided_slice_4/stack_1Const*
valueB: *
dtype0
I
auc/strided_slice_4/stack_2Const*
dtype0*
valueB:
�
auc/strided_slice_4StridedSliceauc/divauc/strided_slice_4/stackauc/strided_slice_4/stack_1auc/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
E
	auc/add_5AddV2auc/strided_slice_3auc/strided_slice_4*
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
auc/strided_slice_5/stackConst*
dtype0*
valueB: 
J
auc/strided_slice_5/stack_1Const*
valueB:�*
dtype0
I
auc/strided_slice_5/stack_2Const*
dtype0*
valueB:
�
auc/strided_slice_5StridedSlice	auc/div_3auc/strided_slice_5/stackauc/strided_slice_5/stack_1auc/strided_slice_5/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
G
auc/strided_slice_6/stackConst*
valueB:*
dtype0
I
auc/strided_slice_6/stack_1Const*
valueB: *
dtype0
I
auc/strided_slice_6/stack_2Const*
dtype0*
valueB:
�
auc/strided_slice_6StridedSlice	auc/div_3auc/strided_slice_6/stackauc/strided_slice_6/stack_1auc/strided_slice_6/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
C
	auc/sub_1Subauc/strided_slice_5auc/strided_slice_6*
T0
G
auc/strided_slice_7/stackConst*
valueB: *
dtype0
J
auc/strided_slice_7/stack_1Const*
dtype0*
valueB:�
I
auc/strided_slice_7/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_7StridedSlice	auc/div_2auc/strided_slice_7/stackauc/strided_slice_7/stack_1auc/strided_slice_7/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
G
auc/strided_slice_8/stackConst*
valueB:*
dtype0
I
auc/strided_slice_8/stack_1Const*
dtype0*
valueB: 
I
auc/strided_slice_8/stack_2Const*
valueB:*
dtype0
�
auc/strided_slice_8StridedSlice	auc/div_2auc/strided_slice_8/stackauc/strided_slice_8/stack_1auc/strided_slice_8/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask 
F

auc/add_11AddV2auc/strided_slice_7auc/strided_slice_8*
T0
<
auc/truediv_1/yConst*
valueB
 *   @*
dtype0
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
J
gradients/loss_grad/ShapeShapelogistic_loss*
T0*
out_type0
s
gradients/loss_grad/TileTilegradients/loss_grad/Reshapegradients/loss_grad/Shape*

Tmultiples0*
T0
L
gradients/loss_grad/Shape_1Shapelogistic_loss*
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
gradients/loss_grad/ProdProdgradients/loss_grad/Shape_1gradients/loss_grad/Const*

Tidx0*
	keep_dims( *
T0
I
gradients/loss_grad/Const_1Const*
dtype0*
valueB: 
�
gradients/loss_grad/Prod_1Prodgradients/loss_grad/Shape_2gradients/loss_grad/Const_1*

Tidx0*
	keep_dims( *
T0
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
W
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0
[
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0
�
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0
�
 gradients/logistic_loss_grad/SumSumgradients/loss_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0
�
"gradients/logistic_loss_grad/Sum_1Sumgradients/loss_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape
�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1
^
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0
]
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0
k
$gradients/logistic_loss/sub_grad/NegNeg5gradients/logistic_loss_grad/tuple/control_dependency*
T0
�
&gradients/logistic_loss/sub_grad/Sum_1Sum$gradients/logistic_loss/sub_grad/Neg8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
*gradients/logistic_loss/sub_grad/Reshape_1Reshape&gradients/logistic_loss/sub_grad/Sum_1(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0
�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
�
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape
�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1
�
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
valueB
 *  �?
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
K
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeadd_1*
T0
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
&gradients/logistic_loss/mul_grad/ShapeShapeadd_1*
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
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0
z
&gradients/logistic_loss/mul_grad/Mul_1Muladd_1;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
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
Y
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0
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
A
gradients/add_1_grad/ShapeShapeadd*
T0*
out_type0
C
gradients/add_1_grad/Shape_1Shapemul*
T0*
out_type0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0
�
gradients/add_1_grad/SumSumgradients/AddN*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0
�
gradients/add_1_grad/Sum_1Sumgradients/AddN,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
z
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
?
gradients/add_grad/ShapeShapeSum*
T0*
out_type0
E
gradients/add_grad/Shape_1Shapew0/read*
T0*
out_type0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0
�
gradients/add_grad/SumSum-gradients/add_1_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sum-gradients/add_1_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
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
A
gradients/mul_grad/ShapeShapemul/x*
T0*
out_type0
C
gradients/mul_grad/Shape_1ShapeSum_3*
T0*
out_type0
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0
^
gradients/mul_grad/MulMul/gradients/add_1_grad/tuple/control_dependency_1Sum_3*
T0
�
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
n
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0
`
gradients/mul_grad/Mul_1Mulmul/x/gradients/add_1_grad/tuple/control_dependency_1*
T0
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
U
gradients/Sum_grad/ShapeShapeembedding_lookup/Identity*
T0*
out_type0
n
gradients/Sum_grad/SizeConst*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :
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
gradients/Sum_grad/range/startConst*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : 
u
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape
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
N*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
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
gradients/Sum_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
T0*
Tshape0
s
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0
A
gradients/Sum_3_grad/ShapeShapesub*
T0*
out_type0
r
gradients/Sum_3_grad/SizeConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_3_grad/addAddV2Sum_3/reduction_indicesgradients/Sum_3_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
�
gradients/Sum_3_grad/modFloorModgradients/Sum_3_grad/addgradients/Sum_3_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
t
gradients/Sum_3_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_3_grad/range/startConst*
dtype0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B : 
y
 gradients/Sum_3_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_3_grad/rangeRange gradients/Sum_3_grad/range/startgradients/Sum_3_grad/Size gradients/Sum_3_grad/range/delta*-
_class#
!loc:@gradients/Sum_3_grad/Shape*

Tidx0
x
gradients/Sum_3_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_3_grad/FillFillgradients/Sum_3_grad/Shape_1gradients/Sum_3_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*

index_type0
�
"gradients/Sum_3_grad/DynamicStitchDynamicStitchgradients/Sum_3_grad/rangegradients/Sum_3_grad/modgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
N
w
gradients/Sum_3_grad/Maximum/yConst*
dtype0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :
�
gradients/Sum_3_grad/MaximumMaximum"gradients/Sum_3_grad/DynamicStitchgradients/Sum_3_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
�
gradients/Sum_3_grad/floordivFloorDivgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
�
gradients/Sum_3_grad/ReshapeReshape-gradients/mul_grad/tuple/control_dependency_1"gradients/Sum_3_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_3_grad/TileTilegradients/Sum_3_grad/Reshapegradients/Sum_3_grad/floordiv*

Tmultiples0*
T0
B
gradients/sub_grad/ShapeShapeSquare*
T0*
out_type0
C
gradients/sub_grad/Shape_1ShapeSum_2*
T0*
out_type0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0
�
gradients/sub_grad/SumSumgradients/Sum_3_grad/Tile(gradients/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0
A
gradients/sub_grad/NegNeggradients/Sum_3_grad/Tile*
T0
�
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
x
%gradients/embedding_lookup_grad/ShapeConst*
_class

loc:@w*%
valueB	"              *
dtype0	
�
$gradients/embedding_lookup_grad/CastCast%gradients/embedding_lookup_grad/Shape*
Truncate( *

DstT0*

SrcT0	*
_class

loc:@w
K
$gradients/embedding_lookup_grad/SizeSizeCast*
T0*
out_type0
X
.gradients/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
*gradients/embedding_lookup_grad/ExpandDims
ExpandDims$gradients/embedding_lookup_grad/Size.gradients/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0
a
3gradients/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
valueB:
c
5gradients/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0
c
5gradients/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
-gradients/embedding_lookup_grad/strided_sliceStridedSlice$gradients/embedding_lookup_grad/Cast3gradients/embedding_lookup_grad/strided_slice/stack5gradients/embedding_lookup_grad/strided_slice/stack_15gradients/embedding_lookup_grad/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
U
+gradients/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0
�
&gradients/embedding_lookup_grad/concatConcatV2*gradients/embedding_lookup_grad/ExpandDims-gradients/embedding_lookup_grad/strided_slice+gradients/embedding_lookup_grad/concat/axis*
T0*
N*

Tidx0
�
'gradients/embedding_lookup_grad/ReshapeReshapegradients/Sum_grad/Tile&gradients/embedding_lookup_grad/concat*
T0*
Tshape0
}
)gradients/embedding_lookup_grad/Reshape_1ReshapeCast*gradients/embedding_lookup_grad/ExpandDims*
T0*
Tshape0
v
gradients/Square_grad/ConstConst,^gradients/sub_grad/tuple/control_dependency*
valueB
 *   @*
dtype0
M
gradients/Square_grad/MulMulSum_1gradients/Square_grad/Const*
T0
s
gradients/Square_grad/Mul_1Mul+gradients/sub_grad/tuple/control_dependencygradients/Square_grad/Mul*
T0
F
gradients/Sum_2_grad/ShapeShapeSquare_1*
T0*
out_type0
r
gradients/Sum_2_grad/SizeConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/addAddV2Sum_2/reduction_indicesgradients/Sum_2_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
�
gradients/Sum_2_grad/modFloorModgradients/Sum_2_grad/addgradients/Sum_2_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
t
gradients/Sum_2_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_2_grad/range/startConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B : *
dtype0
y
 gradients/Sum_2_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/rangeRange gradients/Sum_2_grad/range/startgradients/Sum_2_grad/Size gradients/Sum_2_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
x
gradients/Sum_2_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/FillFillgradients/Sum_2_grad/Shape_1gradients/Sum_2_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape*

index_type0
�
"gradients/Sum_2_grad/DynamicStitchDynamicStitchgradients/Sum_2_grad/rangegradients/Sum_2_grad/modgradients/Sum_2_grad/Shapegradients/Sum_2_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
N
w
gradients/Sum_2_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/MaximumMaximum"gradients/Sum_2_grad/DynamicStitchgradients/Sum_2_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
�
gradients/Sum_2_grad/floordivFloorDivgradients/Sum_2_grad/Shapegradients/Sum_2_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
�
gradients/Sum_2_grad/ReshapeReshape-gradients/sub_grad/tuple/control_dependency_1"gradients/Sum_2_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_2_grad/TileTilegradients/Sum_2_grad/Reshapegradients/Sum_2_grad/floordiv*

Tmultiples0*
T0
Y
gradients/Sum_1_grad/ShapeShapeembedding_lookup_1/Identity*
T0*
out_type0
r
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_1_grad/addAddV2Sum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
t
gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0
y
 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
x
gradients/Sum_1_grad/Fill/valueConst*
dtype0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :
�
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0
�
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N
w
gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
�
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape

gradients/Sum_1_grad/ReshapeReshapegradients/Square_grad/Mul_1"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*
T0*

Tmultiples0
f
gradients/Square_1_grad/ConstConst^gradients/Sum_2_grad/Tile*
valueB
 *   @*
dtype0
g
gradients/Square_1_grad/MulMulembedding_lookup_1/Identitygradients/Square_1_grad/Const*
T0
e
gradients/Square_1_grad/Mul_1Mulgradients/Sum_2_grad/Tilegradients/Square_1_grad/Mul*
T0
�
gradients/AddN_1AddNgradients/Sum_1_grad/Tilegradients/Square_1_grad/Mul_1*
T0*,
_class"
 loc:@gradients/Sum_1_grad/Tile*
N
z
'gradients/embedding_lookup_1_grad/ShapeConst*
_class

loc:@v*%
valueB	"              *
dtype0	
�
&gradients/embedding_lookup_1_grad/CastCast'gradients/embedding_lookup_1_grad/Shape*
Truncate( *

DstT0*

SrcT0	*
_class

loc:@v
M
&gradients/embedding_lookup_1_grad/SizeSizeCast*
T0*
out_type0
Z
0gradients/embedding_lookup_1_grad/ExpandDims/dimConst*
value	B : *
dtype0
�
,gradients/embedding_lookup_1_grad/ExpandDims
ExpandDims&gradients/embedding_lookup_1_grad/Size0gradients/embedding_lookup_1_grad/ExpandDims/dim*
T0*

Tdim0
c
5gradients/embedding_lookup_1_grad/strided_slice/stackConst*
dtype0*
valueB:
e
7gradients/embedding_lookup_1_grad/strided_slice/stack_1Const*
valueB: *
dtype0
e
7gradients/embedding_lookup_1_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
/gradients/embedding_lookup_1_grad/strided_sliceStridedSlice&gradients/embedding_lookup_1_grad/Cast5gradients/embedding_lookup_1_grad/strided_slice/stack7gradients/embedding_lookup_1_grad/strided_slice/stack_17gradients/embedding_lookup_1_grad/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask
W
-gradients/embedding_lookup_1_grad/concat/axisConst*
dtype0*
value	B : 
�
(gradients/embedding_lookup_1_grad/concatConcatV2,gradients/embedding_lookup_1_grad/ExpandDims/gradients/embedding_lookup_1_grad/strided_slice-gradients/embedding_lookup_1_grad/concat/axis*
T0*
N*

Tidx0
�
)gradients/embedding_lookup_1_grad/ReshapeReshapegradients/AddN_1(gradients/embedding_lookup_1_grad/concat*
T0*
Tshape0
�
+gradients/embedding_lookup_1_grad/Reshape_1ReshapeCast,gradients/embedding_lookup_1_grad/ExpandDims*
T0*
Tshape0
E
train_step/learning_rateConst*
valueB
 *��L=*
dtype0
�
)train_step/update_w0/ApplyGradientDescentApplyGradientDescentw0train_step/learning_rate-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@w0
�
train_step/update_w/mulMul'gradients/embedding_lookup_grad/Reshapetrain_step/learning_rate*
T0*
_class

loc:@w
�
train_step/update_w/ScatterSub
ScatterSubw)gradients/embedding_lookup_grad/Reshape_1train_step/update_w/mul*
use_locking( *
Tindices0*
T0*
_class

loc:@w
�
train_step/update_v/mulMul)gradients/embedding_lookup_1_grad/Reshapetrain_step/learning_rate*
T0*
_class

loc:@v
�
train_step/update_v/ScatterSub
ScatterSubv+gradients/embedding_lookup_1_grad/Reshape_1train_step/update_v/mul*
use_locking( *
Tindices0*
T0*
_class

loc:@v
�

train_stepNoOp^train_step/update_v/ScatterSub^train_step/update_w/ScatterSub*^train_step/update_w0/ApplyGradientDescent
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
L
gradients_1/loss_grad/ShapeShapelogistic_loss*
T0*
out_type0
y
gradients_1/loss_grad/TileTilegradients_1/loss_grad/Reshapegradients_1/loss_grad/Shape*

Tmultiples0*
T0
N
gradients_1/loss_grad/Shape_1Shapelogistic_loss*
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
gradients_1/loss_grad/Const_1Const*
dtype0*
valueB: 
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
gradients_1/loss_grad/CastCastgradients_1/loss_grad/floordiv*
Truncate( *

DstT0*

SrcT0
i
gradients_1/loss_grad/truedivRealDivgradients_1/loss_grad/Tilegradients_1/loss_grad/Cast*
T0
Y
$gradients_1/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0
]
&gradients_1/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0
�
4gradients_1/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/logistic_loss_grad/Shape&gradients_1/logistic_loss_grad/Shape_1*
T0
�
"gradients_1/logistic_loss_grad/SumSumgradients_1/loss_grad/truediv4gradients_1/logistic_loss_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
&gradients_1/logistic_loss_grad/ReshapeReshape"gradients_1/logistic_loss_grad/Sum$gradients_1/logistic_loss_grad/Shape*
T0*
Tshape0
�
$gradients_1/logistic_loss_grad/Sum_1Sumgradients_1/loss_grad/truediv6gradients_1/logistic_loss_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
(gradients_1/logistic_loss_grad/Reshape_1Reshape$gradients_1/logistic_loss_grad/Sum_1&gradients_1/logistic_loss_grad/Shape_1*
T0*
Tshape0
`
(gradients_1/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0
_
*gradients_1/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0
�
8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/sub_grad/Shape*gradients_1/logistic_loss/sub_grad/Shape_1*
T0
�
&gradients_1/logistic_loss/sub_grad/SumSum&gradients_1/logistic_loss_grad/Reshape8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_1/logistic_loss/sub_grad/ReshapeReshape&gradients_1/logistic_loss/sub_grad/Sum(gradients_1/logistic_loss/sub_grad/Shape*
T0*
Tshape0
^
&gradients_1/logistic_loss/sub_grad/NegNeg&gradients_1/logistic_loss_grad/Reshape*
T0
�
(gradients_1/logistic_loss/sub_grad/Sum_1Sum&gradients_1/logistic_loss/sub_grad/Neg:gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
,gradients_1/logistic_loss/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss/sub_grad/Sum_1*gradients_1/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0
�
*gradients_1/logistic_loss/Log1p_grad/add/xConst)^gradients_1/logistic_loss_grad/Reshape_1*
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
(gradients_1/logistic_loss/Log1p_grad/mulMul(gradients_1/logistic_loss_grad/Reshape_1/gradients_1/logistic_loss/Log1p_grad/Reciprocal*
T0
M
0gradients_1/logistic_loss/Select_grad/zeros_like	ZerosLikeadd_1*
T0
�
,gradients_1/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual*gradients_1/logistic_loss/sub_grad/Reshape0gradients_1/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_1/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_1/logistic_loss/Select_grad/zeros_like*gradients_1/logistic_loss/sub_grad/Reshape*
T0
Q
(gradients_1/logistic_loss/mul_grad/ShapeShapeadd_1*
T0*
out_type0
U
*gradients_1/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/mul_grad/Shape*gradients_1/logistic_loss/mul_grad/Shape_1*
T0
m
&gradients_1/logistic_loss/mul_grad/MulMul,gradients_1/logistic_loss/sub_grad/Reshape_1input_y*
T0
�
&gradients_1/logistic_loss/mul_grad/SumSum&gradients_1/logistic_loss/mul_grad/Mul8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_1/logistic_loss/mul_grad/ReshapeReshape&gradients_1/logistic_loss/mul_grad/Sum(gradients_1/logistic_loss/mul_grad/Shape*
T0*
Tshape0
m
(gradients_1/logistic_loss/mul_grad/Mul_1Muladd_1,gradients_1/logistic_loss/sub_grad/Reshape_1*
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
[
2gradients_1/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0
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
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*
N
C
gradients_1/add_1_grad/ShapeShapeadd*
T0*
out_type0
E
gradients_1/add_1_grad/Shape_1Shapemul*
T0*
out_type0
�
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0
�
gradients_1/add_1_grad/SumSumgradients_1/AddN,gradients_1/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
z
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
T0*
Tshape0
�
gradients_1/add_1_grad/Sum_1Sumgradients_1/AddN.gradients_1/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
T0*
Tshape0
A
gradients_1/add_grad/ShapeShapeSum*
T0*
out_type0
G
gradients_1/add_grad/Shape_1Shapew0/read*
T0*
out_type0
�
*gradients_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_grad/Shapegradients_1/add_grad/Shape_1*
T0
�
gradients_1/add_grad/SumSumgradients_1/add_1_grad/Reshape*gradients_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients_1/add_grad/ReshapeReshapegradients_1/add_grad/Sumgradients_1/add_grad/Shape*
T0*
Tshape0
�
gradients_1/add_grad/Sum_1Sumgradients_1/add_1_grad/Reshape,gradients_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients_1/add_grad/Reshape_1Reshapegradients_1/add_grad/Sum_1gradients_1/add_grad/Shape_1*
T0*
Tshape0
9
d-w0Identitygradients_1/add_grad/Reshape_1*
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
L
gradients_2/loss_grad/ShapeShapelogistic_loss*
T0*
out_type0
y
gradients_2/loss_grad/TileTilegradients_2/loss_grad/Reshapegradients_2/loss_grad/Shape*
T0*

Tmultiples0
N
gradients_2/loss_grad/Shape_1Shapelogistic_loss*
T0*
out_type0
F
gradients_2/loss_grad/Shape_2Const*
valueB *
dtype0
I
gradients_2/loss_grad/ConstConst*
dtype0*
valueB: 
�
gradients_2/loss_grad/ProdProdgradients_2/loss_grad/Shape_1gradients_2/loss_grad/Const*

Tidx0*
	keep_dims( *
T0
K
gradients_2/loss_grad/Const_1Const*
dtype0*
valueB: 
�
gradients_2/loss_grad/Prod_1Prodgradients_2/loss_grad/Shape_2gradients_2/loss_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
I
gradients_2/loss_grad/Maximum/yConst*
dtype0*
value	B :
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
Y
$gradients_2/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0
]
&gradients_2/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0
�
4gradients_2/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_2/logistic_loss_grad/Shape&gradients_2/logistic_loss_grad/Shape_1*
T0
�
"gradients_2/logistic_loss_grad/SumSumgradients_2/loss_grad/truediv4gradients_2/logistic_loss_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
&gradients_2/logistic_loss_grad/ReshapeReshape"gradients_2/logistic_loss_grad/Sum$gradients_2/logistic_loss_grad/Shape*
T0*
Tshape0
�
$gradients_2/logistic_loss_grad/Sum_1Sumgradients_2/loss_grad/truediv6gradients_2/logistic_loss_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
(gradients_2/logistic_loss_grad/Reshape_1Reshape$gradients_2/logistic_loss_grad/Sum_1&gradients_2/logistic_loss_grad/Shape_1*
T0*
Tshape0
`
(gradients_2/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0
_
*gradients_2/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0
�
8gradients_2/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_2/logistic_loss/sub_grad/Shape*gradients_2/logistic_loss/sub_grad/Shape_1*
T0
�
&gradients_2/logistic_loss/sub_grad/SumSum&gradients_2/logistic_loss_grad/Reshape8gradients_2/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_2/logistic_loss/sub_grad/ReshapeReshape&gradients_2/logistic_loss/sub_grad/Sum(gradients_2/logistic_loss/sub_grad/Shape*
T0*
Tshape0
^
&gradients_2/logistic_loss/sub_grad/NegNeg&gradients_2/logistic_loss_grad/Reshape*
T0
�
(gradients_2/logistic_loss/sub_grad/Sum_1Sum&gradients_2/logistic_loss/sub_grad/Neg:gradients_2/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
,gradients_2/logistic_loss/sub_grad/Reshape_1Reshape(gradients_2/logistic_loss/sub_grad/Sum_1*gradients_2/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0
�
*gradients_2/logistic_loss/Log1p_grad/add/xConst)^gradients_2/logistic_loss_grad/Reshape_1*
valueB
 *  �?*
dtype0
y
(gradients_2/logistic_loss/Log1p_grad/addAddV2*gradients_2/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_2/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_2/logistic_loss/Log1p_grad/add*
T0
�
(gradients_2/logistic_loss/Log1p_grad/mulMul(gradients_2/logistic_loss_grad/Reshape_1/gradients_2/logistic_loss/Log1p_grad/Reciprocal*
T0
M
0gradients_2/logistic_loss/Select_grad/zeros_like	ZerosLikeadd_1*
T0
�
,gradients_2/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual*gradients_2/logistic_loss/sub_grad/Reshape0gradients_2/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_2/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_2/logistic_loss/Select_grad/zeros_like*gradients_2/logistic_loss/sub_grad/Reshape*
T0
Q
(gradients_2/logistic_loss/mul_grad/ShapeShapeadd_1*
T0*
out_type0
U
*gradients_2/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_2/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_2/logistic_loss/mul_grad/Shape*gradients_2/logistic_loss/mul_grad/Shape_1*
T0
m
&gradients_2/logistic_loss/mul_grad/MulMul,gradients_2/logistic_loss/sub_grad/Reshape_1input_y*
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
m
(gradients_2/logistic_loss/mul_grad/Mul_1Muladd_1,gradients_2/logistic_loss/sub_grad/Reshape_1*
T0
�
(gradients_2/logistic_loss/mul_grad/Sum_1Sum(gradients_2/logistic_loss/mul_grad/Mul_1:gradients_2/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
,gradients_2/logistic_loss/mul_grad/Reshape_1Reshape(gradients_2/logistic_loss/mul_grad/Sum_1*gradients_2/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
s
&gradients_2/logistic_loss/Exp_grad/mulMul(gradients_2/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
[
2gradients_2/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0
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
C
gradients_2/add_1_grad/ShapeShapeadd*
T0*
out_type0
E
gradients_2/add_1_grad/Shape_1Shapemul*
T0*
out_type0
�
,gradients_2/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_1_grad/Shapegradients_2/add_1_grad/Shape_1*
T0
�
gradients_2/add_1_grad/SumSumgradients_2/AddN,gradients_2/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_2/add_1_grad/ReshapeReshapegradients_2/add_1_grad/Sumgradients_2/add_1_grad/Shape*
T0*
Tshape0
�
gradients_2/add_1_grad/Sum_1Sumgradients_2/AddN.gradients_2/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_2/add_1_grad/Reshape_1Reshapegradients_2/add_1_grad/Sum_1gradients_2/add_1_grad/Shape_1*
T0*
Tshape0
A
gradients_2/add_grad/ShapeShapeSum*
T0*
out_type0
G
gradients_2/add_grad/Shape_1Shapew0/read*
T0*
out_type0
�
*gradients_2/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_grad/Shapegradients_2/add_grad/Shape_1*
T0
�
gradients_2/add_grad/SumSumgradients_2/add_1_grad/Reshape*gradients_2/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients_2/add_grad/ReshapeReshapegradients_2/add_grad/Sumgradients_2/add_grad/Shape*
T0*
Tshape0
�
gradients_2/add_grad/Sum_1Sumgradients_2/add_1_grad/Reshape,gradients_2/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
z
gradients_2/add_grad/Reshape_1Reshapegradients_2/add_grad/Sum_1gradients_2/add_grad/Shape_1*
T0*
Tshape0
W
gradients_2/Sum_grad/ShapeShapeembedding_lookup/Identity*
T0*
out_type0
r
gradients_2/Sum_grad/SizeConst*-
_class#
!loc:@gradients_2/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_2/Sum_grad/addAddV2Sum/reduction_indicesgradients_2/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_2/Sum_grad/Shape
�
gradients_2/Sum_grad/modFloorModgradients_2/Sum_grad/addgradients_2/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_2/Sum_grad/Shape
t
gradients_2/Sum_grad/Shape_1Const*-
_class#
!loc:@gradients_2/Sum_grad/Shape*
valueB *
dtype0
y
 gradients_2/Sum_grad/range/startConst*-
_class#
!loc:@gradients_2/Sum_grad/Shape*
value	B : *
dtype0
y
 gradients_2/Sum_grad/range/deltaConst*-
_class#
!loc:@gradients_2/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_2/Sum_grad/rangeRange gradients_2/Sum_grad/range/startgradients_2/Sum_grad/Size gradients_2/Sum_grad/range/delta*-
_class#
!loc:@gradients_2/Sum_grad/Shape*

Tidx0
x
gradients_2/Sum_grad/Fill/valueConst*
dtype0*-
_class#
!loc:@gradients_2/Sum_grad/Shape*
value	B :
�
gradients_2/Sum_grad/FillFillgradients_2/Sum_grad/Shape_1gradients_2/Sum_grad/Fill/value*
T0*-
_class#
!loc:@gradients_2/Sum_grad/Shape*

index_type0
�
"gradients_2/Sum_grad/DynamicStitchDynamicStitchgradients_2/Sum_grad/rangegradients_2/Sum_grad/modgradients_2/Sum_grad/Shapegradients_2/Sum_grad/Fill*
T0*-
_class#
!loc:@gradients_2/Sum_grad/Shape*
N
w
gradients_2/Sum_grad/Maximum/yConst*-
_class#
!loc:@gradients_2/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_2/Sum_grad/MaximumMaximum"gradients_2/Sum_grad/DynamicStitchgradients_2/Sum_grad/Maximum/y*
T0*-
_class#
!loc:@gradients_2/Sum_grad/Shape
�
gradients_2/Sum_grad/floordivFloorDivgradients_2/Sum_grad/Shapegradients_2/Sum_grad/Maximum*
T0*-
_class#
!loc:@gradients_2/Sum_grad/Shape
�
gradients_2/Sum_grad/ReshapeReshapegradients_2/add_grad/Reshape"gradients_2/Sum_grad/DynamicStitch*
T0*
Tshape0
y
gradients_2/Sum_grad/TileTilegradients_2/Sum_grad/Reshapegradients_2/Sum_grad/floordiv*

Tmultiples0*
T0
z
'gradients_2/embedding_lookup_grad/ShapeConst*
dtype0	*
_class

loc:@w*%
valueB	"              
�
&gradients_2/embedding_lookup_grad/CastCast'gradients_2/embedding_lookup_grad/Shape*
Truncate( *

DstT0*

SrcT0	*
_class

loc:@w
M
&gradients_2/embedding_lookup_grad/SizeSizeCast*
T0*
out_type0
Z
0gradients_2/embedding_lookup_grad/ExpandDims/dimConst*
dtype0*
value	B : 
�
,gradients_2/embedding_lookup_grad/ExpandDims
ExpandDims&gradients_2/embedding_lookup_grad/Size0gradients_2/embedding_lookup_grad/ExpandDims/dim*

Tdim0*
T0
c
5gradients_2/embedding_lookup_grad/strided_slice/stackConst*
dtype0*
valueB:
e
7gradients_2/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0
e
7gradients_2/embedding_lookup_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
/gradients_2/embedding_lookup_grad/strided_sliceStridedSlice&gradients_2/embedding_lookup_grad/Cast5gradients_2/embedding_lookup_grad/strided_slice/stack7gradients_2/embedding_lookup_grad/strided_slice/stack_17gradients_2/embedding_lookup_grad/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
Index0*
T0
W
-gradients_2/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0
�
(gradients_2/embedding_lookup_grad/concatConcatV2,gradients_2/embedding_lookup_grad/ExpandDims/gradients_2/embedding_lookup_grad/strided_slice-gradients_2/embedding_lookup_grad/concat/axis*

Tidx0*
T0*
N
�
)gradients_2/embedding_lookup_grad/ReshapeReshapegradients_2/Sum_grad/Tile(gradients_2/embedding_lookup_grad/concat*
T0*
Tshape0
�
+gradients_2/embedding_lookup_grad/Reshape_1ReshapeCast,gradients_2/embedding_lookup_grad/ExpandDims*
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
d-w/strided_slice/stack_2Const*
valueB:*
dtype0
�
d-w/strided_sliceStridedSlice&gradients_2/embedding_lookup_grad/Castd-w/strided_slice/stackd-w/strided_slice/stack_1d-w/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
�
	d-w/inputUnsortedSegmentSum)gradients_2/embedding_lookup_grad/Reshape+gradients_2/embedding_lookup_grad/Reshape_1d-w/strided_slice*
Tnumsegments0*
Tindices0*
T0
#
d-wIdentity	d-w/input*
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
X
#gradients_3/loss_grad/Reshape/shapeConst*
dtype0*
valueB"      
v
gradients_3/loss_grad/ReshapeReshapegradients_3/Fill#gradients_3/loss_grad/Reshape/shape*
T0*
Tshape0
L
gradients_3/loss_grad/ShapeShapelogistic_loss*
T0*
out_type0
y
gradients_3/loss_grad/TileTilegradients_3/loss_grad/Reshapegradients_3/loss_grad/Shape*

Tmultiples0*
T0
N
gradients_3/loss_grad/Shape_1Shapelogistic_loss*
T0*
out_type0
F
gradients_3/loss_grad/Shape_2Const*
valueB *
dtype0
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
Y
$gradients_3/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0
]
&gradients_3/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0
�
4gradients_3/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_3/logistic_loss_grad/Shape&gradients_3/logistic_loss_grad/Shape_1*
T0
�
"gradients_3/logistic_loss_grad/SumSumgradients_3/loss_grad/truediv4gradients_3/logistic_loss_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
&gradients_3/logistic_loss_grad/ReshapeReshape"gradients_3/logistic_loss_grad/Sum$gradients_3/logistic_loss_grad/Shape*
T0*
Tshape0
�
$gradients_3/logistic_loss_grad/Sum_1Sumgradients_3/loss_grad/truediv6gradients_3/logistic_loss_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
(gradients_3/logistic_loss_grad/Reshape_1Reshape$gradients_3/logistic_loss_grad/Sum_1&gradients_3/logistic_loss_grad/Shape_1*
T0*
Tshape0
`
(gradients_3/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0
_
*gradients_3/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0
�
8gradients_3/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_3/logistic_loss/sub_grad/Shape*gradients_3/logistic_loss/sub_grad/Shape_1*
T0
�
&gradients_3/logistic_loss/sub_grad/SumSum&gradients_3/logistic_loss_grad/Reshape8gradients_3/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*gradients_3/logistic_loss/sub_grad/ReshapeReshape&gradients_3/logistic_loss/sub_grad/Sum(gradients_3/logistic_loss/sub_grad/Shape*
T0*
Tshape0
^
&gradients_3/logistic_loss/sub_grad/NegNeg&gradients_3/logistic_loss_grad/Reshape*
T0
�
(gradients_3/logistic_loss/sub_grad/Sum_1Sum&gradients_3/logistic_loss/sub_grad/Neg:gradients_3/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
,gradients_3/logistic_loss/sub_grad/Reshape_1Reshape(gradients_3/logistic_loss/sub_grad/Sum_1*gradients_3/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0
�
*gradients_3/logistic_loss/Log1p_grad/add/xConst)^gradients_3/logistic_loss_grad/Reshape_1*
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
(gradients_3/logistic_loss/Log1p_grad/mulMul(gradients_3/logistic_loss_grad/Reshape_1/gradients_3/logistic_loss/Log1p_grad/Reciprocal*
T0
M
0gradients_3/logistic_loss/Select_grad/zeros_like	ZerosLikeadd_1*
T0
�
,gradients_3/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual*gradients_3/logistic_loss/sub_grad/Reshape0gradients_3/logistic_loss/Select_grad/zeros_like*
T0
�
.gradients_3/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_3/logistic_loss/Select_grad/zeros_like*gradients_3/logistic_loss/sub_grad/Reshape*
T0
Q
(gradients_3/logistic_loss/mul_grad/ShapeShapeadd_1*
T0*
out_type0
U
*gradients_3/logistic_loss/mul_grad/Shape_1Shapeinput_y*
T0*
out_type0
�
8gradients_3/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_3/logistic_loss/mul_grad/Shape*gradients_3/logistic_loss/mul_grad/Shape_1*
T0
m
&gradients_3/logistic_loss/mul_grad/MulMul,gradients_3/logistic_loss/sub_grad/Reshape_1input_y*
T0
�
&gradients_3/logistic_loss/mul_grad/SumSum&gradients_3/logistic_loss/mul_grad/Mul8gradients_3/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_3/logistic_loss/mul_grad/ReshapeReshape&gradients_3/logistic_loss/mul_grad/Sum(gradients_3/logistic_loss/mul_grad/Shape*
T0*
Tshape0
m
(gradients_3/logistic_loss/mul_grad/Mul_1Muladd_1,gradients_3/logistic_loss/sub_grad/Reshape_1*
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
[
2gradients_3/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0
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
C
gradients_3/add_1_grad/ShapeShapeadd*
T0*
out_type0
E
gradients_3/add_1_grad/Shape_1Shapemul*
T0*
out_type0
�
,gradients_3/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_1_grad/Shapegradients_3/add_1_grad/Shape_1*
T0
�
gradients_3/add_1_grad/SumSumgradients_3/AddN,gradients_3/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_3/add_1_grad/ReshapeReshapegradients_3/add_1_grad/Sumgradients_3/add_1_grad/Shape*
T0*
Tshape0
�
gradients_3/add_1_grad/Sum_1Sumgradients_3/AddN.gradients_3/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_3/add_1_grad/Reshape_1Reshapegradients_3/add_1_grad/Sum_1gradients_3/add_1_grad/Shape_1*
T0*
Tshape0
C
gradients_3/mul_grad/ShapeShapemul/x*
T0*
out_type0
E
gradients_3/mul_grad/Shape_1ShapeSum_3*
T0*
out_type0
�
*gradients_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_grad/Shapegradients_3/mul_grad/Shape_1*
T0
Q
gradients_3/mul_grad/MulMul gradients_3/add_1_grad/Reshape_1Sum_3*
T0
�
gradients_3/mul_grad/SumSumgradients_3/mul_grad/Mul*gradients_3/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients_3/mul_grad/ReshapeReshapegradients_3/mul_grad/Sumgradients_3/mul_grad/Shape*
T0*
Tshape0
S
gradients_3/mul_grad/Mul_1Mulmul/x gradients_3/add_1_grad/Reshape_1*
T0
�
gradients_3/mul_grad/Sum_1Sumgradients_3/mul_grad/Mul_1,gradients_3/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
z
gradients_3/mul_grad/Reshape_1Reshapegradients_3/mul_grad/Sum_1gradients_3/mul_grad/Shape_1*
T0*
Tshape0
C
gradients_3/Sum_3_grad/ShapeShapesub*
T0*
out_type0
v
gradients_3/Sum_3_grad/SizeConst*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients_3/Sum_3_grad/addAddV2Sum_3/reduction_indicesgradients_3/Sum_3_grad/Size*
T0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape
�
gradients_3/Sum_3_grad/modFloorModgradients_3/Sum_3_grad/addgradients_3/Sum_3_grad/Size*
T0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape
x
gradients_3/Sum_3_grad/Shape_1Const*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*
valueB *
dtype0
}
"gradients_3/Sum_3_grad/range/startConst*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*
value	B : *
dtype0
}
"gradients_3/Sum_3_grad/range/deltaConst*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients_3/Sum_3_grad/rangeRange"gradients_3/Sum_3_grad/range/startgradients_3/Sum_3_grad/Size"gradients_3/Sum_3_grad/range/delta*

Tidx0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape
|
!gradients_3/Sum_3_grad/Fill/valueConst*
dtype0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*
value	B :
�
gradients_3/Sum_3_grad/FillFillgradients_3/Sum_3_grad/Shape_1!gradients_3/Sum_3_grad/Fill/value*
T0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*

index_type0
�
$gradients_3/Sum_3_grad/DynamicStitchDynamicStitchgradients_3/Sum_3_grad/rangegradients_3/Sum_3_grad/modgradients_3/Sum_3_grad/Shapegradients_3/Sum_3_grad/Fill*
T0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*
N
{
 gradients_3/Sum_3_grad/Maximum/yConst*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients_3/Sum_3_grad/MaximumMaximum$gradients_3/Sum_3_grad/DynamicStitch gradients_3/Sum_3_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape
�
gradients_3/Sum_3_grad/floordivFloorDivgradients_3/Sum_3_grad/Shapegradients_3/Sum_3_grad/Maximum*
T0*/
_class%
#!loc:@gradients_3/Sum_3_grad/Shape
�
gradients_3/Sum_3_grad/ReshapeReshapegradients_3/mul_grad/Reshape_1$gradients_3/Sum_3_grad/DynamicStitch*
T0*
Tshape0

gradients_3/Sum_3_grad/TileTilegradients_3/Sum_3_grad/Reshapegradients_3/Sum_3_grad/floordiv*

Tmultiples0*
T0
D
gradients_3/sub_grad/ShapeShapeSquare*
T0*
out_type0
E
gradients_3/sub_grad/Shape_1ShapeSum_2*
T0*
out_type0
�
*gradients_3/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/sub_grad/Shapegradients_3/sub_grad/Shape_1*
T0
�
gradients_3/sub_grad/SumSumgradients_3/Sum_3_grad/Tile*gradients_3/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients_3/sub_grad/ReshapeReshapegradients_3/sub_grad/Sumgradients_3/sub_grad/Shape*
T0*
Tshape0
E
gradients_3/sub_grad/NegNeggradients_3/Sum_3_grad/Tile*
T0
�
gradients_3/sub_grad/Sum_1Sumgradients_3/sub_grad/Neg,gradients_3/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients_3/sub_grad/Reshape_1Reshapegradients_3/sub_grad/Sum_1gradients_3/sub_grad/Shape_1*
T0*
Tshape0
i
gradients_3/Square_grad/ConstConst^gradients_3/sub_grad/Reshape*
valueB
 *   @*
dtype0
Q
gradients_3/Square_grad/MulMulSum_1gradients_3/Square_grad/Const*
T0
h
gradients_3/Square_grad/Mul_1Mulgradients_3/sub_grad/Reshapegradients_3/Square_grad/Mul*
T0
H
gradients_3/Sum_2_grad/ShapeShapeSquare_1*
T0*
out_type0
v
gradients_3/Sum_2_grad/SizeConst*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients_3/Sum_2_grad/addAddV2Sum_2/reduction_indicesgradients_3/Sum_2_grad/Size*
T0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape
�
gradients_3/Sum_2_grad/modFloorModgradients_3/Sum_2_grad/addgradients_3/Sum_2_grad/Size*
T0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape
x
gradients_3/Sum_2_grad/Shape_1Const*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*
valueB *
dtype0
}
"gradients_3/Sum_2_grad/range/startConst*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*
value	B : *
dtype0
}
"gradients_3/Sum_2_grad/range/deltaConst*
dtype0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*
value	B :
�
gradients_3/Sum_2_grad/rangeRange"gradients_3/Sum_2_grad/range/startgradients_3/Sum_2_grad/Size"gradients_3/Sum_2_grad/range/delta*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*

Tidx0
|
!gradients_3/Sum_2_grad/Fill/valueConst*
dtype0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*
value	B :
�
gradients_3/Sum_2_grad/FillFillgradients_3/Sum_2_grad/Shape_1!gradients_3/Sum_2_grad/Fill/value*
T0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*

index_type0
�
$gradients_3/Sum_2_grad/DynamicStitchDynamicStitchgradients_3/Sum_2_grad/rangegradients_3/Sum_2_grad/modgradients_3/Sum_2_grad/Shapegradients_3/Sum_2_grad/Fill*
N*
T0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape
{
 gradients_3/Sum_2_grad/Maximum/yConst*
dtype0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape*
value	B :
�
gradients_3/Sum_2_grad/MaximumMaximum$gradients_3/Sum_2_grad/DynamicStitch gradients_3/Sum_2_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape
�
gradients_3/Sum_2_grad/floordivFloorDivgradients_3/Sum_2_grad/Shapegradients_3/Sum_2_grad/Maximum*
T0*/
_class%
#!loc:@gradients_3/Sum_2_grad/Shape
�
gradients_3/Sum_2_grad/ReshapeReshapegradients_3/sub_grad/Reshape_1$gradients_3/Sum_2_grad/DynamicStitch*
T0*
Tshape0

gradients_3/Sum_2_grad/TileTilegradients_3/Sum_2_grad/Reshapegradients_3/Sum_2_grad/floordiv*

Tmultiples0*
T0
[
gradients_3/Sum_1_grad/ShapeShapeembedding_lookup_1/Identity*
T0*
out_type0
v
gradients_3/Sum_1_grad/SizeConst*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*
value	B :*
dtype0
�
gradients_3/Sum_1_grad/addAddV2Sum_1/reduction_indicesgradients_3/Sum_1_grad/Size*
T0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape
�
gradients_3/Sum_1_grad/modFloorModgradients_3/Sum_1_grad/addgradients_3/Sum_1_grad/Size*
T0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape
x
gradients_3/Sum_1_grad/Shape_1Const*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*
valueB *
dtype0
}
"gradients_3/Sum_1_grad/range/startConst*
dtype0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*
value	B : 
}
"gradients_3/Sum_1_grad/range/deltaConst*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*
value	B :*
dtype0
�
gradients_3/Sum_1_grad/rangeRange"gradients_3/Sum_1_grad/range/startgradients_3/Sum_1_grad/Size"gradients_3/Sum_1_grad/range/delta*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*

Tidx0
|
!gradients_3/Sum_1_grad/Fill/valueConst*
dtype0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*
value	B :
�
gradients_3/Sum_1_grad/FillFillgradients_3/Sum_1_grad/Shape_1!gradients_3/Sum_1_grad/Fill/value*
T0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*

index_type0
�
$gradients_3/Sum_1_grad/DynamicStitchDynamicStitchgradients_3/Sum_1_grad/rangegradients_3/Sum_1_grad/modgradients_3/Sum_1_grad/Shapegradients_3/Sum_1_grad/Fill*
T0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*
N
{
 gradients_3/Sum_1_grad/Maximum/yConst*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape*
value	B :*
dtype0
�
gradients_3/Sum_1_grad/MaximumMaximum$gradients_3/Sum_1_grad/DynamicStitch gradients_3/Sum_1_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape
�
gradients_3/Sum_1_grad/floordivFloorDivgradients_3/Sum_1_grad/Shapegradients_3/Sum_1_grad/Maximum*
T0*/
_class%
#!loc:@gradients_3/Sum_1_grad/Shape
�
gradients_3/Sum_1_grad/ReshapeReshapegradients_3/Square_grad/Mul_1$gradients_3/Sum_1_grad/DynamicStitch*
T0*
Tshape0

gradients_3/Sum_1_grad/TileTilegradients_3/Sum_1_grad/Reshapegradients_3/Sum_1_grad/floordiv*

Tmultiples0*
T0
j
gradients_3/Square_1_grad/ConstConst^gradients_3/Sum_2_grad/Tile*
valueB
 *   @*
dtype0
k
gradients_3/Square_1_grad/MulMulembedding_lookup_1/Identitygradients_3/Square_1_grad/Const*
T0
k
gradients_3/Square_1_grad/Mul_1Mulgradients_3/Sum_2_grad/Tilegradients_3/Square_1_grad/Mul*
T0
�
gradients_3/AddN_1AddNgradients_3/Sum_1_grad/Tilegradients_3/Square_1_grad/Mul_1*
N*
T0*.
_class$
" loc:@gradients_3/Sum_1_grad/Tile
|
)gradients_3/embedding_lookup_1_grad/ShapeConst*
_class

loc:@v*%
valueB	"              *
dtype0	
�
(gradients_3/embedding_lookup_1_grad/CastCast)gradients_3/embedding_lookup_1_grad/Shape*
Truncate( *

DstT0*

SrcT0	*
_class

loc:@v
O
(gradients_3/embedding_lookup_1_grad/SizeSizeCast*
T0*
out_type0
\
2gradients_3/embedding_lookup_1_grad/ExpandDims/dimConst*
dtype0*
value	B : 
�
.gradients_3/embedding_lookup_1_grad/ExpandDims
ExpandDims(gradients_3/embedding_lookup_1_grad/Size2gradients_3/embedding_lookup_1_grad/ExpandDims/dim*

Tdim0*
T0
e
7gradients_3/embedding_lookup_1_grad/strided_slice/stackConst*
valueB:*
dtype0
g
9gradients_3/embedding_lookup_1_grad/strided_slice/stack_1Const*
valueB: *
dtype0
g
9gradients_3/embedding_lookup_1_grad/strided_slice/stack_2Const*
valueB:*
dtype0
�
1gradients_3/embedding_lookup_1_grad/strided_sliceStridedSlice(gradients_3/embedding_lookup_1_grad/Cast7gradients_3/embedding_lookup_1_grad/strided_slice/stack9gradients_3/embedding_lookup_1_grad/strided_slice/stack_19gradients_3/embedding_lookup_1_grad/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask
Y
/gradients_3/embedding_lookup_1_grad/concat/axisConst*
value	B : *
dtype0
�
*gradients_3/embedding_lookup_1_grad/concatConcatV2.gradients_3/embedding_lookup_1_grad/ExpandDims1gradients_3/embedding_lookup_1_grad/strided_slice/gradients_3/embedding_lookup_1_grad/concat/axis*
T0*
N*

Tidx0
�
+gradients_3/embedding_lookup_1_grad/ReshapeReshapegradients_3/AddN_1*gradients_3/embedding_lookup_1_grad/concat*
T0*
Tshape0
�
-gradients_3/embedding_lookup_1_grad/Reshape_1ReshapeCast.gradients_3/embedding_lookup_1_grad/ExpandDims*
T0*
Tshape0
E
d-v/strided_slice/stackConst*
valueB: *
dtype0
G
d-v/strided_slice/stack_1Const*
valueB:*
dtype0
G
d-v/strided_slice/stack_2Const*
valueB:*
dtype0
�
d-v/strided_sliceStridedSlice(gradients_3/embedding_lookup_1_grad/Castd-v/strided_slice/stackd-v/strided_slice/stack_1d-v/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0
�
	d-v/inputUnsortedSegmentSum+gradients_3/embedding_lookup_1_grad/Reshape-gradients_3/embedding_lookup_1_grad/Reshape_1d-v/strided_slice*
Tnumsegments0*
Tindices0*
T0
#
d-vIdentity	d-v/input*
T0
.
initNoOp	^v/Assign	^w/Assign
^w0/Assign
�
init_1NoOp^auc/false_negatives/Assign^auc/false_positives/Assign^auc/true_negatives/Assign^auc/true_positives/Assign

ws_initNoOp^init^init_1"�