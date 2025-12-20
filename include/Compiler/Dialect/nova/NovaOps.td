#ifndef NOVA_OPS
#define NOVA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "NovaDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/BuiltinTypes.td" 
//traits
def NoComplexOperands : NativeOpTrait<"NoComplexOperands">;

// data types
def I4 : Type<CPred<"$_self.isInteger(4)">, "4-bit integer">,
         BuildableType<"$_builder.getIntegerType(4)">;

def Nova_ElementType : AnyTypeOf<[
  I1,I4, I8, I16, I32, I64, 
  F16, BF16, F32, F64,
  Complex<F64>,Complex<F32>]>;
def Nova_Tensor : TensorOf<[Nova_ElementType]>;

// operations start
//enum 
def Nova_Comparisonenum:I32EnumAttr<"ComparisonType","How operands should be compared",[
  I32EnumAttrCase<"EQ",0,"eq">,
  I32EnumAttrCase<"NEQ",1,"neq">,
  I32EnumAttrCase<"LT",2,"lt">,
  I32EnumAttrCase<"GT",3,"gt">,
  I32EnumAttrCase<"LE",4,"le">,
  I32EnumAttrCase<"GE",5,"ge">
  ]>{
    let cppNamespace="::mlir::nova";
  }


//comparison defintion
def Nova_CompareOp:Op<Nova_Dialect,"compare",[Pure,InferTypeOpInterface,NoComplexOperands
  ]> {
  let summary="compare operation";

  let description="elementwise comparison";
  let arguments =(ins
  Nova_Comparisonenum:$kind,
  Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results =(outs Nova_Tensor:$result);
  let assemblyFormat=[{
    `<`$kind `>`$lhs`,`$rhs attr-dict `:`type($lhs)`,`type($rhs)}];
 let builders=[
   OpBuilder <(ins "Value":$lhs,"Value":$rhs,"ComparisonType":$kind)>
 ];
 let skipDefaultBuilders=1;
  let hasVerifier=1;
  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];

}
//---------------------------------------reduction operations---------------------------------------------------------------------------------------

def Nova_ReductionKind : I32EnumAttr<"ReductionKind",
    "Reduction operation kind", [
  I32EnumAttrCase<"MAX", 0, "max">,
  I32EnumAttrCase<"MIN", 1, "min">,
  I32EnumAttrCase<"SUM", 2, "sum">,
  I32EnumAttrCase<"PRODUCT", 3, "product">,
  I32EnumAttrCase<"MEAN", 8, "mean">,
  I32EnumAttrCase<"ALL", 7, "all">,
  I32EnumAttrCase<"ANY", 4, "any">

]> 
{
  let cppNamespace = "::mlir::nova";
}
def Nova_ReduceOp : Op<Nova_Dialect,"reduce", [Pure,InferTypeOpInterface,NoComplexOperands]> {
  let summary = "Generic reduction operation";
  let description = [{
    Performs reduction operation across specified dimensions of a tensor.
    The reduction kind is specified by the 'kind' attribute.
    
    Examples:
    ```mlir
    // Reduce all dimensions with max
    %result = nova.reduce<max> %input : tensor<4x8xf32> -> tensor<f32>
    
    // Reduce along specific dimension with sum
    %result = nova.reduce<sum> %input dimension = [0] : tensor<4x8xf32> -> tensor<8xf32>
    
    // Keep reduced dimensions with min
    %result = nova.reduce<min> %input dimension = [1] keepdims = true ignore_nan=true :  
              tensor<4x8xf32> -> tensor<4x1xf32>
    
    // Product reduction
    %result = nova.reduce<product> %input : tensor<4x8xf32> -> tensor<f32>
    ```}];
      let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
  
  let arguments = (ins 
    Nova_ReductionKind:$kind,
    Nova_Tensor:$input,
    OptionalAttr<I64ArrayAttr>:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    DefaultValuedAttr<BoolAttr, "false">:$ignore_nan
  );

    let results = (outs Nova_Tensor:$output);
  let builders = [
    OpBuilder<(ins "ReductionKind":$kind, "Value":$input, "Type":$resultType,CArg<"bool","false">:$keepdims,
                   CArg<"ArrayRef<int64_t>","{0}">:$dimension,CArg<"bool","false">:$ignore_nan)>
  ];
 let skipDefaultBuilders=1;
  let hasVerifier=1;
  let assemblyFormat = [{
  `<` $kind `>` $input (`dimension` `=` $dimension^)? (`keepdims` `=` $keepdims^)? (`ignore_nan` `=`$ignore_nan^)?
    attr-dict `:` type($input) 
  }];
}

//---------------------------ARGMAX-------------------------------

def Nova_ArgmaxOp:Op <Nova_Dialect,"argmax",[Pure,InferTypeOpInterface]>{
    let arguments = (ins 
    Nova_Tensor:$input,
    OptionalAttr<I64Attr>:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    DefaultValuedAttr<BoolAttr, "false">:$ignore_nan

  );
      let results = (outs Nova_Tensor:$output);
 let skipDefaultBuilders=1;
  let builders = [
    OpBuilder<(ins "Value":$input, 
                   "int64_t":$dimension, "bool":$keepdims,"bool":$ignore_nan, "Type":$resultType)>
  ];
        let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
    let assemblyFormat = [{
     $input (`dimension` `=` $dimension^)? (`keepdims` `=` $keepdims^)?(`ignore_nan` `=` $ignore_nan^)?
    attr-dict `:` type($input) 
  }];
}

//------------------------------ARGMIN----------------------------------------

def Nova_ArgMinOp:Op <Nova_Dialect,"argmin",[Pure,InferTypeOpInterface]>{
    let arguments = (ins 
    Nova_Tensor:$input,
    OptionalAttr<I32Attr>:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims,
    DefaultValuedAttr<BoolAttr, "false">:$ignore_nan
  ); 
      let results = (outs Nova_Tensor:$output);
 let skipDefaultBuilders=1;
  let builders = [
    OpBuilder<(ins "Value":$input, 
                   "int32_t":$dimension, "bool":$keepdims, "bool":$ignore_nan, "Type":$resultType)>
  ];
        let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
    let assemblyFormat = [{
     $input (`dimension` `=` $dimension^)? (`keepdims` `=` $keepdims^)? (`ignore_nan` `=` $ignore_nan^)?
    attr-dict `:` type($input) 
  }];
}
//-----------------------------------------------------------
// Nova broadcasting.
//-----------------------------------------------------------

// TODO : Dont add elementwise int the operation. We will add it in lowering pass

def Nova_BroadcastInDimOp : Op<Nova_Dialect, "broadcast_in_dim", [
    Pure
  ]> {
  let summary = "Broadcast operation";
  let description = [{
    Expands dimensions of input tensor by replicating data.
    
    The `broadcast_dimensions` attribute maps operand dimensions to result dimensions.
    
    Example:
      // Broadcast [3] to [2, 3]
      %result = nova.broadcast_in_dim %input, dims = [1] : (tensor<3xf32>) 
  }];

  let arguments = (ins
    Nova_Tensor:$operand,
    I64ArrayAttr:$broadcast_dimensions
  );
  
  let results = (outs Nova_Tensor:$result);
  
  let assemblyFormat = [{
    $operand `,` `dims` `=` $broadcast_dimensions attr-dict `:`
    `(` type($operand) `)` `->` type($result)
  }];

  let hasVerifier = 1;
}
//-------------------------------------------
//Base class for unary
//------------------------------------
class Nova_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Op<Nova_Dialect,mnemonic, traits # [Pure]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}


def Nova_ConstantOp : Nova_UnaryOp<"constant", [Pure]> {
  let summary = "Constant tensor operation";
  let description = [{
    Creates a constant tensor with a specified value.
    
    Example:
    ```mlir
    %0 = nova.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    ```
  }];
  
  let arguments = (ins ElementsAttr:$value);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
  
  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    /// Set a new value attribute for the constant op.
    void setValue(::mlir::Attribute newValue) {
      (*this)->setAttr("value", newValue);
    }
  }];
}

//-----------ARITHMETIC OPERATIONS-----------------------------

def Nova_AbsOp : Nova_UnaryOp<"abs", [Pure,InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
        let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}

def Nova_SquareOp : Nova_UnaryOp<"square", [Pure, SameOperandsAndResultType,NoComplexOperands]>{
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def Nova_SqrtOp : Nova_UnaryOp<"sqrt", [Pure,InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}

def Nova_SignOp : Nova_UnaryOp<"sign", [Pure,InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
      let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}
def Nova_NegOp : Nova_UnaryOp<"neg", [Pure, SameOperandsAndResultType]>{
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  
}

def Nova_ReciprocalOp : Nova_UnaryOp<"reciprocal", [Pure, SameOperandsAndResultType,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}
//-----------------------------------------------------------------
// EXPONENTS
//-----------------------------------------------------------------
 def Nova_ExpOp : Nova_UnaryOp<"exp", [Pure,InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

 def Nova_Exp2Op : Nova_UnaryOp<"exp2", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
 def Nova_LogOp : Nova_UnaryOp<"log", [Pure, InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
 def Nova_Log2Op : Nova_UnaryOp<"log2", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
 def Nova_Log10Op : Nova_UnaryOp<"log10", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

//----------------------------------------------------------------------
// TRIGNOMETRY
//----------------------------------------------------------------------
 def Nova_SinOp : Nova_UnaryOp<"sin", [Pure, InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

 def Nova_CosOp : Nova_UnaryOp<"cos", [Pure, InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

 def Nova_TanOp : Nova_UnaryOp<"tan", [Pure, InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
 def Nova_AsinOp : Nova_UnaryOp<"asin", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}  
def Nova_AcosOp : Nova_UnaryOp<"acos", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

 def Nova_AtanOp : Nova_UnaryOp<"atan", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
 def Nova_SinhOp : Nova_UnaryOp<"sinh", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
   
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

 def Nova_CoshOp : Nova_UnaryOp<"cosh", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
 def Nova_TanhOp : Nova_UnaryOp<"tanh", [Pure, InferTypeOpInterface]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
 def Nova_AsinhOp : Nova_UnaryOp<"asinh", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

 def Nova_AcoshOp : Nova_UnaryOp<"acosh", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
   
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 

 def Nova_AtanhOp : Nova_UnaryOp<"atanh", [Pure, InferTypeOpInterface,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
 
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
    let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
} 
//--------------------------SPECIAL FUNCTIONS----------------------------

def Nova_ReluOp : Nova_UnaryOp<"relu", [Pure, SameOperandsAndResultType,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$input);
  let results = (outs Nova_Tensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

//===----------------------------------------------------------------------===//
// Base class for binary elementwise tensor ops
//===----------------------------------------------------------------------===//

class Nova_BinaryOp<string mnemonic, list<Trait> commonTraits = []>: 
  Op<Nova_Dialect, mnemonic,
     commonTraits # [Pure,
                     InferTypeOpInterface,
                     ]> {

  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//
def Nova_AddOp : Nova_BinaryOp<"add", [Commutative]> {
  let summary = "Elementwise tensor addition";
  let description = [{
    Adds two tensors elementwise with broadcasting support.
    
    Example:
      %r = nova.add %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
  let hasCanonicalizer =1;
}

def Nova_SubOp : Nova_BinaryOp<"sub", []> {
  let summary = "Elementwise tensor subtraction";
  let description = [{
    Subtracts two tensors elementwise with broadcasting support.

    Example:
      %r = nova.sub %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_MulOp : Nova_BinaryOp<"mul", [Commutative]> {
  let summary = "Elementwise tensor multiplication";
  let description = [{
    Multiplies two tensors elementwise with broadcasting support.
    Example: [1,2,3] * [3,4,5] = [3,8,15]

    Example:
      %r = nova.mul %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_DivOp : Nova_BinaryOp<"div", []> {
  let summary = "Elementwise tensor division";
  let description = [{
    Divides two tensors (dividend and divisor) elementwise with broadcasting support.
    
    Example:
      %r = nova.div %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_ModOp : Nova_BinaryOp<"mod", []> {
  let summary = "Elementwise tensor remainder";
  let description = [{
    Performs element-wise remainder of dividend lhs and divisor rhs.
    Computation example: a=[2,7,10] b=[2,2,4] result=[0,1,2]
  let hasVerifier=1;  
    Example:
      %r = nova.mod %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_PowOp : Nova_BinaryOp<"pow", []> {
  let summary = "Elementwise tensor power";
  let description = [{
    Performs element-wise power: lhs^rhs.
    Computation example: a=[1,2,3] b=[2,2,4] result=[1,4,81]
  
    Example:
      %r = nova.pow %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_MaxOp : Nova_BinaryOp<"max", [Commutative,NoComplexOperands]> {
  let summary = "Elementwise tensor maximum";
  let description = [{
    Performs element-wise max operation on tensors.
    Example: a=[1,2,3] b=[2,4,2] result=[2,4,3]
  let hasVerifier=1;  
    Example:
      %r = nova.max %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_MinOp : Nova_BinaryOp<"min", [Commutative,NoComplexOperands]> {
  let summary = "Elementwise tensor minimum";
  let description = [{
    Performs element-wise min operation on tensors.
    Example: a=[1,2,3] b=[2,4,2] result=[1,2,2]
  let hasVerifier=1;  
    Example:
      %r = nova.min %a, %b : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
  }];
}

def Nova_AndOp : Nova_BinaryOp<"and", [Commutative,NoComplexOperands]> {
  let summary = "Elementwise logical AND";
  let description = [{
    Performs element-wise logical AND operation on tensors.
    Example: a=[true,false,true] b=[true,true,false] result=[true,false,false]
  let hasVerifier=1;  
    Example:
      %r = nova.and %a, %b : tensor<2x3xi1>, tensor<3xi1> -> tensor<2x3xi1>
  }];
}

def Nova_OrOp : Nova_BinaryOp<"or", [Commutative,NoComplexOperands]> {
  let summary = "Elementwise logical OR";
  let description = [{
    Performs element-wise logical OR operation on tensors.
    Example: a=[true,false,true] b=[true,true,false] result=[true,true,true]
  let hasVerifier=1;  
    Example:
      %r = nova.or %a, %b : tensor<2x3xi1>, tensor<3xi1> -> tensor<2x3xi1>
  }];
}

def Nova_XorOp : Nova_BinaryOp<"xor", [Commutative,NoComplexOperands]> {
  let summary = "Elementwise logical XOR";
  let description = [{
    Performs element-wise logical XOR operation on tensors.
    Example: a=[true,false,true] b=[true,true,false] result=[false,true,true]
  let hasVerifier=1;  
    Example:
      %r = nova.xor %a, %b : tensor<2x3xi1>, tensor<3xi1> -> tensor<2x3xi1>
  }];
}

def Nova_NotOp : Nova_UnaryOp<"not", [Commutative,InferTypeOpInterface,NoComplexOperands]> {
  let summary = "Elementwise logical XOR";
  let description = [{
    Performs element-wise logical NOT operation on tensors.
    Example: a=[true,false,true] result=[false,true,false]
  let hasVerifier=1;  
    Example:
      %r = nova.not %a tensor<3xi1> -> tensor<2x3xi1>
  }];
   let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context,
      std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands,
      ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties,
      ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}

def Nova_MatmulOp : Op<Nova_Dialect,"matmul", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$lhs, Nova_Tensor:$rhs);
  let results = (outs Nova_Tensor:$result);
  let hasVerifier=1;  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) 
  }];
  
}
//-------------------------------------ACTIVATION FUNCTIONS-------------------------------------

def Nova_SigmoidOp : Op<Nova_Dialect,"sigmoid", [Pure, SameOperandsAndResultType,NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$lhs);
  let results = (outs Nova_Tensor:$result);

  let assemblyFormat = [{
    $lhs attr-dict `:` type($lhs) }];
}
def Nova_GeluOp : Op<Nova_Dialect,"gelu", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                NoComplexOperands]> {
  let arguments = (ins Nova_Tensor:$lhs);
  let results = (outs Nova_Tensor:$result);
   
  let assemblyFormat = [{
    $lhs attr-dict `:` type($lhs) }];
  
}
def Nova_SoftmaxOp:Op<Nova_Dialect,"softmax",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>,NoComplexOperands]>{
  let arguments=(ins Nova_Tensor:$input,
   OptionalAttr<I32Attr>:$dimension);
  let results =(outs Nova_Tensor:$result);
 

  let assemblyFormat = [{
    $input (`dimension` `=` $dimension^)? attr-dict `:` type($input) }];
  let builders = [
    OpBuilder<(ins "ReductionKind":$kind, "Value":$input, "Type":$resultType,CArg<"bool","false">:$keepdims,
                   CArg<"int32_t","-1">:$dimension,CArg<"bool","false">:$ignore_nan)>
  ];
}




def Nova_TransposeOp : Op<Nova_Dialect,"transpose",[Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Nova_Tensor:$input,
      DefaultValuedAttr<I32Attr, "-1">:$axes1,
    DefaultValuedAttr<I32Attr, "-2">:$axes2);
  let results = (outs Nova_Tensor:$result);
  
  let assemblyFormat = [{
     $input (`axes1` `=` $axes1^)?(`axes2` `=` $axes2^)? attr-dict `:`   type($input) 
  }];
  
}
/*

def Nova_ConvolutionOp : Op<Nova_Dialect, "convolution", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Convolution operation with padding and stride";
  let description = [{
    Performs convolution operation with configurable padding and stride.
    Example:
    ```mlir
    %result = nova.convolution %input, %kernel 
      {padding = [1, 1], stride = [1, 1]} 
      : tensor<1x3x32x32xf32>, tensor<64x3x3x3xf32> -> tensor<1x64x32x32xf32>
    ```
  }];
  
  let arguments = (ins 
    Nova_Tensor:$lhs,
    Nova_Tensor:$rhs,
    ArrayAttr:$padding,
    ArrayAttr:$stride
  );
  
  let results = (outs Nova_Tensor:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)}];
  
  let extraClassDeclaration = [{
    //inferReturnTypes
    
    // Helper method to get padding values
    SmallVector<int64_t> getPaddingValues() {
      SmallVector<int64_t> result;
      for (auto val : getPadding())
        result.push_back(llvm::cast<IntegerAttr>(val).getInt());
      return result;
    }
    
    // Helper method to get stride values
    SmallVector<int64_t> getStrideValues() {
      SmallVector<int64_t> result;
      for (auto val : getStride())
        result.push_back(llvm::cast<IntegerAttr>(val).getInt());
      return result;
    }
  }];

  let hasVerifier=1;
}*/
#endif // NOVA_OPS