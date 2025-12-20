#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/Broadcast.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOpsEnums.cpp.inc"
#include "Compiler/Dialect/nova/NovaOpsAttributes.cpp.inc"
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"

// Helper Functions

// type promotion for result type -heirarchy based

static LogicalResult BinaryTypePromotionReturnType(MLIRContext *context, std::optional<Location> loc, ValueRange operands,
                                                   DictionaryAttr attributes, OpaqueProperties properties,
                                                   RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  mlir::Builder builder(context);
  auto lhstensor = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhstensor = llvm::dyn_cast<TensorType>(operands[1].getType());
  Type lhselemtype = lhstensor.getElementType();
  Type rhselemType = rhstensor.getElementType();
  //if complex
   if( isa<ComplexType>(lhselemtype)){
      auto clhs = dyn_cast<ComplexType>(lhselemtype);
      auto elemtype = clhs.getElementType();
      if(auto ftype = dyn_cast<FloatType>(elemtype)){
        unsigned bitwidth = ftype.getWidth();
        if(bitwidth==64){
    inferredReturnTypes.push_back(
        RankedTensorType::get(computeBroadcastShape(lhstensor.getShape(),
                                                rhstensor.getShape()).value(),
                                              ComplexType::get(builder.getF64Type())));
        return success();
        }
        inferredReturnTypes.push_back(
        RankedTensorType::get(computeBroadcastShape(lhstensor.getShape(),
                                                rhstensor.getShape()).value(),
                                              ComplexType::get(builder.getF32Type())));   
        return success();                                     
    }
       return success();}
  // 1..finding shape
   lhstensor = llvm::dyn_cast<TensorType>(operands[0].getType());
   rhstensor = llvm::dyn_cast<TensorType>(operands[1].getType());
   auto broadcastedShape = computeBroadcastShape(lhstensor.getShape(),
                                                rhstensor.getShape());
  if (!broadcastedShape)
  {
    if (loc)
    {
      mlir::emitError(*loc)
          << "incompatible shapes for broadcasting - "
          << lhstensor << " and " << rhstensor;
    }
    return failure();
  }
  // 2.fiding dtype
  lhselemtype = lhstensor.getElementType();
  rhselemType = rhstensor.getElementType();
  unsigned resultbitwidth = 0;
  auto flhstype = dyn_cast<mlir::FloatType>(lhselemtype);
  auto frhstype = dyn_cast<mlir::FloatType>(rhselemType);
  auto ilhstype = dyn_cast<mlir::IntegerType>(lhselemtype);
  auto irhstype = dyn_cast<mlir::IntegerType>(rhselemType);
  // if both float, get higher bitwidth
  if (flhstype && frhstype)
  {
    unsigned lhsbitwidth = flhstype.getWidth();
    unsigned rhsbitwidth = frhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
  }
  // if lhs float and rhs is int get lhs bitwidth
  else if (flhstype && irhstype)
  {
    unsigned lhsbitwidth = flhstype.getWidth();
    unsigned rhsbitwidth = irhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
  }
  // if rhs float and lhs is int get rhs bitwidth
  else if (ilhstype && frhstype)
  {
    unsigned lhsbitwidth = ilhstype.getWidth();
    unsigned rhsbitwidth = frhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
  }
  // if both integer get higher bitwidth and push back the result int type
  else if(ilhstype && irhstype)
  {
    unsigned lhsbitwidth = ilhstype.getWidth();
    unsigned rhsbitwidth = irhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
    auto resultType = builder.getI8Type();
    switch (resultbitwidth)
    {
    case 64:
      resultType = builder.getI64Type();
      break;
    case 32:
      resultType = builder.getI32Type();
      break;
    case 16:
      resultType = builder.getI16Type();
    }
    inferredReturnTypes.push_back(
        RankedTensorType::get(*broadcastedShape, resultType));
    return success();
  }
  else{
    inferredReturnTypes.push_back(
        RankedTensorType::get(*broadcastedShape, lhselemtype));
        return success();
  }
  auto resulType = builder.getF16Type();
  switch (resultbitwidth)
  {
  case 64:
    resulType = builder.getF64Type();
    break;
  case 32:
    resulType = builder.getF32Type();
  }
  inferredReturnTypes.push_back(
      RankedTensorType::get(*broadcastedShape, resulType));
  return success();
}
// float promotion for result type
static LogicalResult BinaryFloatPromotionReturnType(MLIRContext *context, std::optional<Location> loc, ValueRange operands,
                                                    DictionaryAttr attributes, OpaqueProperties properties,
                                                    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  mlir::Builder builder(context);
  auto lhstensor = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhstensor = llvm::dyn_cast<TensorType>(operands[1].getType());
  Type lhselemtype = lhstensor.getElementType();
  Type rhselemType = rhstensor.getElementType();
  unsigned resultbitwidth = 0;
    if( isa<ComplexType>(lhselemtype)){
      auto clhs = dyn_cast<ComplexType>(lhselemtype);
      auto elemtype = clhs.getElementType();
      if(auto ftype = dyn_cast<FloatType>(elemtype)){
        unsigned bitwidth = ftype.getWidth();
        if(bitwidth==64){
    inferredReturnTypes.push_back(
        RankedTensorType::get(computeBroadcastShape(lhstensor.getShape(),
                            rhstensor.getShape()).value(),
                            builder.getF64Type()));
        return success();
        }
        inferredReturnTypes.push_back(
        RankedTensorType::get(computeBroadcastShape(lhstensor.getShape(),
            rhstensor.getShape()).value(),builder.getF64Type()));                                      
    
       return success();}}
  // 1.finding dtype
  auto flhstype = dyn_cast<mlir::FloatType>(lhselemtype);
  auto frhstype = dyn_cast<mlir::FloatType>(rhselemType);
  auto ilhstype = dyn_cast<mlir::IntegerType>(lhselemtype);
  auto irhstype = dyn_cast<mlir::IntegerType>(rhselemType);
  // if both float, get higher bitwidth
  if (flhstype && frhstype)
  {
    unsigned lhsbitwidth = flhstype.getWidth();
    unsigned rhsbitwidth = frhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
  }
  // if lhs float and rhs is int get lhs bitwidth
  else if (flhstype && irhstype)
  {
    unsigned lhsbitwidth = flhstype.getWidth();
    unsigned rhsbitwidth = irhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
  }
  // if rhs float and lhs is int get rhs bitwidth
  else if (ilhstype && frhstype)
  {
    unsigned rhsbitwidth = frhstype.getWidth();
    unsigned lhsbitwidth = ilhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
  }
  // if both integer get higher bitwidth
  else if (ilhstype && irhstype)
  {
    unsigned lhsbitwidth = ilhstype.getWidth();
    unsigned rhsbitwidth = irhstype.getWidth();
    resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
  }
  auto resultType = builder.getF16Type();
  switch (resultbitwidth)
  {
  case 64:
    resultType = builder.getF64Type();
    break;
  case 32:
    resultType = builder.getF32Type();
  }
  // 2.finding shape
  auto broadcastedShape = computeBroadcastShape(lhstensor.getShape(),
                                                rhstensor.getShape());
  if (!broadcastedShape)
  {
    if (loc)
    {
      mlir::emitError(*loc)
          << "incompatible shapes for broadcasting - "
          << lhstensor << " and " << rhstensor;
    }
    return failure();
  }

  inferredReturnTypes.push_back(
      RankedTensorType::get(*broadcastedShape, resultType));

  return success();
}
// 🫟
// infer return type for unary operations
static LogicalResult unarycastingInferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{

  mlir::Builder builder(context);
  auto opType = llvm::dyn_cast<TensorType>(operands[0].getType());
  // get the element type
  Type inputElementType = opType.getElementType();
  Type resultType = inputElementType;
  if (auto type = dyn_cast<mlir::IntegerType>(inputElementType))
  { // returns true if int or else null
    unsigned bitwidth = type.getWidth();
    if (bitwidth == 64)
    { // if i64 then f64 or else for every other f32.
      resultType = builder.getF64Type();
    }
    else if (bitwidth == 32)
    {
      resultType = builder.getF32Type();
    }
    else
    {
      resultType = builder.getF16Type();
    }
  }
  Type returnTensorType = RankedTensorType::get(opType.getShape(), resultType);
  inferredReturnTypes.push_back(returnTensorType);
  return success();
}

/// Shared implementation for binary elementwise type inference with broadcasting
// template<typename OpType>
// static LogicalResult inferBinaryElementwiseReturnTypes(
//     MLIRContext *context,
//     std::optional<Location> loc,
//     ValueRange operands,
//     DictionaryAttr attributes,
//     OpaqueProperties properties,
//     RegionRange regions,
//     llvm::SmallVectorImpl<Type> &inferredReturnTypes) {

//   if (operands.size() != 2) {
//     if (loc) {
//       mlir::emitError(*loc) << OpType::getOperationName()
//                             << " requires exactly 2 operands";
//     }
//     return failure();
//   }

//   auto lhsType = llvm::dyn_cast<TensorType>(operands[0].getType());
//   auto rhsType = llvm::dyn_cast<TensorType>(operands[1].getType());

//   if (!lhsType || !rhsType) {
//     if (loc) {
//       mlir::emitError(*loc) << OpType::getOperationName()
//                             << " operands must be tensor types";
//     }
//     return failure();
//   }

//   Type elementType = lhsType.getElementType();

//   // if (elementType != rhsType.getElementType()) {
//   //   if (loc) {
//   //     mlir::emitError(*loc) << OpType::getOperationName()
//   //                           << " operands must have the same element type";
//   //   }
//   //   return failure();
//   // }

//   if (!lhsType.hasRank() || !rhsType.hasRank()) {
//     inferredReturnTypes.push_back(UnrankedTensorType::get(elementType));
//     return success();
//   }

//   auto broadcastedShape = computeBroadcastShape(lhsType.getShape(),
//                                                 rhsType.getShape());

//   if (!broadcastedShape) {
//     if (loc) {
//       mlir::emitError(*loc)
//         << OpType::getOperationName()
//         << ": incompatible shapes for broadcasting - "
//         << lhsType << " and " << rhsType;
//     }
//     return failure();
//   }

//   inferredReturnTypes.push_back(
//     RankedTensorType::get(*broadcastedShape, elementType));

//   return success();
// }

/// Generic verify for all binary ops
template <typename OpType>
static LogicalResult verifyBinaryOp(OpType op)
{
  auto lhsType = op.getLhs().getType();
  auto rhsType = op.getRhs().getType();
  auto resultType = op.getResult().getType();

  if (!isa<TensorType>(lhsType) || !isa<TensorType>(rhsType) ||
      !isa<TensorType>(resultType))
  {
    return op.emitOpError("operands and result must be tensor types");
  }
 // if one is complex another must be complex of same element type
  if(isa<ComplexType>(llvm::dyn_cast<TensorType>(lhsType).getElementType()) ||
     isa<ComplexType>(llvm::dyn_cast<TensorType>(rhsType).getElementType())){
       auto clhs = dyn_cast<ComplexType>(llvm::dyn_cast<TensorType>(lhsType).getElementType());
       auto crhs = dyn_cast<ComplexType>(llvm::dyn_cast<TensorType>(rhsType).getElementType());
       if(!clhs || !crhs){
        return op.emitOpError("if one operand is complex, the other must also be complex");
       }
       if(clhs.getElementType() != crhs.getElementType()){
        return op.emitOpError("complex operands must have the same element type");
       }
     }


  return success();
}

// BroadcastInDimOp

LogicalResult BroadcastInDimOp::verify()
{

  auto operandType = dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!operandType || !resultType)
  {
    return success();
  }
  auto broadcastDims = getBroadcastDimensions();

  if (static_cast<int64_t>(broadcastDims.size()) != operandType.getRank())
  {
    return emitOpError("broadcast_dimensions size (")
           << broadcastDims.size() << ") must match operand rank ("
           << operandType.getRank() << ")";
  }

  llvm::SmallVector<bool> seenDims(resultType.getRank(), false);

  for (auto [idx, dimAttr] : llvm::enumerate(broadcastDims))
  {
    int64_t dim = cast<IntegerAttr>(dimAttr).getInt();

    if (dim < 0 || dim >= resultType.getRank())
    {
      return emitOpError("broadcast dimension ") << dim
                                                 << " out of range [0, " << resultType.getRank() << ")";
    }

    if (seenDims[dim])
    {
      return emitOpError("broadcast dimension ") << dim
                                                 << " is used more than once";
    }
    seenDims[dim] = true;

    int64_t operandDim = operandType.getDimSize(idx);
    int64_t resultDim = resultType.getDimSize(dim);

    if (!ShapedType::isDynamic(operandDim) &&
        !ShapedType::isDynamic(resultDim))
    {
      if (operandDim != 1 && operandDim != resultDim)
      {
        return emitOpError() << "operand dimension " << idx
                             << " (size " << operandDim << ") "
                             << "incompatible with result dimension " << dim
                             << " (size " << resultDim << ")";
      }
    }
  }

  return success();
}

// AddOp

LogicalResult AddOp::verify() { return verifyBinaryOp(*this); }

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryTypePromotionReturnType(
      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}




// SubOp

LogicalResult SubOp::verify() { return verifyBinaryOp(*this); }

LogicalResult SubOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryTypePromotionReturnType(
      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}

// MulOp

LogicalResult MulOp::verify() { return verifyBinaryOp(*this); }

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryTypePromotionReturnType(
      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}

// DivOp

LogicalResult DivOp::verify() { return verifyBinaryOp(*this); }

LogicalResult DivOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryFloatPromotionReturnType(
      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}
LogicalResult nova::SqrtOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attrs,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredTypes) {

  // sqrt is unary
  if (operands.size() != 1)
    return failure();

  auto inputType = dyn_cast<TensorType>(operands[0].getType());
  if (!inputType)
    return failure();

  auto elemTy = inputType.getElementType();

  Type outElemTy;

  // Integer → f32
  if (isa<IntegerType>(elemTy)) {
    outElemTy = Float32Type::get(context);
  }
  // Float → same float
  else if (isa<FloatType>(elemTy)) {
    outElemTy = elemTy;
  }
  else {
    return failure();
  }

  inferredTypes.push_back(inputType.clone(outElemTy));
  return success();
}

// ModOp

LogicalResult ModOp::verify() { 
      for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      }
  return verifyBinaryOp(*this); }

LogicalResult ModOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryFloatPromotionReturnType(
      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}

// PowOp

LogicalResult PowOp::verify() { return verifyBinaryOp(*this); }

LogicalResult PowOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryTypePromotionReturnType(
      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}

// MaxOp

LogicalResult MaxOp::verify() {
      for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      } 
  return verifyBinaryOp(*this); }

LogicalResult MaxOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryTypePromotionReturnType(

      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}

// MinOp

LogicalResult MinOp::verify() { 
      for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      }
  return verifyBinaryOp(*this); }

LogicalResult MinOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  return BinaryTypePromotionReturnType(
      context, loc, operands, attributes, properties, regions, inferredReturnTypes);
}

// AndOp

LogicalResult AndOp::verify() { 
      for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      }
  return verifyBinaryOp(*this); }

LogicalResult AndOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  auto shape = llvm::dyn_cast<TensorType>(operands[0].getType()).getShape();
  Type resultType = RankedTensorType::get(shape, IntegerType::get(context, 1));
  inferredReturnTypes.push_back(resultType);
  return success();
}
//not op
LogicalResult NotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  auto shape = llvm::dyn_cast<TensorType>(operands[0].getType()).getShape();
  Type resultType = RankedTensorType::get(shape, IntegerType::get(context, 1));
  inferredReturnTypes.push_back(resultType);
  return success();
}

// OrOp

LogicalResult OrOp::verify() { 
      for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      }
  return verifyBinaryOp(*this); }

LogicalResult OrOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  auto shape = llvm::dyn_cast<TensorType>(operands[0].getType()).getShape();
  Type resultType = RankedTensorType::get(shape, IntegerType::get(context, 1));
  inferredReturnTypes.push_back(resultType);
  return success();
}

// XorOp

LogicalResult XorOp::verify() { 
      for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      }
  return verifyBinaryOp(*this); }

LogicalResult XorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  auto shape = llvm::dyn_cast<TensorType>(operands[0].getType()).getShape();
  Type resultType = RankedTensorType::get(shape, IntegerType::get(context, 1));
  inferredReturnTypes.push_back(resultType);
  return success();
}
LogicalResult AbsOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  auto shape = llvm::dyn_cast<TensorType>(operands[0].getType()).getShape();
  Type elementType = llvm::dyn_cast<TensorType>(operands[0].getType()).getElementType();
  if(isa<FloatType>(elementType) || isa<IntegerType>(elementType)){
    Type resultType = RankedTensorType::get(shape, elementType);
    inferredReturnTypes.push_back(resultType);
    return success();
  }
  else if(isa<ComplexType>(elementType)){
    auto ctype = llvm::dyn_cast<ComplexType>(elementType);
    Type  realtype = ctype.getElementType();
    Type resultType = RankedTensorType::get(shape, realtype);
    inferredReturnTypes.push_back(resultType);
    return success();
  }
  else{
    if (loc)
    {
      mlir::emitError(*loc) << "abs only supports float, integer and complex types";
    }
    return failure();
  }
}
//---------------------------------ConstantOp-----------------

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       Attribute value, Type resultType)
{
  state.addAttribute("value", value);
  state.addTypes(resultType);
}

//---------------------------------comparison-----------------

void CompareOp::build(OpBuilder &builder, OperationState &state,
                      Value lhs, Value rhs, ComparisonType kind)
{
  state.addOperands({lhs, rhs});
  state.addAttribute("kind",
                     builder.getI32IntegerAttr(static_cast<int32_t>(kind)));
  Type resultType = RankedTensorType::get({}, builder.getI1Type());
  state.addTypes(resultType);
}
LogicalResult CompareOp::verify()
{
    for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      }
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();

  // Verify that input shapes match
  if (lhsType.getShape() != rhsType.getShape())
  {
    return emitOpError("operand shapes must match for comparison");
  }

  return success();
}
LogicalResult SignOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
{

  if (operands.size() != 1)
  {
    if (location)
    {
      mlir::emitError(*location) << "sign requires exactly 1 operands";
    }
    return failure();
  }

  // The result type of sign is always int8
  // the shape is tensor of shape same as inputs
  auto shape = llvm::dyn_cast<TensorType>(operands[0].getType()).getShape();
  Type resultType = RankedTensorType::get(shape, IntegerType::get(context, 8));
  inferredReturnTypes.push_back(resultType);
  return success();
}
void ArgmaxOp::build(OpBuilder &builder, OperationState &state,
                     Value input, int64_t dimension,
                     bool keepdims, bool ignore_nan, Type resultType)
{
  state.addOperands(input);
  if (!dimension)
  {
    state.addAttribute("dimension", builder.getIntegerAttr(builder.getI64Type(), dimension));
  }

  if (keepdims)
  {
    state.addAttribute("keepdims", builder.getBoolAttr(keepdims));
  }
  if (ignore_nan)
  {
    state.addAttribute("ignore_nan", builder.getBoolAttr(ignore_nan));
  }
  state.addTypes(resultType);
}
LogicalResult ArgmaxOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
{

  auto inputType = llvm::dyn_cast<TensorType>(operands[0].getType());
  if (!inputType)
    return failure();

  auto inputShape = inputType.getShape();
  size_t inputRank = inputShape.size();

  bool keepDims = false;
  if (auto keepDimsAttr = dyn_cast_or_null<BoolAttr>(attributes.get("keepdims")))
  {
    keepDims = keepDimsAttr.getValue();
  }

  auto dimAttr = dyn_cast_or_null<IntegerAttr>(attributes.get("dimension"));

  llvm::SmallDenseSet<int64_t, 4> dimsToReduce;

  if (dimAttr)
  {
    int64_t axis = dimAttr.getValue().getSExtValue();
    if (axis < 0)
    {
      axis += inputRank;
    }
    if (axis >= 0 && static_cast<size_t>(axis) < inputRank)
    {
      dimsToReduce.insert(axis);
    }
    else
    {
      if (location.has_value())
      {
        return mlir::emitError(*location, "reduction axis is out of bounds");
      }
      return failure();
    }
  }
  else
  {
    // No dimensions specified - reduce all dimensions
    for (size_t i = 0; i < inputRank; ++i)
    {
      dimsToReduce.insert(i);
    }
  }

  llvm::SmallVector<int64_t, 4> resultShape;
  for (size_t i = 0; i < inputRank; ++i)
  {
    if (dimsToReduce.count(i))
    {
      if (keepDims)
      {
        resultShape.push_back(1);
      }
    }
    else
    {
      resultShape.push_back(inputShape[i]);
    }
  }

  if (resultShape.empty() && !keepDims)
  {
    inferredReturnTypes.push_back(RankedTensorType::get({}, IntegerType::get(context, 32)));
  }
  else
  {
    inferredReturnTypes.push_back(RankedTensorType::get(resultShape, IntegerType::get(context, 32)));
  }

  return success();
}
void ArgMinOp::build(OpBuilder &builder, OperationState &state,
                     Value input, int32_t dimension,
                     bool keepdims, bool ignore_nan, Type resultType)
{
  state.addOperands(input);

  state.addAttribute("dimension", builder.getIntegerAttr(builder.getI32Type(), dimension));

  if (keepdims)
  {
    state.addAttribute("keepdims", builder.getBoolAttr(keepdims));
  }
  if (ignore_nan)
  {
    state.addAttribute("ignore_nan", builder.getBoolAttr(ignore_nan));
  }
  state.addTypes(resultType);
}
LogicalResult ArgMinOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
{

  auto inputType = llvm::dyn_cast<TensorType>(operands[0].getType());
  if (!inputType)
    return failure();

  auto inputShape = inputType.getShape();
  size_t inputRank = inputShape.size();

  bool keepDims = false;
  if (auto keepDimsAttr = dyn_cast_or_null<BoolAttr>(attributes.get("keepdims")))
  {
    keepDims = keepDimsAttr.getValue();
  }

  auto dimAttr = dyn_cast_or_null<IntegerAttr>(attributes.get("dimension"));

  llvm::SmallDenseSet<int64_t, 4> dimsToReduce;

  if (dimAttr)
  {
    int64_t axis = dimAttr.getValue().getSExtValue();
    if (axis < 0)
    {
      axis += inputRank;
    }
    if (axis >= 0 && static_cast<size_t>(axis) < inputRank)
    {
      dimsToReduce.insert(axis);
    }
    else
    {
      if (location.has_value())
      {
        return mlir::emitError(*location, "reduction axis is out of bounds");
      }
      return failure();
    }
  }
  else
  {
    // No dimensions specified - reduce all dimensions
    for (size_t i = 0; i < inputRank; ++i)
    {
      dimsToReduce.insert(i);
    }
  }

  llvm::SmallVector<int64_t, 4> resultShape;
  for (size_t i = 0; i < inputRank; ++i)
  {
    if (dimsToReduce.count(i))
    {
      if (keepDims)
      {
        resultShape.push_back(1);
      }
    }
    else
    {
      resultShape.push_back(inputShape[i]);
    }
  }

  if (resultShape.empty() && !keepDims)
  {
    inferredReturnTypes.push_back(RankedTensorType::get({}, IntegerType::get(context, 32)));
  }
  else
  {
    inferredReturnTypes.push_back(RankedTensorType::get(resultShape, IntegerType::get(context, 32)));
  }

  return success();
}
LogicalResult CompareOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
{

  if (operands.size() != 2)
  {
    if (location)
    {
      mlir::emitError(*location) << "compare requires exactly 2 operands";
    }
    return failure();
  }

  // The result type of comparison is always a tensor of i1
  // the shape is tensor of shape same as inputs
  auto shape = llvm::dyn_cast<TensorType>(operands[0].getType()).getShape();
  Type resultType = RankedTensorType::get(shape, IntegerType::get(context, 1));
  inferredReturnTypes.push_back(resultType);
  return success();
}
// Transpose op
LogicalResult TransposeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)
{
  // getting dimensions
  auto axes1 = attributes.get("axes1")? dyn_cast<IntegerAttr>(attributes.get("axes1")).getValue().getSExtValue():-1;
  auto axes2 =attributes.get("axes2")? dyn_cast<IntegerAttr>(attributes.get("axes2")).getValue().getSExtValue():-2;
  // handling negative indexing
  auto inputType = dyn_cast<TensorType>(operands[0].getType());
  auto shape = inputType.getShape();
  int64_t size =shape.size();
  if(size<1){
    mlir::emitError(*loc)<<"transpose only accepts above rank 1";
  }
  if (axes1 < 0)
  {
    axes1 += size;
  }
  if (axes2 < 0)
  {
    axes2 += size;
  }
  llvm::SmallVector<int64_t> resshape;
  for(int64_t i=0;i<size;i++){
    if(i==axes1){
      resshape.push_back(shape[axes2]);
    }
    else if(i==axes2){
      resshape.push_back(shape[axes1]);
    }
    else{
      resshape.push_back(shape[i]);
    }
  }
inferredReturnTypes.push_back(RankedTensorType::get(resshape,inputType.getElementType()));
  return success();
}

// MatmulOp

LogicalResult MatmulOp::verify() { 
      for (Value operand : getOperands()) {
        Type operandType = operand.getType();

        // Check if the operand is a ShapedType (Tensor or MemRef)
        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      }
      return verifyBinaryOp(*this); }

/// Type inference for matrix multiplication
LogicalResult MatmulOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
{

  if (operands.size() != 2)
  {
    if (location)
    {
      mlir::emitError(*location) << "matmul requires exactly 2 operands";
    }
    return failure();
  }

  auto lhsType = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhsType = llvm::dyn_cast<TensorType>(operands[1].getType());

  if (!lhsType || !rhsType)
  {
    if (location)
    {
      mlir::emitError(*location) << "matmul operands must be tensor types";
    }
    return failure();
  }
  // function to find the result element type

  mlir::Builder builder(context);
  auto lhstensor = llvm::dyn_cast<TensorType>(operands[0].getType());
  auto rhstensor = llvm::dyn_cast<TensorType>(operands[1].getType());


  // 2.fiding dtype
  Type lhselemtype = lhstensor.getElementType();
  Type rhselemType = rhstensor.getElementType();
  unsigned resultbitwidth = 0;
  mlir::Type resultType = builder.getI8Type();
  
  // Check for complex types first
  if (isa<ComplexType>(lhselemtype) || isa<ComplexType>(rhselemType))
  {
    // For complex types, extract the element type and use it
    Type complexElemType = builder.getF32Type();
    if (auto clhs = dyn_cast<ComplexType>(lhselemtype)) {
      complexElemType = clhs.getElementType();
    } else if (auto crhs = dyn_cast<ComplexType>(rhselemType)) {
      complexElemType = crhs.getElementType();
    }
    resultType = ComplexType::get(complexElemType);  
  }
  else
  {
    auto flhstype = dyn_cast<mlir::FloatType>(lhselemtype);
    auto frhstype = dyn_cast<mlir::FloatType>(rhselemType);
    auto ilhstype = dyn_cast<mlir::IntegerType>(lhselemtype);
    auto irhstype = dyn_cast<mlir::IntegerType>(rhselemType);
    // if both float, get higher bitwidth
    if (flhstype && frhstype)
    {
      unsigned lhsbitwidth = flhstype.getWidth();
      unsigned rhsbitwidth = frhstype.getWidth();
      resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
    }
    // if lhs float and rhs is int get lhs bitwidth
    else if (flhstype && irhstype)
    {
      unsigned lhsbitwidth = flhstype.getWidth();
      unsigned rhsbitwidth = irhstype.getWidth();
      resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
    }
    // if rhs float and lhs is int get rhs bitwidth
    else if (ilhstype && frhstype)
    {
      unsigned lhsbitwidth = ilhstype.getWidth();
      unsigned rhsbitwidth = frhstype.getWidth();
      resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
    }
    // if both integer get higher bitwidth and push back the result int type
    else if (ilhstype && irhstype)
    {
      unsigned lhsbitwidth = ilhstype.getWidth();
      unsigned rhsbitwidth = irhstype.getWidth();
      resultbitwidth = lhsbitwidth > rhsbitwidth ? lhsbitwidth : rhsbitwidth;
      switch (resultbitwidth)
      {
      case 64:
        resultType = builder.getI64Type();
        break;
      case 32:
        resultType = builder.getI32Type();
        break;
      case 16:
        resultType = builder.getI16Type();
      }
    }
    resultType = builder.getF16Type();
    switch (resultbitwidth)
    {
    case 64:
      resultType = builder.getF64Type();
      break;
    case 32:
      resultType = builder.getF32Type();
    }
  }
  // end of finding element type

  if (!lhsType.hasRank() || !rhsType.hasRank())
  {
    inferredReturnTypes.push_back(UnrankedTensorType::get(resultType));
    return success();
  }

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();

  if (lhsShape.size() < 1 || rhsShape.size() < 1)
  {
    if (location)
    {
      mlir::emitError(*location) << "matmul operands must have at least rank 1";
    }
    return failure();
  }

  SmallVector<int64_t, 4> resultShape;

  // 1D x 1D: dot product -> scalar
  if (lhsShape.size() == 1 && rhsShape.size() == 1)
  {
    if (lhsShape[0] != rhsShape[0] &&
        lhsShape[0] != ShapedType::kDynamic &&
        rhsShape[0] != ShapedType::kDynamic)
    {
      if (location)
      {
        mlir::emitError(*location) << "matmul: incompatible dimensions: "
                                   << lhsShape[0] << " vs " << rhsShape[0];
      }
      return failure();
    }
    inferredReturnTypes.push_back(RankedTensorType::get({}, resultType));
    return success();
  }

  // Matrix multiplication: [..., M, K] x [..., K, N] -> [..., M, N]
  int64_t lhsK = lhsShape[lhsShape.size() - 1];
  int64_t rhsK = (rhsShape.size() == 1) ? rhsShape[0] : rhsShape[rhsShape.size() - 2];

  if (lhsK != rhsK &&
      lhsK != ShapedType::kDynamic &&
      rhsK != ShapedType::kDynamic)
  {
    if (location)
    {
      mlir::emitError(*location) << "matmul: incompatible dimensions: " << lhsK << " vs " << rhsK;
    }
    return failure();
  }

  // Batch dimensions
  // Batch dimensions
  auto lhsBatchShape = lhsShape.drop_back(2);
  auto rhsBatchShape = rhsShape.drop_back(rhsShape.size() == 1 ? 1 : 2);

  auto broadcastedBatchShape = computeBroadcastShape(lhsBatchShape, rhsBatchShape);

  if (!broadcastedBatchShape) {
    if (location) {
      mlir::emitError(*location) << "matmul: incompatible batch dimensions for broadcasting";
    }
    return failure();
  }

  resultShape.append(broadcastedBatchShape->begin(), broadcastedBatchShape->end());

  if (lhsShape.size() >= 2)
  {
    resultShape.push_back(lhsShape[lhsShape.size() - 2]);
  }

  if (rhsShape.size() >= 2)
  {
    resultShape.push_back(rhsShape[rhsShape.size() - 1]);
  }


  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, resultType));
  return success();
}
//---------------------------------reduce op----------------------------------------------------

void ReduceOp::build(OpBuilder &builder, OperationState &state,
                     ReductionKind kind, Value input, Type resultType, bool keepdims, ArrayRef<int64_t> dimension,
                     bool ignore_nan)
{
  state.addOperands(input);
  state.addAttribute("kind", builder.getI32IntegerAttr(static_cast<int32_t>(kind)));

  if (!dimension.empty())
  {
    state.addAttribute("dimension", builder.getI64ArrayAttr(dimension));
  }

  if (keepdims)
  {
    state.addAttribute("keepdims", builder.getBoolAttr(keepdims));
  }
  if (ignore_nan)
  {
    state.addAttribute("ignore_nan", builder.getBoolAttr(ignore_nan));
  }
  state.addTypes(resultType);
}
LogicalResult ReduceOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
{

  auto inputType = llvm::dyn_cast<TensorType>(operands[0].getType());
  if (!inputType)
    return failure();

  auto inputShape = inputType.getShape();
  Type elementType = inputType.getElementType();
  size_t inputRank = inputShape.size();

  bool keepDims = false;
  if (auto keepDimsAttr = dyn_cast_or_null<BoolAttr>(attributes.get("keepdims")))
  {
    keepDims = keepDimsAttr.getValue();
  }

  auto dimAttr = dyn_cast_or_null<ArrayAttr>(attributes.get("dimension"));

  llvm::SmallDenseSet<int64_t, 4> dimsToReduce;

  if (dimAttr)
  {
    for (auto axisAttr : dimAttr.getAsValueRange<IntegerAttr>())
    {
      int64_t axis = axisAttr.getSExtValue();
      if (axis < 0)
      {
        axis += inputRank;
      }
      if (axis >= 0 && static_cast<size_t>(axis) < inputRank)
      {
        dimsToReduce.insert(axis);
      }
      else
      {
        if (location.has_value())
        {
          return mlir::emitError(*location, "reduction axis is out of bounds");
        }
        return failure();
      }
    }
  }
  else
  {
    // No dimensions specified - reduce all dimensions
    for (size_t i = 0; i < inputRank; ++i)
    {
      dimsToReduce.insert(i);
    }
  }

  llvm::SmallVector<int64_t, 4> resultShape;
  for (size_t i = 0; i < inputRank; ++i)
  {
    if (dimsToReduce.count(i))
    {
      if (keepDims)
      {
        resultShape.push_back(1);
      }
    }
    else
    {
      resultShape.push_back(inputShape[i]);
    }
  }

  if (resultShape.empty() && !keepDims)
  {
    inferredReturnTypes.push_back(RankedTensorType::get({}, elementType));
  }
  else
  {
    inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elementType));
  }

  return success();
}

LogicalResult ReduceOp::verify()
{
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType)
  {
    return emitOpError("input must be a ranked tensor");
  }

  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!outputType)
  {
    return emitOpError("output must be a ranked tensor");
  }
  
        auto operand = getInput();
        Type operandType = operand.getType();

        if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
            // Get the elemental type
            Type elementType = shapedType.getElementType();

            // Check if the element type is complex
            if (isa<ComplexType>(elementType)) {
                return emitOpError("does not support complex number operands, but found type: ")
                       << elementType;
            }
        } 
      

  int64_t inputRank = inputType.getRank();

  // Verify Dimension are valid
  if (auto dimensionAttr = getDimensionAttr())
  {
    llvm::SmallVector<int64_t> DimensionVec;
    for (auto dimension : dimensionAttr.getAsValueRange<IntegerAttr>())
    {
      int64_t dimensionVal = dimension.getZExtValue();
      if (dimensionVal < 0 || dimensionVal >= inputRank)
      {
        return emitOpError("axis ") << dimensionVal << " is out of range [0, "
                                    << inputRank << ")";
      }
      DimensionVec.push_back(dimensionVal);
    }

    // Check for duplicate Dimension
    llvm::SmallSet<int64_t, 4> uniqueDimension(DimensionVec.begin(), DimensionVec.end());
    if (uniqueDimension.size() != DimensionVec.size())
    {
      return emitOpError("duplicate axis found in Dimension attribute");
    }
  }

  // For argmax/argmin, output element type should be integer
  // auto kind = getKind();
  // if (kind == ReductionKind::ARGMAX || kind == ReductionKind::ARGMIN) {
  //   if (!outputType.getElementType().isInteger(64)) {
  //     return emitOpError("argmax/argmin output must have i64 element type, got ")
  //            << outputType.getElementType();
  //   }
  // } else {
  // For other reductions, element types should match
  //   if (inputType.getElementType() != outputType.getElementType()) {
  //     return emitOpError("input and output element types must match for non-arg reductions, got input ")
  //            << inputType.getElementType() << " and output "
  //            << outputType.getElementType();
  //   }
  // }

  // Verify output shape matches expected reduced shape
  llvm::SmallVector<int64_t> expectedShape;
  if (auto dimensionAttr = getDimensionAttr())
  {
    llvm::SmallSet<int64_t, 4> reduceDimension;
    for (auto axis : dimensionAttr.getAsValueRange<IntegerAttr>())
    {
      reduceDimension.insert(axis.getZExtValue());
    }

    for (int64_t i = 0; i < inputRank; ++i)
    {
      if (reduceDimension.contains(i))
      {
        if (getKeepdims())
        {
          expectedShape.push_back(1);
        }
      }
      else
      {
        expectedShape.push_back(inputType.getDimSize(i));
      }
    }
  }
  else
  {
    // No Dimension specified - reduce all dimensions
    if (getKeepdims())
    {
      expectedShape.assign(inputRank, 1);
    }
    // else expectedShape is empty (scalar result)
  }

  auto outputShape = outputType.getShape();
  if (outputShape.size() != expectedShape.size())
  {
    return emitOpError("output rank does not match expected rank, expected ")
           << expectedShape.size() << " got " << outputShape.size();
  }

  for (size_t i = 0; i < expectedShape.size(); ++i)
  {
    if (expectedShape[i] != outputShape[i] && expectedShape[i] != ShapedType::kDynamic)
    {
      return emitOpError("output shape mismatch at dimension ") << i
                                                                << ", expected " << expectedShape[i] << " got " << outputShape[i];
    }
  }

  return success();
}
//------------------------operand not support complex number check---------------------------------
// #define VERIFY_UNARY_OP_NO_COMPLEX(Op)                    \
//   LogicalResult Op::verify()                              \
//   {                                                       \
//     auto operand = getOperand();                          \
//     Type operandType = operand.getType();                 \
//                                                           \
//     if (auto shapedType = dyn_cast<ShapedType>(operandType)) \
//     { \                     
//         Type elementType = shapedType.getElementType();   \
//         if (isa<ComplexType>(elementType)) {              \
//             return emitOpError("does not support complex number operands, but found type: ") \
//                    << elementType;                        \
//         }                                                 \
//     }                                               \
//     return success();                                     \
//   }


//------------------------unary casting infer return types---------------------------------
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                        \
  LogicalResult Op::inferReturnTypes(                                         \
      MLIRContext *context, std::optional<Location> loc, ValueRange operands, \
      DictionaryAttr attributes, OpaqueProperties properties,                 \
      RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes)  \
  {                                                                           \
    return unarycastingInferReturnTypes(                                      \
        context, loc, operands, attributes, properties, regions,              \
        inferredReturnTypes);                                                 \
  }
// exp operations

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ExpOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Exp2Op);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LogOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Log2Op);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Log10Op);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CosOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinhOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CoshOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanhOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AsinOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AcosOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AtanOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AsinhOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AcoshOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AtanhOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(GeluOp);
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SoftmaxOp);

void SoftmaxOp::build(OpBuilder &builder, OperationState &state,
                      ReductionKind kind, Value input, Type resultType,
                      bool keepdims, int32_t dimension, bool ignore_nan) {
  state.addOperands(input);
  if (dimension != -1) {
    state.addAttribute("dimension", builder.getI32IntegerAttr(dimension));
  }
  state.addTypes(resultType);
}

