

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Complex/IR/Complex.h"

#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"

#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
namespace mlir
{
  namespace nova
  {

    // functions which will be called inside template
    struct NovaOpTosaOp
    {
      // helper function
      static SmallVector<int64_t> shapeFind(Type currType, int64_t axis) //if 2x3,axis=1 is given returns 
      {
        SmallVector<int64_t> newshape; // paramters=>inputshape(auto) and axis(int32)
        auto rankedType = cast<RankedTensorType>(currType);
        for (int64_t i = 0; i < rankedType.getRank(); ++i)
        {
          if (i == axis)
          {
            newshape.push_back(1); // TOSA keeps reduced dimension as size 1
          }
          else
          {
            newshape.push_back(rankedType.getDimSize(i));
          }
        }
        return newshape;
      }
      static SmallVector<int64_t> shapeFindargmax(Type currType, int64_t axis)
      {
        SmallVector<int64_t> newshape; // paramters=>inputshape(auto) and axis(int32)
        auto rankedType = cast<RankedTensorType>(currType);
        for (int64_t i = 0; i < rankedType.getRank(); ++i)
        {
          if (i == axis)
          {
            // newshape.push_back(1); // TOSA keeps reduced dimension as size 1
          }
          else
          {
            newshape.push_back(rankedType.getDimSize(i));
          }
        }
        return newshape;
      }

      static int64_t shapeFindforargmax(Type currType)
      {
        int64_t newshape = 1; // paramters=>inputshape(auto) and axis(int32)
        auto rankedType = cast<RankedTensorType>(currType);
        for (int64_t i = 0; i < rankedType.getRank(); ++i)
        {
          newshape *= rankedType.getDimSize(i);
        }
        return newshape;
      }
      template <typename OpTy>
      static Value maptop(OpTy op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return mappingtosa(op, resultType, input, builder);
      }

    private:
      template <typename OpTy>
      static Value mappingtosa(OpTy op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return nullptr;
      }
      static Value mappingtosa(nova::MaxOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {

        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

        return builder->create<tosa::MaximumOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::MinOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);
        return builder->create<tosa::MinimumOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::TransposeOp op,Type resultType,ValueRange input,OpBuilder *builder)
      {
  auto loc = op.getLoc();

auto inputType = dyn_cast<mlir::TensorType>(input[0].getType());
 auto resultTensorType =dyn_cast<mlir::TensorType>(resultType);
if (!inputType || !resultTensorType) {
    op.emitError("expected ranked tensor types");
    return nullptr;
}

int64_t rank = inputType.getRank();
auto inShape  = inputType.getShape();
auto outShape = resultTensorType.getShape();

// ---- derive permutation ----
llvm::SmallVector<int32_t> perms;
perms.reserve(rank);
llvm::SmallVector<bool> used(rank, false);

for (int64_t i = 0; i < rank; ++i) {
    bool found = false;
    for (int64_t j = 0; j < rank; ++j) {
        if (!used[j] && inShape[j] == outShape[i]) {
            perms.push_back(static_cast<int32_t>(j));
            used[j] = true;
            found = true;
            break;
        }
    }
}

// ---- create transpose ----
auto permsAttr = builder->getDenseI32ArrayAttr(perms);

return builder->create<tosa::TransposeOp>(
    loc, resultTensorType, input[0], permsAttr);

      }
      static Value mappingtosa(nova::AndOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);
        return builder->create<tosa::LogicalAndOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::OrOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

        return builder->create<tosa::LogicalOrOp>(op.getLoc(), resultType, v, w);
      }
        //log op
      static Value mappingtosa(nova::LogOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        //if complex type use complex.exp

        auto tensorTy = llvm::dyn_cast<TensorType>(input[0].getType());
        if(isa<ComplexType>(tensorTy.getElementType())){
          // Need to use linalg.generic to apply complex.exp element-wise
          auto loc = op.getLoc();
          auto resultTensorType = llvm::cast<RankedTensorType>(resultType);
          
          Value emptyTensor = builder->create<tensor::EmptyOp>(
              loc, resultTensorType.getShape(), resultTensorType.getElementType());
          
          auto identityMap = builder->getMultiDimIdentityMap(resultTensorType.getRank());
          SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
          SmallVector<utils::IteratorType> iteratorTypes(
              resultTensorType.getRank(), utils::IteratorType::parallel);
          
          auto genericOp = builder->create<linalg::GenericOp>(
              loc, TypeRange{resultType}, input[0], emptyTensor,
              indexingMaps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args[0] is complex<f32> (scalar)
                Value exp = b.create<complex::LogOp>(loc, args[0]);
                b.create<linalg::YieldOp>(loc, exp);
              });
          
          return genericOp.getResult(0);
        }
        //cast operation to result data type
        auto restensor= dyn_cast<TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

        return builder->create<tosa::LogOp>(op.getLoc(), resultType, v);
      }
      //exp op
      static Value mappingtosa(nova::ExpOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        //if complex type use complex.exp

        auto tensorTy = llvm::dyn_cast<TensorType>(input[0].getType());
        if(isa<ComplexType>(tensorTy.getElementType())){
          // Need to use linalg.generic to apply complex.exp element-wise
          auto loc = op.getLoc();
          auto resultTensorType = llvm::cast<RankedTensorType>(resultType);
          
          Value emptyTensor = builder->create<tensor::EmptyOp>(
              loc, resultTensorType.getShape(), resultTensorType.getElementType());
          
          auto identityMap = builder->getMultiDimIdentityMap(resultTensorType.getRank());
          SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
          SmallVector<utils::IteratorType> iteratorTypes(
              resultTensorType.getRank(), utils::IteratorType::parallel);
          
          auto genericOp = builder->create<linalg::GenericOp>(
              loc, TypeRange{resultType}, input[0], emptyTensor,
              indexingMaps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args[0] is complex<f32> (scalar)
                Value exp = b.create<complex::ExpOp>(loc, args[0]);
                b.create<linalg::YieldOp>(loc, exp);
              });
          
          return genericOp.getResult(0);
        }
        //cast operation to result data type
        auto restensor= dyn_cast<TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

        return builder->create<tosa::ExpOp>(op.getLoc(), resultType, v);
      }
      //abs op
      static Value mappingtosa(nova::AbsOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
    //if complex type use complex.abs 
        auto tensorTy = llvm::dyn_cast<TensorType>(input[0].getType());
        if(isa<ComplexType>(tensorTy.getElementType())){
          // Need to use linalg.generic to apply complex.abs element-wise
          auto loc = op.getLoc();
          auto resultTensorType = llvm::cast<RankedTensorType>(resultType);
          
          Value emptyTensor = builder->create<tensor::EmptyOp>(
              loc, resultTensorType.getShape(), resultTensorType.getElementType());
          
          auto identityMap = builder->getMultiDimIdentityMap(resultTensorType.getRank());
          SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
          SmallVector<utils::IteratorType> iteratorTypes(
              resultTensorType.getRank(), utils::IteratorType::parallel);
          
          auto genericOp = builder->create<linalg::GenericOp>(
              loc, TypeRange{resultType}, input[0], emptyTensor,
              indexingMaps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args[0] is complex<f32> (scalar)
                Value abs = b.create<complex::AbsOp>(loc, args[0]);
                b.create<linalg::YieldOp>(loc, abs);
              });
          
          return genericOp.getResult(0);
        }
        return builder->create<tosa::AbsOp>(op.getLoc(), resultType, input[0]);
      }
      static Value mappingtosa(nova::XorOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

        return builder->create<tosa::LogicalXorOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::NegOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto tensorTy = llvm::dyn_cast<TensorType>(input[0].getType());
        if (isa<IntegerType>(tensorTy.getElementType()) || isa<FloatType>(tensorTy.getElementType())) {
          return builder->create<tosa::NegateOp>(op.getLoc(), resultType, input[0]);
        }
        if (isa<ComplexType>(tensorTy.getElementType())) {
          // Need to use linalg.generic to apply complex.neg element-wise
          auto loc = op.getLoc();
          auto resultTensorType = llvm::cast<RankedTensorType>(resultType);
          
          Value emptyTensor = builder->create<tensor::EmptyOp>(
              loc, resultTensorType.getShape(), resultTensorType.getElementType());
          
          auto identityMap = builder->getMultiDimIdentityMap(resultTensorType.getRank());
          SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
          SmallVector<utils::IteratorType> iteratorTypes(
              resultTensorType.getRank(), utils::IteratorType::parallel);
          
          auto genericOp = builder->create<linalg::GenericOp>(
              loc, TypeRange{resultType}, input[0], emptyTensor,
              indexingMaps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args[0] is complex<f32> (scalar)
                Value neg = b.create<complex::NegOp>(loc, args[0]);
                b.create<linalg::YieldOp>(loc, neg);
              });
          
          return genericOp.getResult(0);
        }
        return nullptr;
      }
      //=================================
      //TRIGNOMENTARY
      //=================================
      //sin op
      static Value mappingtosa(nova::SinOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        //if complex type use complex.exp

        auto tensorTy = llvm::dyn_cast<TensorType>(input[0].getType());
        if(isa<ComplexType>(tensorTy.getElementType())){
          // Need to use linalg.generic to apply complex.exp element-wise
          auto loc = op.getLoc();
          auto resultTensorType = llvm::cast<RankedTensorType>(resultType);
          
          Value emptyTensor = builder->create<tensor::EmptyOp>(
              loc, resultTensorType.getShape(), resultTensorType.getElementType());
          
          auto identityMap = builder->getMultiDimIdentityMap(resultTensorType.getRank());
          SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
          SmallVector<utils::IteratorType> iteratorTypes(
              resultTensorType.getRank(), utils::IteratorType::parallel);
          
          auto genericOp = builder->create<linalg::GenericOp>(
              loc, TypeRange{resultType}, input[0], emptyTensor,
              indexingMaps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args[0] is complex<f32> (scalar)
                Value exp = b.create<complex::SinOp>(loc, args[0]);
                b.create<linalg::YieldOp>(loc, exp);
              });
          
          return genericOp.getResult(0);
        }
        //cast operation to result data type
        auto restensor= dyn_cast<TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

        return builder->create<tosa::SinOp>(op.getLoc(), resultType, v);
      }
      //cos op
      static Value mappingtosa(nova::CosOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        //if complex type use complex.exp

        auto tensorTy = llvm::dyn_cast<TensorType>(input[0].getType());
        if(isa<ComplexType>(tensorTy.getElementType())){
          // Need to use linalg.generic to apply complex.exp element-wise
          auto loc = op.getLoc();
          auto resultTensorType = llvm::cast<RankedTensorType>(resultType);
          
          Value emptyTensor = builder->create<tensor::EmptyOp>(
              loc, resultTensorType.getShape(), resultTensorType.getElementType());
          
          auto identityMap = builder->getMultiDimIdentityMap(resultTensorType.getRank());
          SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
          SmallVector<utils::IteratorType> iteratorTypes(
              resultTensorType.getRank(), utils::IteratorType::parallel);
          
          auto genericOp = builder->create<linalg::GenericOp>(
              loc, TypeRange{resultType}, input[0], emptyTensor,
              indexingMaps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args[0] is complex<f32> (scalar)
                Value exp = b.create<complex::CosOp>(loc, args[0]);
                b.create<linalg::YieldOp>(loc, exp);
              });
          
          return genericOp.getResult(0);
        }
        //cast operation to result data type
        auto restensor= dyn_cast<TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

        return builder->create<tosa::CosOp>(op.getLoc(), resultType, v);
      }
      //tanh
       static Value mappingtosa(nova::TanhOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        //if complex type use complex.exp

        auto tensorTy = llvm::dyn_cast<TensorType>(input[0].getType());
        if(isa<ComplexType>(tensorTy.getElementType())){
          // Need to use linalg.generic to apply complex.exp element-wise
          auto loc = op.getLoc();
          auto resultTensorType = llvm::cast<RankedTensorType>(resultType);
          
          Value emptyTensor = builder->create<tensor::EmptyOp>(
              loc, resultTensorType.getShape(), resultTensorType.getElementType());
          
          auto identityMap = builder->getMultiDimIdentityMap(resultTensorType.getRank());
          SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
          SmallVector<utils::IteratorType> iteratorTypes(
              resultTensorType.getRank(), utils::IteratorType::parallel);
          
          auto genericOp = builder->create<linalg::GenericOp>(
              loc, TypeRange{resultType}, input[0], emptyTensor,
              indexingMaps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args[0] is complex<f32> (scalar)
                Value exp = b.create<complex::TanhOp>(loc, args[0]);
                b.create<linalg::YieldOp>(loc, exp);
              });
          
          return genericOp.getResult(0);
        }
        //cast operation to result data type
        auto restensor= dyn_cast<TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);

        return builder->create<tosa::TanhOp>(op.getLoc(), resultType, v);
      }
      static Value mappingtosa(nova::ReciprocalOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return builder->create<tosa::ReciprocalOp>(op.getLoc(), resultType, input[0]);
      }
      static Value mappingtosa(nova::NotOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restype = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restype, input[0]);
        return builder->create<tosa::LogicalNotOp>(op.getLoc(), resultType, v);
      }
      static Value mappingtosa(nova::SigmoidOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return builder->create<tosa::SigmoidOp>(op.getLoc(), resultType, input[0]);
      }
      //-------------------------reduce case helper----------------------
      static Value mappincasereduce(nova::ReduceOp op, Type temresult, Value v, mlir::IntegerAttr axisAttr, OpBuilder *builder, mlir::StringAttr nanmode)
      {
        nova::ReductionKind rk = op.getKind();
        switch (rk)
        {
        case nova::ReductionKind::MAX:
          return builder->create<tosa::ReduceMaxOp>(op.getLoc(), temresult, v, axisAttr, nanmode);
        case nova::ReductionKind::MIN:
          return builder->create<tosa::ReduceMinOp>(op.getLoc(), temresult, v, axisAttr, nanmode);
        case nova::ReductionKind::PRODUCT:
          return builder->create<tosa::ReduceProductOp>(op.getLoc(), temresult, v, axisAttr);
        case nova::ReductionKind::SUM:
          return builder->create<tosa::ReduceSumOp>(op.getLoc(), temresult, v, axisAttr);

        case nova::ReductionKind::MEAN:
        {
          auto sum = builder->create<tosa::ReduceSumOp>(op.getLoc(), temresult, v, axisAttr);
          int64_t axis = axisAttr.getInt();
          auto inputType = cast<RankedTensorType>(v.getType());
          int64_t dimSize = inputType.getDimSize(axis);

          Value divisor;
          auto elementType = inputType.getElementType();

          if (inputType.isDynamicDim(axis))
          {
            Value dimVal = builder->create<tensor::DimOp>(op.getLoc(), v, axis);
            if (isa<FloatType>(elementType))
            {
              divisor = builder->create<mlir::arith::IndexCastOp>(op.getLoc(), builder->getI64Type(), dimVal);
              divisor = builder->create<mlir::arith::UIToFPOp>(op.getLoc(), elementType, divisor);
            }
            else
            {
              divisor = builder->create<mlir::arith::IndexCastOp>(op.getLoc(), elementType, dimVal);
            }
          }
          else
          {
            if (isa<FloatType>(elementType))
            {
              divisor = builder->create<tosa::ConstOp>(op.getLoc(),
                                                       RankedTensorType::get({}, elementType),
                                                       DenseElementsAttr::get(RankedTensorType::get({}, elementType),
                                                                              builder->getFloatAttr(elementType, static_cast<double>(dimSize))));
            }
            else
            {
              divisor = builder->create<tosa::ConstOp>(op.getLoc(),
                                                       RankedTensorType::get({}, elementType),
                                                       DenseElementsAttr::get(RankedTensorType::get({}, elementType),
                                                                              builder->getIntegerAttr(elementType, dimSize)));
            }
          }

          // Reshape divisor to match rank of sum for broadcasting
          auto resultType = cast<RankedTensorType>(temresult);
          int64_t rank = resultType.getRank();
          SmallVector<int64_t> newShape(rank, 1);

          auto shapeType = RankedTensorType::get({rank}, builder->getIndexType());
          auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
          auto shapeConst = builder->create<tosa::ConstShapeOp>(op.getLoc(),
                                                                mlir::tosa::shapeType::get(builder->getContext(), rank),
                                                                shapeAttr);

          auto reshapedDivisorType = RankedTensorType::get(newShape, elementType);
          auto reshapedDivisor = builder->create<tosa::ReshapeOp>(op.getLoc(), reshapedDivisorType, divisor, shapeConst);

          if (isa<FloatType>(elementType))
          {
            auto reciprocal = builder->create<tosa::ReciprocalOp>(op.getLoc(), reshapedDivisorType, reshapedDivisor);
            auto shift = builder->create<tosa::ConstOp>(op.getLoc(),
                                                        RankedTensorType::get({1}, builder->getI8Type()),
                                                        DenseElementsAttr::get(RankedTensorType::get({1}, builder->getI8Type()),
                                                                               builder->getI8IntegerAttr(0)));
            return builder->create<tosa::MulOp>(op.getLoc(), temresult, sum, reciprocal, shift);
          }
          else
          {
            return builder->create<tosa::IntDivOp>(op.getLoc(), temresult, sum, reshapedDivisor);
          }
        }
        case nova::ReductionKind::ALL:
          return builder->create<tosa::ReduceAllOp>(op.getLoc(), temresult, v, axisAttr);
        case nova::ReductionKind::ANY:
          return builder->create<tosa::ReduceAnyOp>(op.getLoc(), temresult, v, axisAttr);
        }
        return nullptr;
      }

      //-------------------------Reduce-ArgMax---------------------------

      static Value mappingtosa(nova::ArgmaxOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        Value v = input[0]; // the final result is store in this
        auto inputType = dyn_cast<RankedTensorType>(v.getType());
        auto resultdt = dyn_cast<RankedTensorType>(resultType);
        auto ignorenanAttr = op.getIgnoreNan();
        mlir::StringAttr nanmode = builder->getStringAttr("PROPAGATE");

        if (ignorenanAttr)
        {
          nanmode = builder->getStringAttr("IGNORE");
        }
        // getting dimension
        auto dimensionAttr = op.getDimension();
        // reducing with dimesion-----------
        if (dimensionAttr.has_value())
        {
          int64_t axisValue = dimensionAttr.value();
          // getting value for axis attribute
          auto axisAttr = builder->getI32IntegerAttr(axisValue);
          // cretaing result tensor
          auto tempshape = shapeFindargmax(inputType, axisValue);
          auto temptype = RankedTensorType::get(tempshape, resultdt.getElementType());

          // we have to replac the resulttype
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), temptype, v, axisAttr, nanmode);
          //  op.emitOpError("dimension attribute missing for TOSA mapping");
        }
        // No dimension - reduce all dimension
        else
        {
          auto finalShape = shapeFindforargmax(v.getType());
          // Create the final result type
          auto finalType = RankedTensorType::get({}, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {1}, builder->getIndexType());
          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              finalShape);
          // flatten it and
          //  Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1), shapeAttr);
          // Perform reshape
          Value reshapedres = builder->create<tosa::ReshapeOp>(
              op.getLoc(), v, shapeValue);
          auto axisAttr = builder->getI32IntegerAttr(0);
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), finalType, reshapedres, axisAttr, nanmode);
        }
        // KEEP DIMS
        if (op.getKeepdims())
        {
          auto finalShape = resultdt.getShape();

          // Create the final result type
          auto finalType = RankedTensorType::get(finalShape, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {static_cast<int64_t>(finalShape.size())}, builder->getIndexType());

          SmallVector<int64_t> shapeValues(finalShape.begin(), finalShape.end());

          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              llvm::ArrayRef(shapeValues.data(), shapeValues.size()));

          // Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), finalShape.size()), shapeAttr);

          // Perform reshape
          v = builder->create<tosa::ReshapeOp>(
              op.getLoc(), finalType, v, shapeValue);
        }

        return v;
      }
      //----------------------------------------ARGMIN-----------------------------

      static Value mappingtosa(nova::ArgMinOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        Value v = input[0]; // the final result is store in this
        auto inputType = dyn_cast<RankedTensorType>(v.getType());
        auto resultdt = dyn_cast<RankedTensorType>(resultType);
        auto ignorenanAttr = op.getIgnoreNan();
        mlir::StringAttr nanmode = builder->getStringAttr("PROPAGATE");

        if (ignorenanAttr)
        {
          nanmode = builder->getStringAttr("IGNORE");
        }
        v = builder->create<tosa::NegateOp>(op.getLoc(), v.getType(), v);

        // getting dimension
        auto dimensionAttr = op.getDimension();
        // reducing with dimesion-----------
        if (dimensionAttr.has_value())
        {
          int64_t axisValue = dimensionAttr.value();
          // getting value for axis attribute
          auto axisAttr = builder->getI32IntegerAttr(axisValue);
          // cretaing result tensor
          auto tempshape = shapeFindargmax(inputType, axisValue);
          auto temptype = RankedTensorType::get(tempshape, resultdt.getElementType());

          // we have to replac the resulttype
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), temptype, v, axisAttr, nanmode);
          //  op.emitOpError("dimension attribute missing for TOSA mapping");
        }
        // No dimension - reduce all dimension
        else
        {
          auto finalShape = shapeFindforargmax(v.getType());
          // Create the final result type
          auto finalType = RankedTensorType::get({}, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {1}, builder->getIndexType());
          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              finalShape);
          // flatten it and
          //  Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1), shapeAttr);
          // Perform reshape
          Value reshapedres = builder->create<tosa::ReshapeOp>(
              op.getLoc(), v, shapeValue);
          auto axisAttr = builder->getI32IntegerAttr(0);
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), finalType, reshapedres, axisAttr, nanmode);
        }
        // KEEP DIMS
        if (op.getKeepdims())
        {
          auto finalShape = resultdt.getShape();

          // Create the final result type
          auto finalType = RankedTensorType::get(finalShape, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {static_cast<int64_t>(finalShape.size())}, builder->getIndexType());

          SmallVector<int64_t> shapeValues(finalShape.begin(), finalShape.end());

          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              llvm::ArrayRef(shapeValues.data(), shapeValues.size()));

          // Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), finalShape.size()), shapeAttr);
          // Perform reshape
          v = builder->create<tosa::ReshapeOp>(
              op.getLoc(), finalType, v, shapeValue);
        }

        return v;
      }
      //---------------------------Reduce-Op---------------------------------

      static Value mappingtosa(nova::ReduceOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        Value v = input[0]; // the final result is store in this
        // getting the axis from dimension
        auto dimensionAttr = op.getDimension();
        auto inputType = dyn_cast<RankedTensorType>(v.getType());
        auto result1Type = dyn_cast<RankedTensorType>(resultType);
        // setting ignore nan attribute to nan mode
        auto ignorenanAttr = op.getIgnoreNan();
        mlir::StringAttr nanmode = builder->getStringAttr("PROPAGATE");

        if (ignorenanAttr)
        {
          nanmode = builder->getStringAttr("IGNORE");
        }

        // 🪻👍🏻🪻
        // reducing with dimesion-----------
        if (dimensionAttr.has_value())
        {
          auto dimension = dimensionAttr.value();
          for (auto dim : dimension)
          {
            // getting value for axis attribute
            int64_t axisValue = dyn_cast<IntegerAttr>(dim).getInt();
            auto axisAttr = builder->getI32IntegerAttr(axisValue);
            // getting temp shape
            auto tempshape = shapeFind(v.getType(), axisValue); // placeholder for now
            auto currType = cast<RankedTensorType>(v.getType());
            auto tempresult = RankedTensorType::get(tempshape, currType.getElementType());
            // getting the correct operation
            v = mappincasereduce(op, tempresult, v, axisAttr, builder, nanmode);
          }
        }
        // No dimension - reduce all dimension
        else
        {
          auto inputRank = inputType.getRank();
          for (int64_t axis = inputRank - 1; axis >= 0; --axis)
          {
            auto axisAttr = builder->getI32IntegerAttr(axis);
            auto tempShape = shapeFind(v.getType(), axis);
            auto currType = cast<RankedTensorType>(v.getType());
            auto tempresult = RankedTensorType::get(tempShape, currType.getElementType());
            v = mappincasereduce(op, tempresult, v, axisAttr, builder, nanmode);
          }
        }
        // NEED TO ADD KEEP DIMS HERE
        if (!op.getKeepdims())
        {
          auto currentType = cast<RankedTensorType>(v.getType());
          auto finalShape = result1Type.getShape();

          // Create the final result type
          auto finalType = RankedTensorType::get(finalShape, currentType.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {static_cast<int64_t>(finalShape.size())}, builder->getIndexType());

          SmallVector<int64_t> shapeValues(finalShape.begin(), finalShape.end());

          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              llvm::ArrayRef(shapeValues.data(), shapeValues.size()));

          // Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), finalShape.size()), shapeAttr);

          // Perform reshape
          v = builder->create<tosa::ReshapeOp>(
              op.getLoc(), finalType, v, shapeValue);
        }

        return v;
      }
    };
    // pattern to convert nova.gelu to seauence of operations
    /// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    struct NovaGeluOpLowering : public OpConversionPattern<GeluOp>
    {
      using OpConversionPattern<GeluOp>::OpConversionPattern;
      LogicalResult matchAndRewrite(GeluOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override
      {
        Location loc = op.getLoc();
        Value input = adaptor.getLhs();
        // if input is integer, cast to float and update type for following ops
        auto inputType = cast<RankedTensorType>(input.getType());
        if (isa<IntegerType>(inputType.getElementType())) {
          auto newInputType = RankedTensorType::get(inputType.getShape(), rewriter.getF32Type());
          input = rewriter.create<tosa::CastOp>(loc, newInputType, input);
          inputType = newInputType;
        }
        // op0 = pow(x, 3)
        Value cst_3 = rewriter.create<nova::ConstantOp>(
            loc, inputType, DenseElementsAttr::get(inputType, {3.0f}));
        auto op0 =
            rewriter.create<tosa::PowOp>(loc, inputType, input, cst_3);
        // op1 = mul(op0, 0.044715)
        Value cst_004 = rewriter.create<nova::ConstantOp>(
            loc, inputType, DenseElementsAttr::get(inputType, {4.471500e-02f}));
        auto op1 =
            rewriter.create<nova::MulOp>(loc, inputType, op0.getResult(),
                                         cst_004);
        // op2 = add(x, op1)
        auto op2 = rewriter.create<tosa::AddOp>(loc, inputType,
                                                input, op1);
        // op3 = mul(op2, sqrt(2/pi))
        Value cst_sqrt2pi = rewriter.create<nova::ConstantOp>(
            loc, inputType, DenseElementsAttr::get(inputType, {0.797884583f}));
        auto op3 =
            rewriter.create<nova::MulOp>(loc, inputType, op2,
                                         cst_sqrt2pi);
        // op4 = tanh(op3)
        auto op4 = rewriter.create<tosa::TanhOp>(loc, inputType,
                                                 op3);
        // op5 = add(op4 ,1)
        Value cst_1 = rewriter.create<nova::ConstantOp>(
            loc, inputType, DenseElementsAttr::get(inputType, {1.0f}));
        auto op5 = rewriter.create<tosa::AddOp>(loc, inputType,
                                                op4, cst_1);
        // op6 = mul(x, 0.5)
        Value cst_05 = rewriter.create<nova::ConstantOp>(
            loc, inputType, DenseElementsAttr::get(inputType, {0.5f}));
        auto op6 = rewriter.create<nova::MulOp>(
            loc, inputType, input, cst_05);

        auto op7 = rewriter.create<nova::MulOp>(loc, inputType, op6, op5);

        rewriter.replaceOp(op, {op7.getResult()});

        return success();
      }
    };
    // Pattern to convert nova.relu to tosa.relu
    struct NovaReluOpLowering : public OpConversionPattern<ReluOp>
    {
      using OpConversionPattern<ReluOp>::OpConversionPattern;

      LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const override
      {
        Location loc = op.getLoc();
        Value input = adaptor.getInput();
        auto inputType = cast<RankedTensorType>(input.getType());
        Type elementType = inputType.getElementType();

        // Create zero constant tensor with the same shape as input
        Attribute zeroAttr;

        if (auto floatType = dyn_cast<FloatType>(elementType))
        {
          APFloat zeroVal = APFloat::getZero(floatType.getFloatSemantics());
          zeroAttr = rewriter.getFloatAttr(floatType, zeroVal);
        }
        else if (auto intType = dyn_cast<IntegerType>(elementType))
        {
          zeroAttr = rewriter.getIntegerAttr(intType, 0);
        }
        else
        {
          return failure();
        }
        DenseElementsAttr zeroTensor = DenseElementsAttr::get(inputType, zeroAttr);
        Value zero = rewriter.create<nova::ConstantOp>(loc, inputType, zeroTensor);
        Value result = rewriter.create<tosa::MaximumOp>(
            loc, inputType, input, zero);

        rewriter.replaceOp(op, result);
        return success();
      }
    };
    //creating a  lowering for softmax
    struct NovaSoftmaxLoweringPattern:public OpConversionPattern<SoftmaxOp>
    {
      using OpConversionPattern<SoftmaxOp>::OpConversionPattern;
      LogicalResult matchAndRewrite (SoftmaxOp op,OpAdaptor adaptor,ConversionPatternRewriter &rewriter)const override{

        Location loc=op.getLoc();
        Value input=adaptor.getInput();
        auto restype=cast<RankedTensorType>(op.getType());
        auto inputType=cast<RankedTensorType>(input.getType());
        auto size=inputType.getShape().size();
        int32_t dimension=op.getDimension().has_value()?op.getDimension().value():-1;
        if(dimension<0){
        //  auto size1=inputType.getShape().size() - 1;
          dimension+=size;
        }
        SmallVector<int64_t> dim;
        dim.push_back(dimension);

        bool keepdims=true;
        nova::ReductionKind kind= nova::ReductionKind::MAX;
        auto shape=NovaOpTosaOp::shapeFind(inputType,dimension);
        auto tempresult = RankedTensorType::get(shape, restype.getElementType());
        //creating cast - only if element types differ
        if (inputType.getElementType() != restype.getElementType()) {
          input=rewriter.create<tosa::CastOp>(loc,restype,input);
        }

        auto axisAttr = rewriter.getI32IntegerAttr(dimension);
        Value op1=rewriter.create<tosa::ReduceMaxOp>(loc,tempresult,input,axisAttr);

        //step2  
        //create a TOSA sub op with input and op1
        Value op2=rewriter.create<tosa::SubOp>(loc,restype,input,op1);
        //step3
        //create  a TOSA exp op
        Value op3=rewriter.create<tosa::ExpOp>(loc,restype,op2);
        //step4
        //create a TOSA reduce sum
        Value op4=rewriter.create<tosa::ReduceSumOp>(loc,tempresult,op3,axisAttr);
        
        //step 5  
        //Explicitly broadcast op4 to match op3's shape for division
        //op3 is tensor<8x128x128xf32>, op4 is tensor<8x128x1xf32>
        //We need to tile op4 along dimension 2 to broadcast it
        SmallVector<int64_t> multiples;
        auto op3Shape = cast<RankedTensorType>(op3.getType()).getShape();
        auto op4Shape = cast<RankedTensorType>(op4.getType()).getShape();
        for (size_t i = 0; i < op3Shape.size(); ++i) {
          if (i < op4Shape.size()) {
            multiples.push_back(op3Shape[i] / op4Shape[i]);
          } else {
            multiples.push_back(1);
          }
        }
        
        auto shapeType = RankedTensorType::get({static_cast<int64_t>(multiples.size())}, 
                                                rewriter.getIndexType());
        auto shapeAttr = DenseIntElementsAttr::get(shapeType, multiples);
        Value multiplesConst = rewriter.create<tosa::ConstShapeOp>(
            loc,
            mlir::tosa::shapeType::get(rewriter.getContext(), multiples.size()),
            shapeAttr);
        
        Value op4_broadcast = rewriter.create<tosa::TileOp>(
            loc, restype, op4, multiplesConst);
        
        //create TOSA div: reciprocal(op4_broadcast) * op3
        Value recip = rewriter.create<tosa::ReciprocalOp>(loc,restype,op4_broadcast);
        auto shift = rewriter.create<mlir::arith::ConstantOp>(
            loc,
            DenseElementsAttr::get(RankedTensorType::get({1}, rewriter.getI8Type()),
                                  rewriter.getI8IntegerAttr(0)));
        Value op5 = rewriter.create<tosa::MulOp>(loc,restype,op3,recip,shift);
        rewriter.replaceOp(op,op5);


        return success();

      }
    };


    //-----------------------------------------------------------------------------
    // Matmul lowering
    //-----------------------------------------------------------------------------
    
    struct NovaMatmulOpTosaLowering : public OpConversionPattern<MatmulOp>
    {
      using OpConversionPattern<MatmulOp>::OpConversionPattern;
      
      // Helper function to create a tosa.ConstShapeOp
      Value createConstShapeOp(ConversionPatternRewriter &rewriter, Location loc,
                               ArrayRef<int64_t> shape) const {
        auto shapeType = RankedTensorType::get({static_cast<int64_t>(shape.size())}, 
                                                rewriter.getIndexType());
        auto shapeAttr = DenseIntElementsAttr::get(shapeType, shape);
        return rewriter.create<tosa::ConstShapeOp>(
            loc, 
            mlir::tosa::shapeType::get(rewriter.getContext(), shape.size()), 
            shapeAttr);
      }


      // Helper function to broadcast input to match target batch shape
      Value broadcastToShape(Value input, ArrayRef<int64_t> targetBatchShape,
                             ConversionPatternRewriter &rewriter, Location loc) const {
        auto inputType = cast<RankedTensorType>(input.getType());
        auto inputShape = inputType.getShape();
        int64_t inputRank = inputType.getRank();
        int64_t targetBatchRank = targetBatchShape.size();
        int64_t inputBatchRank = inputRank - 2;

        // Reshape to align ranks 
        SmallVector<int64_t> reshapedShape;
        int64_t rankDiff = targetBatchRank - inputBatchRank;
        
        if (rankDiff > 0) {
          for (int64_t i = 0; i < rankDiff; ++i) reshapedShape.push_back(1);
        }
        for (int64_t dim : inputShape) reshapedShape.push_back(dim);

        Value currentVal = input;
        if (rankDiff > 0) {
          auto reshapedType = RankedTensorType::get(reshapedShape, inputType.getElementType());
          Value shapeConst = createConstShapeOp(rewriter, loc, reshapedShape);
          currentVal = rewriter.create<tosa::ReshapeOp>(loc, reshapedType, input, shapeConst);
        }

        // Tile to broadcast dimensions
        SmallVector<int64_t> multiples;
        bool needsTiling = false;
        
        for (int64_t i = 0; i < targetBatchRank; ++i) {
          int64_t inputDim = reshapedShape[i];
          int64_t targetDim = targetBatchShape[i];
          
          if (inputDim == 1 && targetDim > 1) {
            multiples.push_back(targetDim);
            needsTiling = true;
          } else {
            multiples.push_back(1);
          }
        }
        // Matrix dims
        multiples.push_back(1);
        multiples.push_back(1);

        if (needsTiling) {
          Value multiplesConst = createConstShapeOp(rewriter, loc, multiples);
          
          SmallVector<int64_t> tiledShape;
          for (int64_t dim : targetBatchShape) tiledShape.push_back(dim);
          tiledShape.push_back(reshapedShape[reshapedShape.size()-2]);
          tiledShape.push_back(reshapedShape[reshapedShape.size()-1]);
          
          auto tiledType = RankedTensorType::get(tiledShape, inputType.getElementType());
          currentVal = rewriter.create<tosa::TileOp>(loc, tiledType, currentVal, multiplesConst);
        }
        
        return currentVal;
      }
      
      LogicalResult matchAndRewrite(MatmulOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override
      {
        auto operands = adaptor.getOperands();

        if (operands.size() != 2)
        {
          return rewriter.notifyMatchFailure(op, "expected exactly 2 operands");
        }

        Value lhs = operands[0];
        Value rhs = operands[1];

        auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
        auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());

        if (!lhsType || !rhsType || !resultType)
        {
          return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
        }

        Location loc = op.getLoc();
        int64_t resultRank = resultType.getRank();

        // Case 1: 2D matmul - reshape to 3D, do matmul, reshape back to 2D
        if (resultRank == 2)
        {
          // [M, K] x [K, N] -> [M, N]
          // Reshape to [1, M, K] x [1, K, N] -> [1, M, N]
          SmallVector<int64_t> lhs3DShape = {1, lhsType.getShape()[0], lhsType.getShape()[1]};
          SmallVector<int64_t> rhs3DShape = {1, rhsType.getShape()[0], rhsType.getShape()[1]};
          SmallVector<int64_t> result3DShape = {1, resultType.getShape()[0], resultType.getShape()[1]};

          auto lhs3DType = RankedTensorType::get(lhs3DShape, lhsType.getElementType());
          auto rhs3DType = RankedTensorType::get(rhs3DShape, rhsType.getElementType());
          auto result3DType = RankedTensorType::get(result3DShape, resultType.getElementType());

          Value lhs3DShapeValue = createConstShapeOp(rewriter, loc, lhs3DShape);
          Value rhs3DShapeValue = createConstShapeOp(rewriter, loc, rhs3DShape);
          SmallVector<int64_t> result2DShapeVec(resultType.getShape().begin(), resultType.getShape().end());
          Value result2DShapeValue = createConstShapeOp(rewriter, loc, result2DShapeVec);

          Value lhs3D = rewriter.create<tosa::ReshapeOp>(loc, lhs3DType, lhs, lhs3DShapeValue);
          Value rhs3D = rewriter.create<tosa::ReshapeOp>(loc, rhs3DType, rhs, rhs3DShapeValue);

          Value matmul3D = rewriter.create<tosa::MatMulOp>(loc, result3DType, lhs3D, rhs3D);

          rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, resultType, matmul3D, result2DShapeValue);
          return success();
        }

        // Case 2: 3D matmul - may need broadcasting
        if (resultRank == 3)
        {
          int64_t batchDim = resultType.getShape()[0];

          // Broadcast lhs if needed
          if (lhsType.getRank() == 2) {
            // [M, K] -> [1, M, K] -> [B, M, K]
            // Step 1: Reshape to add batch dimension
            SmallVector<int64_t> lhs3DShapeWith1 = {1, lhsType.getShape()[0], lhsType.getShape()[1]};
            auto lhs3DTypeWith1 = RankedTensorType::get(lhs3DShapeWith1, lhsType.getElementType());
            Value lhs3DShapeWith1Value = createConstShapeOp(rewriter, loc, lhs3DShapeWith1);
            Value lhsReshaped = rewriter.create<tosa::ReshapeOp>(loc, lhs3DTypeWith1, lhs, lhs3DShapeWith1Value);
            
            // Step 2: Tile to broadcast if batch > 1
            if (batchDim > 1) {
              SmallVector<int64_t> multiples = {static_cast<int64_t>(batchDim), 1, 1};
              // Create const for multiples 
              Value multiplesValue = createConstShapeOp(rewriter, loc, multiples);

              SmallVector<int64_t> lhsBroadcastShape = {batchDim, lhsType.getShape()[0], lhsType.getShape()[1]};
              auto lhsBroadcastType = RankedTensorType::get(lhsBroadcastShape, lhsType.getElementType());
              lhs = rewriter.create<tosa::TileOp>(loc, lhsBroadcastType, lhsReshaped, multiplesValue);
            } else {
              lhs = lhsReshaped;
            }
          } else if (lhsType.getRank() == 3 && lhsType.getShape()[0] == 1 && batchDim > 1) {
            // [1, M, K] -> [B, M, K] - just tile
            SmallVector<int64_t> multiples = {batchDim, 1, 1};
            Value multiplesValue = createConstShapeOp(rewriter, loc, multiples);
            
            SmallVector<int64_t> lhsBroadcastShape = {batchDim, lhsType.getShape()[1], lhsType.getShape()[2]};
            auto lhsBroadcastType = RankedTensorType::get(lhsBroadcastShape, lhsType.getElementType());
            lhs = rewriter.create<tosa::TileOp>(loc, lhsBroadcastType, lhs, multiplesValue);
          }

          // Broadcast rhs if needed
          if (rhsType.getRank() == 2) {
            // [K, N] -> [1, K, N] -> [B, K, N]
            // Step 1: Reshape to add batch dimension
            SmallVector<int64_t> rhs3DShapeWith1 = {1, rhsType.getShape()[0], rhsType.getShape()[1]};
            auto rhs3DTypeWith1 = RankedTensorType::get(rhs3DShapeWith1, rhsType.getElementType());
            Value rhs3DShapeWith1Value = createConstShapeOp(rewriter, loc, rhs3DShapeWith1);
            Value rhsReshaped = rewriter.create<tosa::ReshapeOp>(loc, rhs3DTypeWith1, rhs, rhs3DShapeWith1Value);
            
            // Step 2: Tile to broadcast if batch > 1
            if (batchDim > 1) {
              SmallVector<int64_t> multiples = {static_cast<int64_t>(batchDim), 1, 1};
              Value multiplesValue = createConstShapeOp(rewriter, loc, multiples);
              
              SmallVector<int64_t> rhsBroadcastShape = {batchDim, rhsType.getShape()[0], rhsType.getShape()[1]};
              auto rhsBroadcastType = RankedTensorType::get(rhsBroadcastShape, rhsType.getElementType());
              rhs = rewriter.create<tosa::TileOp>(loc, rhsBroadcastType, rhsReshaped, multiplesValue);
            } else {
              rhs = rhsReshaped;
            }
          } else if (rhsType.getRank() == 3 && rhsType.getShape()[0] == 1 && batchDim > 1) {
            // [1, K, N] -> [B, K, N] - just tile
            SmallVector<int64_t> multiples = {batchDim, 1, 1};
            Value multiplesValue = createConstShapeOp(rewriter, loc, multiples);
           
            SmallVector<int64_t> rhsBroadcastShape = {batchDim, rhsType.getShape()[1], rhsType.getShape()[2]};
            auto rhsBroadcastType = RankedTensorType::get(rhsBroadcastShape, rhsType.getElementType());
            rhs = rewriter.create<tosa::TileOp>(loc, rhsBroadcastType, rhs, multiplesValue);
          }

          rewriter.replaceOpWithNewOp<tosa::MatMulOp>(op, resultType, lhs, rhs);
          return success();
        }

        // Case 3: rank > 3 - flatten batch dimensions, do matmul, reshape back

        // Broadcast inputs to match result batch dimensions
        SmallVector<int64_t> targetBatchShape;
        for (int64_t i = 0; i < resultRank - 2; ++i) {
          targetBatchShape.push_back(resultType.getShape()[i]);
        }
        
        Value broadcastLhs = broadcastToShape(lhs, targetBatchShape, rewriter, loc);
        Value broadcastRhs = broadcastToShape(rhs, targetBatchShape, rewriter, loc);
        
        // Update types after broadcasting
        auto broadcastLhsType = cast<RankedTensorType>(broadcastLhs.getType());
        //auto broadcastRhsType = cast<RankedTensorType>(broadcastRhs.getType());

        // Calculate the flattened batch size
        int64_t N = 1;
        for (int64_t dim : targetBatchShape) {
          N *= dim;
        }

        int64_t M = resultType.getShape()[resultRank - 2];
        int64_t K = broadcastLhsType.getShape()[broadcastLhsType.getRank() - 1];
        int64_t N_cols = resultType.getShape()[resultRank - 1];

        SmallVector<int64_t> rank3_lhs_shape({N, M, K});
        SmallVector<int64_t> rank3_rhs_shape({N, K, N_cols});
        SmallVector<int64_t> rank3_output_shape({N, M, N_cols});

        auto rank3LhsType = RankedTensorType::get(rank3_lhs_shape, lhsType.getElementType());
        auto rank3RhsType = RankedTensorType::get(rank3_rhs_shape, rhsType.getElementType());
        auto rank3OutputType = RankedTensorType::get(rank3_output_shape, resultType.getElementType());

        Value rank3LhsShapeValue = createConstShapeOp(rewriter, loc, rank3_lhs_shape);
        Value rank3RhsShapeValue = createConstShapeOp(rewriter, loc, rank3_rhs_shape);
        SmallVector<int64_t> resultShapeVec(resultType.getShape().begin(), resultType.getShape().end());
        Value resultShapeValue = createConstShapeOp(rewriter, loc, resultShapeVec);

        Value lhsReshaped = rewriter.create<tosa::ReshapeOp>(loc, rank3LhsType, broadcastLhs, rank3LhsShapeValue);
        Value rhsReshaped = rewriter.create<tosa::ReshapeOp>(loc, rank3RhsType, broadcastRhs, rank3RhsShapeValue);

        Value matmul = rewriter.create<tosa::MatMulOp>(loc, rank3OutputType, lhsReshaped, rhsReshaped);

        rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, resultType, matmul, resultShapeValue);
        return success();
      }
    };
    
    
    
    // creating a template
    template <typename NovaTopTy>
    class NovaToTosaLoweringTemplate : public OpConversionPattern<NovaTopTy>
    {
    public:
      using OpConversionPattern<NovaTopTy>::OpConversionPattern;
      using OpAdaptor = typename NovaTopTy::Adaptor; // for getting all meta data dynamically using adaptor
      LogicalResult matchAndRewrite(NovaTopTy op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override
      {
        ValueRange operands = adaptor.getOperands();
        // checking operand is empty or not
        if (operands.empty())
          return rewriter.notifyMatchFailure(op, "expected operands for tosa lowering operations");
        // getting resultType
        auto resultType = op.getResult().getType();
        Value result = NovaOpTosaOp::maptop(
            op, resultType, operands, &rewriter);
        if (!result)
          return rewriter.notifyMatchFailure(op, "failed to map to TOSA operation");

        rewriter.replaceOp(op, result);
        return success();
      }
    };

    // pass definition
    namespace
    {
      struct NovaToTosaLoweringPass
          : public PassWrapper<NovaToTosaLoweringPass, OperationPass<ModuleOp>>
      {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToTosaLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override
        {
          registry.insert<tosa::TosaDialect>();
          registry.insert<func::FuncDialect>();
          registry.insert<nova::NovaDialect>();
        }

        StringRef getArgument() const final { return "convert-nova-to-tosa"; }

        StringRef getDescription() const final
        {
          return "Lower Nova dialect operations to Tosa dialect";
        }

        void runOnOperation() override
        {
          ModuleOp module = getOperation();
          ConversionTarget target(getContext());

          target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
          target.addLegalOp<nova::ConstantOp>();
          target.addIllegalOp<nova::ReluOp>();
          target.addIllegalOp<nova::ExpOp>();
          target.addIllegalOp<nova::LogOp>();
          target.addIllegalOp<nova::AbsOp>();
          target.addIllegalOp<nova::MaxOp>();
          target.addIllegalOp<nova::MinOp>();
          target.addIllegalOp<nova::SubOp>();
          target.addIllegalOp<nova::MulOp>();
          target.addIllegalOp<nova::PowOp>();
          target.addIllegalOp<nova::SqrtOp>();
          target.addIllegalOp<nova::AndOp>();
          target.addIllegalOp<nova::OrOp>();
          target.addIllegalOp<nova::XorOp>();
          target.addIllegalOp<nova::NegOp>();
          target.addIllegalOp<nova::NotOp>();
          target.addIllegalOp<nova::SinOp>();
          target.addIllegalOp<nova::CosOp>();
          target.addIllegalOp<nova::TanhOp>();
          target.addIllegalOp<nova::ReciprocalOp>();
          target.addIllegalOp<nova::ReduceOp>();
          target.addIllegalOp<nova::ArgmaxOp>();
          target.addIllegalOp<nova::ArgMinOp>();
          target.addIllegalOp<nova::SigmoidOp>();
          target.addIllegalOp<nova::GeluOp>();
          target.addIllegalOp<nova::SoftmaxOp>();
          target.addIllegalOp<nova::MatmulOp>();
          target.addIllegalOp<nova::AddOp>();
          target.addIllegalOp<nova::ConstantOp>();
          target.addIllegalOp<nova::TransposeOp>();
          target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
          TypeConverter typeConverter;
          typeConverter.addConversion([](Type type)
                                      { return type; });
          RewritePatternSet patterns(&getContext());
          populateNovaToTosaConversionPatterns(patterns);
          populateNovaToTosaTemplatePatterns(patterns);
          populateNovaToArithConversionPatterns(patterns);
       //   populateNovaToLinalgPatterns(patterns);
         // populateNovaToLinalgPatternsTemplate(patterns);


          if (failed(applyPartialConversion(module, target, std::move(patterns))))
          {
            signalPassFailure();
            return;
          }
        }
      };

    }

    void populateNovaToTosaConversionPatterns(RewritePatternSet &patterns)
    {
      patterns.add<NovaReluOpLowering, 
                   NovaGeluOpLowering,
                   NovaSoftmaxLoweringPattern,
                   NovaMatmulOpTosaLowering,
                   NovaToTosaLoweringTemplate<nova::MaxOp>,
                   NovaToTosaLoweringTemplate<nova::LogOp>,
                   NovaToTosaLoweringTemplate<nova::AbsOp>,
                   NovaToTosaLoweringTemplate<nova::ExpOp>,
                   NovaToTosaLoweringTemplate<nova::MinOp>,
                   NovaToTosaLoweringTemplate<nova::AndOp>,
                   NovaToTosaLoweringTemplate<nova::SinOp>,
                   NovaToTosaLoweringTemplate<nova::CosOp>,
                   NovaToTosaLoweringTemplate<nova::TanhOp>,
                   NovaToTosaLoweringTemplate<nova::OrOp>,
                   NovaToTosaLoweringTemplate<nova::XorOp>,
                   NovaToTosaLoweringTemplate<nova::NotOp>,
                   NovaToTosaLoweringTemplate<nova::NegOp>,
                   NovaToTosaLoweringTemplate<nova::TransposeOp>,
                   NovaToTosaLoweringTemplate<nova::ReciprocalOp>,
                   NovaToTosaLoweringTemplate<nova::ReduceOp>,
                   NovaToTosaLoweringTemplate<nova::ArgmaxOp>,
                   NovaToTosaLoweringTemplate<nova::ArgMinOp>,
                   NovaToTosaLoweringTemplate<nova::SigmoidOp>>(
          patterns.getContext());
    }

    // creating a pointer for this pass
    std::unique_ptr<Pass> createNovaToTosaLoweringPass()
    {
      return std::make_unique<NovaToTosaLoweringPass>();
    }

    // Register the pass
    void registerNovaToTosaLoweringPass()
    {
      PassRegistration<NovaToTosaLoweringPass>();
    }

  } // namespace nova
} // namespace mlir