//===----------------------------------------------------------------------===//
// NovaCanonicalizations.cpp - Canonicalization patterns for Nova dialect
//===----------------------------------------------------------------------===//

#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/Broadcast.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::nova;

namespace {

//===----------------------------------------------------------------------===//
// Broadcast Insertion Pattern (Generic for all binary ops)
//===----------------------------------------------------------------------===//

template<typename OpType>
struct InsertBroadcastPattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!lhsType || !rhsType || !resultType) {
      return failure();
    }

    if (lhsType.getShape() == resultType.getShape() &&
        rhsType.getShape() == resultType.getShape()) {
      return failure();
    }

    Value newLhs = op.getLhs();
    Value newRhs = op.getRhs();
    bool changed = false;

    if (lhsType.getShape() != resultType.getShape()) {
      if (lhsType.getRank() > resultType.getRank()) {
        return failure();
      }
      if (!isBroadcastCompatible(lhsType.getShape(), resultType.getShape())) {
        return failure();
      }
      
      auto broadcastDims = computeBroadcastDimensions(
          lhsType.getRank(), resultType.getRank());
      
      auto broadcastDimsAttr = rewriter.getI64ArrayAttr(broadcastDims);
      auto restype=resultType.clone(lhsType.getElementType());

      newLhs = rewriter.create<BroadcastInDimOp>(
          op.getLoc(), restype, newLhs, broadcastDimsAttr).getResult();
      changed = true;
    }

    if (rhsType.getShape() != resultType.getShape()) {
      if (rhsType.getRank() > resultType.getRank()) {
        return failure();
      }
      if (!isBroadcastCompatible(rhsType.getShape(), resultType.getShape())) {
        return failure();
      }
      
      auto broadcastDims = computeBroadcastDimensions(
          rhsType.getRank(), resultType.getRank());
      
      auto broadcastDimsAttr = rewriter.getI64ArrayAttr(broadcastDims);
      //here instead of creating the op with result type we should 
      //make thedata type of rhs to rhs itself
      auto restype=resultType.clone(rhsType.getElementType());
      newRhs = rewriter.create<BroadcastInDimOp>(
          op.getLoc(), restype, newRhs, broadcastDimsAttr).getResult();
      changed = true;
    }

    if (!changed) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<OpType>(op, op.getResult().getType(), newLhs, newRhs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddOp Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Eliminate A + 0 -> A
struct EliminateAddZero : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {

    // Check if RHS is a zero
    if (auto rhsDefOp = op.getRhs().getDefiningOp<nova::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(rhsDefOp.getValue())) {
        if (denseAttr.isSplat() && isSplatZero(denseAttr)) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }

    // Check if LHS is zero
    if (auto lhsDefOp = op.getLhs().getDefiningOp<nova::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(lhsDefOp.getValue())) {
        if (denseAttr.isSplat() && isSplatZero(denseAttr)) {
          rewriter.replaceOp(op, op.getRhs());
          return success();
        }
      }
    }

    return failure();
  }

private:
  bool isSplatZero(DenseElementsAttr attr) const {
    auto elementType = attr.getElementType();
    if (isa<FloatType>(elementType)) {
      APFloat val = attr.getSplatValue<APFloat>();
      return val.isZero();
    } 
    else if (isa<IntegerType>(elementType)) {
      APInt val = attr.getSplatValue<APInt>();
      return val.isZero();
    }
    return false;
  }
};

/// Combine consecutive additions: (A + c1) + c2 -> A + (c1 + c2)
struct CombineAddConstants : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // Check if outer RHS is constant
    auto rhsConst = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!rhsConst)
      return failure();

    // Check if LHS is another addition
    auto lhsAdd = op.getLhs().getDefiningOp<AddOp>();
    if (!lhsAdd)
      return failure();

    // For checking both operands
    arith::ConstantOp innerConst = nullptr;
    Value otherOperand;

    // Constant on RHS of inner add
    if (auto rhsInnerConst = lhsAdd.getRhs().getDefiningOp<arith::ConstantOp>()) {
      innerConst = rhsInnerConst;
      otherOperand = lhsAdd.getLhs();
    }
    // Constant on LHS of inner add  
    else if (auto lhsInnerConst = lhsAdd.getLhs().getDefiningOp<arith::ConstantOp>()) {
      innerConst = lhsInnerConst;
      otherOperand = lhsAdd.getRhs();
    }
    
    if (!innerConst)
      return failure();

    auto rhsAttr = dyn_cast<DenseElementsAttr>(rhsConst.getValue());
    auto innerAttr = dyn_cast<DenseElementsAttr>(innerConst.getValue());
    
    if (!rhsAttr || !innerAttr || !rhsAttr.isSplat() || !innerAttr.isSplat())
      return failure();

    if (rhsAttr.getElementType() != innerAttr.getElementType())
      return failure();

    TypedAttr newConstAttr;
    auto elementType = rhsAttr.getElementType();
    
    if (isa<FloatType>(elementType)) {
      APFloat val1 = rhsAttr.getSplatValue<APFloat>();
      APFloat val2 = innerAttr.getSplatValue<APFloat>();
      APFloat combined = val1;
      combined.add(val2, APFloat::rmNearestTiesToEven);
      newConstAttr = DenseElementsAttr::get(cast<ShapedType>(rhsConst.getType()), combined);
      
    } else if (isa<IntegerType>(elementType)) {
      APInt val1 = rhsAttr.getSplatValue<APInt>();
      APInt val2 = innerAttr.getSplatValue<APInt>();
      APInt combined = val1 + val2;  // Integer addition is exact
      newConstAttr = DenseElementsAttr::get(cast<ShapedType>(rhsConst.getType()), combined);
      
    } else {
      return failure();
    }

    auto newConst = rewriter.create<arith::ConstantOp>(op.getLoc(), newConstAttr);

    // Replace with new addition
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), otherOperand, newConst);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SubOp Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Eliminate A - 0 -> A
struct EliminateSubZero : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    if (auto rhsDefOp = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(rhsDefOp.getValue())) {
        if (denseAttr.isSplat() && isSplatZero(denseAttr)) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    return failure();
  }

private:
  bool isSplatZero(DenseElementsAttr attr) const {
    auto elementType = attr.getElementType();

    if (isa<FloatType>(elementType)) {
      APFloat value = attr.getSplatValue<APFloat>();
      return value.isZero();
    } else if (isa<IntegerType>(elementType)) {
      APInt value = attr.getSplatValue<APInt>();
      return value.isZero();
    }
    return false;
  }
};

/// Eliminate A - A -> 0
struct EliminateSubSelf : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return failure();

    auto resultType = cast<ShapedType>(op.getResult().getType());
    auto elementType = resultType.getElementType();
    
    TypedAttr zeroAttr;
    if (isa<FloatType>(elementType)) {
      zeroAttr = rewriter.getFloatAttr(elementType, 0.0);
    } else if (isa<IntegerType>(elementType)) {
      zeroAttr = rewriter.getIntegerAttr(elementType, 0);
    } else {
      return failure();
    }
    
    auto denseZeroAttr = DenseElementsAttr::get(resultType, zeroAttr);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, denseZeroAttr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MulOp Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Eliminate A * 1 -> A
struct EliminateMulOne : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
    // Check RHS for constant 1
    if (auto rhsDefOp = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(rhsDefOp.getValue())) {
        if (denseAttr.isSplat() && isSplatOne(denseAttr)) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }

    // Check LHS for constant 1
    if (auto lhsDefOp = op.getLhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(lhsDefOp.getValue())) {
        if (denseAttr.isSplat() && isSplatOne(denseAttr)) {
          rewriter.replaceOp(op, op.getRhs());
          return success();
        }
      }
    }

    return failure();
  }

private:
  bool isSplatOne(DenseElementsAttr attr) const {
    auto elementType = attr.getElementType();
    
    if (isa<FloatType>(elementType)) {
      APFloat value = attr.getSplatValue<APFloat>();
      return value.isExactlyValue(1.0);
    } else if (isa<IntegerType>(elementType)) {
      APInt value = attr.getSplatValue<APInt>();
      return value.isOne();
    }
    return false;
  }
};

/// Eliminate A * 0 -> 0
struct EliminateMulZero : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
    Value zeroOperand = nullptr;
    
    // Check RHS for zero constant
    if (auto rhsDefOp = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(rhsDefOp.getValue())) {
        if (denseAttr.isSplat() && isSplatZero(denseAttr)) {
          zeroOperand = op.getRhs();
        }
      }
    }

    // Check LHS for zero constant
    if (!zeroOperand) {
      if (auto lhsDefOp = op.getLhs().getDefiningOp<arith::ConstantOp>()) {
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(lhsDefOp.getValue())) {
          if (denseAttr.isSplat() && isSplatZero(denseAttr)) {
            zeroOperand = op.getLhs();
          }
        }
      }
    }

    if (zeroOperand) {
      rewriter.replaceOp(op, zeroOperand);
      return success();
    }

    return failure();
  }

private:
  bool isSplatZero(DenseElementsAttr attr) const {
    auto elementType = attr.getElementType();
    
    if (isa<FloatType>(elementType)) {
      APFloat value = attr.getSplatValue<APFloat>();
      return value.isZero();
    } else if (isa<IntegerType>(elementType)) {
      APInt value = attr.getSplatValue<APInt>();
      return value.isZero();
    }
    return false;
  }
};

/// Combine consecutive multiplications: (A * c1) * c2 -> A * (c1 * c2)
struct CombineMulConstants : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
    // Check if outer RHS is constant
    auto rhsConst = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!rhsConst)
      return failure();

    // Check if LHS is another multiplication
    auto lhsMul = op.getLhs().getDefiningOp<MulOp>();
    if (!lhsMul)
      return failure();

    // checking to find constant in inner mul
    arith::ConstantOp innerConst = nullptr;
    Value otherOperand;

    // Constant on RHS of inner mul
    if (auto rhsInnerConst = lhsMul.getRhs().getDefiningOp<arith::ConstantOp>()) {
      innerConst = rhsInnerConst;
      otherOperand = lhsMul.getLhs();
    }
    // Constant on LHS of inner mul  
    else if (auto lhsInnerConst = lhsMul.getLhs().getDefiningOp<arith::ConstantOp>()) {
      innerConst = lhsInnerConst;
      otherOperand = lhsMul.getRhs();
    }
    
    if (!innerConst)
      return failure();

    // Extract and validate both constants
    auto rhsAttr = dyn_cast<DenseElementsAttr>(rhsConst.getValue());
    auto innerAttr = dyn_cast<DenseElementsAttr>(innerConst.getValue());
    
    if (!rhsAttr || !innerAttr || !rhsAttr.isSplat() || !innerAttr.isSplat())
      return failure();

    // Check if types are compatible
    if (rhsAttr.getElementType() != innerAttr.getElementType())
      return failure();

    TypedAttr newConstAttr;
    auto elementType = rhsAttr.getElementType();
    
    if (isa<FloatType>(elementType)) {
      APFloat val1 = rhsAttr.getSplatValue<APFloat>();
      APFloat val2 = innerAttr.getSplatValue<APFloat>();
      APFloat combined = val1;
      combined.multiply(val2, APFloat::rmNearestTiesToEven);
      newConstAttr = DenseElementsAttr::get(cast<ShapedType>(rhsConst.getType()), combined);
      
    } else if (isa<IntegerType>(elementType)) {
      APInt val1 = rhsAttr.getSplatValue<APInt>();
      APInt val2 = innerAttr.getSplatValue<APInt>();
      APInt combined = val1 * val2; 
      newConstAttr = DenseElementsAttr::get(cast<ShapedType>(rhsConst.getType()), combined);
      
    } else {
      return failure();
    }

    auto newConst = rewriter.create<arith::ConstantOp>(op.getLoc(), newConstAttr);
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), otherOperand, newConst);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// DivOp Canonicalization Patterns
//===----------------------------------------------------------------------===//

//// Eliminate A / 1 -> A
struct EliminateDivOne : public OpRewritePattern<DivOp> {
  using OpRewritePattern<DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOp op, PatternRewriter &rewriter) const override {
    // Only check RHS - division is NOT commutative
    if (auto rhsDefOp = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(rhsDefOp.getValue())) {
        if (denseAttr.isSplat() && isSplatOne(denseAttr)) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    return failure();
  }

private:
  bool isSplatOne(DenseElementsAttr attr) const {
    auto elementType = attr.getElementType();
    
    if (isa<FloatType>(elementType)) {
      APFloat value = attr.getSplatValue<APFloat>();
      return value.isExactlyValue(1.0);
    } else if (isa<IntegerType>(elementType)) {
      APInt value = attr.getSplatValue<APInt>();
      return value.isOne();
    }
    return false;
  }
};

//===----------------------------------------------------------------------===//
// MaxOp/MinOp Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Simplify max(A, A) -> A
struct SimplifyMaxSelf : public OpRewritePattern<MaxOp> {
  using OpRewritePattern<MaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return failure();

    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};

/// Simplify min(A, A) -> A
struct SimplifyMinSelf : public OpRewritePattern<MinOp> {
  using OpRewritePattern<MinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MinOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return failure();

    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};

} // namespace

static Attribute subIntAttrs(Builder &builder, Attribute lhs, Attribute rhs) {
  auto lhsAttr = dyn_cast<DenseElementsAttr>(lhs);
  auto rhsAttr = dyn_cast<DenseElementsAttr>(rhs);
  
  if (!lhsAttr || !rhsAttr) return {};
  
  auto elementType = lhsAttr.getElementType();
  if (elementType != rhsAttr.getElementType()) return {};
  
  if (isa<IntegerType>(elementType)) {
    if (lhsAttr.isSplat() && rhsAttr.isSplat()) {
      APInt val1 = lhsAttr.getSplatValue<APInt>();
      APInt val2 = rhsAttr.getSplatValue<APInt>();
      APInt result = val1 - val2;
      return DenseElementsAttr::get(cast<ShapedType>(lhsAttr.getType()), result);
    }
  }
  else if(isa<FloatType>(elementType)){
    if (lhsAttr.isSplat() && rhsAttr.isSplat()){
      APFloat val1 = lhsAttr.getSplatValue<APFloat>();
      APFloat val2 = rhsAttr.getSplatValue<APFloat>();
      APFloat result = val1;
      result.subtract(val2, APFloat::rmNearestTiesToEven);
      return DenseElementsAttr::get(cast<ShapedType>(lhsAttr.getType()), result);
    }
  }
  return {};
}

static bool canFoldSub(Attribute lhs, Attribute rhs) {
  auto lhsAttr = dyn_cast<DenseElementsAttr>(lhs);
  auto rhsAttr = dyn_cast<DenseElementsAttr>(rhs);
  
  if (!lhsAttr || !rhsAttr) return false;
  if (lhsAttr.getElementType() != rhsAttr.getElementType()) return false;
  
  if (!lhsAttr.isSplat() || !rhsAttr.isSplat()) return false;
  
  return isa<IntegerType>(lhsAttr.getElementType()) || isa<FloatType>(lhsAttr.getElementType());
}

//===----------------------------------------------------------------------===//
// Populate Canonicalization Patterns (called from each Op)
//===----------------------------------------------------------------------===//

#include "Compiler/Dialect/nova/NovaCanonicalization.inc"

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, 
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<AddOp>>(context);
  results.add<EliminateAddZero>(context);
  results.add<CombineAddConstants>(context);
  populateWithGenerated(results);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results, 
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<SubOp>>(context);
  results.add<EliminateSubZero>(context);
  results.add<EliminateSubSelf>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results, 
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<MulOp>>(context);
  results.add<EliminateMulOne>(context);
  results.add<EliminateMulZero>(context);
  results.add<CombineMulConstants>(context);
}

void DivOp::getCanonicalizationPatterns(RewritePatternSet &results, 
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<DivOp>>(context);
  results.add<EliminateDivOne>(context);
}

void ModOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<ModOp>>(context);
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<PowOp>>(context);
}

void MaxOp::getCanonicalizationPatterns(RewritePatternSet &results, 
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<MaxOp>>(context);
  results.add<SimplifyMaxSelf>(context);
}

void MinOp::getCanonicalizationPatterns(RewritePatternSet &results, 
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<MinOp>>(context);
  results.add<SimplifyMinSelf>(context);
}

void AndOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<AndOp>>(context);
}

void OrOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<InsertBroadcastPattern<OrOp>>(context);
}

void XorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<InsertBroadcastPattern<XorOp>>(context);
}