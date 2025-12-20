#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Complex/IR/Complex.h"

#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"

namespace mlir
{
  namespace nova
  {

    // Helper Utilities
    inline bool isScalar(Value v)
    {
      auto type = dyn_cast<RankedTensorType>(v.getType());
      return !type || type.getRank() == 0;
    }

    inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n)
    {
      return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
    }
    //-----------------------------------------------
    // For getting Type builder
    //-----------------------------------------------
    using namespace mlir;
    static mlir::IntegerType getinttype(int bw, OpBuilder *builder)
    {
      switch (bw)
      {
      case 64:
        return builder->getI64Type();
      case 32:
        return builder->getI32Type();
      }
      return nullptr;
    }

    static mlir::FloatType getfloattype(int bw, OpBuilder *builder)
    {
      switch (bw)
      {
      case 64:
        return builder->getF64Type();
      case 32:
        return builder->getF32Type();
      }
      return nullptr;
    }
    static std::optional<arith::CmpIPredicate>
    getArithCmpiPredicate(nova::ComparisonType type)
    {
      switch (type)
      {
      case nova::ComparisonType::EQ:
        return arith::CmpIPredicate::eq;
      case nova::ComparisonType::NEQ:
        return arith::CmpIPredicate::ne;
      case nova::ComparisonType::LT:
        return arith::CmpIPredicate::slt;
      case nova::ComparisonType::GT:
        return arith::CmpIPredicate::sgt;
      case nova::ComparisonType::LE:
        return arith::CmpIPredicate::sle;
      case nova::ComparisonType::GE:
        return arith::CmpIPredicate::sge;
      }
      return std::nullopt;
    }
    static std::optional<arith::CmpFPredicate>
    getArithCmpfPredicate(nova::ComparisonType type)
    {
      switch (type)
      {
      case nova::ComparisonType::EQ:
        return arith::CmpFPredicate::UEQ;
      case nova::ComparisonType::NEQ:
        return arith::CmpFPredicate::UNE;
      case nova::ComparisonType::LT:
        return arith::CmpFPredicate::ULT;
      case nova::ComparisonType::GT:
        return arith::CmpFPredicate::UGT;
      case nova::ComparisonType::LE:
        return arith::CmpFPredicate::ULE;
      case nova::ComparisonType::GE:
        return arith::CmpFPredicate::UGE;
      }
      return std::nullopt;
    }
    // function to select operation
    // if add choose add
    static Value opdispatcher(nova::AddOp op, Value lhs, Value rhs, OpBuilder *builder)
    {

      if (isa<IntegerType>(lhs.getType()))
        return builder->create<arith::AddIOp>(op.getLoc(), lhs, rhs);
      else if(isa<FloatType>(lhs.getType()))
      return builder->create<arith::AddFOp>(op.getLoc(), lhs, rhs);
      return builder->create<complex::AddOp>(op.getLoc(), lhs, rhs);
      return nullptr;
    }
    static Value opdispatcher(nova::SubOp op, Value lhs, Value rhs, OpBuilder *builder)
    {

      if (isa<IntegerType>(lhs.getType()))
        return builder->create<arith::SubIOp>(op.getLoc(), lhs, rhs);
      else if(isa<FloatType>(lhs.getType()))
      return builder->create<arith::SubFOp>(op.getLoc(), lhs, rhs);
      return builder->create<complex::SubOp>(op.getLoc(), lhs, rhs);
      return nullptr;
    }
    static Value opdispatcher(nova::MulOp op, Value lhs, Value rhs, OpBuilder *builder)
    {

      if (isa<IntegerType>(lhs.getType()))
        return builder->create<arith::MulIOp>(op.getLoc(), lhs, rhs);
      if(isa<FloatType>(lhs.getType()))
      return builder->create<arith::MulFOp>(op.getLoc(), lhs, rhs);
      if(isa<ComplexType>(lhs.getType())){
        return builder->create<complex::MulOp>(op.getLoc(), lhs, rhs);
      }
      return nullptr;
    }
    static Value opdispatcher(nova::PowOp op, Value lhs, Value rhs, OpBuilder *builder)
    {

      if (isa<IntegerType>(lhs.getType()))
        return builder->create<math::IPowIOp>(op.getLoc(), lhs, rhs);
      else if(isa<FloatType>(lhs.getType()))
      return builder->create<math::PowFOp>(op.getLoc(), lhs, rhs);
      else if(isa<ComplexType>(lhs.getType())&&isa<ComplexType>(rhs.getType()))
      return builder->create<complex::PowOp>(op.getLoc(), lhs, rhs);
      return nullptr;
    }
    static Value opdispatcher(nova::CompareOp op,Value lhs,Value rhs,OpBuilder *builder){
      nova::ComparisonType compareType = op.getKind();
        if (isa<IntegerType>(lhs.getType()))
        {
          std::optional<arith::CmpIPredicate> arithPred = getArithCmpiPredicate(compareType);
          return builder->create<arith::CmpIOp>(op.getLoc(), *arithPred, lhs, rhs);
        }
        if (isa<FloatType>(lhs.getType()))
        {
          std::optional<arith::CmpFPredicate> arithPred = getArithCmpfPredicate(compareType);
          return builder->create<arith::CmpFOp>(op.getLoc(), *arithPred,lhs, rhs);
        }
        return nullptr;
    }
    // TYPE PROMOTION LOWERING
    template <typename top>
    static Value TypePromotionLowering(top op, Type resultType, ArrayRef<Value> args, OpBuilder *builder)
    { // need to find parameters
      // 1..fiding dtype
      auto flhstype = dyn_cast<mlir::FloatType>(args[0].getType());
      auto frhstype = dyn_cast<mlir::FloatType>(args[1].getType());
      auto ilhstype = dyn_cast<mlir::IntegerType>(args[0].getType());
      auto irhstype = dyn_cast<mlir::IntegerType>(args[1].getType());
      Value v;
      // checking if lhs and rhs are same
      if (isa<FloatType>(args[0].getType()) && isa<FloatType>(args[1].getType()))
      {
        // check both bitwidth
        auto lhsbw = flhstype.getWidth();
        auto rhsbw = frhstype.getWidth();
        // selecting bigger one
        if (lhsbw == rhsbw)
          return opdispatcher(op, args[0], args[1], builder);
        else if (lhsbw > rhsbw)
        {
          v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
          return opdispatcher(op, args[0], v, builder);
        }
        else
        {
          v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
          return opdispatcher(op, v, args[1], builder);
        }
      }
      else if (isa<IntegerType>(args[0].getType()) && isa<FloatType>(args[1].getType()))
      {

        auto lhsbw = ilhstype.getWidth();
        auto rhsbw = frhstype.getWidth();
        if (lhsbw == rhsbw)
        {
          v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
          return opdispatcher(op, v, args[1], builder);
        }
        else if (lhsbw > rhsbw)
        {
          v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
          auto lhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[0]);
          return opdispatcher(op, lhs, v, builder);
        }
        else
        {
          v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(rhsbw, builder), args[0]);
          auto lhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), v);
          return opdispatcher(op, lhs, args[1], builder);
        }
      }
      // lhs if float and rhs is int
      else if (isa<FloatType>(args[0].getType()) && isa<IntegerType>(args[1].getType()))
      {
        auto lhsbw = flhstype.getWidth();
        auto rhsbw = irhstype.getWidth();
        if (lhsbw == rhsbw)
        {
          v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
          return opdispatcher(op, args[0], v, builder);
        }
        else if (lhsbw > rhsbw)
        {
          v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(lhsbw, builder), args[1]);
          auto rhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), v);
          return opdispatcher(op, args[0], rhs, builder);
        }
        else
        {
          v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
          auto rhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[1]);
          return opdispatcher(op, v, rhs, builder);
        }
      }

      else if (isa<IntegerType>(args[0].getType()) && isa<IntegerType>(args[1].getType()))
      {
        auto lhsbw = ilhstype.getWidth();
        auto rhsbw = irhstype.getWidth();
        if (lhsbw == rhsbw)
        {
          return opdispatcher(op, args[0], args[1], builder);
        }

        else if (lhsbw > rhsbw)
        {
          v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(lhsbw, builder), args[1]);
          return opdispatcher(op, args[0], v, builder);
        }

        else
        {
          v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(rhsbw, builder), args[0]);
          return opdispatcher(op, v, args[1], builder);
        }
      }
      else{
        return opdispatcher(op, args[0], args[1], builder);
      }

      return nullptr;
    }
    // Scalar Operation Mapper

    struct NovaOpToStdScalarOp
    {
      template <typename OpTy>
      // kind of main function
      static Value mapOp(OpTy op, Type resultType, ArrayRef<Value> args,
                         OpBuilder *builder)
      {
        return mapOpImpl(op, resultType, args, builder);
      }

      // default function to return null ptr if the operation didn't match
    private:
      template <typename OpTy>
      static Value mapOpImpl(OpTy op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        return nullptr;
      }

      // Add operation
      static Value mapOpImpl(nova::AddOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        return TypePromotionLowering(op, resultType, args, builder);
      }
      // sub operation
      static Value mapOpImpl(nova::SubOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        return TypePromotionLowering(op, resultType, args, builder);
      }
      // mul operation
      static Value mapOpImpl(nova::MulOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        return TypePromotionLowering(op, resultType, args, builder);
      }
      // pow operation
      static Value mapOpImpl(nova::PowOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        return TypePromotionLowering(op, resultType, args, builder);
      }
      // Square operation
      static Value mapOpImpl(nova::SquareOp op, Type resultType, ArrayRef<Value> args, OpBuilder *builder)
      {
        if (isa<FloatType>(resultType))
          return builder->create<arith::MulFOp>(op.getLoc(), args[0], args[0]);
        if (isa<IntegerType>(resultType))
          return builder->create<arith::MulIOp>(op.getLoc(), args[0], args[0]);
        return nullptr;
      }
      // abs operation
      static Value mapOpImpl(nova::AbsOp op, Type resultType, ArrayRef<Value> args, OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::AbsFOp>(op.getLoc(), args[0]);
        else if (isa<IntegerType>(resultType))
          return builder->create<math::AbsIOp>(op.getLoc(), args[0]);
        else
          return builder->create<complex::AbsOp>(op.getLoc(), args[0]);
        return nullptr;
      }
      // sqrt operation
      static Value mapOpImpl(nova::SqrtOp op, Type resultType, ArrayRef<Value> args, OpBuilder *builder)
      {
        if(isa<ComplexType>(args[0].getType())){
          return builder->create<complex::SqrtOp>(op.getLoc(),args[0]);
        }
        return builder->create<math::SqrtOp>(op.getLoc(), args[0]);
      }
      // div operation
      static Value mapOpImpl(nova::DivOp op, Type resultType, ArrayRef<Value> args, OpBuilder *builder)
      {
        //if complex directly lower it
        if(isa<ComplexType>(args[0].getType())){
          return builder->create<complex::DivOp>(op.getLoc(),args[0],args[1]);
        }
        // 1..fiding dtype
        auto flhstype = dyn_cast<mlir::FloatType>(args[0].getType());
        auto frhstype = dyn_cast<mlir::FloatType>(args[1].getType());
        auto ilhstype = dyn_cast<mlir::IntegerType>(args[0].getType());
        auto irhstype = dyn_cast<mlir::IntegerType>(args[1].getType());
        Value v;
        // checking if lhs and rhs are same
        if (isa<FloatType>(args[0].getType()) && isa<FloatType>(args[1].getType()))
        {
          // check both bitwidth
          auto lhsbw = flhstype.getWidth();
          auto rhsbw = frhstype.getWidth();
          // selecting bigger one
          if (lhsbw == rhsbw)
            return builder->create<arith::DivFOp>(op.getLoc(), args[0], args[1]);
          else if (lhsbw > rhsbw)
          {
            v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
            return builder->create<arith::DivFOp>(op.getLoc(), args[0], v);
          }
          else
          {
            v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
            return builder->create<arith::DivFOp>(op.getLoc(), v, args[1]);
          }
        }
        else if (isa<IntegerType>(args[0].getType()) && isa<FloatType>(args[1].getType()))
        {

          auto lhsbw = ilhstype.getWidth();
          auto rhsbw = frhstype.getWidth();
          if (lhsbw == rhsbw)
          {
            v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
            return builder->create<arith::DivFOp>(op.getLoc(), v, args[1]);
          }
          else if (lhsbw > rhsbw)
          {
            v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
            auto lhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[0]);
            return builder->create<arith::DivFOp>(op.getLoc(), lhs, v);
          }
          else
          {
            v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(rhsbw, builder), args[0]);
            auto lhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), v);
            return builder->create<arith::DivFOp>(op.getLoc(), lhs, args[1]);
          }
        }
        // lhs if float and rhs is int
        else if (isa<FloatType>(args[0].getType()) && isa<IntegerType>(args[1].getType()))
        {
          auto lhsbw = flhstype.getWidth();
          auto rhsbw = irhstype.getWidth();
          if (lhsbw == rhsbw)
          {
            v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
            return builder->create<arith::DivFOp>(op.getLoc(), args[0], v);
          }
          else if (lhsbw > rhsbw)
          {
            v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(lhsbw, builder), args[1]);
            auto rhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), v);
            return builder->create<arith::DivFOp>(op.getLoc(), args[0], rhs);
          }
          else
          {
            v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
            auto rhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[1]);
            return builder->create<arith::DivFOp>(op.getLoc(), v, rhs);
          }
        }

        else if (isa<IntegerType>(args[0].getType()) && isa<IntegerType>(args[1].getType()))
        {
          auto lhsbw = ilhstype.getWidth();
          auto rhsbw = irhstype.getWidth();
          if (lhsbw == rhsbw)
          {
            v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
            auto w = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
            return builder->create<arith::DivFOp>(op.getLoc(), v, w);
          }

          else if (lhsbw > rhsbw)
          {
            v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(lhsbw, builder), args[1]);
            auto r = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[0]);
            auto w = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), v);
            return builder->create<arith::DivFOp>(op.getLoc(), r, w);
          }

          else
          {
            v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(rhsbw, builder), args[0]);
            auto r = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), v);
            auto w = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[1]);
            return builder->create<arith::DivFOp>(op.getLoc(), r, w);
          }
        }

        return nullptr;
      }

      // mod operation
      static Value mapOpImpl(nova::ModOp op, Type resultType, ArrayRef<Value> args, OpBuilder *builder)
      {
        if (isa<FloatType>(resultType))
        {
          auto flhstype = dyn_cast<mlir::FloatType>(args[0].getType());
          auto frhstype = dyn_cast<mlir::FloatType>(args[1].getType());
          auto ilhstype = dyn_cast<mlir::IntegerType>(args[0].getType());
          auto irhstype = dyn_cast<mlir::IntegerType>(args[1].getType());
          Value v;
          if (isa<FloatType>(args[0].getType()) && isa<FloatType>(args[1].getType()))
          {
            // check both bitwidth
            auto lhsbw = flhstype.getWidth();
            auto rhsbw = frhstype.getWidth();
            // selecting bigger one
            if (lhsbw == rhsbw)
              return builder->create<arith::RemFOp>(op.getLoc(), args[0], args[1]);
            else if (lhsbw > rhsbw)
            {
              v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
              return builder->create<arith::RemFOp>(op.getLoc(), args[0], v);
            }
            else
            {
              v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
              return builder->create<arith::RemFOp>(op.getLoc(), v, args[1]);
            }
          }
          else if (isa<IntegerType>(args[0].getType()) && isa<FloatType>(args[1].getType()))
          {

            auto lhsbw = ilhstype.getWidth();
            auto rhsbw = frhstype.getWidth();
            if (lhsbw == rhsbw)
            {
              v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
              return builder->create<arith::RemFOp>(op.getLoc(), v, args[1]);
            }
            else if (lhsbw > rhsbw)
            {
              v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
              auto lhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[0]);
              return builder->create<arith::RemFOp>(op.getLoc(), lhs, v);
            }
            else
            {
              v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(rhsbw, builder), args[0]);
              auto lhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), v);
              return builder->create<arith::RemFOp>(op.getLoc(), lhs, args[1]);
            }
          }
          else if (isa<FloatType>(args[0].getType()) && isa<IntegerType>(args[1].getType()))
          {
            auto lhsbw = flhstype.getWidth();
            auto rhsbw = irhstype.getWidth();
            if (lhsbw == rhsbw)
            {
              v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
              return builder->create<arith::RemFOp>(op.getLoc(), args[0], v);
            }
            else if (lhsbw > rhsbw)
            {
              v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(lhsbw, builder), args[1]);
              auto rhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), v);
              return builder->create<arith::RemFOp>(op.getLoc(), args[0], rhs);
            }
            else
            {
              v = builder->create<arith::ExtFOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
              auto rhs = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[1]);
              return builder->create<arith::RemFOp>(op.getLoc(), v, rhs);
            }
          }

          else if (isa<IntegerType>(args[0].getType()) && isa<IntegerType>(args[1].getType()))
          {
            auto lhsbw = ilhstype.getWidth();
            auto rhsbw = irhstype.getWidth();
            if (lhsbw == rhsbw)
            {
              v = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[0]);
              auto w = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[1]);
              return builder->create<arith::RemFOp>(op.getLoc(), v, w);
            }

            else if (lhsbw > rhsbw)
            {
              v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(lhsbw, builder), args[1]);
              auto r = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), args[0]);
              auto w = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(lhsbw, builder), v);
              return builder->create<arith::RemFOp>(op.getLoc(), r, w);
            }

            else
            {
              v = builder->create<arith::ExtSIOp>(op.getLoc(), getinttype(rhsbw, builder), args[0]);
              auto r = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), v);
              auto w = builder->create<arith::BitcastOp>(op.getLoc(), getfloattype(rhsbw, builder), args[1]);
              return builder->create<arith::RemFOp>(op.getLoc(), r, w);
            }
          }
        }
        return nullptr;
      }
      // and operation
      //  //only integer type
      //  static Value mapOpImpl(nova::AndOp op,Type resultType,ArrayRef<Value> args,OpBuilder* builder){
      //    if(isa<IntegerType>(resultType))
      //    return builder ->create<arith::AndIOp>(op.getLoc(),args[0],args[1]);
      //    return nullptr;
      //  }
      //  // //or operation
      //    static Value mapOpImpl(nova::OrOp op,Type resultType,ArrayRef<Value> args,OpBuilder* builder){
      //    if(isa<IntegerType>(resultType))
      //    return builder ->create<arith::OrIOp>(op.getLoc(),args[0],args[1]);
      //    return nullptr;
      //  }
      //  // //xor operation
      //    static Value mapOpImpl(nova::XorOp op,Type resultType,ArrayRef<Value> args,OpBuilder* builder){
      //    if(isa<IntegerType>(resultType))
      //    return builder ->create<arith::XOrIOp>(op.getLoc(),args[0],args[1]);
      //    return nullptr;
      //  }

      //--------------------------------------------------------
      // EXPONENTS
      //-----------------------------------------------------------
      // exp operaton
      static Value mapOpImpl(nova::ExpOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::ExpOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::ExpOp>(op.getLoc(),
                                              builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        if(isa<ComplexType>(args[0].getType())){
          return builder->create<complex::ExpOp>(op.getLoc(),args[0]);
        }
        return nullptr;
      }
      // exp2 operaton
      static Value mapOpImpl(nova::Exp2Op op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::Exp2Op>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::Exp2Op>(op.getLoc(),
                                               builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // log operaton
      static Value mapOpImpl(nova::LogOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::LogOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::LogOp>(op.getLoc(),
                                              builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        if(isa<ComplexType>(args[0].getType())){
          return builder->create<complex::LogOp>(op.getLoc(),args[0]);
        }
        return nullptr;
      }
      //----------------------------------------------------------------
      // log2 operaton
      static Value mapOpImpl(nova::Log2Op op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::Log2Op>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::Log2Op>(op.getLoc(),
                                               builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // log10 operaton
      static Value mapOpImpl(nova::Log10Op op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::Log10Op>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::Log10Op>(op.getLoc(),
                                                builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      //----------------------------------------------------
      // TRIGNOMENTARY OPERATIONS
      //--------------------------------------------------------------------
      // sin operaton
      static Value mapOpImpl(nova::SinOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::SinOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::SinOp>(op.getLoc(),
                                              builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        if (isa<ComplexType>(args[0].getType()))
          return builder->create<complex::SinOp>(op.getLoc(), args[0]);
        return nullptr;
      }

      // cos operation
      static Value mapOpImpl(nova::CosOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::CosOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::CosOp>(op.getLoc(),
                                              builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        if (isa<ComplexType>(args[0].getType()))
          return builder->create<complex::CosOp>(op.getLoc(), args[0]);
        
        return nullptr;
      }

      // tan operation
      static Value mapOpImpl(nova::TanOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::TanOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::TanOp>(op.getLoc(),
                                              builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        if(isa<ComplexType>(args[0].getType())){
          return builder->create<complex::TanOp>(op.getLoc(), args[0]);
        }
        return nullptr;
      }
      // asin operation
      static Value mapOpImpl(nova::AsinOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::AsinOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::AsinOp>(op.getLoc(),
                                               builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // acos operation
      static Value mapOpImpl(nova::AcosOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::AcosOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::AcosOp>(op.getLoc(),
                                               builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // atan operation
      static Value mapOpImpl(nova::AtanOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::AtanOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::AtanOp>(op.getLoc(),
                 builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));

        return nullptr;
      }
      // sinh operation
      static Value mapOpImpl(nova::SinhOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::SinhOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::SinhOp>(op.getLoc(),
                                               builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // cosh operation
      static Value mapOpImpl(nova::CoshOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::CoshOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::CoshOp>(op.getLoc(),
                                               builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // tanh operation
      static Value mapOpImpl(nova::TanhOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::TanhOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::TanhOp>(op.getLoc(),
                                               builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        if (isa<ComplexType>(args[0].getType()))
          return builder->create<complex::TanhOp>(op.getLoc(), args[0]);
        return nullptr;
      }
      // asinh operation
      static Value mapOpImpl(nova::AsinhOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::AsinhOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::AsinhOp>(op.getLoc(),
                                                builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // acosh operation
      static Value mapOpImpl(nova::AcoshOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::AcoshOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::AcoshOp>(op.getLoc(),
                                                builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }
      // atanh operation
      static Value mapOpImpl(nova::AtanhOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        if (isa<FloatType>(args[0].getType()))
          return builder->create<math::AtanhOp>(op.getLoc(), args[0]);
        if (isa<IntegerType>(args[0].getType()))
          return builder->create<math::AtanhOp>(op.getLoc(),
                                                builder->create<arith::SIToFPOp>(op.getLoc(), builder->getF32Type(), args[0]));
        return nullptr;
      }

      // Compare operation
      static Value mapOpImpl(nova::CompareOp op, Type resultType, ArrayRef<Value> args,
                             OpBuilder *builder)
      {
        // assume example  if compareType is eq of nova dialect them arthpred will be eq of arith dialect
                return TypePromotionLowering(op, resultType, args, builder);

      }

      // Reduction Operation(mean)

      // sign operation
      static Value mapOpImpl(nova::SignOp op, Type resultType, ArrayRef<Value> args, OpBuilder *builder)
      {
        mlir::Value input = args[0];
        mlir::Location loc = op.getLoc();
        if (auto floatType = llvm::dyn_cast<mlir::FloatType>(input.getType()))
        {
          // Get 1.0 constant of the correct type
          mlir::Value zeroF = builder->create<mlir::arith::ConstantOp>(
              loc, floatType, builder->getFloatAttr(floatType, 0.0));

          mlir::Value greaterThanZero = builder->create<mlir::arith::CmpFOp>(
              loc, mlir::arith::CmpFPredicate::OGT, input, zeroF);

          mlir::Value signPos = builder->create<mlir::arith::ExtUIOp>(loc, resultType, greaterThanZero);

          mlir::Value lessThanZero = builder->create<mlir::arith::CmpFOp>(
              loc, mlir::arith::CmpFPredicate::OLT, input, zeroF);

          mlir::Value signNeg = builder->create<mlir::arith::ExtUIOp>(loc, resultType, lessThanZero);

          return builder->create<mlir::arith::SubIOp>(loc, signPos, signNeg);
        }
        else if (auto intType = llvm::dyn_cast<mlir::IntegerType>(input.getType()))
        {

          mlir::Value zero = builder->create<mlir::arith::ConstantOp>(
              loc, intType, builder->getIntegerAttr(intType, 0));
          mlir::Value greaterThanZero = builder->create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::sgt, input, zero);

          mlir::Value signPos = builder->create<mlir::arith::ExtUIOp>(loc, resultType, greaterThanZero);
          mlir::Value lessThanZero = builder->create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::slt, input, zero);
          mlir::Value signNeg = builder->create<mlir::arith::ExtUIOp>(loc, resultType, lessThanZero);

          return builder->create<mlir::arith::SubIOp>(loc, signPos, signNeg);
        }
        return nullptr;
      }
    };

    //----------------------------------------------------------------
    //                          Argmin
    //----------------------------------------------------------------
    static TypedAttr createInitialValueForReduceOp(Operation *op, Type elementTy,
                                                   PatternRewriter &rewriter)
    {
      if (isa<nova::ArgMinOp>(op) && isa<FloatType>(elementTy))
        return rewriter.getFloatAttr(
            elementTy, APFloat::getLargest(
                           cast<FloatType>(elementTy).getFloatSemantics(), false));

      if (isa<nova::ArgMinOp>(op) && isa<IntegerType>(elementTy))
        return rewriter.getIntegerAttr(
            elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));

      return {};
    }

    class ArgMinConverter : public OpRewritePattern<nova::ArgMinOp>
    {
    public:
      using OpRewritePattern<nova::ArgMinOp>::OpRewritePattern;

      LogicalResult matchAndRewrite(nova::ArgMinOp argminOp,
                                    PatternRewriter &rewriter) const final
      {
        auto loc = argminOp.getLoc();
        Value input = argminOp.getInput();
        auto inputTy = cast<ShapedType>(input.getType());
        auto resultTy = cast<ShapedType>(argminOp.getOutput().getType());
        auto inElementTy = inputTy.getElementType();
        auto outElementTy = resultTy.getElementType();
        int axis = static_cast<int>(argminOp.getDimension().value());
        auto resultMinTy = RankedTensorType::get(resultTy.getShape(), inElementTy);

        if (!isa<IntegerType>(outElementTy))
          return rewriter.notifyMatchFailure(
              argminOp,
              "nova.arg_min to linalg.* requires integer-like result type");

        SmallVector<Value> dynDims;
        for (int i = 0; i < inputTy.getRank(); i++)
        {
          if (inputTy.isDynamicDim(i) && i != axis)
          {
            dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
          }
        }

        // First fill the output buffer for the index.
        auto emptyTensorIdx = rewriter
                                  .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                           outElementTy, dynDims)
                                  .getResult();
        auto fillValueIdx = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(outElementTy, 0));
        auto filledTensorIdx =
            rewriter
                .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                        ValueRange{emptyTensorIdx})
                .result();

        // Second fill the output buffer for the running min.
        auto emptyTensorMin = rewriter
                                  .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                           inElementTy, dynDims)
                                  .getResult();
        auto fillValueMinAttr =
            createInitialValueForReduceOp(argminOp, inElementTy, rewriter);

        if (!fillValueMinAttr)
          return rewriter.notifyMatchFailure(
              argminOp, "unsupported nova.argmin element type");

        auto fillValueMin =
            rewriter.create<arith::ConstantOp>(loc, fillValueMinAttr);
        auto filledTensorMin =
            rewriter
                .create<linalg::FillOp>(loc, ValueRange{fillValueMin},
                                        ValueRange{emptyTensorMin})
                .result();

        // We need to reduce along the arg-min axis, with parallel operations along
        // the rest.
        SmallVector<utils::IteratorType, 4> iteratorTypes;
        iteratorTypes.resize(inputTy.getRank(), utils::IteratorType::parallel);
        iteratorTypes[axis] = utils::IteratorType::reduction;

        SmallVector<AffineExpr, 2> srcExprs;
        SmallVector<AffineExpr, 2> dstExprs;
        for (int i = 0, rank = inputTy.getRank(); i != rank; ++i)
        {
          srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
          if (axis != i)
            dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
        }

        bool didEncounterError = false;
        auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs, dstExprs},
                                                 rewriter.getContext());
        auto linalgOp = rewriter.create<linalg::GenericOp>(
            loc, ArrayRef<Type>({resultTy, resultMinTy}), input,
            ValueRange({filledTensorIdx, filledTensorMin}), maps, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc,
                ValueRange blockArgs)
            {
              auto newValue = blockArgs[0];
              auto oldIndex = blockArgs[1];
              auto oldValue = blockArgs[2];

              Value newIndex = rewriter.create<arith::IndexCastOp>(
                  nestedLoc, oldIndex.getType(),
                  rewriter.create<linalg::IndexOp>(loc, axis));

              Value predicate;
              if (isa<FloatType>(inElementTy))
              {
                if (argminOp.getIgnoreNan())
                {
                  // Only update index & min value for non NaN values. If all
                  // values are NaNs, the initial index will be return which is 0.
                  predicate = rewriter.create<arith::CmpFOp>(
                      nestedLoc, arith::CmpFPredicate::OLT, newValue, oldValue);
                }
                else
                {
                  // Update min value if either of the following is true:
                  // - new value is bigger
                  // - cur min is not NaN and new value is NaN
                  Value lt = rewriter.create<arith::CmpFOp>(
                      nestedLoc, arith::CmpFPredicate::ULT, newValue, oldValue);
                  Value oldNonNaN = rewriter.create<arith::CmpFOp>(
                      nestedLoc, arith::CmpFPredicate::ORD, oldValue, oldValue);
                  predicate = rewriter.create<arith::AndIOp>(
                      nestedLoc, rewriter.getI1Type(), lt, oldNonNaN);
                }
              }
              else if (isa<IntegerType>(inElementTy))
              {
                predicate = rewriter.create<arith::CmpIOp>(
                    nestedLoc, arith::CmpIPredicate::slt, newValue, oldValue);
              }
              else
              {
                didEncounterError = true;
                return;
              }

              auto resultMin = rewriter.create<arith::SelectOp>(
                  nestedLoc, predicate, newValue, oldValue);
              auto resultIndex = rewriter.create<arith::SelectOp>(
                  nestedLoc, predicate, newIndex, oldIndex);
              nestedBuilder.create<linalg::YieldOp>(
                  nestedLoc, ValueRange({resultIndex, resultMin}));
            });

        if (didEncounterError)
          return rewriter.notifyMatchFailure(
              argminOp, "unsupported nova.argmin element type");

        rewriter.replaceOp(argminOp, linalgOp.getResult(0));
        return success();
      }
    };

    // generic pattern definition

    template <typename NovaOpTy>
    class NovaToLinalgElementwiseConverter : public OpConversionPattern<NovaOpTy>
    {
    public:
      using OpConversionPattern<NovaOpTy>::OpConversionPattern; // creates a constructor
      using OpAdaptor = typename NovaOpTy::Adaptor;             // for getting data type dynamically using adaptor

      LogicalResult
      matchAndRewrite(NovaOpTy op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override
      {
        auto operands = adaptor.getOperands();
        if (operands.empty())
          return rewriter.notifyMatchFailure(op, "expected operands for linalg lowering operations");
        // checking if operand is ranked tensortype
        auto resultType = dyn_cast<RankedTensorType>(op.getType());
        if (!resultType)
          return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
        // each element type
        auto resultDataType = resultType.getElementType();
        // casting

        // Create output tensor
        Value out = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultDataType);

        // Prepare affine maps
        int64_t rank = resultType.getRank();
        AffineMap scalarMap = AffineMap::get(rank, 0, rewriter.getContext());
        AffineMap idMap = rewriter.getMultiDimIdentityMap(rank);
        SmallVector<AffineMap> maps;
        for (Value v : operands)
          maps.push_back(isScalar(v) ? scalarMap : idMap);
        maps.push_back(idMap);

        // Create Linalg generic
        auto linalgOp = rewriter.create<linalg::GenericOp>(
            op.getLoc(), out.getType(), operands, out, maps,
            getNParallelLoopsAttrs(rank),
            [&](OpBuilder &b, Location loc, ValueRange args)
            {
              Type elemType = getElementTypeOrSelf(out);
              SmallVector<Value> argVec(args.begin(), args.end());
              // call our custom lowering functions
              Value inner = NovaOpToStdScalarOp::mapOp(op, elemType, argVec, &b);
              if (!inner)
                return; // op failed to map
              b.create<linalg::YieldOp>(loc, inner);
            });

        rewriter.replaceOp(op, linalgOp->getResults());
        return success();
      }
    };

    //---------------=-------------------=------------------=---------------------
    // Pass Definition

    namespace
    {
      struct NovaToLinalgLoweringPassTemplate
          : public PassWrapper<NovaToLinalgLoweringPassTemplate, OperationPass<func::FuncOp>>
      {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToLinalgLoweringPassTemplate)

        void getDependentDialects(DialectRegistry &registry) const override
        {
          registry.insert<linalg::LinalgDialect,
                          tensor::TensorDialect,
                          arith::ArithDialect,
                          func::FuncDialect>();
        }

        StringRef getArgument() const final { return "convert-nova-to-linalg"; }

        StringRef getDescription() const final
        {
          return "Lower Nova dialect operations to Linalg dialect";
        }

        void runOnOperation() override
        {
          MLIRContext *context = &getContext();
          func::FuncOp funcOp = getOperation();
          ConversionTarget target(*context);

          target.addLegalDialect<linalg::LinalgDialect>();
          target.addLegalDialect<tensor::TensorDialect>();
          target.addLegalDialect<arith::ArithDialect>();
          target.addLegalDialect<func::FuncDialect>();
          target.addLegalDialect<math::MathDialect>();

          target.addIllegalOp<nova::AbsOp>();
          target.addIllegalOp<nova::AddOp>();
          target.addIllegalOp<nova::SubOp>();
          target.addIllegalOp<nova::MulOp>();
          target.addIllegalOp<nova::MatmulOp>();
          target.addIllegalOp<nova::PowOp>();
          target.addIllegalOp<nova::BroadcastInDimOp>();
          target.addIllegalOp<nova::DivOp>();
          target.addIllegalOp<nova::ModOp>();
          target.addIllegalOp<nova::SquareOp>();
          target.addIllegalOp<nova::SqrtOp>();
          target.addIllegalOp<nova::LogOp>();
          target.addIllegalOp<nova::ExpOp>();
          target.addIllegalOp<nova::Exp2Op>();
          target.addIllegalOp<nova::Log2Op>();
          target.addIllegalOp<nova::Log10Op>();
          target.addIllegalOp<nova::SinOp>();
          target.addIllegalOp<nova::CosOp>();
          target.addIllegalOp<nova::TanOp>();
          target.addIllegalOp<nova::SinhOp>();
          target.addIllegalOp<nova::CoshOp>();
          target.addIllegalOp<nova::TanhOp>();
          target.addIllegalOp<nova::AsinOp>();
          target.addIllegalOp<nova::AcosOp>();
          target.addIllegalOp<nova::AtanOp>();
          target.addIllegalOp<nova::AsinhOp>();
          target.addIllegalOp<nova::AcoshOp>();
          target.addIllegalOp<nova::AtanhOp>();
          target.addIllegalOp<nova::CompareOp>();
          target.addIllegalOp<nova::SignOp>();
          target.addIllegalOp<nova::ArgMinOp>();
          target.addIllegalOp<nova::TransposeOp>();
          target.markUnknownOpDynamicallyLegal([](Operation *)
                                               { return true; });
          RewritePatternSet patterns(context);
          populateNovaToLinalgPatterns(patterns);
          populateNovaToLinalgPatternsTemplate(patterns);
          if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
          {
            signalPassFailure();
            return;
          }
        }
      };

    } // namespace

    //===----------------------------------------------------------------------===//
    // Pass Registration & Pattern Population
    //===----------------------------------------------------------------------===//

    std::unique_ptr<Pass> createNovaToLinalgLoweringPass()
    {
      return std::make_unique<NovaToLinalgLoweringPassTemplate>();
    }

    void regsiterNovaToLinalgLoweringTemplatePass()
    {
      PassRegistration<NovaToLinalgLoweringPassTemplate>();
    }

    void populateNovaToLinalgPatternsTemplate(RewritePatternSet &patterns)
    {
      // Use generic converters for pointwise ops
      patterns.add<
          NovaToLinalgElementwiseConverter<nova::AddOp>,
          NovaToLinalgElementwiseConverter<nova::SubOp>,
          NovaToLinalgElementwiseConverter<nova::MulOp>,
          NovaToLinalgElementwiseConverter<nova::PowOp>,
          NovaToLinalgElementwiseConverter<nova::AbsOp>,
          NovaToLinalgElementwiseConverter<nova::DivOp>,
          NovaToLinalgElementwiseConverter<nova::ModOp>,
          NovaToLinalgElementwiseConverter<nova::SquareOp>,
          NovaToLinalgElementwiseConverter<nova::SqrtOp>,
          NovaToLinalgElementwiseConverter<nova::LogOp>,
          NovaToLinalgElementwiseConverter<nova::ExpOp>,
          NovaToLinalgElementwiseConverter<nova::Exp2Op>,
          NovaToLinalgElementwiseConverter<nova::Log2Op>,
          NovaToLinalgElementwiseConverter<nova::Log10Op>,
          NovaToLinalgElementwiseConverter<nova::SinOp>,
          NovaToLinalgElementwiseConverter<nova::CosOp>,
          NovaToLinalgElementwiseConverter<nova::TanOp>,
          NovaToLinalgElementwiseConverter<nova::AsinOp>,
          NovaToLinalgElementwiseConverter<nova::AcosOp>,
          NovaToLinalgElementwiseConverter<nova::AtanOp>,
          NovaToLinalgElementwiseConverter<nova::SinhOp>,
          NovaToLinalgElementwiseConverter<nova::CoshOp>,
          NovaToLinalgElementwiseConverter<nova::TanhOp>,
          NovaToLinalgElementwiseConverter<nova::AsinhOp>,
          NovaToLinalgElementwiseConverter<nova::AcoshOp>,
          NovaToLinalgElementwiseConverter<nova::AtanhOp>,
          NovaToLinalgElementwiseConverter<nova::CompareOp>,
          NovaToLinalgElementwiseConverter<nova::SignOp>,
  
          ArgMinConverter>(patterns.getContext());
    }

  } // namespace nova
} // namespace mlir
