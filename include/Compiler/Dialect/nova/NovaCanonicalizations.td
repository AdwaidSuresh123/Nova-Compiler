#ifndef NOVA_CANONICALIZATIONS_PATTERNS
#define NOVA_CANONICALIZATIONS_PATTERNS

include "mlir/IR/PatternBase.td"
include "NovaOps.td"

def SubIntAttrs : NativeCodeCall<"subIntAttrs($_builder, $0, $1)">;
def GetType : NativeCodeCall<"$0.getType()">;
def CreateConstant : NativeCodeCall<"$_builder.create<nova::ConstantOp>($_loc, $0, $1)">;

//-------------------------------------------------------------------------
//   AddOp patterns
//-------------------------------------------------------------------------

def CanFoldSub : Constraint<CPred<"canFoldSub($0, $1)">>;

def AddISubConstantRHS :
    Pat<(Nova_AddOp:$res
          (Nova_SubOp $x, (Nova_ConstantOp ElementsAttr:$c0)),
          (Nova_ConstantOp ElementsAttr:$c1)),
        (Nova_AddOp $x, (CreateConstant (SubIntAttrs $c1, $c0), (GetType $res))),
        [(CanFoldSub $c1, $c0)]>;



#endif // NOVA_CANONICALIZATIONS_PATTERNS