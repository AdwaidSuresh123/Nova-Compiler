#ifndef MLIR_CONVERSION_NOVATOTOSA_H
#define MLIR_CONVERSION_NOVATOTOSA_H

#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir{
    class Pass;
    class RewritePatternSet;
    class TypeConverter;
    namespace nova{
        std::unique_ptr<Pass> createNovaToTosaLoweringPass();
        void registerNovaToTosaLoweringPass();
        void populateNovaToTosaConversionPatterns(RewritePatternSet &patterns);
        void populateNovaToTosaTemplatePatterns(RewritePatternSet &Patterns);

    }
}
#endif