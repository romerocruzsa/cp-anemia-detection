#!/bin/bash

# Each line represents: MODEL_NAME   SIGNATURE           QUANTIZATION   PRUNING   DISTILLATION

# FINGERNAIL-ANEMIA
# multilayer-perceptron       fingernail-anemia   base            base      base

# # FINGERNAIL-ANEMIA
mobilenetv2       fingernail-anemia   base            base      base
resnet18          fingernail-anemia   base            base      base
vit-tiny          fingernail-anemia   base            base      base
shufflenetv2-0.5x fingernail-anemia   base            base      base
regnety-400mf     fingernail-anemia   base            base      base
mnasnet0_5        fingernail-anemia   base            base      base
ghostnetv2        fingernail-anemia   base            base      base

# mobilenetv2       fingernail-anemia   qat            base      base
# resnet18          fingernail-anemia   qat            base      base
# vit-tiny          fingernail-anemia   qat            base      base
# shufflenetv2-0.5x fingernail-anemia   qat            base      base
# regnety-400mf     fingernail-anemia   qat            base      base
# mnasnet0_5        fingernail-anemia   qat            base      base
# ghostnetv2        fingernail-anemia   qat            base      base

# mobilenetv2       fingernail-anemia   base            structured      base
# resnet18          fingernail-anemia   base            structured      base
# vit-tiny          fingernail-anemia   base            structured      base
# shufflenetv2-0.5x fingernail-anemia   base            structured      base
# regnety-400mf     fingernail-anemia   base            structured      base
# mnasnet0_5        fingernail-anemia   base            structured      base
# ghostnetv2        fingernail-anemia   base            structured      base

# mobilenetv2       fingernail-anemia   base            unstructured      base
# resnet18          fingernail-anemia   base            unstructured      base
# vit-tiny          fingernail-anemia   base            unstructured      base
# shufflenetv2-0.5x fingernail-anemia   base            unstructured      base
# regnety-400mf     fingernail-anemia   base            unstructured      base
# mnasnet0_5        fingernail-anemia   base            unstructured      base
# ghostnetv2        fingernail-anemia   base            unstructured      base
