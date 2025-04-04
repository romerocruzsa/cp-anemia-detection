# main.sh
# Each line represents: MODEL_NAME       QUANTIZATION     PRUNING         DISTILLATION

mobilenetv2        base              base            base
mobilenetv2        qat               base            base
mobilenetv2        base              structured      base
mobilenetv2        base              unstructured    base
mobilenetv2        base              base            self-distil
# mobilenetv2        qat               structured      base
# mobilenetv2        qat               base            self-distil
mobilenetv2        base              structured      self-distil
# mobilenetv2        qat               structured      self-distil

resnet18           base              base            base
resnet18           qat               base            base
resnet18           base              structured      base
resnet18           base              unstructured    base
resnet18           base              base            self-distil
# resnet18           qat               structured      base
# resnet18           qat               base            self-distil
resnet18           base              structured      self-distil
# resnet18           qat               structured      self-distil

densenet121        base              base            base
densenet121        qat               base            base
densenet121        base              structured      base
densenet121        base              unstructured    base
densenet121        base              base            self-distil
# densenet121        qat               structured      base
# densenet121        qat               base            self-distil
densenet121        base              structured      self-distil
# densenet121        qat               structured      self-distil

vgg16              base              base            base
vgg16              qat               base            base
vgg16              base              structured      base
vgg16              base              unstructured    base
vgg16              base              base            self-distil
# vgg16              qat               structured      base
# vgg16              qat               base            self-distil
vgg16              base              structured      self-distil
# vgg16              qat               structured      self-distil

vit-tiny           base              base            base
vit-tiny           qat               base            base
vit-tiny           base              structured      base
vit-tiny           base              unstructured    base
vit-tiny           base              base            self-distil
# vit-tiny           qat               structured      base
# vit-tiny           qat               base            self-distil
vit-tiny           base              structured      self-distil
# vit-tiny           qat               structured      self-distil

convnext-tiny      base              base            base
convnext-tiny      qat               base            base
convnext-tiny      base              structured      base
convnext-tiny      base              unstructured    base
convnext-tiny      base              base            self-distil
# convnext-tiny      qat               structured      base
# convnext-tiny      qat               base            self-distil
convnext-tiny      base              structured      self-distil
# convnext-tiny      qat               structured      self-distil

efficientnet-b0    base              base            base
efficientnet-b0    qat               base            base
efficientnet-b0    base              structured      base
efficientnet-b0    base              unstructured    base
efficientnet-b0    base              base            self-distil
# efficientnet-b0    qat               structured      base
# efficientnet-b0    qat               base            self-distil
efficientnet-b0    base              structured      self-distil
# efficientnet-b0    qat               structured      self-distil

shufflenetv2-0.5x  base              base            base
shufflenetv2-0.5x  qat               base            base
shufflenetv2-0.5x  base              structured      base
shufflenetv2-0.5x  base              unstructured    base
shufflenetv2-0.5x  base              base            self-distil
# shufflenetv2-0.5x  qat               structured      base
# shufflenetv2-0.5x  qat               base            self-distil
shufflenetv2-0.5x  base              structured      self-distil
# shufflenetv2-0.5x  qat               structured      self-distil

regnety-400mf      base              base            base
regnety-400mf      qat               base            base
regnety-400mf      base              structured      base
regnety-400mf      base              unstructured    base
regnety-400mf      base              base            self-distil
# regnety-400mf      qat               structured      base
# regnety-400mf      qat               base            self-distil
regnety-400mf      base              structured      self-distil
# regnety-400mf      qat               structured      self-distil

mnasnet0_5         base              base            base
mnasnet0_5         qat               base            base
mnasnet0_5         base              structured      base
mnasnet0_5         base              unstructured    base
mnasnet0_5         base              base            self-distil
# mnasnet0_5         qat               structured      base
# mnasnet0_5         qat               base            self-distil
mnasnet0_5         base              structured      self-distil
# mnasnet0_5         qat               structured      self-distil

ghostnetv2         base              base            base
ghostnetv2         qat               base            base
ghostnetv2         base              structured      base
ghostnetv2         base              unstructured    base
ghostnetv2         base              base            self-distil
# ghostnetv2         qat               structured      base
# ghostnetv2         qat               base            self-distil
ghostnetv2         base              structured      self-distil
# ghostnetv2         qat               structured      self-distil

tinynet-a          base              base            base
tinynet-a          qat               base            base
tinynet-a          base              structured      base
tinynet-a          base              unstructured    base
tinynet-a          base              base            self-distil
# tinynet-a          qat               structured      base
# tinynet-a          qat               base            self-distil
tinynet-a          base              structured      self-distil
# tinynet-a          qat               structured      self-distil
