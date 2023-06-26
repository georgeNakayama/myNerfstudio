#!/bin/bash

for i in 500 2500 5000
do 
    for lr in 0.01 0.1
    do
        ns-train cimle-nerfacto --pipeline.model.cimle-cache-interval $i --optimizers.cimle.optimizer.lr $lr --pipeline.model.pretrained_path pretrained/step-000012000.ckpt  --optimizers.valid-param-groups.valid-pgs cimle --experiment_name=fern_from_pretrained  llff --data data/llff/fern --train_ratio 0.2
    done
done 

echo "Done"

ns-train cimle-nerfacto --pipeline.model.pretrained_path pretrained/step-000012000.ckpt  --optimizers.valid-param-groups.valid-pgs="" --experiment_name=test_fern_from_pretrained  llff --data data/llff/fern --train_ratio 0.2