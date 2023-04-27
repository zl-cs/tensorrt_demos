# !/bin/bash

model_list="yolov3-tiny-288 yolov3-tiny-416 yolov3-288 yolov3-416 yolov3-608 yolov4-tiny-288 yolov4-tiny-416 yolov4-tiny-608 yolov4-288 yolov4-416 yolov4-608"

# cd /home/nvidia/software/tensorrt_demos/yolo
# for model in $model_list
# do
#     echo "==================== $model ===================="
#     python3 yolo_to_onnx.py -m $model 
# done

cd /home/nvidia/software/tensorrt_demos/yolo
for model in $model_list
do
    echo "==================== $model ===================="
    python3 onnx_to_tensorrt.py -m $model
done
