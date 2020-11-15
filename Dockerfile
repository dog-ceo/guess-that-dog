FROM tensorflow/tensorflow:latest

WORKDIR /train

COPY ./images /train/images

RUN pip install tensorflow-hub[make_image_classifier]
RUN pip install tensorflowjs[wizard]

RUN make_image_classifier --image_dir /train/images --tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 --image_size 224 --saved_model_dir data/model/1 --labels_output_file data/class_labels.txt

RUN tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model data/model/1 web/model/1
