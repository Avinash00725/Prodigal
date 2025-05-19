from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load("resnet18.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("resnet18_savedmodel_tf")
