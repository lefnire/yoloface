import dlib, cv2, sys, os, time, pdb, boto3
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
from pathlib import Path

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)

# OpenVINO stuff
OPENVINO_DIR = f'{Path.home()}/intel/computer_vision_sdk/deployment_tools'
# MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.
# OPENVINO_CPU_SO = f"{OPENVINO_DIR}/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so"
OPENVINO_CPU_SO = f'{Path.home()}/inference_engine_samples/intel64/Release/lib/libcpu_extension.so'
# Path to a plugin folder
PLUGIN_DIR = None

class Model(object):
    color = GREEN

    def __init__(self, args):
        self.args = args

    def draw_boxes(self, frame):
        raise NotImplementedError("This model not implemented (override in class)")


# --- OpenVINO ---
class OpenVINO_Model(Model):
    def __del__(self):
        del self.exec_net
        del self.plugin
        del self

class OpenVINO_YOLO(OpenVINO_Model):
    def __init__(self, args):
        super().__init__(args)
        device = 'MYRIAD' if 'MYRIAD' in args.device else args.device
        fp = 'FP16' if device == 'MYRIAD' else 'FP32'
        #xml += f'/yolo_v3/{fp}/fa.xml'
        xml += f'./yolo_v3.xml'
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        model_xml = xml
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        log.info("Initializing plugin for {} device...".format(device))
        plugin = IEPlugin(device=device, plugin_dirs=PLUGIN_DIR)
        if 'CPU' in device:
            plugin.add_cpu_extension(OPENVINO_CPU_SO)

        # Read IR
        log.info("Reading IR...")
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)

        if "CPU" in plugin.device:
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(plugin.device, ', '.join(not_supported_layers)))
                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        log.info("Loading IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=2)
        # Read and pre-process input image
        self.dims = net.inputs[self.input_blob].shape
        self.plugin = plugin
        del net

        # <removed labelMap>

        self.cur_request_id = 0
        self.next_request_id = 1
        # <removed async management>
        self.is_async_mode = True
        self.render_time = 0

    def draw_boxes(self, frame):
        args = self.args
        n, c, h, w = self.dims
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = time.time()
        if self.is_async_mode:
            self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: in_frame})
        found_faces = False
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    found_faces = True
                    xmin = int(obj[3] * args.init_w)
                    ymin = int(obj[4] * args.init_h)
                    xmax = int(obj[5] * args.init_w)
                    ymax = int(obj[6] * args.init_h)
                    # <removed class(color,label)>
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.color, 2)

            # <removed drawing performance stats>

        render_start = time.time()
        # cv2.imshow("Detection Results", frame)
        render_end = time.time()
        self.render_time = render_end - render_start
        self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
        return found_faces
