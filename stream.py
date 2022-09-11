from argparse import ArgumentParser
from gstreamer import GstCommandPipeline
import cv2
import time
from gi.repository import Gst
import pyds
from pyds import NvDsMetaType, NvDsInferDataType
import numpy as np
import ctypes
import sys
sys.path.append('/snap/pycharm-community/293/plugins/python-ce/helpers/pydev/')
from pydevd_pycharm import settrace


class Gprobe:
    """
    write a function in the form

    def fname(numpy_array, layer_info, tensor_meta):

    it need not return anything

    to process data attach to ...

    probe_frame_meta_list: to process frame_meta_list items

    probe_user_output_meta: to process user_output_meta

    example:

        def draw_array(array, layer_info, tensor_meta):
            norm_image = cv2.normalize(array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
            cv2.imshow('hello', norm_image)
            cv2.waitKey(1)

        probe = Gprobe(probe_frame_meta_list=draw_array)
        nvvideoconvert1_sink.add_probe(Gst.PadProbeType.BUFFER, probe)

    """
    def __init__(self, probe_frame_meta_list=None, probe_user_output_meta=None):
        self.probe_frame_meta_list = probe_frame_meta_list
        self.probe_user_output_meta = probe_user_output_meta

    @staticmethod
    def convert_type(type):
        if type == NvDsInferDataType.FLOAT:
            return ctypes.c_float
        if type == NvDsInferDataType.INT32:
            return ctypes.c_int32
        if type == NvDsInferDataType.HALF:
            # we probably need to use pytorch instead of numpy here
            raise NotImplemented
        if type == NvDsInferDataType.INT8:
            return ctypes.c_int8

    @staticmethod
    def unpack(tensor_meta):
        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, 0)
        shape = tuple(layer_info.dims.d[:layer_info.dims.numDims])
        ptr = ctypes.cast(pyds.get_ptr(layer_info.buffer), ctypes.POINTER(Gprobe.convert_type(layer_info.dataType)))
        return np.array(np.ctypeslib.as_array(ptr, shape=shape), copy=True), layer_info, tensor_meta

    def __call__(self, pad, info):
        gst_buffer = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        """ run probe on the meta_list if it's not one """
        if self.probe_frame_meta_list is not None:
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    frame_user_meta = frame_meta.frame_user_meta_list

                    while frame_user_meta is not None:
                        user_meta = pyds.NvDsUserMeta.cast(frame_user_meta.data)
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        array, layer_info, tensor_meta = Gprobe.unpack(tensor_meta)
                        self.probe_frame_meta_list(array, layer_info, tensor_meta)
                        try:
                            frame_user_meta = frame_user_meta.next
                        except StopIteration:
                            break

                    l_frame = l_frame.next
                except StopIteration:
                    break

        if self.probe_user_output_meta is not None:
            l_user = batch_meta.batch_user_meta_list

            while l_user is not None:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                base_meta = pyds.NvDsBaseMeta.cast(user_meta.base_meta)

                if base_meta.meta_type == NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    array, layer_info, tensor_meta = Gprobe.unpack(tensor_meta)
                    self.probe_user_output_meta(array, layer_info, tensor_meta)

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

        return Gst.PadProbeReturn.OK


def draw_array(array, layer_info, tensor_meta):
    norm_image = cv2.normalize(array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)
    cv2.imshow('hello', norm_image)
    cv2.waitKey(1)


class DeepstreamPreProcessor(GstCommandPipeline):
    def __init__(self, data_dir, height, width, start_index=0):
        self.data_dir = data_dir
        self.counter = start_index
        self.height, self.width = height, width
        super().__init__(
            f'multifilesrc location = {self.data_dir}/frame_%05d.png start-index={start_index} '
            f'caps=image/png,framerate=(fraction)4/1 '
            f'! pngdec '
            f'! videoconvert '
            f'! videorate '
            f'! nvvideoconvert '
            f'! m.sink_0 nvstreammux name=m batch-size=1 width={width} height={height} nvbuf-memory-type=2 '
            #f'! nvdspreprocess config-file=roll_classifier_prepro.txt enable=1 '
            f'! nvinfer name=nvinfer config-file-path=checkpoints/MSNet2D/infer.txt raw-output-file-write=0 '
            f'! nvvideoconvert name=nvvideoconvert1 '
            f'! nveglglessink'
        )

    def on_pipeline_init(self) -> None:
        nvvideoconvert1 = self.get_by_name('nvvideoconvert1')
        if nvvideoconvert1 is not None:
            nvvideoconvert1_sink = nvvideoconvert1.get_static_pad('sink')
            probe = Gprobe(probe_frame_meta_list=draw_array)
            nvvideoconvert1_sink.add_probe(Gst.PadProbeType.BUFFER, probe)
        # nvinfer = self.get_by_name('nvinfer')
        # if nvinfer is not None:
        #     print('adding src pad probe to nvinfer')
        #     nvinfer_src = nvinfer.get_static_pad('src')
        #     nvinfer_src.add_probe(Gst.PadProbeType.BUFFER, probe_user_output_meta)
        # if prepro is not None:
        #     prepro_src = prepro.get_static_pad('src')
        #     prepro_src.add_probe(Gst.PadProbeType.BUFFER, probe_nvdspreprocess_pad_src_data, self)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datapath', required=True, help='path to directory containing frame_00001.png')
    args = parser.parse_args()

    # discover the height and width
    img = cv2.imread(f'{args.datapath}/frame_00001.png')
    height, width = img.shape[0], img.shape[1]

    with DeepstreamPreProcessor(args.datapath, height, width) as pipeline:
        while not pipeline.is_done:
            time.sleep(1)
            # message = pipeline.bus.timed_pop_filtered(10000, Gst.MessageType.EOS)
            # if message is not None:
            #    break


