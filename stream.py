from argparse import ArgumentParser
from gstreamer import GstCommandPipeline
import cv2
import time
from gi.repository import Gst
import pyds
from pyds import NvDsMetaType
import numpy as np
import ctypes


def probe_user_output_meta(pad, info):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_user = batch_meta.batch_user_meta_list

    print(l_user)

    while l_user is not None:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        base_meta = pyds.NvDsBaseMeta.cast(user_meta.base_meta)

        if base_meta.meta_type == NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            array = np.array(np.ctypeslib.as_array(ptr, shape=(layer.dims.numElements,)), copy=True)
            print(array.shape)

        try:
            l_user = l_user.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


class DeepstreamPreProcessor(GstCommandPipeline):
    def __init__(self, data_dir, height, width, start_index=0):
        self.data_dir = data_dir
        self.counter = start_index
        self.height, self.width = height, width
        super().__init__(
            f'multifilesrc location = {self.data_dir}/frame_%05d.png start-index={start_index} '
            f'caps=image/png,framerate=(fraction)1/1 '
            f'! pngdec '
            f'! videoconvert '
            f'! videorate '
            f'! nvvideoconvert '
            f'! m.sink_0 nvstreammux name=m batch-size=1 width={width} height={height} nvbuf-memory-type=2 '
            #f'! nvdspreprocess config-file=roll_classifier_prepro.txt enable=1 '
            f'! nvinfer name=nvinfer config-file-path=checkpoints/MSNet2D/infer.txt raw-output-file-write=1 '
            f'! nvvideoconvert name=nvvideoconvert1 '
            f'! nveglglessink'
        )

    def on_pipeline_init(self) -> None:
        nvvideoconvert1 = self.get_by_name('nvvideoconvert1')
        if nvvideoconvert1 is not None:
            nvvideoconvert1_sink = nvvideoconvert1.get_static_pad('sink')
            nvvideoconvert1_sink.add_probe(Gst.PadProbeType.BUFFER, probe_user_output_meta)
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


