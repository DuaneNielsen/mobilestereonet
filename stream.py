from argparse import ArgumentParser
from gstreamer import GstCommandPipeline
import cv2
import time
from gi.repository import Gst


class DeepstreamPreProcessor(GstCommandPipeline):
    def __init__(self, data_dir, height, width, start_index=0):
        self.data_dir = data_dir
        self.counter = start_index
        self.height, self.width = height, width
        super().__init__(
            f'multifilesrc location = {self.data_dir}/frame_%05d.png start-index={start_index} '
            f'caps=image/png,framerate=(fraction)12/1 '
            f'! pngdec '
            f'! videoconvert '
            f'! videorate '
            f'! nvvideoconvert '
            f'! m.sink_0 nvstreammux name=m batch-size=1 width={width} height={height} nvbuf-memory-type=2 '
            #f'! nvdspreprocess config-file=roll_classifier_prepro.txt enable=1 '
            f'! nvinfer name=nvinfer config-file-path=checkpoints/MSNet2D/infer.txt '
            f'! nvvideoconvert '
            f'! nveglglessink'
        )

    def on_pipeline_init(self) -> None:
        pass
        # prepro = self.get_by_name('nvdspreprocess0')
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


