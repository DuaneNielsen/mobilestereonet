import torch
from models import __models__
from torch import nn
from pathlib import Path
from argparse import ArgumentParser

class Split(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapped_model = model

    def forward(self, img):
        left, right = torch.chunk(img, 2, dim=3)
        return self.wrapped_model(left, right)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
    parser.add_argument('--model', default='MSNet2D', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    args = parser.parse_args()

    # model, optimizer
    model = __models__[args.model](args.maxdisp)
    model = nn.DataParallel(model)
    model.to(torch.float32)

    # load parameters
    print("Loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])

    checkpt_path = Path(args.loadckpt)
    onnx_file = checkpt_path.parent / Path(checkpt_path.stem + '.onnx')

    # set the model to inference mode
    model.eval()

    # make the model take a single input of left and right images concatenated to create a single wide image
    model = Split(model.module)

    # Let's create a dummy input tensor
    dummy_left = torch.ones(1, 3, 384, 1248).float()
    dummy_right = torch.ones(1, 3, 384, 1248).float()
    dummy_img = torch.cat((dummy_left, dummy_right), dim=3)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_img,  # model input (or a tuple for multiple inputs)
                      str(onnx_file),  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['disparity'])  # the model's output names
    # dynamic_axes={'modelInput': {1: 'batch_size'},  # variable length axes
    #               'modelOutput': {1: 'batch_size'}})
    print("*************************************")
    print(f'Saving onnx model to {onnx_file}')
    print("*************************************")
    print("to convert onnx model to tensorRT engine")
    print(f"trtexec --onnx={checkpt_path.stem}.onnx --explicitBatch --workspace=2048 --saveEngine={checkpt_path.stem}.engine")

