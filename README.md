torchrun --nnodes=1 --nproc_per_node=1 .\src\animatediff\__main__.py train animatediff -w -c config\training\training_ad.yaml
torchrun --nnodes=1 --nproc_per_node=1 .\src\animatediff\__main__.py train motionpredictor -w -c config\training\training_mp.yaml
python .\src\animatediff\__main__.py generate -c config\prompts\IPAImageTest.json  

python ./src/animatediff/__main__.py generate -c config/prompts/IPAImageTest.json 

torchrun --nnodes=1 --nproc_per_node=1 ./src/animatediff/__main__.py train motionpredictor -w -c config/training/training_mp.yaml
python .\src\animatediff\__main__.py generate -c config\prompts\IPAImageTestCow.json

python ./src/animatediff/__main__.py generate -c config/prompts/IPAImageTestCow.json
python ./src/animatediff/__main__.py generate -c config/prompts/ValidationCow.json
python ./src/animatediff/__main__.py generate -c config/prompts/IPAVideoTest.json
python ./src/animatediff/__main__.py train motionpredictor -w -c config/training/training_mp.yaml
python ./src/animatediff/__main__.py train animatediff -w -c config/training/training_ad.yaml

python ./src/animatediff/__main__.py filter --input /workspace/animatediff-cli/training_ad-2024-09-07T16-50-25_animatediff.pth --output data/models/motion-module/0013-24c-24fps-512-b1-img-rope.pth

python ./src/animatediff/__main__.py generate -c config/prompts/ValidationCowRealVision.json -W 256 -H 256 -L 24
python ./src/animatediff/__main__.py generate -c config/prompts/IPAVideoTest.json -W 256 -H 256 -L 64


python ./src/animatediff/__main__.py generate -c config/prompts/ValidationCowRealVision.json -W 512 -H 512 -L 16
python ./src/animatediff/__main__.py generate -c config/prompts/IPAImageTest.json -W 512 -H 512 -L 256

python ./src/animatediff/__main__.py generate -c 0000
python ./src/animatediff/__main__.py train animatediff -w -c config/training/training_ad.yaml
python ./src/animatediff/__main__.py train animatediff -w -c config/training/image_finetune.yaml

apt update -y
apt upgrade -y
apt install sudo -y
apt install ffmpeg -y
apt install iftop -y
apt-get install git-lfs -y
apt install unzip -y

# Copy aws.zip to ~ and unzip
# Copy secrets.dat to ~
pip install wandb; source ~/secrets.dat; wandb login $WANDB_API_KEY;

sudo /workspace/aws/install
cd animatediff-cli
source .venv/bin/activate


git clone https://github.com/neggles/animatediff-cli
cd animatediff-cli
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[dev]'

# official PPA comes with ffmpeg 2.8, which lacks tons of features, we use ffmpeg 4.0 here
sudo add-apt-repository ppa:jonathonf/ffmpeg-4 # for ubuntu20.04 official PPA is already version 4.2, you may skip this step
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake
sudo apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
# note: make sure you have cmake 3.8 or later, you can install from cmake official website if it's too old

cd /workspace
git clone --recursive https://github.com/dmlc/decord

ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/local/cuda/libnvcuvid.so
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release 
make
cd ../python
pip install wheel
pip install .



cd /workspace/animatediff-cli/data/models/sd
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
cd /workspace/animatediff-cli/data/models/
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
git clone https://huggingface.co/h94/IP-Adapter





# animatediff
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/neggles/animatediff-cli/main.svg)](https://results.pre-commit.ci/latest/github/neggles/animatediff-cli/main)

animatediff refactor, ~~because I can.~~ with significantly lower VRAM usage.

Also, **infinite generation length support!** yay!

# LoRA loading is ABSOLUTELY NOT IMPLEMENTED YET!

PRs welcome! 😆😅

This can theoretically run on CPU, but it's not recommended. Should work fine on a GPU, nVidia or otherwise,
but I haven't tested on non-CUDA hardware. Uses PyTorch 2.0 Scaled-Dot-Product Attention (aka builtin xformers)
by default, but you can pass `--xformers` to force using xformers if you *really* want.

## How to use

I should write some more detailed steps, but here's the gist of it:

```sh
git clone https://github.com/neggles/animatediff-cli
cd animatediff-cli
python3.10 -m venv .venv
source .venv/bin/activate
# install Torch. Use whatever your favourite torch version >= 2.0.0 is, but, good luck on non-nVidia...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install the rest of all the things (probably! I may have missed some deps.)
python -m pip install -e '.[dev]'
# you should now be able to
animatediff --help
# There's a nice pretty help screen with a bunch of info that'll print here.
```

From here you'll need to put whatever checkpoint you want to use into `data/models/sd`, copy
one of the prompt configs in `config/prompts`, edit it with your choices of prompt and model (model
paths in prompt .json files are **relative to `data/`**, e.g. `models/sd/vanilla.safetensors`), and
off you go.

Then it's something like (for an 8GB card):
```sh
animatediff generate -c 'config/prompts/waifu.json' -W 576 -H 576 -L 128 -C 16
```
You may have to drop `-C` down to 8 on cards with less than 8GB VRAM, and you can raise it to 20-24
on cards with more. 24 is max.

N.B. generating 128 frames is _**slow...**_

## RiFE!

I have added experimental support for [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)
using the `animatediff rife interpolate` command. It has fairly self-explanatory help, and it has
been tested on Linux, but I've **no idea** if it'll work on Windows.

Either way, you'll need ffmpeg installed on your system and present in PATH, and you'll need to
download the rife-ncnn-vulkan release for your OS of choice from the GitHub repo (above). Unzip it, and
place the extracted folder at `data/rife/`. You should have a `data/rife/rife-ncnn-vulkan` executable, or `data\rife\rife-ncnn-vulkan.exe` on Windows.

You'll also need to reinstall the repo/package with:
```py
python -m pip install -e '.[rife]'
```
or just install `ffmpeg-python` manually yourself.

Default is to multiply each frame by 8, turning an 8fps animation into a 64fps one, then encode
that to a 60fps WebM. (If you pick GIF mode, it'll be 50fps, because GIFs are cursed and encode
frame durations as 1/100ths of a second).

Seems to work pretty well...

## TODO:

In no particular order:

- [x] Infinite generation length support
- [x] RIFE support for motion interpolation (`rife-ncnn-vulkan` isn't the greatest implementation)
- [x] Export RIFE interpolated frames to a video file (webm, mp4, animated webp, hevc mp4, gif, etc.)
- [x] Generate infinite length animations on a 6-8GB card (at 512x512 with 8-frame context, but hey it'll do)
- [x] Torch SDP Attention (makes xformers optional)
- [x] Support for `clip_skip` in prompt config
- [x] Experimental support for `torch.compile()` (upstream Diffusers bugs slow this down a little but it's still zippy)
- [x] Batch your generations with `--repeat`! (e.g. `--repeat 10` will repeat all your prompts 10 times)
- [x] Call the `animatediff.cli.generate()` function from another Python program without reloading the model every time
- [x] Drag remaining old Diffusers code up to latest (mostly)
- [ ] Add a webUI (maybe, there are people wrapping this already so maybe not?)
- [ ] img2img support (start from an existing image and continue)
- [ ] Stop using custom modules where possible (should be able to use Diffusers for almost all of it)
- [ ] Automatic generate-then-interpolate-with-RIFE mode

## Credits:

see [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) (very little of this is my work)

n.b. the copyright notice in `COPYING` is missing the original authors' names, solely because
the original repo (as of this writing) has no name attached to the license. I have, however,
used the same license they did (Apache 2.0).
