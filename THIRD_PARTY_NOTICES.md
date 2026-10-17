# Third-party notices

No third-party source code is redistributed in this repository. The repositories below are cloned at pinned commits by
`src/third_party/fetch.sh` (list: `src/third_party_VERSIONS.txt`) and keep their own licences. A line-level comparison
of every file in `src/` against all five repositories at those commits finds no copied code apart from trivial lines
(imports, `if __name__ == "__main__":`, one one-line expression).

| Repository | Commit | Licence | How it is used here |
|---|---|---|---|
| [Poirazi-Lab/dendritic_anns](https://github.com/Poirazi-Lab/dendritic_anns) | fadc384 | GPL-3.0 | Optional runtime dependency, fetched separately and not distributed: `opt.make_masks` (dANN-LRF mask construction) is imported at run time by `src/t1/dann_fmnist.py` and `src/mcu/common.py` (path: `src/third_party/dendritic_anns/codes_` or `$DENDRITIC_ANNS`). Only these two code paths need it; nothing of it is copied, and the network itself is re-written in PyTorch. |
| [numenta/nupic.research](https://github.com/numenta/nupic.research) | 7a287be | AGPL-3.0 | Reference only: `src/t1/active_dendrites.py` and `src/t6/ad_cache.py` are independent re-implementations of the published active-dendrites model (Iyer et al., 2022), checked against this code for hyper-parameters and initialisation. Nothing is imported or copied. |
| [eva1801/DH-SNN](https://github.com/eva1801/DH-SNN) | 79ab1fb | MIT | Reference for the PyTorch port of DH-SRNN / LIF-SRNN in `src/t1/shd.py` (same model definitions and training recipe, re-written; deviations listed in its docstring). Not imported. Notice below. |
| [robodhruv/visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) | dca7981 | MIT | Imported at run time by `src/t4a/*` (NoMaD / ViNT / GNM model classes, `filter_backwards`, `quat_to_yaw`); `src/t4a/process_recon.py`, `process_huron.py` and `build_bench.py` follow its data-processing logic. Notice below. |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 5ba07ac | MIT | Imported at run time by `src/t4a/common.py` (`ConditionalUnet1D`); not copied. |

**dendritic_anns (GPL-3.0) is an optional runtime dependency.** It is not part of this repository and is not
distributed with it; `src/third_party/fetch.sh` clones it from its upstream repository on the user's machine, and it is
only needed to build the dANN-LRF masks (T1 G1 and the 8051 dANN models). Users who obtain and run the GPL-3.0 and
AGPL-3.0 code do so under those licences.

`src/t4a/lowlight.py` re-implements the synthetic low-light model of our earlier release
https://github.com/zjianwu261/llie-robot-bench (MIT, same authors); nothing is imported from it.

Datasets and weights are not redistributed: Fashion-MNIST, MNIST, Spiking Heidelberg Digits (SHD), RECON,
SACSoN/HuRoN and the public NoMaD checkpoint are obtained from their original sources under their own terms.

## DH-SNN — port in `src/t1/shd.py` (and the DH-SNN surrogate / readout reused in `src/t6/dh_ssm.py`)

MIT License

Copyright (c) 2023 eva1801

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## visualnav-transformer — data processing in `src/t4a/process_recon.py`, `process_huron.py`, `build_bench.py`

MIT License

Copyright (c) 2023 Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, Sergey Levine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
