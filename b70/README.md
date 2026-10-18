# B70 branch

vLLM release tag + the B70 changes from [riceharvest/b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook),
one commit each. The cookbook image overlays this branch's changed files on the stock `vllm/vllm-openai-xpu` image; the
Python side of two changes lives on riceharvest/vllm-xpu-kernels `b70/v0.1.15.4`.

Every change is off until its `B70_*` environment variable turns it on (switches: cookbook `docs/patches.md`).

The commits marked `Co-authored-by: SergiioB` port code from
[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
(commit `1a76136ea1b3f96bba3b20e42041a6fcc95f56fd`), MIT License: see `LICENSE.SergiioB`.

New vLLM release: `git rebase --onto vX.Y.Z v0.32.0` into a new branch `b70/vX.Y`, then bump the pins in the cookbook's
`docker/fork.env`.
