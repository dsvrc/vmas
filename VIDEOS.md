# Videos: train on the server, render on the laptop

Rendering on a compute node does not work and is not worth fixing. VMAS draws
through pyglet, and pyglet needs an X or EGL context to make an OpenGL surface
*even for an invisible window* — `rgb_array` mode still constructs
`pyglet.window.Window(visible=False)`. A compute node has no display server, no
`xvfb-run`, and no GL driver to fall back on, so a render there dies in the GL
layer before any Python code of ours runs. Nothing about "I don't need to watch
it" removes that requirement: it is the driver that needs somewhere to draw.

So the work is split, and neither half needs a video dependency on the cluster:

| where | does | needs |
| --- | --- | --- |
| server | trains, writes checkpoints | nothing extra |
| server | `scripts/pack_checkpoints.sh` | `tar` |
| laptop | `scripts/make_videos.py` | torch, torchrl, tensordict, vmas, this repo |

## 1. On the server: train

Every launcher now leaves a run renderable. No flags to remember:

```bash
bash scripts/run_checkpoints.sh          # just the two arms a clip is wanted of
bash scripts/ns_main.sh                  # the PACT ladder
bash scripts/run_baselines.sh            # the baseline table
bash scripts/run_extra_baselines.sh      # X1..X6
```

What that guarantees, per run, is the two files a replay needs and nothing else:

```
<run>/config.pkl                          the task, its kwargs (severity, the
                                          PACT switch, the observation flags),
                                          the algorithm and model configs, the
                                          seed
<run>/checkpoints/checkpoint_<frames>.pt  the weights
```

`config.pkl` is written by BenchMARL at startup and always was. The three
settings that were added to every launcher are:

- **`checkpoint_interval`** — `CKPT_EVERY`, default 600 000 frames, so a 3M run
  leaves six checkpoints. This is also the crash insurance: with
  `checkpoint_at_end` alone, a job killed at 90% of its budget leaves nothing to
  replay. The value is rounded down to a multiple of the row's own batch by
  `ckpt_interval`, because `ExperimentConfig` rejects anything else — which is
  why the `m3w` rows, at `off_policy_collected_frames_per_batch=3200`, get
  598 400 rather than 600 000.
- **`keep_checkpoints_num=null`** — `KEEP_CKPT`, keep all of them. The first one
  is the "before" of a before/after clip, and a rolling window of three would
  have deleted it by the end of the run.
- **`exclude_buffer_from_checkpoint=true`** — what keeps the files
  downloadable. Off-policy rows would otherwise write the whole replay buffer
  into every `.pt` (up to `off_policy_memory_size` transitions, gigabytes), and
  a replay needs the weights, not the buffer.

Turn the interval off with `CKPT_EVERY=0` (end of run only) or thin it with
`KEEP_CKPT=3`.

## 2. On the server: pack

```bash
bash scripts/pack_checkpoints.sh runs/clips
```

Picks up every folder holding a `config.pkl` with checkpoints beside it, and
tars the `config.pkl` plus the `.pt` files — not the csv logs, not the hydra
output, not `pact_debug.csv`, which is what makes a naive `scp -r` slow. It
prints the `scp` line to run and writes a `MANIFEST_checkpoints.txt` into the
tarball recording the **commit**.

```bash
DRY=1 bash scripts/pack_checkpoints.sh runs/clips        # list, pack nothing
LAST=1 bash scripts/pack_checkpoints.sh runs/clips       # newest ckpt per run
PATTERN=pact bash scripts/pack_checkpoints.sh runs/clips
```

The commit matters as much as the files: `config.pkl` is a pickle, so unpickling
it imports `benchmarl`, the `simple_ns` task class and the algorithm config
dataclass *by name*. Code that has moved on shows up as an `AttributeError` on
load, not as a bad video. Check out the commit in the manifest on the laptop.

## 3. On the laptop: once, set it up

This checkout currently has `torch` and no `torchrl`, so the replay needs one
install. **Match the server's versions**, do not take the newest: the `.pt`
holds torchrl loss `state_dict`s, and parameter key names have moved between
torchrl releases, so a mismatch shows up as missing/unexpected keys on load.
Read the versions off the server and pin them here:

```bash
python -c "import torchrl, tensordict, vmas; print(torchrl.__version__, tensordict.__version__, vmas.__version__)"
```

```bash
pip install "torchrl==<that>" "tensordict==<that>" "vmas==<that>"
```

`setup.py` pins `torchrl>=0.10,<0.12`, but the cluster env is on 0.7.x — the
server's number is the one that counts here, because it wrote the file.

## 4. On the laptop: render

```bash
tar xzf checkpoints_<stamp>.tar.gz
python scripts/make_videos.py runs/clips/balance/sigma3.0/s0/pact_sev3 --out videos
```

Point it at a run folder (its newest checkpoint is used), a row folder, or a
`.pt` directly; several at once is fine. It reloads the experiment with the
pickled task and config, replays one episode of the trained policy, and writes
`<row>_seed<N>_<frames>.mp4`. No training happens.

```bash
python scripts/make_videos.py <row folder> --all       # every checkpoint: before/after
python scripts/make_videos.py <run> --static           # same clip every time
python scripts/make_videos.py <run> --stochastic       # sample, do not take the mode
python scripts/make_videos.py <run> --dry-run          # what would be rendered
```

**It has no video dependency to install.** The frames are captured through the
task's own `render_callback` and written by whichever writer is present —
imageio-ffmpeg, then OpenCV, then Pillow — falling back to a PNG sequence (with
the `ffmpeg` line to join it) and then to an `.npz` of raw frames. A missing
codec costs you the container format, never the render. BenchMARL's own render
path is deliberately left off, because its video goes through the torchrl
logger, which wants `torchvision` and `av`.

Three things are handled for you because they are the ones that bite:

- **GPU tensors.** A cluster run holds `cuda` tensors; `torch.load` raises on a
  machine with no GPU. `restore_map_location` maps them to CPU unless
  `--device cuda` finds one.
- **The server's `save_folder`.** It is baked into the pickled config and does
  not exist on the laptop, and BenchMARL `mkdir`s it with `parents=False`. It is
  repointed at `<out>/_reload`.
- **The 300-worker collector.** A reload rebuilds the training collector and
  loads its state. `collect_with_grad` skips both and the env width drops to
  one, which is what makes this a laptop-sized job rather than a 300-world one.
