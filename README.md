# Chroma Subsampling Before You Upscale: What 4:2:0 Actually Costs

Measured: 4:2:0 leaves luma bit-identical and drops chroma PSNR by 9.7 dB at 2x. Bilinear chroma reconstruction beats lanczos by 0.45 dB. And on smooth-chroma photos, 4:4:4 encoded 2.3% SMALLER than 4:2:0 at the same CRF — with the mechanism measured.

**Read it with the tables and the reproduction snippets:** <https://celticgemini001.github.io/upgolab-chroma-subsampling/>

## About this repo

This repository exists to serve one measured article. Every number in it came from
ffmpeg and numpy run on real files on this machine — the commands are in the page so
you can re-run them and get the same rows. Where the measurement contradicted the
hypothesis, the page says so.

## The tool behind the notes

<https://upgolab.com> — browser-based AI upscaler for photos and short video.

## Index

All the guides are cross-linked from the hub:
<https://celticgemini001.github.io/upgolab-tools/>
