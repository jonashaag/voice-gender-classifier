#!/bin/bash -eu
curl "https://commonvoice.mozilla.org/api/v1/bucket/dataset/cv-corpus-6.1-2020-12-11%2F$1.tar.gz/true" | \
  cut -d '"' -f4
