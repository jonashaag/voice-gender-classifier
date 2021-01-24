#!/bin/bash -eu
for l in `cat commonvoice-locales`; do
  wget "$(./commonvoice-geturl.sh $l)" -O - | pv -Ss 500M | tar xfz -
done
