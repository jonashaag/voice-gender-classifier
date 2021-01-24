# Voice gender classifier 

Around 95% validation accuracy.

## To use

- Install requirements (see `requirements.txt`)
- Run `python infer.py FILES...`:

  ```
  $ python infer.py male.wav female.mp3
  Male probabilities:
  0.999 male.wav
  0.104 female.mp3
  ```

## To train

- Install requirements (see `requirements.txt`)
- Install `pv`
- Download data from Common Voice using `./commonvoice-download.sh`.
  By default the script downloads only 500 MiB from each language (that's more
  than enough to train a good classifier).
- Run `python genderclassifier.py`.
