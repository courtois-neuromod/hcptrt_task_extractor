# INSTALL

pip install -r requirements.txt

# EXTRACT HCPTRT TASK

```
usage: extract_hcptrt.py [-h] [-v] in_file in_task out_file

    Convert txt data from eprime to tsv using convert_eprime.
    git@github.com:tsalo/convert-eprime.git

    HCPTRT tasks
    https://github.com/hbp-brain-charting/public_protocols

positional arguments:
  in_file     BIDS folder to convert.
  in_task     task you want to convert (wm, emotion, gambling...).
  out_file    output tsv file.

optional arguments:
  -h, --help  show this help message and exit
  -v          If set, produces verbose output.
```

# WRITE YOUR OWN CONFIG FILE
