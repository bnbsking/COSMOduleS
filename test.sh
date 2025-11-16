# /bin/bash

projectPath="/home/james/Desktop/code/COSMOduleS"

docker run -it --rm \
  -v "${projectPath}:/app" \
  -w /app \
  --name lapp \
  3.12-slim \
  /bin/bash
