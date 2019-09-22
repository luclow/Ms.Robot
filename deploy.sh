#!/usr/bin/env bash
# Deployment script with two options:
# 1) Can be used to deploy ALL demos --> ./deploy.sh
# 2) Pass single argument for single demo deployment --> ./deploy.sh fashion-mnist
#  - `yarn build` script generates a dist/ folder in the repo directory

if [ -z "$1" ]
  then
    EXAMPLES=$(ls -d */)
else
  EXAMPLES=$1
  if [ ! -d "$EXAMPLES" ]; then
    echo "Error: Could not find example $1"
    echo "Make sure the first argument to this script matches the example dir"
    exit 1
  fi
fi

for i in $EXAMPLES; do
  cd ${i}
  # Strip any trailing slashes.
  EXAMPLE_NAME=${i%/}

  echo "building ${EXAMPLE_NAME}..."
  yarn
  rm -rf dist .cache
  yarn build
  # Remove files in the example directory (but not sub-directories).
  gsutil -m rm gs://tfjs-examples/$EXAMPLE_NAME/*
  # Gzip and copy all the dist files.
  # The trailing slash is important so we get $EXAMPLE_NAME/dist/.
  gsutil -m cp -Z -r dist gs://tfjs-examples/$EXAMPLE_NAME/
  cd ..
done
