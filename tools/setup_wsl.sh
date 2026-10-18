#!/bin/bash
# One-time WSL setup for the match harness: Node 24 (for the site-rules referee) and its npm deps.
set -e
NODE_VER=v24.22.0
NODE_SHA=cc2f601cee82d9a5a233302fe307abc600727035fd7d2c3a3571755989e95008
ROOT=/mnt/c/Users/honza/Desktop/InfiniteChess

mkdir -p ~/.local
if [ ! -x ~/.local/node/bin/node ]; then
  cd ~/.local
  curl -sSfLO https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-linux-x64.tar.xz
  echo "$NODE_SHA  node-$NODE_VER-linux-x64.tar.xz" | sha256sum -c -
  tar xf node-$NODE_VER-linux-x64.tar.xz
  rm node-$NODE_VER-linux-x64.tar.xz
  mv node-$NODE_VER-linux-x64 node
fi
grep -q '.local/node/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/node/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/node/bin:$PATH"
node --version

# The site's shared rules code imports these; installing them in ref/ lets Node resolve them from ref/site/src.
cd $ROOT/ref
npm install --no-fund --no-audit
