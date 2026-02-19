export IP=86.38.238.149


rsync -av \
  --exclude='*.pt' \
  --exclude='sam3_logs/' \
  --exclude='/data/' \
  --exclude='.venv/' \
  --exclude='.git/' \
  /home/frank/Desktop/AGENT/sam3/ \
  root@$IP:/sam3


rsync -av \
  --include='*/' \
  --include='*.png' \
  --include='*.pth' \
  --exclude='*' \
  root@$IP:/sam3/sam3_logs/ \
  /home/frank/Desktop/AGENT/sam3/sam3_logs/
