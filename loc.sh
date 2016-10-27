find src | \
  grep "\(\.h\|\.cpp\|.py\)" | \
  grep -v "CImg" | 
  xargs cloc
