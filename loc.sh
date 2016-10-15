find src | \
  grep "\(\.h\|\.cpp\)" | \
  grep -v "CImg" | 
  xargs cloc
