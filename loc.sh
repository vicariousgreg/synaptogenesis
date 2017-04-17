find src/core | \
  grep "\(\.h\|\.cpp\)" | grep -v "main\.cpp" | \
  xargs cloc

find src/ui | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cloc

find src/core | \
  grep "\(main\.cpp\)" | \
  xargs cloc
