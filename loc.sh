echo ==========
echo Core:
find src/core | \
  grep "\(\.h\|\.cpp\)" | grep -v "main\.cpp" | \
  xargs cloc

echo ==========
echo UI:
find src/ui | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cloc

echo ==========
echo Main:
find src/core | \
  grep "\(main\.cpp\)" | \
  xargs cloc

echo ==========
echo ==========
echo Total:
find src/core src/ui | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cloc
