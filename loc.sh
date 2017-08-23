echo ==============
echo === Core: ====
echo ==============
find src/core | \
  grep "\(\.h\|\.cpp\)" | grep -v "main\.cpp" | \
  xargs cloc

echo
echo ==============
echo === UI: ======
echo ==============
find src/ui | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cloc

echo
echo ==============
echo === Main: ====
echo ==============
find src/core | \
  grep "\(main\.cpp\)" | \
  xargs cloc

echo
echo ===============
echo ===============
echo === Total: ====
echo ===============
echo ===============
find src/core src/ui | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cloc
