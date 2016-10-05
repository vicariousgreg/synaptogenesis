find src | \
  grep "\(\.h\|\.cpp\)" | \
  grep -v "CImg" | 
  xargs cat | \
  wc -l | \
  xargs echo Total lines:

find src | \
  grep "\(\.h\|\.cpp\)" | \
  grep -v "CImg" | 
  xargs cat | \
  sed '/^\s*$/d' | \
  wc -l | \
  xargs echo Non-blank lines:

find src | \
  grep "\(\.h\|\.cpp\)" | \
  grep -v "CImg" | 
  xargs cat | \
  sed '/^\s*\(#\|\/\/\|\/\*\|\*\)/d;/^\s*$/d' | \
  wc -l | \
  xargs echo Lines of code:

find src | \
  grep "\(\.h\|\.cpp\)" | \
  grep -v "CImg" | 
  xargs cat | \
  sed '/^\s*\(#\|\/\/\|\/\*\|\*\)/!d;/^\s*$/d' | \
  wc -l | \
  xargs echo Comments and preprocessor directives: 
