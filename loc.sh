find src/main.cpp src/framework src/implementation | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  wc -l | \
  xargs echo Total lines:

find src/main.cpp src/framework src/implementation | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  sed '/^\s*$/d' | \
  wc -l | \
  xargs echo Non-blank lines:

find src/main.cpp src/framework src/implementation | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  sed '/^\s*\(#\|\/\/\|\/\*\|\*\)/d;/^\s*$/d' | \
  wc -l | \
  xargs echo Lines of code:

find src/main.cpp src/framework src/implementation | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  sed '/^\s*\(#\|\/\/\|\/\*\|\*\)/!d;/^\s*$/d' | \
  wc -l | \
  xargs echo Comments and preprocessor directives: 
