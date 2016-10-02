find -maxdepth 1 -name '[a-z]*' | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  wc -l | \
  xargs echo Total lines:

find -maxdepth 1 -name '[a-z]*' | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  sed '/^\s*$/d' | \
  wc -l | \
  xargs echo Non-blank lines:

find -maxdepth 1 -name '[a-z]*' | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  sed '/^\s*\(#\|\/\/\|\/\*\|\*\)/d;/^\s*$/d' | \
  wc -l | \
  xargs echo Lines of code:

find -maxdepth 1 -name '[a-z]*' | \
  grep "\(\.h\|\.cpp\)" | \
  xargs cat | \
  sed '/^\s*\(#\|\/\/\|\/\*\|\*\)/!d;/^\s*$/d' | \
  wc -l | \
  xargs echo Comments and preprocessor directives: 
