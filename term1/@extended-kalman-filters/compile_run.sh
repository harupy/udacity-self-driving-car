srcfile=$1
outfile=${srcfile//.cpp/}
g++ "$1" -o "$outfile" -I /usr/local/include/eigen3
./"$outfile"