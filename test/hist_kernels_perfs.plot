#!/usr/bin/gnuplot
IF_LATEX = system("echo \$LATEX_PLOTS")
IF_LATEX = IF_LATEX eq "1"
LATEX_TEST = system("echo \$LATEX_TEST")
LATEX_TEST = LATEX_TEST eq "1"
wrap_str(s) = (IF_LATEX ? sprintf("\$%s\$", s) : s)
###
PLOT_NAME = ARGV[1]
PLOT_TITLE = ARGV[2]
FILE_COUNT = (ARGC - 2) / 2
array DATA_FILES[FILE_COUNT]
array DATA_LEGENDS[FILE_COUNT]
do for [i=1:FILE_COUNT] {
    DATA_FILES[i] = ARGV[1+i*2]
    DATA_LEGENDS[i] = ARGV[1+i*2+1]
}
###
if (IF_LATEX) {
    set terminal epslatex color size 10cm, 8cm
    if (LATEX_TEST) {
        plot_path = sprintf("%s.pdf", PLOT_NAME)
    } else {
        plot_path = sprintf("%s.tex", PLOT_NAME)
    }
    set output plot_path
    set ylabel 'Performance [cell/s]' offset 1,0,0
} else {
    set terminal pdfcairo color size 10in, 6in
    plot_path = sprintf("%s.pdf", PLOT_NAME)
    set output plot_path
    set ylabel 'Performance [cell/s]'
}
set title PLOT_TITLE
unset xlabel
set key left top
set style fill solid 1.00 border 0
set xtics rotate by 45 right
set logscale y
set grid
set grid mxtics mytics
set format y wrap_str("10^{%L}")
plot for [i=1:FILE_COUNT] DATA_FILES[i] u 3:xtic(1) skip 1 w histogram t DATA_LEGENDS[i]
