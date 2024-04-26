# Homework 1
## Matrix Transposition

## Compile
From the source directory, compile the executable with:
```
make all
```
You can also make some changes to the `Makefile`, for example you can change the optimization flag *OPT_FLAG*.
If you have already built the executable you can run:
```
make clean; make all
```
to remove the previous binary and build the new one

## Usage
The default usage is as follows:
<pre>
./transpose <i>n</i>
</pre>
where *n* will be used to determine the size of the matrix as 2<sup>n</sup> x 2<sup>n</sup>. <br>
Also the <b>Na&iuml;ve algorithm</b> will be the default one. <br>
To use the blocked algorithm instead, use:
<pre>
./transpose <i>n</i> --block=yes
</pre>

<b>Be aware:</b> since in the main function there are some print statements that will allow you to state the transpose functions work, keep in mind that with big matrices there could be some problems with the visualization.
