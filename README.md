# Gaussian boson sampling with matrix-product states

Julia implementation of the tensor-network-based algorithm for simulating
Gaussian boson sampling experiments proposed in [[1]](#1), and of the algorithm
in [[2]](#2) for computing elements of Gaussian operations in the Fock number
basis.

The functions for computing hafnians and loop hafnians of square matrices are
ported from _The Walrus_ [[3]](#3).

## References

<a id="1">[1]</a>
Changun Oh, Minzhao Liu, Yuri Alexeev, Bill Ferrerman and Liang Jiang,
‘Classical algorithm for simulating experimental Gaussian boson sampling’.
[_Nature Physics_ 20, 1461–1468 (2024)](https://doi.org/10.1038/s41567-024-02535-8)

<a id="2">[2]</a>
Nicolás Quesada.
‘Franck-Condon factors by counting perfect matchings of graphs with loops’.
[_The Journal of Chemical Physics_ 150.16
(2019)](https://doi.org/10.1063/1.5086387)

<a id="3">[3]</a>
Brajesh Gupt, Josh Izaac and Nicolás Quesada.
‘The Walrus: a library for the calculation of hafnians, Hermite polynomials and
Gaussian boson sampling.’
[_Journal of Open Source Software_, 4(44), 1705
(2019)](https://joss.theoj.org/papers/10.21105/joss.01705)
